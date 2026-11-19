#---------------------------------------------------------------------------------------
# Phase-field fracture coupled to (visco-)elastic mechanics on the staggered grid.
#
# Unknowns: Vx, Vy (faces), Pt (centroids) and the crack phase field d ∈ [0,1] (centroids).
# Same layout and machinery as the TwoPhases module: d takes the slot of Pf (4th block).
#
# Model (Miehe et al., 2010, AT2 functional):
#   undamaged (effective) stresses, Maxwell visco-elasticity in incremental form
#       τ̃  = 2ηᵛᵉ (ε̇' + τ̃⁰/(2GΔt)),   ηᵛᵉ = (1/η + 1/(GΔt))⁻¹   (η = Inf → linear elasticity)
#   P̃ :  Δt ∇⋅V + (P̃ - P̃⁰)/K = 0   (increment form)
#   degraded stress
#       σ  = g(d) τ̃ - g(d) P̃ I,      g(d) = (1-d)² + k_res
#       (amor = true: volumetric/deviatoric split of Amor et al. (2009), compressive P̃ > 0 is not degraded)
#   momentum: ∇⋅σ = 0
#   phase field:
#       gc/l d - ∇⋅(gc l ∇d) - 2(1-d) (H + βp Wp) + ηd ∂d/∂t = 0,     H = max(H⁰, ⟨ψ⁺ - ψc⟩)
#       ψ = τ̃:τ̃/(4G) + P̃²/(2K)   (elastic strain energy, tensile part only for amor = true)
#   plasticity (optional, materials.plasticity, e.g. DruckerPrager of StagFDTools, return mapping of the
#   Rheology module): checked on the degraded stress g τ̃ (Hageman, 2026, CMAME 454, 118857), at centroids and
#   vertices (missing components averaged). Plastic work Wp = ∫ τII λ̇ dt drives damage with weight βp
#   (Hageman: β_p(d) = β(1-d) + 1-β; here β(1-d)² + 1-β, consistent with AT2, which keeps d < 1 without a bound).
#   Optional strength threshold ψc (Miehe et al., 2015; ψc = 0: Miehe et al., 2010).
#   damage = false: d evolves (driven by plastic work) but does not degrade the material.
#   Gradient regularisation of plasticity (Kd > 0): F = τII - P sinϕ - C cosϕ - ηvp λ̇ + Kd ∇²d, cf. gradient
#   plasticity (de Borst & Mühlhaus, 1992; Duretz et al., 2023) with d as a proxy of the plastic strain; ∇²d is
#   taken from the phase-field equation (gc l ∇²d = gc/l d - 2(1-d)(H + βp Wp) + ηd ∂d/∂t), so λ̇ stays local.
#
# The full (Vx, Vy, Pt, d) system is solved with Newton's method, local Jacobians are obtained by AD.
#---------------------------------------------------------------------------------------
using Base.Threads, SparseArrays, StaticArrays, LinearAlgebra
import StagFDTools.TwoPhases: PatternBlock, FixedPattern, _target, set_sparsity_pattern!, zero_values!
import StagFDTools.Rheology: return_mapping

struct Fields{Tx,Ty,Tp,Td}
    Vx::Tx
    Vy::Ty
    Pt::Tp
    d::Td
end

@inline function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 5
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
    i == 4 && return x.d
end

SparseArrays.dropzeros(M::Fields) = Fields(ntuple(a -> Fields(ntuple(b -> dropzeros(M[a][b]), 4)...), 4)...)

"""
    pattern_storage()

Per-thread `PatternBlock` storage to pass to the `Assemble*2D!` kernels before calling `set_sparsity_pattern!`.
"""
pattern_storage() = [Fields(ntuple(_ -> Fields(ntuple(_ -> PatternBlock(), 4)...), 4)...) for _ in 1:nthreads()]

"""
    fixed_pattern(M)

Wrap each block of `M` so the `Assemble*2D!` kernels write directly into its nonzeros.
"""
fixed_pattern(M) = Fields(ntuple(a -> Fields(ntuple(b -> FixedPattern(M[a][b]), 4)...), 4)...)

#---------------------------------------------------------------------------------------
# Global Jacobian: a single CSC matrix with a fixed, minimal sparsity pattern
#---------------------------------------------------------------------------------------

# Records only nonzero entries (structural zeros of the stencils are not stored)
struct NonzeroPatternBlock
    I::Vector{Int}
    J::Vector{Int}
end
NonzeroPatternBlock() = NonzeroPatternBlock(Int[], Int[])
@inline function Base.setindex!(A::NonzeroPatternBlock, val, i::Integer, j::Integer)
    iszero(val) && return val
    push!(A.I, i)
    push!(A.J, j)
    return val
end

nonzero_pattern_storage() = [Fields(ntuple(_ -> Fields(ntuple(_ -> NonzeroPatternBlock(), 4)...), 4)...) for _ in 1:nthreads()]

# Block (a,b) of the global matrix: writes accumulate into the existing nonzeros of A
struct GlobalBlock{T<:SparseMatrixCSC}
    A::T
    roff::Int
    coff::Int
end

@inline function Base.setindex!(K::GlobalBlock, val, i::Integer, j::Integer)
    iszero(val) && return val   # zero derivatives (stencil corners, inactive branches) are not stored
    A  = K.A
    ig, jg = i + K.roff, j + K.coff
    rv = rowvals(A)
    r  = nzrange(A, jg)
    k  = first(r) - 1 + searchsortedfirst(view(rv, r), ig)
    (k <= last(r) && rv[k] == ig) || error("entry ($ig, $jg) is not in the sparsity pattern")
    nonzeros(A)[k] += val
    return val
end

"""
    JacobianCache(number, record!)
    JacobianCache(V, P, d, old, rheo, materials, number, type, BC, nc, Δ)

Global Jacobian (single CSC matrix) with the minimal sparsity pattern, and an LU factorisation whose symbolic
analysis is reused. `record!(K)` assembles (possibly several times) into the pattern storage `K`, the union of the
nonzero entries defines the pattern. The second method records it from random states (both pressure signs,
active history branch), so that entries which vanish only for particular states are kept.
"""
struct JacobianCache{TA,TK}
    A::TA
    K::TK
    rng::NTuple{4,UnitRange{Int}}   # rows/columns of the Vx, Vy, Pt, d blocks
    F::Base.RefValue{Any}           # LU factorisation (symbolic analysis reused)
    gcr::Base.RefValue{Any}         # GCR workspace (symbolic Cholesky of the preconditioner reused)
end

function JacobianCache(number, record!::Function)
    off   = block_offsets(number)
    ndof  = off[4] + maximum(number.d)
    K_pat = nonzero_pattern_storage()
    record!(K_pat)
    I = Int[]; J = Int[]
    for Kt in K_pat, a in 1:4, b in 1:4
        append!(I, Kt[a][b].I .+ off[a])
        append!(J, Kt[a][b].J .+ off[b])
    end
    # Diagonal always present (pinned/degenerate rows)
    append!(I, 1:ndof); append!(J, 1:ndof)
    A = sparse(I, J, zeros(length(I)), ndof, ndof)
    K = Fields(ntuple(a -> Fields(ntuple(b -> GlobalBlock(A, off[a], off[b]), 4)...), 4)...)
    n   = (off[2] - off[1], off[3] - off[2], off[4] - off[3], ndof - off[4])
    rng = ntuple(a -> off[a]+1:off[a]+n[a], 4)
    return JacobianCache(A, K, rng, Ref{Any}(nothing), Ref{Any}(nothing))
end

"""
    jacobian_blocks(J::JacobianCache)

The 4×4 sparse blocks (Vx, Vy, Pt, d) of the global Jacobian, as `Fields` of `Fields` (M[a][b]).
"""
jacobian_blocks(J::JacobianCache) = Fields(ntuple(a -> Fields(ntuple(b -> J.A[J.rng[a], J.rng[b]], 4)...), 4)...)

function JacobianCache(V, P, d, old, rheo, materials, number, type, BC, nc, Δ; nstates=4)
    record!(K) = for s in 1:nstates
        sgn = isodd(s) ? 1.0 : -1.0
        Vr  = (x = V.x .+ (1 .+ rand(size(V.x)...)), y = V.y .+ (1 .+ rand(size(V.y)...)))
        Pr  = (t = sgn .* (1 .+ rand(size(P.t)...)),)
        dr  = ifelse.(type.d .=== :constant, d, 0.5 .* rand(size(d)...))
        oldr = merge(old, (τ = map(A -> rand(size(A)...), old.τ), H = zero(old.H)))
        Assemble!(K, Vr, Pr, dr, oldr, rheo, materials, number, type, BC, nc, Δ)
    end
    return JacobianCache(number, record!)
end

zero_values!(J::JacobianCache) = (fill!(nonzeros(J.A), 0.0); J)

"""
    solve!(dx, J::JacobianCache, r; solver=:lu, reltol=1e-10, abstol=1e-30, restart=25, noisy=false)

Newton correction dx = -J⁻¹ r.

- `solver = :lu` : sparse LU of the full Jacobian, the symbolic analysis is computed once and reused.
- `solver = :gcr`: GCR of the TwoPhases module (`KSP_GCR_TwoPhases_opt!`) with d in the place of Pf.
  Block-triangular preconditioner: Cholesky of the d–d block and of the velocity Schur complement
  J̃vv = Jvv - Jvp Dpp Jpv (Dpp = diag(Jpp)⁻¹ = K). Both blocks are symmetric positive definite for the
  phase-field problem (d frozen: degraded elasticity), so the Newton blocks are used directly.
  Returns the number of GCR iterations.
"""
function solve!(dx, J::JacobianCache, r; solver=:lu, reltol=1e-10, abstol=1e-30, restart=50, maxit=1000, coupled=true, refresh=true, noisy=false)
    if solver === :lu
        if J.F[] === nothing
            J.F[] = lu(J.A)
        else
            lu!(J.F[], J.A)
        end
        ldiv!(dx, J.F[], r)
        dx .*= -1
        return 1
    elseif solver === :gcr     # allocation-free GCR, Cholesky block preconditioner (PhaseFieldMechanics_GCR.jl)
        J.gcr[] === nothing && (J.gcr[] = GCRWorkspace(J; restart, maxit, coupled))
        its = lagged_gcr_solve!(dx, J.A, r, J.gcr[]; refresh, reltol, abstol)   # J dx = r, then dx ← -dx
        dx .*= -1
        noisy && @printf("GCR: %d iterations\n", its)
        return its
    elseif solver === :gcr_twophases   # original TwoPhases solver (rebuilds the preconditioner each call)
        Mb = jacobian_blocks(J)
        J.gcr[] === nothing && (J.gcr[] = KSP_GCR_TwoPhases_setup(Mb; restart=restart))
        fill!(dx, 0.0)
        return KSP_GCR_TwoPhases_opt!(dx, J.A, -r, noisy, Mb, J.gcr[]; reltol=reltol, abstol=abstol)
    else
        error("Unknown solver $(solver)")
    end
end

@inline function Ranges(nc)
    return (inx_Vx = 2:nc.x+2, iny_Vx = 3:nc.y+2, inx_Vy = 3:nc.x+2, iny_Vy = 2:nc.y+2, inx_c = 2:nc.x+1, iny_c = 2:nc.y+1, inx_v = 2:nc.x+2, iny_v = 2:nc.y+2, size_x = (nc.x+3, nc.y+4), size_y = (nc.x+4, nc.y+3), size_c = (nc.x+2, nc.y+2), size_v = (nc.x+3, nc.y+3))
end

# Local stencil extraction: M×N block of A starting at (i0, j0)
@inline loc(A, i0, j0, ::Val{M}, ::Val{N}) where {M,N} = SMatrix{M,N}(A[i0+ii-1, j0+jj-1] for ii in 1:M, jj in 1:N)

#---------------------------------------------------------------------------------------
# Boundary condition templates
#---------------------------------------------------------------------------------------

"""
    set_BC_template(config, nc)

Node types for the (Vx, Vy, Pt, d) fields. The phase field has zero flux (∂d/∂n = 0) on all sides;
cells with a prescribed phase field (e.g. an initial notch) are flagged `:constant` in the script.

- `:free_slip`      : normal velocity prescribed on all sides, zero shear stress (FDMIC_Geodynamics tension test)
- `:clamped_tension`: S/N tangential and normal velocity prescribed, W/E traction free (Miehe et al., 2010)
"""
function set_BC_template(config, nc)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    type = Fields(
        fill(:out, size_x),
        fill(:out, size_y),
        fill(:out, size_c),
        fill(:out, size_c),
    )

    # Pt and d: zero gradient ghosts on all sides
    for A in (type.Pt, type.d)
        A[2:end-1,2:end-1] .= :in
        A[1,:]             .= :Neumann
        A[end,:]           .= :Neumann
        A[:,1]             .= :Neumann
        A[:,end]           .= :Neumann
    end

    if config == :free_slip
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal
        type.Vx[inx_Vx,2]       .= :Neumann_tangent
        type.Vx[inx_Vx,end-1]   .= :Neumann_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in
        type.Vy[2,iny_Vy]       .= :Neumann_tangent
        type.Vy[end-1,iny_Vy]   .= :Neumann_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal
        type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal
    elseif config == :clamped_tension
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in
        type.Vx[1,iny_Vx]       .= :normal_stress
        type.Vx[end,iny_Vx]     .= :normal_stress
        type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in
        type.Vy[2,iny_Vy]       .= :tangential_stress
        type.Vy[end-1,iny_Vy]   .= :tangential_stress
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal
        type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal
    else
        error("Unknown BC template $(config)")
    end
    return type
end

#---------------------------------------------------------------------------------------
# Constitutive relations
#---------------------------------------------------------------------------------------

@inline degradation(d, k_res) = (1 - d)^2 + k_res

@inline viscoelastic_viscosity(G, η, Δt) = inv(inv(η) + inv(G * Δt))

# Undamaged deviatoric normal stresses (plane strain: ε̇zz = 0, τzz = -(τxx + τyy))
@inline function normal_stresses(Dxx, Dyy, τ0xx, τ0yy, G, η, Δt)
    ηve = viscoelastic_viscosity(G, η, Δt)
    ε̇kk = Dxx + Dyy
    τxx = 2ηve * (Dxx - ε̇kk/3 + τ0xx / (2G*Δt))
    τyy = 2ηve * (Dyy - ε̇kk/3 + τ0yy / (2G*Δt))
    return τxx, τyy
end

@inline function shear_stress(Dxy, Dyx, τ0xy, G, η, Δt)
    ηve = viscoelastic_viscosity(G, η, Δt)
    return 2ηve * ((Dxy + Dyx)/2 + τ0xy / (2G*Δt))
end

# Degraded pressure contribution to the total stress
@inline degraded_pressure(P, g, amor) = ifelse(amor & (P > 0), P, g * P)

# Optional model components (absent from `materials`: Miehe et al., 2010)
@inline _plasticity(materials) = hasproperty(materials, :plasticity) ? materials.plasticity : nothing
@inline _βp(materials)         = hasproperty(materials, :βp) ? materials.βp : 0.0
@inline _ψc(materials)         = hasproperty(materials, :ψc) ? materials.ψc : 0.0
@inline _damage(materials)     = hasproperty(materials, :damage) ? materials.damage : true
@inline _Kd(materials)         = hasproperty(materials, :Kd) ? materials.Kd : 0.0

# Stiffness degradation, g = 1 if the damage coupling is switched off (materials.damage = false)
@inline stiffness_degradation(d, materials) = _damage(materials) ? degradation(d, materials.k_res) : one(d)

@inline avg4(A) = (A[1,1] + A[2,1] + A[1,2] + A[2,2]) / 4

# Second invariant of the deviatoric stress (plane strain, τzz = -(τxx + τyy)); regularised at 0 for AD
@inline second_invariant(xx, yy, xy) = sqrt(((xx^2 + yy^2 + (xx + yy)^2) / 2 + xy^2) + 1e-40)

"""
    point_stress(τ̃xx, τ̃yy, τ̃xy, P, g, ηve, K, ph, nl, materials, Δt)

Degraded stress at a point from the trial (undamaged, visco-elastic) deviatoric stresses τ̃ and pressure P̃.
Without plasticity: g τ̃ and the degraded pressure. With plasticity, the yield function is evaluated on the degraded
stress (trial g τ̃, viscosity g ηve, bulk modulus g K; Hageman, 2026, eq. B.4) with the return mapping of the
Rheology module, or with `gradient_return_mapping` if Kd > 0 (`nl`: phase field data at the point).
Returns the degraded τxx, τyy, τxy, corrected pressure, λ̇, τII and the ratio f = τII/τII_trial.
"""
@inline function point_stress(τ̃xx, τ̃yy, τ̃xy, P, g, ηve, K, ph, nl, materials, Δt)
    Pd = degraded_pressure(P, g, materials.amor)
    τxx, τyy, τxy = g * τ̃xx, g * τ̃yy, g * τ̃xy
    pl = _plasticity(materials)
    if pl === nothing
        z = zero(τxx)
        return τxx, τyy, τxy, Pd, z, second_invariant(τxx, τyy, τxy), one(z)
    end
    τII_t = second_invariant(τxx, τyy, τxy)
    if _Kd(materials) > 0
        τII, Pc, λ̇ = gradient_return_mapping(τII_t, Pd, g * ηve, g * K, pl, ph, nl, materials, Δt)
    else
        # Common (Dual) type for the return mapping: with a single Dual input, λ̇ = zero(τII) would change type
        T = promote_type(typeof(τII_t), typeof(Pd), typeof(g))
        τII, Pc, λ̇ = return_mapping(T(τII_t), T(Pd), 0.0, 0.0, 0.0, T(g * ηve), T(inv(g * K)), Δt, 1.0, pl, ph)
    end
    f = τII / τII_t
    return f * τxx, f * τyy, f * τxy, Pc, λ̇, τII, f
end

"""
    gradient_return_mapping(τII_t, Pd, ηe, Ke, pl, ph, nl, materials, Δt)

Drucker-Prager return mapping (parameters of `pl` for phase `ph`) with the gradient term Kd ∇²d:

    F = τII - Pc sinϕ - C cosϕ - ηvp λ̇ + Kd ∇²d,   τII = τII_t - ηe λ̇,   Pc = Pd + λ̇ sinψ Δt Ke

∇²d is expressed with the phase-field equation at the point, gc l ∇²d = gc/l d - 2(1-d)(H⁰ + βp Wp) + ηd ḋ,
with the current plastic work Wp = Wp⁰ + τII λ̇ Δt: the local plastic work hardens, the (non-local) phase field
promotes yielding around existing bands. F(λ̇) is quadratic, its smallest positive root is used.
`nl = (d, d0, H, Wp0, gc, l, ηd)` at the point (vertex: means of the 4 centroid values).
"""
@inline function gradient_return_mapping(τII_t, Pd, ηe, Ke, pl, ph, nl, materials, Δt)
    C, cosϕ, sinϕ, sinψ, ηvp = pl.C[ph], pl.cosϕ[ph], pl.sinϕ[ph], pl.sinψ[ph], pl.ηvp[ph]
    βp  = _βp(materials)
    A   = _Kd(materials) / (nl.gc * nl.l)
    Fg0 = A * (nl.gc / nl.l * nl.d - 2 * (1 - nl.d) * (nl.H + βp * nl.Wp0) + nl.ηd * (nl.d - nl.d0) / Δt)
    κ   = 2 * A * (1 - nl.d) * βp * Δt
    c   = τII_t - Pd * sinϕ - C * cosϕ + Fg0
    b   = -(ηe + sinψ * Δt * Ke * sinϕ + ηvp + κ * τII_t)
    a   = κ * ηe
    # no real root (F > 0 for all λ̇, e.g. far from equilibrium during the Newton iterations): discriminant clipped
    λ̇t  = 2c / (-b + sqrt(max(b^2 - 4 * a * c, zero(c)) + 1e-40))
    λ̇   = ifelse(c > 0, λ̇t, zero(λ̇t))
    return τII_t - ηe * λ̇, Pd + λ̇ * sinψ * Δt * Ke, λ̇
end

# Elastic strain energy (tensile part if amor) at a centroid; τxy² is the mean of the 4 vertex values / (2G)
@inline function strain_energy(τxx, τyy, ψxy, P, G, K, amor)
    τzz  = -(τxx + τyy)
    ψvol = ifelse(amor & (P > 0), zero(P), P^2 / (2K))
    return (τxx^2 + τyy^2 + τzz^2) / (4G) + ψxy + ψvol
end

#---------------------------------------------------------------------------------------
# Local residual functions
#---------------------------------------------------------------------------------------

# Stress components entering the momentum residuals, from the windows of gather_x / gather_y (ghosts set).
# x-momentum: centroids W/E = column 2 of the 2×3 centroid windows (vertex windows 3×2),
#             vertices S/N  = row 2 of the 3×2 vertex windows (centroid windows 2×2)
@inline function σxx_x(Vx, Vy, Pt, d, τ0, mat, k, materials, Δ)
    v2, v3 = Val(2), Val(3)
    τ0c  = (xx = τ0.xx[k,2], yy = τ0.yy[k,2], xy = loc(τ0.xy, k, 1, v2, v2))
    matc = (Gc = mat.Gc[k,2], ηc = mat.ηc[k,2], K = mat.K[k,2], phc = mat.phc[k,2],
            Gv = loc(mat.Gv, k, 1, v2, v2), ηv = loc(mat.ηv, k, 1, v2, v2))
    nl   = (d = d[k,2], d0 = mat.d0[k,2], H = mat.H[k,2], Wp0 = mat.Wp[k,2], gc = mat.gc[k,2], l = mat.l[k,2], ηd = mat.ηd[k,2])
    st = centroid_vep(loc(Vx, k, 1, v2, v3), loc(Vy, k, 2, v3, v2), Pt[k,2], d[k,2], τ0c, matc, nl, materials, Δ)
    return st.τxx - st.Pc
end

@inline function τxy_x(Vx, Vy, Pt, d, τ0, mat, k, materials, Δ)
    v2, v3 = Val(2), Val(3)
    τ0v  = (xx = loc(τ0.xx, 1, k, v2, v2), yy = loc(τ0.yy, 1, k, v2, v2), xy = τ0.xy[2,k])
    matv = (Gc = loc(mat.Gc, 1, k, v2, v2), ηc = loc(mat.ηc, 1, k, v2, v2), K = loc(mat.K, 1, k, v2, v2),
            Gv = mat.Gv[2,k], ηv = mat.ηv[2,k], phv = mat.phv[2,k],
            d0 = loc(mat.d0, 1, k, v2, v2), H = loc(mat.H, 1, k, v2, v2), Wp = loc(mat.Wp, 1, k, v2, v2),
            gc = loc(mat.gc, 1, k, v2, v2), l = loc(mat.l, 1, k, v2, v2), ηd = loc(mat.ηd, 1, k, v2, v2))
    return first(vertex_vep(loc(Vx, 1, k, v3, v2), loc(Vy, 2, k, v2, v3), loc(Pt, 1, k, v2, v2), loc(d, 1, k, v2, v2), τ0v, matv, materials, Δ))
end

# y-momentum: centroids S/N = row 2 of the 3×2 centroid windows (vertex windows 2×3),
#             vertices W/E  = column 2 of the 2×3 vertex windows (centroid windows 2×2)
@inline function σyy_y(Vx, Vy, Pt, d, τ0, mat, k, materials, Δ)
    v2, v3 = Val(2), Val(3)
    τ0c  = (xx = τ0.xx[2,k], yy = τ0.yy[2,k], xy = loc(τ0.xy, 1, k, v2, v2))
    matc = (Gc = mat.Gc[2,k], ηc = mat.ηc[2,k], K = mat.K[2,k], phc = mat.phc[2,k],
            Gv = loc(mat.Gv, 1, k, v2, v2), ηv = loc(mat.ηv, 1, k, v2, v2))
    nl   = (d = d[2,k], d0 = mat.d0[2,k], H = mat.H[2,k], Wp0 = mat.Wp[2,k], gc = mat.gc[2,k], l = mat.l[2,k], ηd = mat.ηd[2,k])
    st = centroid_vep(loc(Vx, 2, k, v2, v3), loc(Vy, 1, k, v3, v2), Pt[2,k], d[2,k], τ0c, matc, nl, materials, Δ)
    return st.τyy - st.Pc
end

@inline function τxy_y(Vx, Vy, Pt, d, τ0, mat, k, materials, Δ)
    v2, v3 = Val(2), Val(3)
    τ0v  = (xx = loc(τ0.xx, k, 1, v2, v2), yy = loc(τ0.yy, k, 1, v2, v2), xy = τ0.xy[k,2])
    matv = (Gc = loc(mat.Gc, k, 1, v2, v2), ηc = loc(mat.ηc, k, 1, v2, v2), K = loc(mat.K, k, 1, v2, v2),
            Gv = mat.Gv[k,2], ηv = mat.ηv[k,2], phv = mat.phv[k,2],
            d0 = loc(mat.d0, k, 1, v2, v2), H = loc(mat.H, k, 1, v2, v2), Wp = loc(mat.Wp, k, 1, v2, v2),
            gc = loc(mat.gc, k, 1, v2, v2), l = loc(mat.l, k, 1, v2, v2), ηd = loc(mat.ηd, k, 1, v2, v2))
    return first(vertex_vep(loc(Vx, k, 2, v3, v2), loc(Vy, k, 1, v2, v3), loc(Pt, k, 1, v2, v2), loc(d, k, 1, v2, v2), τ0v, matv, materials, Δ))
end

@inline function Momentum_x(Vx_loc, Vy_loc, Pt_loc, d_loc, aux, materials, Δ)

    τ0, mat, type, bcv = aux
    invΔx, invΔy, BC_sym = inv(Δ.x), inv(Δ.y), 1.0

    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)
    Pt = SetBCPt1(Pt_loc, type.pt, bcv.pt, Δ, 0.0)
    d  = SetBCPt1( d_loc, type.d,  bcv.d,  Δ, 0.0)

    σxx = SVector(σxx_x(Vx, Vy, Pt, d, τ0, mat, 1, materials, Δ), σxx_x(Vx, Vy, Pt, d, τ0, mat, 2, materials, Δ))
    τxy = SVector(τxy_x(Vx, Vy, Pt, d, τ0, mat, 1, materials, Δ), τxy_x(Vx, Vy, Pt, d, τ0, mat, 2, materials, Δ))

    # Normal stress BC
    if type.x[1,2] == :normal_stress
        σxx = SVector{2}(2*bcv.x[2,2] - σxx[2], σxx[2])
        BC_sym = 1/2
    end
    if type.x[end,2] == :normal_stress
        σxx = SVector{2}(σxx[1], 2*bcv.x[end-1,2] - σxx[1])
        BC_sym = 1/2
    end
    # Shear stress BC
    if type.x[2,1] == :tangential_stress
        τxy = SVector{2}(bcv.x[2,1], τxy[2])
    end
    if type.x[2,end] == :tangential_stress
        τxy = SVector{2}(τxy[1], bcv.x[2,end])
    end

    fx  = (σxx[2] - σxx[1]) * invΔx + (τxy[2] - τxy[1]) * invΔy
    fx *= -Δ.x * Δ.y * BC_sym
    return fx
end

@inline function Momentum_y(Vx_loc, Vy_loc, Pt_loc, d_loc, aux, materials, Δ)

    τ0, mat, type, bcv = aux
    invΔx, invΔy, BC_sym = inv(Δ.x), inv(Δ.y), 1.0

    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)
    Pt = SetBCPt1(Pt_loc, type.pt, bcv.pt, Δ, 0.0)
    d  = SetBCPt1( d_loc, type.d,  bcv.d,  Δ, 0.0)

    σyy = SVector(σyy_y(Vx, Vy, Pt, d, τ0, mat, 1, materials, Δ), σyy_y(Vx, Vy, Pt, d, τ0, mat, 2, materials, Δ))
    τxy = SVector(τxy_y(Vx, Vy, Pt, d, τ0, mat, 1, materials, Δ), τxy_y(Vx, Vy, Pt, d, τ0, mat, 2, materials, Δ))

    # Normal stress BC
    if type.y[2,1] == :normal_stress
        σyy = SVector{2}(2*bcv.y[2,2] - σyy[2], σyy[2])
        BC_sym = 1/2
    end
    if type.y[2,end] == :normal_stress
        σyy = SVector{2}(σyy[1], 2*bcv.y[2,end-1] - σyy[1])
        BC_sym = 1/2
    end
    # Shear stress BC
    if type.y[1,2] == :tangential_stress
        τxy = SVector{2}(bcv.y[1,2], τxy[2])
    end
    if type.y[end,2] == :tangential_stress
        τxy = SVector{2}(τxy[1], bcv.y[end,2])
    end

    fy  = (σyy[2] - σyy[1]) * invΔy + (τxy[2] - τxy[1]) * invΔx
    fy *= -Δ.x * Δ.y * BC_sym
    return fy
end

# Written in increment form (Δt ∇⋅V + ΔP/K): its round-off does not grow as Δt → 0 (relaxation steps)
@inline function Continuity(Vx_loc, Vy_loc, Pt_loc, d_loc, aux, materials, Δ)
    P0, K = aux
    divV  = (Vx_loc[2,1] - Vx_loc[1,1]) * inv(Δ.x) + (Vy_loc[1,2] - Vy_loc[1,1]) * inv(Δ.y)
    return Δ.t * divV + (Pt_loc[1,1] - P0) / K
end

"""
    centroid_vep(Vx, Vy, P, dc, τ0, mat, nl, materials, Δ)

State at a centroid from its (2×3 Vx, 3×2 Vy) window (ghosts set): trial stresses (τxy: mean of the 4 vertex
trial values), degraded (visco-elasto-plastic) stresses, plastic rate and the elastic strain energy of the
effective stresses (deviatoric part scaled by f², f = τII/τII_trial). Returns a NamedTuple.
"""
@inline function centroid_vep(Vx, Vy, P, dc, τ0, mat, nl, materials, Δ)
    invΔx, invΔy, Δt = inv(Δ.x), inv(Δ.y), Δ.t
    Dxx = (Vx[2,2] - Vx[1,2]) * invΔx
    Dyy = (Vy[2,2] - Vy[2,1]) * invΔy
    τ̃xx, τ̃yy = normal_stresses(Dxx, Dyy, τ0.xx, τ0.yy, mat.Gc, mat.ηc, Δt)
    τ̃xy = zero(τ̃xx)
    ψxy = zero(τ̃xx)
    for b in 1:2, a in 1:2
        Dxy  = (Vx[a,b+1] - Vx[a,b]) * invΔy
        Dyx  = (Vy[a+1,b] - Vy[a,b]) * invΔx
        τxyv = shear_stress(Dxy, Dyx, τ0.xy[a,b], mat.Gv[a,b], mat.ηv[a,b], Δt)
        τ̃xy += τxyv / 4
        ψxy += τxyv^2 / (2*mat.Gv[a,b]) / 4
    end
    g   = stiffness_degradation(dc, materials)
    ηve = viscoelastic_viscosity(mat.Gc, mat.ηc, Δt)
    τxx, τyy, τxy, Pc, λ̇, τII, f = point_stress(τ̃xx, τ̃yy, τ̃xy, P, g, ηve, mat.K, mat.phc, nl, materials, Δt)
    ψ = strain_energy(f * τ̃xx, f * τ̃yy, f^2 * ψxy, P, mat.Gc, mat.K, materials.amor)
    # Undamaged pressure including the plastic dilatancy (P - P⁰ = -KΔt(∇⋅V - λ̇ sinψ)): history P⁰ of the next step
    P̃c = P + (Pc - degraded_pressure(P, g, materials.amor)) / g
    return (τxx = τxx, τyy = τyy, Pc = Pc, P̃c = P̃c, λ̇ = λ̇, τII = τII, ψ = ψ, τ̃xx = f * τ̃xx, τ̃yy = f * τ̃yy, f = f)
end

"""
    vertex_vep(Vx, Vy, P, d, τ0, mat, materials, Δ)

State at a vertex from its (3×2 Vx, 2×3 Vy) window and the 2×2 surrounding centroids (P, d, τ0.xx, τ0.yy,
mat.Gc, mat.ηc, mat.K): τxx, τyy are the means of the 4 centroid trial values, g and P̃ the means of the centroid
values. Returns the degraded shear stress, the plastic rate and the effective shear stress.
"""
@inline function vertex_vep(Vx, Vy, P, d, τ0, mat, materials, Δ)
    invΔx, invΔy, Δt = inv(Δ.x), inv(Δ.y), Δ.t
    Dxy  = (Vx[2,2] - Vx[2,1]) * invΔy
    Dyx  = (Vy[2,2] - Vy[1,2]) * invΔx
    τ̃xy  = shear_stress(Dxy, Dyx, τ0.xy, mat.Gv, mat.ηv, Δt)
    g    = (stiffness_degradation(d[1,1], materials) + stiffness_degradation(d[2,1], materials) +
            stiffness_degradation(d[1,2], materials) + stiffness_degradation(d[2,2], materials)) / 4
    if _plasticity(materials) === nothing
        return g * τ̃xy, zero(τ̃xy), τ̃xy
    end
    τ̃xx, τ̃yy = zero(τ̃xy), zero(τ̃xy)
    for b in 1:2, a in 1:2
        Dxx = (Vx[a+1,b] - Vx[a,b]) * invΔx
        Dyy = (Vy[a,b+1] - Vy[a,b]) * invΔy
        txx, tyy = normal_stresses(Dxx, Dyy, τ0.xx[a,b], τ0.yy[a,b], mat.Gc[a,b], mat.ηc[a,b], Δt)
        τ̃xx += txx / 4
        τ̃yy += tyy / 4
    end
    Pv  = (P[1,1] + P[2,1] + P[1,2] + P[2,2]) / 4
    Kv  = (mat.K[1,1] + mat.K[2,1] + mat.K[1,2] + mat.K[2,2]) / 4
    ηve = viscoelastic_viscosity(mat.Gv, mat.ηv, Δt)
    nl  = (d = avg4(d), d0 = avg4(mat.d0), H = avg4(mat.H), Wp0 = avg4(mat.Wp), gc = avg4(mat.gc), l = avg4(mat.l), ηd = avg4(mat.ηd))
    _, _, τxy, _, λ̇, _, f = point_stress(τ̃xx, τ̃yy, τ̃xy, Pv, g, ηve, Kv, mat.phv, nl, materials, Δt)
    return τxy, λ̇, f * τ̃xy
end

@inline function PhaseField(Vx_loc, Vy_loc, Pt_loc, d_loc, aux, materials, Δ)

    τ0, mat, old, type, bcv = aux
    invΔx, invΔy, Δt = inv(Δ.x), inv(Δ.y), Δ.t

    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)
    d  = SetBCPt1( d_loc, type.d, bcv.d, Δ, 0.0)

    # Crack driving force: history of the elastic strain energy above the threshold ψc, and plastic work
    nl = (d = d[2,2], d0 = old.d, H = old.H, Wp0 = old.Wp, gc = mat.gc[2,2], l = mat.l[2,2], ηd = mat.ηd)
    st = centroid_vep(Vx, Vy, Pt_loc[1,1], d[2,2], τ0, mat, nl, materials, Δ)
    H  = max(old.H, max(st.ψ - _ψc(materials), zero(st.ψ)))
    Wp = old.Wp + st.λ̇ * st.τII * Δt

    # Diffusion term, face coefficients gc⋅l
    gcl = mat.gc .* mat.l
    qW  = (gcl[1,2] + gcl[2,2]) / 2 * (d[2,2] - d[1,2]) * invΔx
    qE  = (gcl[2,2] + gcl[3,2]) / 2 * (d[3,2] - d[2,2]) * invΔx
    qS  = (gcl[2,1] + gcl[2,2]) / 2 * (d[2,2] - d[2,1]) * invΔy
    qN  = (gcl[2,2] + gcl[2,3]) / 2 * (d[2,3] - d[2,2]) * invΔy
    div_q = (qE - qW) * invΔx + (qN - qS) * invΔy

    dc = d[2,2]
    fd = mat.gc[2,2] / mat.l[2,2] * dc - div_q - 2 * (1 - dc) * (H + _βp(materials) * Wp) + mat.ηd * (dc - old.d) / Δt
    return fd * Δ.x * Δ.y
end

#---------------------------------------------------------------------------------------
# Stencil gathering (shared by residual and assembly kernels)
#---------------------------------------------------------------------------------------

@inline function gather_x(i, j, V, P, d, old, rheo, type, BC)
    v3, v4, v2, v1 = Val(3), Val(4), Val(2), Val(1)
    Vx  = loc(V.x, i-1, j-1, v3, v3)
    Vy  = loc(V.y, i-1, j-2, v4, v4)
    Pt  = loc(P.t, i-1, j-2, v2, v3)
    dl  = loc(d,   i-1, j-2, v2, v3)
    # centroids c[i-1:i, j-2:j] (2×3), vertices v[i-1:i+1, j-1:j] (3×2)
    τ0  = (xx = loc(old.τ.xx, i-1, j-2, v2, v3), yy = loc(old.τ.yy, i-1, j-2, v2, v3), xy = loc(old.τ.xy, i-1, j-1, v3, v2))
    mat = (Gc = loc(rheo.G.c, i-1, j-2, v2, v3), ηc = loc(rheo.η.c, i-1, j-2, v2, v3), K = loc(rheo.K.c, i-1, j-2, v2, v3), phc = loc(rheo.ph.c, i-1, j-2, v2, v3),
           Gv = loc(rheo.G.v, i-1, j-1, v3, v2), ηv = loc(rheo.η.v, i-1, j-1, v3, v2), phv = loc(rheo.ph.v, i-1, j-1, v3, v2),
           d0 = loc(old.d, i-1, j-2, v2, v3), H = loc(old.H, i-1, j-2, v2, v3), Wp = loc(old.Wp, i-1, j-2, v2, v3),
           gc = loc(rheo.gc.c, i-1, j-2, v2, v3), l = loc(rheo.l.c, i-1, j-2, v2, v3), ηd = loc(rheo.ηd.c, i-1, j-2, v2, v3))
    tl  = (x = loc(type.Vx, i-1, j-1, v3, v3), y = loc(type.Vy, i-1, j-2, v4, v4), pt = loc(type.Pt, i-1, j-2, v2, v3), d = loc(type.d, i-1, j-2, v2, v3))
    bl  = (x = loc(BC.Vx,   i-1, j-1, v3, v3), y = loc(BC.Vy,   i-1, j-2, v4, v4), pt = loc(BC.Pt,   i-1, j-2, v2, v3), d = loc(BC.d,   i-1, j-2, v2, v3))
    return Vx, Vy, Pt, dl, (τ0, mat, tl, bl)
end

@inline function numbers_x(i, j, num)
    v3, v4, v2 = Val(3), Val(4), Val(2)
    return loc(num.Vx, i-1, j-1, v3, v3), loc(num.Vy, i-1, j-2, v4, v4), loc(num.Pt, i-1, j-2, v2, v3), loc(num.d, i-1, j-2, v2, v3)
end

@inline function gather_y(i, j, V, P, d, old, rheo, type, BC)
    v3, v4, v2, v1 = Val(3), Val(4), Val(2), Val(1)
    Vx  = loc(V.x, i-2, j-1, v4, v4)
    Vy  = loc(V.y, i-1, j-1, v3, v3)
    Pt  = loc(P.t, i-2, j-1, v3, v2)
    dl  = loc(d,   i-2, j-1, v3, v2)
    # centroids c[i-2:i, j-1:j] (3×2), vertices v[i-1:i, j-1:j+1] (2×3)
    τ0  = (xx = loc(old.τ.xx, i-2, j-1, v3, v2), yy = loc(old.τ.yy, i-2, j-1, v3, v2), xy = loc(old.τ.xy, i-1, j-1, v2, v3))
    mat = (Gc = loc(rheo.G.c, i-2, j-1, v3, v2), ηc = loc(rheo.η.c, i-2, j-1, v3, v2), K = loc(rheo.K.c, i-2, j-1, v3, v2), phc = loc(rheo.ph.c, i-2, j-1, v3, v2),
           Gv = loc(rheo.G.v, i-1, j-1, v2, v3), ηv = loc(rheo.η.v, i-1, j-1, v2, v3), phv = loc(rheo.ph.v, i-1, j-1, v2, v3),
           d0 = loc(old.d, i-2, j-1, v3, v2), H = loc(old.H, i-2, j-1, v3, v2), Wp = loc(old.Wp, i-2, j-1, v3, v2),
           gc = loc(rheo.gc.c, i-2, j-1, v3, v2), l = loc(rheo.l.c, i-2, j-1, v3, v2), ηd = loc(rheo.ηd.c, i-2, j-1, v3, v2))
    tl  = (x = loc(type.Vx, i-2, j-1, v4, v4), y = loc(type.Vy, i-1, j-1, v3, v3), pt = loc(type.Pt, i-2, j-1, v3, v2), d = loc(type.d, i-2, j-1, v3, v2))
    bl  = (x = loc(BC.Vx,   i-2, j-1, v4, v4), y = loc(BC.Vy,   i-1, j-1, v3, v3), pt = loc(BC.Pt,   i-2, j-1, v3, v2), d = loc(BC.d,   i-2, j-1, v3, v2))
    return Vx, Vy, Pt, dl, (τ0, mat, tl, bl)
end

@inline function numbers_y(i, j, num)
    v3, v4, v2 = Val(3), Val(4), Val(2)
    return loc(num.Vx, i-2, j-1, v4, v4), loc(num.Vy, i-1, j-1, v3, v3), loc(num.Pt, i-2, j-1, v3, v2), loc(num.d, i-2, j-1, v3, v2)
end

@inline function gather_p(i, j, V, P, d, old, rheo, type, BC)
    v2, v1 = Val(2), Val(1)
    Vx = loc(V.x, i,   j+1, v2, v1)
    Vy = loc(V.y, i+1, j,   v1, v2)
    Pt = loc(P.t, i,   j,   v1, v1)
    dl = loc(d,   i,   j,   v1, v1)
    return Vx, Vy, Pt, dl, (old.P[i,j], rheo.K.c[i,j])
end

@inline function numbers_p(i, j, num)
    v2, v1 = Val(2), Val(1)
    return loc(num.Vx, i, j+1, v2, v1), loc(num.Vy, i+1, j, v1, v2), loc(num.Pt, i, j, v1, v1), loc(num.d, i, j, v1, v1)
end

@inline function gather_d(i, j, V, P, d, old, rheo, type, BC)
    v3, v2, v1 = Val(3), Val(2), Val(1)
    Vx  = loc(V.x, i,   j,   v2, v3)
    Vy  = loc(V.y, i,   j,   v3, v2)
    Pt  = loc(P.t, i,   j,   v1, v1)
    dl  = loc(d,   i-1, j-1, v3, v3)
    τ0  = (xx = old.τ.xx[i,j], yy = old.τ.yy[i,j], xy = loc(old.τ.xy, i, j, v2, v2))
    mat = (Gc = rheo.G.c[i,j], ηc = rheo.η.c[i,j], K = rheo.K.c[i,j], phc = rheo.ph.c[i,j],
           Gv = loc(rheo.G.v, i, j, v2, v2), ηv = loc(rheo.η.v, i, j, v2, v2),
           gc = loc(rheo.gc.c, i-1, j-1, v3, v3), l = loc(rheo.l.c, i-1, j-1, v3, v3), ηd = rheo.ηd.c[i,j])
    oldl = (d = old.d[i,j], H = old.H[i,j], Wp = old.Wp[i,j])
    tl  = (x = loc(type.Vx, i, j, v2, v3), y = loc(type.Vy, i, j, v3, v2), d = loc(type.d, i-1, j-1, v3, v3))
    bl  = (x = loc(BC.Vx,   i, j, v2, v3), y = loc(BC.Vy,   i, j, v3, v2), d = loc(BC.d,   i-1, j-1, v3, v3))
    return Vx, Vy, Pt, dl, (τ0, mat, oldl, tl, bl)
end

@inline function numbers_d(i, j, num)
    v3, v2, v1 = Val(3), Val(2), Val(1)
    return loc(num.Vx, i, j, v2, v3), loc(num.Vy, i, j, v3, v2), loc(num.Pt, i, j, v1, v1), loc(num.d, i-1, j-1, v3, v3)
end

#---------------------------------------------------------------------------------------
# Residual and assembly kernels
#---------------------------------------------------------------------------------------

# Loop ranges and "is this an equation" predicates for each block row
@inline eq_ranges(::Val{1}, nc) = (2:nc.x+2, 3:nc.y+2)
@inline eq_ranges(::Val{2}, nc) = (3:nc.x+2, 2:nc.y+2)
@inline eq_ranges(::Val{3}, nc) = (2:nc.x+1, 2:nc.y+1)
@inline eq_ranges(::Val{4}, nc) = (2:nc.x+1, 2:nc.y+1)

@inline is_equation(::Val{1}, type, i, j) = type.Vx[i,j] === :in
@inline is_equation(::Val{2}, type, i, j) = type.Vy[i,j] === :in
@inline is_equation(::Val{3}, type, i, j) = type.Pt[i,j] === :in
@inline is_equation(::Val{4}, type, i, j) = type.d[i,j]  === :in

@inline gather(::Val{1}, args...) = gather_x(args...)
@inline gather(::Val{2}, args...) = gather_y(args...)
@inline gather(::Val{3}, args...) = gather_p(args...)
@inline gather(::Val{4}, args...) = gather_d(args...)

@inline numbers(::Val{1}, args...) = numbers_x(args...)
@inline numbers(::Val{2}, args...) = numbers_y(args...)
@inline numbers(::Val{3}, args...) = numbers_p(args...)
@inline numbers(::Val{4}, args...) = numbers_d(args...)

@inline local_residual(::Val{1}) = Momentum_x
@inline local_residual(::Val{2}) = Momentum_y
@inline local_residual(::Val{3}) = Continuity
@inline local_residual(::Val{4}) = PhaseField

@inline residual_array(::Val{1}, R) = R.x
@inline residual_array(::Val{2}, R) = R.y
@inline residual_array(::Val{3}, R) = R.pt
@inline residual_array(::Val{4}, R) = R.d

@inline row_number(::Val{1}, num, i, j) = num.Vx[i,j]
@inline row_number(::Val{2}, num, i, j) = num.Vy[i,j]
@inline row_number(::Val{3}, num, i, j) = num.Pt[i,j]
@inline row_number(::Val{4}, num, i, j) = num.d[i,j]

function Residual2D!(blk::Val{B}, R, V, P, d, old, rheo, materials, type, BC, nc, Δ) where {B}
    f        = local_residual(blk)
    Rb       = residual_array(blk, R)
    irange, jrange = eq_ranges(blk, nc)
    Threads.@threads for j in jrange
        @inbounds for i in irange
            is_equation(blk, type, i, j) || continue
            Vx, Vy, Pt, dl, aux = gather(blk, i, j, V, P, d, old, rheo, type, BC)
            Rb[i,j] = f(Vx, Vy, Pt, dl, aux, materials, Δ)
        end
    end
    return nothing
end

@inline function write_block!(K, a, b, row, Local, ∂R)
    @inbounds for jj in axes(Local,2), ii in axes(Local,1)
        if Local[ii,jj] > 0
            K[a][b][row, Local[ii,jj]] = ∂R[ii,jj]
        end
    end
end

function Assemble2D!(blk::Val{B}, K_loc, V, P, d, old, rheo, materials, num, type, BC, nc, Δ) where {B}
    f        = local_residual(blk)
    irange, jrange = eq_ranges(blk, nc)
    Threads.@threads for j in jrange
        @inbounds for i in irange
            is_equation(blk, type, i, j) || continue
            row = row_number(blk, num, i, j)
            row > 0 || continue
            K = _target(K_loc, threadid())

            Vx, Vy, Pt, dl, aux = gather(blk, i, j, V, P, d, old, rheo, type, BC)
            ∂R∂Vx = ad_gradient(x -> f(x,  Vy, Pt, dl, aux, materials, Δ), Vx)
            ∂R∂Vy = ad_gradient(x -> f(Vx, x,  Pt, dl, aux, materials, Δ), Vy)
            ∂R∂Pt = ad_gradient(x -> f(Vx, Vy, x,  dl, aux, materials, Δ), Pt)
            ∂R∂d  = ad_gradient(x -> f(Vx, Vy, Pt, x,  aux, materials, Δ), dl)

            nVx, nVy, nPt, nd = numbers(blk, i, j, num)
            write_block!(K, B, 1, row, nVx, ∂R∂Vx)
            write_block!(K, B, 2, row, nVy, ∂R∂Vy)
            write_block!(K, B, 3, row, nPt, ∂R∂Pt)
            write_block!(K, B, 4, row, nd,  ∂R∂d )
        end
    end
    return nothing
end

"""
    Residuals!(R, V, P, d, old, rheo, materials, type, BC, nc, Δ)

Evaluate the four residual fields `R.x`, `R.y`, `R.pt`, `R.d`.
"""
function Residuals!(R, V, P, d, old, rheo, materials, type, BC, nc, Δ)
    for B in 1:4
        Residual2D!(Val(B), R, V, P, d, old, rheo, materials, type, BC, nc, Δ)
    end
    return nothing
end

"""
    Assemble!(K, V, P, d, old, rheo, materials, number, type, BC, nc, Δ)

Assemble the Jacobian into `K` (a `fixed_pattern` of the block matrix, or a `pattern_storage`).
"""
function Assemble!(K, V, P, d, old, rheo, materials, number, type, BC, nc, Δ)
    for B in 1:4
        Assemble2D!(Val(B), K, V, P, d, old, rheo, materials, number, type, BC, nc, Δ)
    end
    return nothing
end

#---------------------------------------------------------------------------------------
# Numbering, RHS, solution update
#---------------------------------------------------------------------------------------

function Numbering!(N, type, nc)
    ndof = 0
    for j=3:nc.y+2, i=2:nc.x+2
        if !(type.Vx[i,j] === :Dirichlet_normal || type.Vx[i,j] === :constant)
            ndof += 1
            N.Vx[i,j] = ndof
        end
    end
    ndof = 0
    for j=2:nc.y+2, i=3:nc.x+2
        if !(type.Vy[i,j] === :Dirichlet_normal || type.Vy[i,j] === :constant)
            ndof += 1
            N.Vy[i,j] = ndof
        end
    end
    ndof = 0
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] !== :constant
            ndof += 1
            N.Pt[i,j] = ndof
        end
    end
    ndof = 0
    for j=2:nc.y+1, i=2:nc.x+1
        if type.d[i,j] === :in
            ndof += 1
            N.d[i,j] = ndof
        end
    end
    return nothing
end

block_offsets(number) = (0, maximum(number.Vx), maximum(number.Vx) + maximum(number.Vy), maximum(number.Vx) + maximum(number.Vy) + maximum(number.Pt))

function SetRHS!(r, R, number, type, nc)
    off = block_offsets(number)
    # One call per block (function barrier: the loop over a heterogeneous tuple is type-unstable)
    _set_rhs!(Val(1), r, R.x,  number.Vx, type, nc, off[1])
    _set_rhs!(Val(2), r, R.y,  number.Vy, type, nc, off[2])
    _set_rhs!(Val(3), r, R.pt, number.Pt, type, nc, off[3])
    _set_rhs!(Val(4), r, R.d,  number.d,  type, nc, off[4])
    return nothing
end

function _set_rhs!(::Val{B}, r, Rb, nb, type, nc, off) where {B}
    irange, jrange = eq_ranges(Val(B), nc)
    @inbounds for j in jrange, i in irange
        if is_equation(Val(B), type, i, j) && nb[i,j] > 0
            r[nb[i,j] + off] = Rb[i,j]
        end
    end
    return nothing
end

function UpdateSolution!(V, P, d, dx, number, type, nc)
    off = block_offsets(number)
    _update_solution!(Val(1), V.x, dx, number.Vx, type, nc, off[1])
    _update_solution!(Val(2), V.y, dx, number.Vy, type, nc, off[2])
    _update_solution!(Val(3), P.t, dx, number.Pt, type, nc, off[3])
    _update_solution!(Val(4), d,   dx, number.d,  type, nc, off[4])
    return nothing
end

function _update_solution!(::Val{B}, X, dx, nb, type, nc, off) where {B}
    irange, jrange = eq_ranges(Val(B), nc)
    @inbounds for j in jrange, i in irange
        if is_equation(Val(B), type, i, j) && nb[i,j] > 0
            X[i,j] += dx[nb[i,j] + off]
        end
    end
    return nothing
end

#---------------------------------------------------------------------------------------
# Post-processing: undamaged stresses, strain energy
#---------------------------------------------------------------------------------------

"""
    UpdateStress!(τ, ψ, V, P, d, old, rheo, materials, type, BC, nc, Δ; plastic=nothing)

Effective (undamaged, after return mapping) deviatoric stresses (`τ.xx`, `τ.yy` at centroids, `τ.xy` at vertices)
and elastic strain energy `ψ` (centroids) of the current solution. If `plastic = (λ̇ = (c, v), τII = (c,), Pc = (c,))`
is given, the plastic rates (centroids, vertices), the degraded stress invariant and the undamaged pressure
corrected for plastic dilatancy (the P⁰ of the next step) at centroids are stored too. Ghost centroids of τ.xx, τ.yy are set to their neighbours (used by the
vertex averages at the boundaries). After convergence of a step: τ⁰ ← τ, H⁰ ← max(H⁰, ⟨ψ - ψc⟩),
Wp ← Wp + τII λ̇ Δt.
"""
function UpdateStress!(τ, ψ, V, P, d, old, rheo, materials, type, BC, nc, Δ; plastic=nothing)
    v1, v2, v3 = Val(1), Val(2), Val(3)
    # Centroids
    Threads.@threads for j in 2:nc.y+1
        @inbounds for i in 2:nc.x+1
            Vx, Vy, Pt, dl, (τ0, mat, oldl, tl, bl) = gather_d(i, j, V, P, d, old, rheo, type, BC)
            Vx = SetBCVx1(Vx, tl.x, bl.x, Δ)
            Vy = SetBCVy1(Vy, tl.y, bl.y, Δ)
            nl = (d = d[i,j], d0 = oldl.d, H = oldl.H, Wp0 = oldl.Wp, gc = mat.gc[2,2], l = mat.l[2,2], ηd = mat.ηd)
            st = centroid_vep(Vx, Vy, Pt[1,1], d[i,j], τ0, mat, nl, materials, Δ)
            τ.xx[i,j], τ.yy[i,j], ψ[i,j] = st.τ̃xx, st.τ̃yy, st.ψ
            if plastic !== nothing
                plastic.λ̇.c[i,j], plastic.τII.c[i,j], plastic.Pc.c[i,j] = st.λ̇, st.τII, st.P̃c
            end
        end
    end
    # Vertices: windows Vx v[i-1:i+1, j:j+1] (3×2), Vy [i:i+1, j-1:j+1] (2×3), centroids c[i-1:i, j-1:j]
    Threads.@threads for j in 2:nc.y+2
        @inbounds for i in 2:nc.x+2
            Vx  = SetBCVx1(loc(V.x, i-1, j, v3, v2), loc(type.Vx, i-1, j, v3, v2), loc(BC.Vx, i-1, j, v3, v2), Δ)
            Vy  = SetBCVy1(loc(V.y, i, j-1, v2, v3), loc(type.Vy, i, j-1, v2, v3), loc(BC.Vy, i, j-1, v2, v3), Δ)
            Pt  = SetBCPt1(loc(P.t, i-1, j-1, v2, v2), loc(type.Pt, i-1, j-1, v2, v2), loc(BC.Pt, i-1, j-1, v2, v2), Δ, 0.0)
            dl  = SetBCPt1(loc(d,   i-1, j-1, v2, v2), loc(type.d,  i-1, j-1, v2, v2), loc(BC.d,  i-1, j-1, v2, v2), Δ, 0.0)
            τ0  = (xx = loc(old.τ.xx, i-1, j-1, v2, v2), yy = loc(old.τ.yy, i-1, j-1, v2, v2), xy = old.τ.xy[i,j])
            mat = (Gc = loc(rheo.G.c, i-1, j-1, v2, v2), ηc = loc(rheo.η.c, i-1, j-1, v2, v2), K = loc(rheo.K.c, i-1, j-1, v2, v2),
                   Gv = rheo.G.v[i,j], ηv = rheo.η.v[i,j], phv = rheo.ph.v[i,j],
                   d0 = loc(old.d, i-1, j-1, v2, v2), H = loc(old.H, i-1, j-1, v2, v2), Wp = loc(old.Wp, i-1, j-1, v2, v2),
                   gc = loc(rheo.gc.c, i-1, j-1, v2, v2), l = loc(rheo.l.c, i-1, j-1, v2, v2), ηd = loc(rheo.ηd.c, i-1, j-1, v2, v2))
            _, λ̇, τ̃xy = vertex_vep(Vx, Vy, Pt, dl, τ0, mat, materials, Δ)
            τ.xy[i,j] = τ̃xy
            plastic !== nothing && (plastic.λ̇.v[i,j] = λ̇)
        end
    end
    # Ghost centroids (zero gradient)
    for A in (τ.xx, τ.yy)
        A[1,:] .= A[2,:]; A[end,:] .= A[end-1,:]
        A[:,1] .= A[:,2]; A[:,end] .= A[:,end-1]
    end
    return nothing
end

"""
    set_ghosts!(A)

Zero-gradient ghost centroids of a centroid array (history fields used in vertex averages at the boundaries).
"""
function set_ghosts!(A)
    A[1,:] .= A[2,:]; A[end,:] .= A[end-1,:]
    A[:,1] .= A[:,2]; A[:,end] .= A[:,end-1]
    return A
end

"""
    total_stress(τ, P, d, materials)

Degraded total normal stresses σxx, σyy at centroids.
"""
function total_stress(τ, P, d, materials)
    g   = stiffness_degradation.(d, Ref(materials))
    Pd  = degraded_pressure.(P, g, materials.amor)
    return (xx = g .* τ.xx .- Pd, yy = g .* τ.yy .- Pd)
end

#---------------------------------------------------------------------------------------
# Line search
#---------------------------------------------------------------------------------------

function residual_norms(R, number, nc)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c = Ranges(nc)
    nVx, nVy, nPt, nd = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt), max(maximum(number.d), 1)
    return @views (x  = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx), y = norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy),
                   pt = norm(R.pt[inx_c,iny_c])/sqrt(nPt),  d = norm(R.d[inx_c,iny_c])/sqrt(nd))
end

"""
    BackTrackingLineSearch!(R, dx, V, P, d, Vi, Pi, di, old, rheo, materials, number, type, BC, nc, Δ, ϵ_ref; α0=1.0, ρ_LS=0.5, αmin=1/16, ϕ_accept=1e-9)

Backtracking on the sum of the residual norms of each field, each normalised by the reference norms `ϵ_ref`. A step is accepted if it decreases the merit function or brings it below `ϕ_accept`
(round-off level). Leaves the solution updated with the accepted step and the residuals evaluated there.
"""
function BackTrackingLineSearch!(R, dx, V, P, d, Vi, Pi, di, old, rheo, materials, number, type, BC, nc, Δ, ϵ_ref;
    α0=1.0, ρ_LS=0.5, αmin=1/16, ϕ_accept=1e-9)

    merit() = (e = residual_norms(R, number, nc); e.x/ϵ_ref.x + e.y/ϵ_ref.y + e.pt/ϵ_ref.pt + e.d/ϵ_ref.d)

    Vi.x .= V.x; Vi.y .= V.y; Pi.t .= P.t; di .= d
    ϕ0 = merit()
    α  = α0
    while true
        V.x .= Vi.x; V.y .= Vi.y; P.t .= Pi.t; d .= di
        UpdateSolution!(V, P, d, α .* dx, number, type, nc)
        Residuals!(R, V, P, d, old, rheo, materials, type, BC, nc, Δ)
        ϕ = merit()
        (ϕ < ϕ0 || ϕ < ϕ_accept || α * ρ_LS < αmin) && return α, ϕ / ϕ0
        α *= ρ_LS
    end
end
