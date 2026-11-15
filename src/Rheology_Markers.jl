# Constitutive update on the markers (materials.mixing = :markers, :markers_stress, :markers_work), MPM-like
#
# Every marker carries its own history: deviatoric stress τ0ₘ, pressure P0ₘ and plastic strain εpₘ (TensorMarkers,
# tm.p.εp), and the rheology of its own phase. The stress is computed on the markers and averaged on the grid:
#     τ(control volume) = (1/n) Σₘ τₘ(ε̇),   D = (1/n) Σₘ ∂τₘ/∂ε̇
# over the n markers of the control volume of the stress point (centroid cell for τxx, τyy, P; dual cell of the vertex
# for τxy). All markers of a control volume see the strain rate of that control volume (piecewise constant, as the
# staggered FD stencil): this is iso-strain rate (Voigt) with the markers as the "phases". It keeps the FD stencil and a
# local consistent tangent; interpolating ε̇ to the markers (bilinear, as MPM) would couple neighbouring stress points
# through the tangent.
# Between the phases of a control volume, :markers_stress and :markers_work partition the strain rate (iso-stress,
# iso-work; see marker_partition below), :markers keeps it uniform (iso-strain).
# Differences with the grid schemes:
#  - no phase fractions, no mixing rule, no per-phase grid fields: the material is resolved at the marker scale;
#  - the marker stress is the computed one (no grid → marker interpolation of increments, no subgrid diffusion);
#  - plastic strain (hardening) and pressure are carried by each marker.
# Pressure: marker trial pressure P0ₘ + (β̄/βₘ)(P - P0) with β̄ the harmonic mean of the βₘ of the control volume and
# P0 = mean P0ₘ (set by stress_ToGrid_Markers!): the trial pressures average to the cell trial pressure P.
# Cost: one rheology evaluation (with derivatives) per marker and per stress point, i.e. ~2 × markers per cell
# (vs. 1 to a few phase evaluations for the grid schemes).

struct MarkerEngine{P,PH,TM,O,X}
    particles::P
    phases::PH
    tm::TM
    out::O      # marker results of the last evaluation: τxx, τyy, P, ε̇pl (centroids), τxy (vertices)
    xv::X       # vertex coordinates (with ghosts), as the vertex arrays
    Δx::Float64
    Δy::Float64
end

const marker_engine_cache = Ref{Any}(nothing)

function marker_engine(a, particles, phases, tm)
    e = marker_engine_cache[]
    if e === nothing || e.particles !== particles
        out = cell_fields(particles, :τxx, :τyy, :τxy, :P, :ε̇pl)
        X = a.X.v_e
        e = MarkerEngine(particles, phases, tm, out, (x=collect(X.x), y=collect(X.y)), step(X.x), step(X.y))
        marker_engine_cache[] = e
    end
    return e
end

# Stress of one marker of phase p. ε̇ = (ε̇xx, ε̇yy, ε̇xy, P) true deviatoric strain rate of the control volume and its
# trial pressure; P0c: old pressure of the control volume; βr = β̄/βₚ
@inline function marker_stress(ε̇, ε̇kk, P0c, τ0xx, τ0yy, τ0xy, P0m, βr, materials, p, Δ, Tk, εpm, Ω)
    G = materials.G[p]
    _2GΔt = inv(2 * G * Δ.t)
    ϵ̇xx = ε̇[1] + τ0xx * _2GΔt
    ϵ̇yy = ε̇[2] + τ0yy * _2GΔt
    ϵ̇xy = ε̇[3] + τ0xy * _2GΔt
    ϵ̇II = sqrt((ϵ̇xx^2 + ϵ̇yy^2 + (-ϵ̇xx - ϵ̇yy)^2) / 2 + ϵ̇xy^2 + 1e-34)
    Ptr = materials.compressible ? P0m + βr * (ε̇[4] - P0c) : one(P0m) * ε̇[4]
    τII, Pp, λ̇, ε̇pl = phase_vep(ϵ̇II, Ptr, ε̇kk, P0m, G, materials, p, Δ, Tk, εpm, Ω)
    η = τII / (2 * ϵ̇II)
    return SVector{4}(2η * ϵ̇xx, 2η * ϵ̇yy, 2η * ϵ̇xy, Pp), η, λ̇, τII, ε̇pl
end

@inline _in_cv(x, y, xc, yc, hx, hy) = abs(x - xc) <= hx && abs(y - yc) <= hy

const mk_maxph = 8 # max. number of phases (marker partition)

# Strain rate of a phase: cell strain rate scaled by the multiplier a (deviatoric part only), or its own deviatoric
# strain rate e = (ε̇xx, ε̇yy, ε̇xy) (tensor iso-stress); the trial pressure stays the cell one
@inline _scaled(ε̇, a::Real) = SVector{4}(a * ε̇[1], a * ε̇[2], a * ε̇[3], one(a) * ε̇[4])
@inline _scaled(ε̇, e::SVector{3}) = SVector{4}(e[1], e[2], e[3], one(e[1]) * ε̇[4])

# Sum over the markers of cell (ic, jc) inside the box (xc ± hx, yc ± hy) (no box: hx = Inf)
# mode Val(1): count, Σ 1/β, count per phase
#      Val(2): iso-strain, stress and tangent of every marker
#      Val(3): stresses with the strain rate multipliers A[phase] (values or duals), results stored if store
#      Val(4): stresses of the markers of phase pf only, multiplier A[pf]
@inline function _cv_cell(mk, acc, mode, ic, jc, xc, yc, hx, hy, ε̇, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, loc, store, A, pf)
    (; particles, phases, tm, out) = mk
    (; index, coords) = particles
    (1 <= ic <= size(index, 1) && 1 <= jc <= size(index, 2)) || return acc
    for ip in cellaxes(index)
        @index(index[ip, ic, jc]) || continue
        x = @index coords[1][ip, ic, jc]
        y = @index coords[2][ip, ic, jc]
        _in_cv(x, y, xc, yc, hx, hy) || continue
        p = round(Int, @index(phases[ip, ic, jc]))
        if mode isa Val{1}
            acc = (acc[1] + 1, acc[2] + inv(materials.β[p]), setindex(acc[3], acc[3][p] + 1, p))
            continue
        end
        mode isa Val{4} && p != pf && continue
        τ0xx = @index tm.p.τxx[ip, ic, jc]
        τ0yy = @index tm.p.τyy[ip, ic, jc]
        τ0xy = @index tm.p.τxy[ip, ic, jc]
        P0m  = @index tm.p.P[ip, ic, jc]
        εpm  = @index tm.p.εp[ip, ic, jc]
        βr   = βbar / materials.β[p]
        if mode isa Val{2}
            τm, ηm, λ̇m, _, ε̇plm = marker_stress(ε̇, ε̇kk, P0c, τ0xx, τ0yy, τ0xy, P0m, βr, materials, p, Δ, Tk, εpm, Ω)
            _, Jm = fd_jacobian_first(marker_stress, ε̇, ε̇kk, P0c, τ0xx, τ0yy, τ0xy, P0m, βr, materials, p, Δ, Tk, εpm, Ω)
            store && _store!(out, loc, ip, ic, jc, τm, ε̇plm)
            acc = (acc[1] + 1, acc[2] + τm, acc[3] + Jm, acc[4] + ηm, acc[5] + λ̇m)
        else
            τm, ηm, λ̇m, _, ε̇plm = marker_stress(_scaled(ε̇, A[p]), ε̇kk, P0c, τ0xx, τ0yy, τ0xy, P0m, βr, materials, p, Δ, Tk, εpm, Ω)
            if mode isa Val{3}
                store && _store!(out, loc, ip, ic, jc, τm, ε̇plm)
                acc = (acc[1] + 1, acc[2] + τm, acc[3] + ηm, acc[4] + λ̇m)
            else
                acc = (acc[1] + 1, acc[2] + τm)
            end
        end
    end
    return acc
end

@inline function _store!(out, loc, ip, ic, jc, τm, ε̇plm)
    if loc === :c
        @index out.τxx[ip, ic, jc] = value_of(τm[1])
        @index out.τyy[ip, ic, jc] = value_of(τm[2])
        @index out.P[ip, ic, jc]   = value_of(τm[4])
        @index out.ε̇pl[ip, ic, jc] = value_of(ε̇plm)
    else
        @index out.τxy[ip, ic, jc] = value_of(τm[3])
    end
    return nothing
end
@inline value_of(x::Real) = x

# Loop over the cells of the control volume: centroid (i, j): cell (i, j); vertex (i, j): cells (i-1:i, j-1:j), markers
# within half a cell of the vertex. If a vertex control volume holds no marker (boundary), the markers of the
# 4 surrounding cells are used (results not stored: they belong to the neighbouring vertices).
function _cv_sum(mk, mode, i, j, loc, ε̇, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, acc0, wide=false, store=true, A=nothing, pf=0)
    acc = acc0
    if loc === :c
        acc = _cv_cell(mk, acc, mode, i, j, 0.0, 0.0, Inf, Inf, ε̇, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, loc, store, A, pf)
    else
        xv, yv = mk.xv.x[i], mk.xv.y[j]
        hx, hy = wide ? (Inf, Inf) : (0.5 * mk.Δx, 0.5 * mk.Δy)
        for jc in (j - 1):j, ic in (i - 1):i
            acc = _cv_cell(mk, acc, mode, ic, jc, xv, yv, hx, hy, ε̇, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, loc, store && !wide, A, pf)
        end
    end
    return acc
end

# ------------------------------------------------------------------------------------------------------------------
# Partition between the phases of a control volume (mixing = :markers_stress, :markers_work)
#
# The markers of phase i all see the strain rate aᵢ ε̇, with Σᵢ φᵢ aᵢ = 1 (φᵢ: fraction of the markers of the control
# volume that belong to phase i). With τ̄ᵢ(aᵢ) the mean stress of the markers of phase i:
#   iso-stress: tensor partition, see below (the scalar version τ̄ᵢ(aᵢ):ε̇ is ill-conditioned when the stresses
#               carried by the phases are not aligned with ε̇)
#   iso-work:   hᵢ = aᵢ (τ̄ᵢ(aᵢ) - τ̄ᵢ(0)) : ε̇  (as the grid iso-work, Rheology_IsoWork.jl)
# both increasing in aᵢ (positive tangents); hᵢ is the same for all phases (Newton, closed form as phase_multipliers).
# Markers of the same phase share the strain rate (the rule acts between phases).

struct MkStress end
struct MkWork end

# Mean stress of the markers of phase p for the multiplier a
function _mk_phase_mean(a, p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
    T = promote_type(eltype(a), eltype(ε̇))
    A = SVector{mk_maxph}(ntuple(_ -> a, mk_maxph)) # only entry p is used (markers of phase p)
    n, τs = _cv_sum(mk, Val(4), i, j, loc, ε̇, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, (0, zero(SVector{4,T})), wide, false, A, p)
    return τs / n
end

function _mk_h(a, ::MkWork, p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
    τa = _mk_phase_mean(a, p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
    τr = _mk_phase_mean(zero(a), p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
    return a * _power(τa - τr, ε̇)
end

# iso-work: hᵢ(0) = 0 and increasing for aᵢ > 0, the multipliers stay positive
@inline _mk_update(::MkWork, a, δ) = max(a + δ, 0.1 * a)

function marker_partition(rule::MkWork, ε̇::SVector{4,T}, cnt, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide; tol=1e-10, niter=30) where {T}
    a = SVector{mk_maxph,T}(ntuple(_ -> one(T), mk_maxph))
    nphases = length(materials.n)
    nact = count(>(0), cnt)
    (nact <= 1 || _power(ε̇, ε̇) == 0) && return a
    n = sum(cnt)
    h  = SVector{mk_maxph,T}(ntuple(_ -> zero(T), mk_maxph))
    dh = h
    for it = 1:niter
        for p = 1:nphases
            cnt[p] > 0 || continue
            hp, dhp = fd_value_and_derivative(_mk_h, a[p], rule, p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
            dhp > 0 || return SVector{mk_maxph,T}(ntuple(_ -> one(T), mk_maxph)) # safeguard: iso-strain
            h  = setindex(h, hp, p)
            dh = setindex(dh, dhp, p)
        end
        num, den, Σφa = zero(T), zero(T), zero(T)
        for p = 1:nphases
            cnt[p] > 0 || continue
            φ = cnt[p] / n
            num += φ * h[p] / dh[p]
            den += φ / dh[p]
            Σφa += φ * a[p]
        end
        H = (1 - Σφa + num) / den
        err = zero(T)
        for p = 1:nphases
            cnt[p] > 0 || continue
            err = max(err, abs(h[p] - H))
            a = setindex(a, _mk_update(rule, a[p], (H - h[p]) / dh[p]), p)
        end
        (err < tol * abs(H) && abs(Σφa - 1) < tol) && break
    end
    return a
end

# Tensor iso-stress (MkStress): phase i has its own deviatoric strain rate eᵢ, all phases carry the same mean stress τ̄
# and Σφᵢeᵢ = ε̇. Newton with the phase tangents Dᵢ = ∂τ̄ᵢ/∂eᵢ (3×3), closed form:
#     τ* = (ΣφᵢDᵢ⁻¹)⁻¹ (ε̇ - Σφᵢeᵢ + ΣφᵢDᵢ⁻¹τ̄ᵢ),   eᵢ += Dᵢ⁻¹(τ* - τ̄ᵢ)
function _mk_phase_tau3(e, p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
    τ = _mk_phase_mean(e, p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
    return SVector{3}(τ[1], τ[2], τ[3])
end

function marker_partition(rule::MkStress, ε̇::SVector{4,T}, cnt, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide; tol=1e-10, niter=30) where {T}
    e3 = SVector{3}(ε̇[1], ε̇[2], ε̇[3])
    E  = SVector{mk_maxph}(ntuple(_ -> e3, mk_maxph))
    nphases = length(materials.n)
    count(>(0), cnt) <= 1 && return E
    n = sum(cnt)
    scale = sqrt(_power(ε̇, ε̇)) + 1e-30
    for it = 1:niter
        S  = zero(SMatrix{3,3,T})
        r  = e3
        τs = SVector{mk_maxph}(ntuple(_ -> zero(e3), mk_maxph))
        Cs = SVector{mk_maxph}(ntuple(_ -> zero(SMatrix{3,3,T}), mk_maxph))
        for p = 1:nphases
            cnt[p] > 0 || continue
            φ = cnt[p] / n
            τp, Dp = fd_value_and_jacobian(_mk_phase_tau3, E[p], p, ε̇, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
            Cp = inv(Dp)
            τs = setindex(τs, τp, p)
            Cs = setindex(Cs, Cp, p)
            S += φ * Cp
            r += φ * (Cp * τp - E[p])
        end
        τstar = S \ r
        err, Σφe = zero(T), zero(e3)
        for p = 1:nphases
            cnt[p] > 0 || continue
            err = max(err, maximum(abs, τs[p] - τstar))
            Σφe += cnt[p] / n * E[p]
            E = setindex(E, E[p] + Cs[p] * (τstar - τs[p]), p)
        end
        (err < tol * (maximum(abs, τstar) + 1e-30) && maximum(abs, Σφe - e3) < tol * scale) && break
    end
    return E
end

# Mean stress (and η, λ̇) of the control volume with the partition: differentiated as a whole for the tangent
function cv_stress(ε̇::SVector{4,T}, rule, cnt, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide) where {T}
    A = marker_partition(rule, ε̇, cnt, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
    n, τs, ηs, λ̇s = _cv_sum(mk, Val(3), i, j, loc, ε̇, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, (0, zero(SVector{4,T}), zero(T), zero(T)), wide, false, A)
    return τs / n, ηs / n, λ̇s / n, A
end

@inline _marker_rule(materials) = _mixing(materials) === :markers_stress ? MkStress() :
                                  _mixing(materials) === :markers_work ? MkWork() : nothing

# Called by TangentOperator! (Stokes.jl) at a centroid (loc = :c) or vertex (loc = :v) when materials.mixing is one of
# :markers (iso-strain), :markers_stress, :markers_work.
# ε̇vec: true deviatoric strain rate and trial pressure of the control volume. Writes the equivalent τ0 used by the
# momentum residual (as Voigt) and returns what the default branch gets from StressVector!.
function TangentOperator_Markers!(mk, τ0, i, j, loc::Symbol, ε̇vec::SVector{4,T}, ε̇kk, P0c, Ggrid, materials, Δ, Tk, Ω) where {T}
    # Harmonic mean of the marker compressibilities, marker count per phase
    wide = false
    acc1 = (0, 0.0, zero(SVector{mk_maxph,Int}))
    n, sβ, cnt = _cv_sum(mk, Val(1), i, j, loc, ε̇vec, ε̇kk, P0c, 1.0, materials, Δ, Tk, Ω, acc1)
    if n == 0 && loc === :v
        wide = true
        n, sβ, cnt = _cv_sum(mk, Val(1), i, j, loc, ε̇vec, ε̇kk, P0c, 1.0, materials, Δ, Tk, Ω, acc1, true)
    end
    n == 0 && error("TangentOperator_Markers!: no marker around $loc point ($i, $j)")
    βbar = n / sβ
    rule = _marker_rule(materials)
    if rule === nothing || count(>(0), cnt) <= 1
        # Iso-strain (or a single phase): sum of the marker stresses and tangents
        acc0 = (0, zero(SVector{4,T}), zero(SMatrix{4,4,T}), zero(T), zero(T))
        n, τs, Js, ηs, λ̇s = _cv_sum(mk, Val(2), i, j, loc, ε̇vec, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, acc0, wide)
        τ_vec, jac, η, λ̇ = τs / n, Js / n, ηs / n, λ̇s / n
    else
        τ_vec, η, λ̇, A = cv_stress(ε̇vec, rule, cnt, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
        _, jac = fd_jacobian_first(cv_stress, ε̇vec, rule, cnt, mk, i, j, loc, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, wide)
        # store the marker results
        _cv_sum(mk, Val(3), i, j, loc, ε̇vec, ε̇kk, P0c, βbar, materials, Δ, Tk, Ω, (0, zero(SVector{4,T}), zero(T), zero(T)), wide, true, A)
    end

    # Equivalent old stress for the momentum residual: 2η(ε̇ + τ0_eq/(2GΔt)) = τ (see Rheology_Voigt.jl)
    s = Ggrid * Δ.t / η
    I = CartesianIndex(i, j)
    if loc === :c
        τ0.xx[I] = (τ_vec[1] - 2 * η * ε̇vec[1]) * s
        τ0.yy[I] = (τ_vec[2] - 2 * η * ε̇vec[2]) * s
    else
        τ0.xy[I] = (τ_vec[3] - 2 * η * ε̇vec[3]) * s
    end
    τII = sqrt((τ_vec[1]^2 + τ_vec[2]^2 + (-τ_vec[1] - τ_vec[2])^2) / 2 + τ_vec[3]^2)
    return τ_vec, jac, η, λ̇, τII
end

# ------------------------------------------------------------------------------------------------------------------
# Marker updates

# Markers → grid: τ0 (diagnostics and Picard initial state), and the old pressure of each cell as the plain mean of
# its markers (consistent with the marker trial pressures, see above)
function stress_ToGrid_Markers!(tm::TensorMarkers, a, mk)
    stress_ToGrid!(tm, a, mk.particles)
    (; index) = mk.particles
    Threads.@threads for j in axes(index, 2)
        for i in axes(index, 1)
            n, s = 0, 0.0
            for ip in cellaxes(index)
                @index(index[ip, i, j]) || continue
                n += 1
                s += @index tm.p.P[ip, i, j]
            end
            n > 0 && (a.Pt0[i, j] = s / n)
        end
    end
    correct_boundaries!(a)
    return nothing
end

# After the solve: the markers take the stresses computed on them (last, converged, evaluation), + objective rotation;
# plastic strain εpₘ += Δt ε̇II_plₘ (plastic part of the deviatoric strain rate invariant of the marker)
function stress_ToParticles_Markers!(a, tm::TensorMarkers, mk, Δ; rotation::Symbol=:upper_convected)
    (; particles, out) = mk
    (; index) = particles
    compute_vorticity!(a, Δ)
    Threads.@threads for j in axes(index, 2)
        for i in axes(index, 1), ip in cellaxes(index)
            @index(index[ip, i, j]) || continue
            τxx, τyy, τxy, P = @index(out.τxx[ip, i, j]), @index(out.τyy[ip, i, j]), @index(out.τxy[ip, i, j]), @index(out.P[ip, i, j])
            @index tm.Δ.τxx[ip, i, j] = τxx - @index(tm.p.τxx[ip, i, j])
            @index tm.Δ.τyy[ip, i, j] = τyy - @index(tm.p.τyy[ip, i, j])
            @index tm.Δ.τxy[ip, i, j] = τxy - @index(tm.p.τxy[ip, i, j])
            @index tm.Δ.P[ip, i, j]   = P - @index(tm.p.P[ip, i, j])
            @index tm.t.τxx[ip, i, j] = τxx
            @index tm.t.τyy[ip, i, j] = τyy
            @index tm.t.τxy[ip, i, j] = τxy
            @index tm.p.εp[ip, i, j]  = @index(tm.p.εp[ip, i, j]) + Δ.t * @index(out.ε̇pl[ip, i, j])
        end
    end
    centroid2particle!(tm.t.P, a.Pt, particles)
    grid2particle!(tm.t.ω, a.ω.xy_v, particles)
    centroid2particle!(tm.t.ε̇xx, a.ε̇.xx, particles)
    centroid2particle!(tm.t.ε̇yy, a.ε̇.yy, particles)
    grid2particle!(tm.t.ε̇xy, a.ε̇.xy, particles)
    rotation_increments!(tm, particles, Δ.t; rotation)
    update_stress_markers!(tm, particles)
    return nothing
end
