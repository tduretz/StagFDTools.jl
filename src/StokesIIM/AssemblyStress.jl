# Alternative IIM formulation in the style of the StagFDTools Stokes module ("stress form"):
# unknowns v and p, residuals are the divergence of the stress.
#   - Vx is interpolated to the Vy nodes and Vy to the Vx nodes (4-point averages), so that all
#     components of ∇v are available at the 2 centroids and the 2 vertices of a momentum stencil;
#   - τ = 𝐷 ε̇ with the deviatoric ε̇ = (ε̇xx, ε̇yy, ε̇xy) (isotropic: 𝐷 = 2η I), σ = τ - p I;
#   - x-momentum: ∂x σxx + ∂y τxy = 0 (σxx at centroids, τxy at vertices), likewise in y;
#   - continuity: ∇·v + p/ξ = 0, or ∇·v = 0 with a Lagrange multiplier.
# One-sided stencils: in the equation of a node on side s, every value of the stencil on the
# other side is replaced by the extension of the side-s field to x_k (δ = x_k - x_Γ), and all
# stresses of the equation use the material of side s. The kernel therefore only sees smooth
# fields: the strain-rate tensor at every centroid and vertex is one-sided. Two extensions:
#   extend = :z   v_k -> (η_k v_k + s([z] + [∇z]·δ + ½ δᵀ[∇∇z]δ)) / η_s     (z = ηv, default)
#   extend = :v   v_k -> v_k + s([∇v]·δ + ½ δᵀ[∇∇v]δ)                         ([v] = 0)
#   pressure      p_k -> p_k + s([p] + [∇p]·δ)
# (second-order terms with order = 2). Both are exact up to the Taylor remainder, but the
# remainder of :v is O(δ³∇³v) of the other side, which a stiff-side equation multiplies by
# η_s: the error grows with the viscosity contrast. With :z the remainder is O(δ³∇³z), a stress
# gradient, and the error is contrast independent (this is why Landry uses z = ηv).
# The jumps come from `InterfaceJumpsC` / `InterfaceJumpsV` (they need the traction σn,
# reconstructed from one-sided fits) and are linear in the cloud values. Row coefficients are
# obtained by forward-mode AD of the local residual.

import ForwardDiff

# Crossing for the stencil pair (node, member): the one on the fine-lattice edges between them
# closest to the member (as `pair_crossing`); members outside the fine lattice (ghost nodes of
# the wider stencils, near the boundary) take the crossing closest to them
function member_crossing(lookup, F0, F1, xk, J)
    Imin, Imax = minmax(F0[1], F1[1])
    Jmin, Jmax = minmax(F0[2], F1[2])
    best, dbest = 0, Inf
    for I in Imin:Imax, Jj in Jmin:Jmax, (dir, dI, dJ) in ((1, 1, 0), (2, 0, 1))
        (I + dI <= Imax && Jj + dJ <= Jmax) || continue
        c = get(lookup, (I, Jj, dir), 0)
        c == 0 && continue
        d = norm(J[c].xΓ - xk)
        d < dbest && ((best, dbest) = (c, d))
    end
    best == 0 && (best = argmin(c -> norm(J[c].xΓ - xk), eachindex(J)))
    return best
end

# Grid node of field f at position x (positions of the stencil are exact multiples of Δ/2)
function node_index(f, x, X, Δ)
    g = f == :x ? X.vx_e : f == :y ? X.vy_e : X.c_e
    return CartesianIndex(round(Int, (x[1] - g.x[1]) / Δ.x) + 1, round(Int, (x[2] - g.y[1]) / Δ.y) + 1)
end

# Velocity gradient (∂x vx, ∂y vy, ∂y vx, ∂x vy) at a centroid or a vertex, as in the Stokes
# module: direct differences, and differences of V̄x (Vx averaged to Vy nodes) and V̄y (Vy
# averaged to Vx nodes). `vx(x)`, `vy(x)` return the (one-sided) values at the nodes at x.
function velocity_gradient(vx, vy, x, Δ, ::Val{:c})
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    V̄y(x) = (vy(x + ex + ey) + vy(x - ex + ey) + vy(x + ex - ey) + vy(x - ex - ey)) / 4
    V̄x(x) = (vx(x + ex + ey) + vx(x - ex + ey) + vx(x + ex - ey) + vx(x - ex - ey)) / 4
    return ((vx(x + ex) - vx(x - ex)) / Δ.x, (vy(x + ey) - vy(x - ey)) / Δ.y,
            (V̄x(x + ey) - V̄x(x - ey)) / Δ.y, (V̄y(x + ex) - V̄y(x - ex)) / Δ.x)
end
function velocity_gradient(vx, vy, x, Δ, ::Val{:v})
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    V̄y(x) = (vy(x + ex + ey) + vy(x - ex + ey) + vy(x + ex - ey) + vy(x - ex - ey)) / 4
    V̄x(x) = (vx(x + ex + ey) + vx(x - ex + ey) + vx(x + ex - ey) + vx(x - ex - ey)) / 4
    return ((V̄x(x + ex) - V̄x(x - ex)) / Δ.x, (V̄y(x + ey) - V̄y(x - ey)) / Δ.y,
            (vx(x + ey) - vx(x - ey)) / Δ.y, (vy(x + ex) - vy(x - ex)) / Δ.x)
end

# Deviatoric stress (τxx, τyy, τxy) from the velocity gradient
function deviatoric_stress(D, 𝐷)
    Dxx, Dyy, Dxy, Dyx = D
    θ = Dxx + Dyy
    return 𝐷 * SVector(Dxx - θ / 3, Dyy - θ / 3, (Dxy + Dyx) / 2)
end

# Local residuals; vx, vy, p: one-sided value getters by position, mat = (𝐷, ξ) of the row's side
function residual(::Val{:x}, vx, vy, p, x0, Δ, mat)
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    σxx(xc) = deviatoric_stress(velocity_gradient(vx, vy, xc, Δ, Val(:c)), mat.𝐷)[1] - p(xc)
    τxy(xv) = deviatoric_stress(velocity_gradient(vx, vy, xv, Δ, Val(:v)), mat.𝐷)[3]
    return (σxx(x0 + ex) - σxx(x0 - ex)) / Δ.x + (τxy(x0 + ey) - τxy(x0 - ey)) / Δ.y
end
function residual(::Val{:y}, vx, vy, p, x0, Δ, mat)
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    σyy(xc) = deviatoric_stress(velocity_gradient(vx, vy, xc, Δ, Val(:c)), mat.𝐷)[2] - p(xc)
    τxy(xv) = deviatoric_stress(velocity_gradient(vx, vy, xv, Δ, Val(:v)), mat.𝐷)[3]
    return (τxy(x0 + ex) - τxy(x0 - ex)) / Δ.x + (σyy(x0 + ey) - σyy(x0 - ey)) / Δ.y
end
function residual(::Val{:p}, vx, vy, p, x0, Δ, mat)
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    return (vx(x0 + ex) - vx(x0 - ex)) / Δ.x + (vy(x0 + ey) - vy(x0 - ey)) / Δ.y + p(x0) / mat.ξ
end

"""
    StressJumpFunctionals(tr, η, ξ)

For every crossing, the jumps `z[a, m]`, `∇z[a, b, m]`, `∇∇z[a, b, c, m]` (`InterfaceJumpsC`),
`∇v[a, b, m]`, `∇∇v[a, b, c, m]` (`InterfaceJumpsV`), `p[m]` and `∇p[b, m]` as linear functions
of the cloud values u = [vx(idx_x); vy(idx_y); p(idx_p)] (`field[m]` ∈ :x, :y, :p). The traces of `InterfaceTraces` act on z = ηv; on v they are scaled by
the viscosity of each node's side. σn is averaged from the one-sided fits of both sides.
"""
function StressJumpFunctionals(tr, η, ξ)
    k  = isnothing(ξ) ? (neg=0.0, pos=0.0) : (neg=η.neg / ξ.neg, pos=η.pos / ξ.pos)
    ηs(s) = s < 0 ? η.neg : η.pos
    return map(tr) do T
        nx, ny, np = length(T.x.idx), length(T.y.idx), length(T.pt.idx)
        N  = nx + ny + np
        n, κ = T.p.n, T.p.κ
        t  = tangent(n)
        J∇v, J∇∇v, Jp, J∇p = zeros(2, 2, N), zeros(2, 2, 2, N), zeros(N), zeros(2, N)
        Jz, J∇z, J∇∇z = zeros(2, N), zeros(2, 2, N), zeros(2, 2, 2, N)
        for m in 1:N
            a, kk, Tr = m <= nx ? (1, m, T.x) : m <= nx + ny ? (2, m - nx, T.y) : (0, m - nx - ny, T.pt)
            is = Tr.s[kk] < 0 ? 1 : 2
            G, dG, ps, dps = zeros(2, 2, 2), zeros(2, 2, 2), zeros(2), zeros(2)
            vΓ, w, w2 = zeros(2), zeros(2), zeros(2)
            if a > 0                                       # unit vx or vy value: z = η v there
                ηk = ηs(Tr.s[kk])
                vΓ[a], w[a], w2[a] = ηk * Tr.v[kk], ηk * Tr.t[kk], ηk * Tr.tt[kk]
                G[a, :, is]  .= ηk .* Tr.g[:, kk]
                dG[a, :, is] .= ηk .* (Tr.H[:, :, kk] * t)
            else
                ps[is], dps[is] = Tr.λ[kk], dot(Tr.g[:, kk], t)
            end
            σs(G, p) = -p * I + G + G' - 2 / 3 * LinearAlgebra.tr(G) * I   # `tr` is the trace list here
            Tn  = sum(σs(SMatrix{2,2}(G[:, :, i]), ps[i]) * n for i in 1:2) / 2
            dTn = sum(σs(SMatrix{2,2}(dG[:, :, i]), dps[i]) * n + κ * σs(SMatrix{2,2}(G[:, :, i]), ps[i]) * t for i in 1:2) / 2
            J = InterfaceJumpsV(η, k, SVector(w...), SVector(w2...), Tn, dTn, n, κ)
            Jc = InterfaceJumpsC(η, k, SVector(vΓ...), SVector(w...), SVector(w2...), Tn, dTn, n, κ)
            Jz[:, m] .= Jc.z
            J∇z[:, :, m] .= Jc.∇z
            J∇∇z[:, :, :, m] .= Jc.∇∇z
            J∇v[:, :, m] .= J.∇v
            J∇∇v[:, :, :, m] .= J.∇∇v
            Jp[m] = J.p
            J∇p[:, m] .= J.∇p
        end
        (xΓ=T.p.x, idx=vcat(T.x.idx, T.y.idx, T.pt.idx), field=vcat(fill(:x, nx), fill(:y, ny), fill(:p, np)),
         z=Jz, ∇z=J∇z, ∇∇z=J∇∇z, ∇v=J∇v, ∇∇v=J∇∇v, p=Jp, ∇p=J∇p)
    end
end

# Free-slip ghosts (`bc = :free_slip`): a tangential velocity ghost outside a wall takes the
# value of its mirror node inside the domain (∂ₙvₜ = 0, zero tangential stress for a straight
# wall with constant normal velocity). Returns the mirror index, or `nothing`.
function mirror_ghost(f, I, type, nc)
    if f == :x
        i, j = Tuple(I)
        j < 3 && type.Vx[i, 2] == :Neumann_tangent && return CartesianIndex(i, 5 - j)
        j > nc.y + 2 && type.Vx[i, nc.y + 3] == :Neumann_tangent && return CartesianIndex(i, 2nc.y + 5 - j)
    elseif f == :y
        i, j = Tuple(I)
        i < 3 && type.Vy[2, j] == :Neumann_tangent && return CartesianIndex(5 - i, j)
        i > nc.x + 2 && type.Vy[nc.x + 3, j] == :Neumann_tangent && return CartesianIndex(2nc.x + 5 - i, j)
    end
    return nothing
end

# One-sided stencil of a kernel evaluated for a node of side `s0` at fine-lattice index `F0`.
# `kernel(vx, vy, p)` reads node values through getters by position. Pass 1 records, for every
# node it reads, a linear form Σ w u[loc] + c over local node values (`nodes`): the node itself
# on side s0, its extension when it lies across Γ, its mirror node for a free-slip ghost.
# Returns `value(u, f, x)`, the getter of those forms for local values `u`, and `nodes`.
function onesided_forms(ctx, s0, F0, kernel)
    (; X, Δ, φf, J, tr, lookup, ηof, order, extend, corrections, Jexact, type, nc) = ctx
    loc   = Dict{Tuple{Symbol,CartesianIndex{2}},Int}()
    nodes = Tuple{Symbol,CartesianIndex{2}}[]
    local!(key) = get!(loc, key) do
        push!(nodes, key); length(nodes)
    end
    forms = Dict{Tuple{Symbol,CartesianIndex{2}},Tuple{Vector{Int},Vector{Float64},Float64}}()
    function member(f, x)
        I   = node_index(f, x, X, Δ)
        key = (f, I)
        haskey(forms, key) && return 0.0
        Im = mirror_ghost(f, I, type, nc)
        if !isnothing(Im)
            member(f, node_position(Val(f), Im, X))
            forms[key] = forms[(f, Im)]
            return 0.0
        end
        ids, ws, c0 = [local!(key)], [1.0], 0.0
        if corrections && side(φf[f][I]) != s0
            xk = node_position(Val(f), I, X)
            c  = member_crossing(lookup, F0, fine_index(Val(f), I), xk, J)
            Jc = J[c]
            δ  = xk - Jc.xΓ
            a  = f == :x ? 1 : 2
            # z-extension: v = (η_k v_k + s(...)) / η_s, the jump terms scaled by 1/η_s
            sc = f != :p && extend == :z ? 1 / ηof(s0) : 1.0
            f != :p && extend == :z && (ws[1] = ηof(-s0) / ηof(s0))
            jump(z, ∇z, ∇∇z, ∇v, ∇∇v, p, ∇p) = f == :p ? p + (order == 2 ? dot(∇p, δ) : 0.0) :
                extend == :z ? z[a] + dot(∇z[a, :], δ) + (order == 2 ? dot(δ, ∇∇z[a, :, :] * δ) / 2 : 0.0) :
                               dot(∇v[a, :], δ) + (order == 2 ? dot(δ, ∇∇v[a, :, :] * δ) / 2 : 0.0)
            if !isnothing(Jexact)
                Je = Jexact(tr[c].p)
                c0 = s0 * sc * jump(Je.z, Je.∇z, Je.∇∇z, Je.∇v, Je.∇∇v, Je.p, Je.∇p)
            else
                for m in eachindex(Jc.p)
                    wm = jump(view(Jc.z, :, m), view(Jc.∇z, :, :, m), view(Jc.∇∇z, :, :, :, m),
                              view(Jc.∇v, :, :, m), view(Jc.∇∇v, :, :, :, m), Jc.p[m], view(Jc.∇p, :, m))
                    push!(ids, local!((Jc.field[m], Jc.idx[m]))); push!(ws, s0 * sc * wm)
                end
            end
        end
        forms[key] = (ids, ws, c0)
        return 0.0
    end
    kernel(x -> member(:x, x), x -> member(:y, x), x -> member(:p, x))
    function value(u, f, x)
        ids, ws, c0 = forms[(f, node_index(f, x, X, Δ))]
        return sum(ws[i] * u[ids[i]] for i in eachindex(ids)) + c0
    end
    return value, nodes
end

"""
    AssembleIIMStress(φ, X, Δ, η, nc, vbc; ξ=nothing, order=1, fit=:linear, extend=:z, corrections=true, Jexact=nothing, bc=:all_Dirichlet)

Assemble the stress-form IIM system for v and p. `vbc = (x, y)` holds v on the full Vx/Vy
arrays (values of the non-numbered nodes). `bc`: `:all_Dirichlet` (ghost values from `vbc`) or
`:free_slip` (normal velocities from `vbc`, tangential ghosts mirrored: zero shear stress).
`η`, `ξ` as in `AssembleIIM`; isotropic viscosity (𝐷 = 2η I). `extend`: `:z` or `:v` (see above).
Diagnostic: `Jexact(p)` returns exact jumps at crossing `p` (fields of `InterfaceJumpsC` and of
`InterfaceJumpsV`). Returns the matrix, right-hand side, numbering, interface data and the
context `ctx` used by `CentroidStress`.
"""
function AssembleIIMStress(φ, X, Δ, η, nc, vbc; ξ=nothing, order=1, fit=:linear, extend=:z, corrections=true, Jexact=nothing, bc=:all_Dirichlet)
    order == 2 && fit == :linear && isnothing(Jexact) && error("order = 2 requires fit = :quadratic or :cubic")
    size_x, size_y, size_c = size(φ.x), size(φ.y), size(φ.c)
    type   = Fields(fill(:out, size_x...), fill(:out, size_y...), fill(:out, size_c...))
    number = Fields(zeros(Int, size_x...), zeros(Int, size_y...), zeros(Int, size_c...))
    set_boundaries_template!(type, bc, nc)
    Numbering!(number, type, nc)
    nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
    off  = (x=0, y=nVx, p=nVx + nVy)
    comp = !isnothing(ξ)
    ndof = nVx + nVy + nPt + !comp
    num  = (x=number.Vx, y=number.Vy, p=number.Pt)
    φf   = (x=φ.x, y=φ.y, p=φ.c)
    val0 = (x=vbc.x, y=vbc.y, p=zeros(size_c))
    ηof(s) = s < 0 ? η.neg : η.pos
    mats = (neg=(𝐷=2η.neg * SMatrix{3,3}(1.0I), ξ=comp ? ξ.neg : Inf),
            pos=(𝐷=2η.pos * SMatrix{3,3}(1.0I), ξ=comp ? ξ.pos : Inf))

    Γ  = InterfaceCrossings(FineLattice(φ, X, Δ))
    tr = InterfaceTraces(Γ, φ, X, Δ, η, nc; fit, pressure=true, borrow=true)
    J  = StressJumpFunctionals(tr, η, ξ)
    lookup = Dict((T.p.I, T.p.J, T.p.dir == :x ? 1 : 2) => c for (c, T) in enumerate(tr))
    ctx = (; X, Δ, φ, φf, J, tr, lookup, ηof, mats, order, extend, corrections, Jexact, type, nc)

    Is, Js, Vs = Int[], Int[], Float64[]
    b = zeros(ndof)
    for f0 in (:x, :y, :p), I0 in CartesianIndices(num[f0])
        (row = num[f0][I0]) > 0 || continue
        s0  = side(φf[f0][I0])
        x0  = node_position(Val(f0), I0, X)
        mat = s0 < 0 ? mats.neg : mats.pos
        kernel(vx, vy, p) = residual(Val(f0), vx, vy, p, x0, Δ, mat)
        value, nodes = onesided_forms(ctx, s0, fine_index(Val(f0), I0), kernel)
        # residual as a function of the local node values, differentiated by AD
        R(u) = kernel(x -> value(u, :x, x), x -> value(u, :y, x), x -> value(u, :p, x))
        u0 = [num[f][I] > 0 ? 0.0 : val0[f][I] for (f, I) in nodes]
        g  = ForwardDiff.gradient(R, u0)
        r  = off[f0] + row
        b[r] -= R(u0)
        for (j, (f, I)) in enumerate(nodes)
            num[f][I] > 0 && g[j] != 0 && (push!(Is, r); push!(Js, off[f] + num[f][I]); push!(Vs, g[j]))
        end
        f0 == :p && !comp && (push!(Is, r, ndof); push!(Js, ndof, r); push!(Vs, 1.0, 1.0))
    end
    K = sparse(Is, Js, Vs, ndof, ndof)
    return (K=K, b=b, num=num, off=off, Γ=Γ, tr=tr, J=J, lookup=lookup, ctx=ctx)
end

"""
    CentroidStress(S, v, p)

Deviatoric stress (τxx, τyy, τxy) at the centroids (grid `c_e`) of a stress-form solution, with
the one-sided stencils of the equations: every velocity of the centroid's strain-rate stencil
that lies across Γ is replaced by the extension of the centroid side's field (jumps evaluated on
the solution), and τ = 𝐷 ε̇ uses the centroid side's material. τxy uses the Vx/Vy interpolations
of the Stokes module (no vertex averaging), so the field contains no mixed-phase strain rates.
`S` is the output of `AssembleIIMStress` (`SolveIIMStress(...).S`). Returns three arrays on `c_e`
(NaN on the ghost centroids).
"""
function CentroidStress(S, v, p)
    (; X, Δ, φ, mats) = S.ctx
    val = (x=v.x, y=v.y, p=p)
    τ   = ntuple(_ -> fill(NaN, size(φ.c)), 3)
    for j in 2:(size(φ.c, 2) - 1), i in 2:(size(φ.c, 1) - 1)
        I0  = CartesianIndex(i, j)
        s0  = side(φ.c[I0])
        x0  = node_position(Val(:p), I0, X)
        mat = s0 < 0 ? mats.neg : mats.pos
        kernel(vx, vy, pp) = deviatoric_stress(velocity_gradient(vx, vy, x0, Δ, Val(:c)), mat.𝐷)
        value, nodes = onesided_forms(S.ctx, s0, fine_index(Val(:p), I0), kernel)
        u  = [val[f][I] for (f, I) in nodes]
        τc = kernel(x -> value(u, :x, x), x -> value(u, :y, x), x -> value(u, :p, x))
        for k in 1:3
            τ[k][I0] = τc[k]
        end
    end
    return (xx=τ[1], yy=τ[2], xy=τ[3])
end

"""
    VertexPressure(S, v, p, φ, X; onesided=true, order=2)

Pressure at the vertices (grid `v_e`) by averaging the 4 surrounding centroid pressures, as the
Stokes module does for the pressure column of 𝐷 (`Pt_v = av(Pt)`). With `onesided`, centroids
on the other side of Γ than the vertex are first replaced by the extension of the vertex side's
pressure, p_k + s([p] + [∇p]·δ) (δ = x_k - x_Γ, [∇p] term with `order = 2`), the same rule as for
the stencil values of `AssembleIIMStress`; otherwise the plain average, which mixes p₋ and p₊.
`S` is the output of `AssembleIIMStress`; `v`, `p` the solution (jumps are evaluated on it).
Returns an array on `v_e`, NaN where the 4 centroids are not all inside the domain.
"""
function VertexPressure(S, v, p, φ, X; onesided=true, order=2)
    val = (x=v.x, y=v.y, p=p)
    Pv  = fill(NaN, size(φ.v))
    for j in 3:(size(φ.v, 2) - 2), i in 3:(size(φ.v, 1) - 2)
        sv, xv = side(φ.v[i, j]), SVector(X.v_e.x[i], X.v_e.y[j])
        Pv[i, j] = sum(CartesianIndex(i + di, j + dj) for di in -1:0, dj in -1:0) do K
            (onesided && side(φ.c[K]) != sv) || return p[K] / 4
            xk = node_position(Val(:p), K, X)
            Jc = S.J[member_crossing(S.lookup, (2i - 1, 2j - 1), fine_index(Val(:p), K), xk, S.J)]
            δ  = xk - Jc.xΓ
            jp = sum((Jc.p[m] + (order == 2 ? dot(Jc.∇p[:, m], δ) : 0.0)) * val[Jc.field[m]][Jc.idx[m]] for m in eachindex(Jc.p))
            (p[K] + sv * jp) / 4
        end
    end
    return Pv
end

"""
    SolveIIMStress(φ, X, Δ, η, nc, vbc; ξ=nothing, order=1, fit=:linear, extend=:z, corrections=true, Jexact=nothing, bc=:all_Dirichlet)

Assemble the stress-form system and solve with a sparse direct solver. Returns v (with the
boundary values of `vbc`) and p on the full staggered arrays. The solver sees the velocity
columns scaled by η and the momentum rows by 1/η (η of each node's side), i.e. unknowns z = ηv:
without it, the fill-in of the LU factors grows with the contrast (2.4 times more at 10³).
"""
function SolveIIMStress(φ, X, Δ, η, nc, vbc; ξ=nothing, order=1, fit=:linear, extend=:z, corrections=true, Jexact=nothing, bc=:all_Dirichlet)
    S   = AssembleIIMStress(φ, X, Δ, η, nc, vbc; ξ, order, fit, extend, corrections, Jexact, bc)
    dη  = ones(size(S.K, 1))
    for (f, φf) in ((:x, φ.x), (:y, φ.y)), I in CartesianIndices(φf)
        S.num[f][I] > 0 && (dη[S.off[f] + S.num[f][I]] = side(φf[I]) < 0 ? η.neg : η.pos)
    end
    sol = (Diagonal(1 ./ dη) * S.K * Diagonal(1 ./ dη) \ (S.b ./ dη)) ./ dη
    v   = (x=copy(vbc.x), y=copy(vbc.y))
    p   = zeros(size(φ.c))
    for (f, a) in ((:x, v.x), (:y, v.y), (:p, p)), I in CartesianIndices(a)
        S.num[f][I] > 0 && (a[I] = sol[S.off[f] + S.num[f][I]])
    end
    return (v=v, p=p, S=S)
end
