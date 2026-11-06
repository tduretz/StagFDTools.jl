# IIM operator for Stokes with piecewise constant viscosity, in z = ηv form.
#
# Regular equations (constant η on each side), cd = 1 (incompressible) or ⅓ (compressible):
#   x-momentum: (1 + cd)∂xx zx + ∂yy zx + cd ∂xy zy - ∂x p = 0
#   y-momentum: ∂xx zy + (1 + cd)∂yy zy + cd ∂xy zx - ∂y p = 0
#   continuity: ∂x zx + ∂y zy + k p = 0,  k = η/ξ (0: incompressible)
# Correction: in the equation of a node on side s, every stencil member k on the other side is
# replaced by the extension of the s-side field to x_k (δ = x_k - x_Γ):
#   order 1:  q_k -> q_k + s ([q] + [∇q]·δ)               ([p] only for pressure members)
#   order 2:  q_k -> q_k + s ([q] + [∇q]·δ + ½ δᵀ[∇∇q]δ)  ([p] + [∇p]·δ for pressure members)
# with [q] = q₊ - q₋ at a crossing x_Γ between the node and k. The jumps are linear in z through
# the interface traces, so the corrections are matrix coefficients on the cloud of each crossing.

using SparseArrays
import StagFDTools.Stokes: Fields, Numbering!, set_boundaries_template!

# Fine-lattice indices of the staggered nodes (see Geometry.jl)
fine_index(::Val{:x}, I) = (2I[1] - 1, 2I[2] - 2)
fine_index(::Val{:y}, I) = (2I[1] - 2, 2I[2] - 1)
fine_index(::Val{:p}, I) = (2I[1], 2I[2])

node_position(::Val{:x}, I, X) = SVector(X.vx_e.x[I[1]], X.vx_e.y[I[2]])
node_position(::Val{:y}, I, X) = SVector(X.vy_e.x[I[1]], X.vy_e.y[I[2]])
node_position(::Val{:p}, I, X) = SVector(X.c_e.x[I[1]], X.c_e.y[I[2]])

"""
    JumpFunctionals(tr, η; ξ=nothing)

For every crossing, the jumps as linear functions of the cloud values u = [zx(idx_x); zy(idx_y)]
of its traces: `z[a, m]`, `∇z[a, b, m]`, `∇∇z[a, b, c, m]`, `p[m]`, `∇p[b, m]`, with
[z_a] = Σ_m z[a, m] u_m, etc. `field[m]` tells whether cloud value m is a zx or a zy node.
Compressible (`ξ = (neg, pos)`, bulk viscosities): `InterfaceJumpsC`, with the traction σn on Γ
averaged from the one-sided fits of both sides; the cloud then also holds pressure nodes (`:p`).
"""
function JumpFunctionals(tr, η; ξ=nothing)
    Δη = η.pos - η.neg
    k  = isnothing(ξ) ? nothing : (neg=η.neg / ξ.neg, pos=η.pos / ξ.pos)
    return map(tr) do T
        nx, ny = length(T.x.idx), length(T.y.idx)
        np = isnothing(k) ? 0 : length(T.pt.idx)
        N  = nx + ny + np
        Mv  = vcat(hcat(T.x.v', zeros(1, ny + np)), hcat(zeros(1, nx), T.y.v', zeros(1, np)))    # v_Γ    = Mv u
        Mt  = vcat(hcat(T.x.t', zeros(1, ny + np)), hcat(zeros(1, nx), T.y.t', zeros(1, np)))    # ∂ₛv_Γ  = Mt u
        Mtt = vcat(hcat(T.x.tt', zeros(1, ny + np)), hcat(zeros(1, nx), T.y.tt', zeros(1, np)))  # ∂ₛₛv_Γ = Mtt u
        Jz, JG, JH, Jp, Jgp = zeros(2, N), zeros(2, 2, N), zeros(2, 2, 2, N), zeros(N), zeros(2, N)
        n, κ = T.p.n, T.p.κ
        t  = tangent(n)
        for m in 1:N
            vΓ, w, w2 = SVector(Mv[1, m], Mv[2, m]), SVector(Mt[1, m], Mt[2, m]), SVector(Mtt[1, m], Mtt[2, m])
            J = if isnothing(k)
                InterfaceJumps2(Δη, vΓ, w, w2, n, κ)
            else
                # one-sided ∇z, ∂ₜ∇z, p, ∂ₜp of the unit cloud vector m, then σn and d(σn)/ds per side
                G, dG, ps, dps = zeros(2, 2, 2), zeros(2, 2, 2), zeros(2), zeros(2)
                a, kk, Tr = m <= nx ? (1, m, T.x) : m <= nx + ny ? (2, m - nx, T.y) : (0, m - nx - ny, T.pt)
                is = Tr.s[kk] < 0 ? 1 : 2
                if a > 0
                    G[a, :, is]  .= Tr.g[:, kk]
                    dG[a, :, is] .= Tr.H[:, :, kk] * t
                else
                    ps[is], dps[is] = Tr.λ[kk], dot(Tr.g[:, kk], t)
                end
                σs(G, p)  = -p * I + G + G' - 2 / 3 * LinearAlgebra.tr(G) * I   # `tr` is the trace list here
                Tn  = sum(σs(SMatrix{2,2}(G[:, :, i]), ps[i]) * n for i in 1:2) / 2
                dTn = sum(σs(SMatrix{2,2}(dG[:, :, i]), dps[i]) * n + κ * σs(SMatrix{2,2}(G[:, :, i]), ps[i]) * t for i in 1:2) / 2
                InterfaceJumpsC(η, k, vΓ, w, w2, Tn, dTn, n, κ)
            end
            Jz[:, m] .= J.z
            JG[:, :, m] .= J.∇z
            JH[:, :, :, m] .= J.∇∇z
            Jp[m] = J.p
            Jgp[:, m] .= J.∇p
        end
        idx   = isnothing(k) ? vcat(T.x.idx, T.y.idx) : vcat(T.x.idx, T.y.idx, T.pt.idx)
        field = vcat(fill(:x, nx), fill(:y, ny), fill(:p, np))
        (xΓ=T.p.x, idx=idx, field=field, z=Jz, ∇z=JG, ∇∇z=JH, p=Jp, ∇p=Jgp)
    end
end

# Crossing used for the stencil pair (node, member): the crossing on the fine-lattice edges
# between them that is closest to their midpoint; if that crossing is outside the domain (Γ
# meets the boundary, member is a ghost node), the crossing closest to the midpoint
function pair_crossing(lookup, F0, F1, xmid, J)
    Imin, Imax = minmax(F0[1], F1[1])
    Jmin, Jmax = minmax(F0[2], F1[2])
    best, dbest = 0, Inf
    for I in Imin:Imax, Jj in Jmin:Jmax, (dir, dI, dJ) in ((1, 1, 0), (2, 0, 1))
        (I + dI <= Imax && Jj + dJ <= Jmax) || continue
        c = get(lookup, (I, Jj, dir), 0)
        c == 0 && continue
        d = norm(J[c].xΓ - xmid)
        d < dbest && ((best, dbest) = (c, d))
    end
    best == 0 && (best = argmin(c -> norm(J[c].xΓ - xmid), eachindex(J)))
    return best
end

"""
    AssembleIIM(φ, X, Δ, η, nc, zbc; ξ=nothing, order=1, fit=:linear, corrections=true, Jexact=nothing)

Assemble the IIM system for z = ηv and p with Dirichlet velocities: `zbc = (x, y)` holds z on
the full Vx/Vy arrays and provides the values of the non-numbered (boundary and ghost) nodes.
- `ξ`: `nothing` (incompressible; the pressure null space is removed with a Lagrange multiplier,
  Σ p = 0) or `(neg, pos)` bulk viscosities (compressible: τ = 2η(ε̇ - ⅓(∇·v)I), p = -ξ∇·v)
- `order`: 1 (Landry) or 2 (adds ½δᵀ[∇∇z]δ and [∇p]·δ; requires `fit=:quadratic`)
- `fit`: `:linear`, `:quadratic` or `:cubic` one-sided fits for v_Γ, ∂ₛv_Γ, ∂ₛₛv_Γ (see `InterfaceTrace`)
Diagnostic: `Jexact(p)` returns given jumps (as `InterfaceJumps2`) at crossing `p`; they then
enter the right-hand side instead of being reconstructed from z.
Returns the matrix, right-hand side, numbering and interface data.
"""
function AssembleIIM(φ, X, Δ, η, nc, zbc; ξ=nothing, order=1, fit=:linear, corrections=true, Jexact=nothing)
    order == 2 && fit == :linear && isnothing(Jexact) && error("order = 2 requires fit = :quadratic or :cubic")
    size_x, size_y, size_c = size(φ.x), size(φ.y), size(φ.c)
    type   = Fields(fill(:out, size_x...), fill(:out, size_y...), fill(:out, size_c...))
    number = Fields(zeros(Int, size_x...), zeros(Int, size_y...), zeros(Int, size_c...))
    set_boundaries_template!(type, :all_Dirichlet, nc)
    Numbering!(number, type, nc)
    nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
    off  = (x=0, y=nVx, p=nVx + nVy)
    comp = !isnothing(ξ)
    ndof = nVx + nVy + nPt + !comp
    num  = (x=number.Vx, y=number.Vy, p=number.Pt)
    φf   = (x=φ.x, y=φ.y, p=φ.c)

    Γ  = InterfaceCrossings(FineLattice(φ, X, Δ))
    tr = InterfaceTraces(Γ, φ, X, Δ, η, nc; fit, pressure=comp)
    J  = JumpFunctionals(tr, η; ξ)
    lookup = Dict((T.p.I, T.p.J, T.p.dir == :x ? 1 : 2) => c for (c, T) in enumerate(tr))

    Is, Js, Vs = Int[], Int[], Float64[]
    b = zeros(ndof)
    function add!(row, f, I, coef)
        col = num[f][I]
        if col > 0
            push!(Is, row); push!(Js, off[f] + col); push!(Vs, coef)
        else
            b[row] -= coef * zbc[f][I]          # only velocity nodes are non-numbered
        end
    end
    function equation!(row, f0, I0, members)
        s0 = side(φf[f0][I0])
        x0 = node_position(Val(f0), I0, X)
        for (f, I, coef) in members
            add!(row, f, I, coef)
            (corrections && side(φf[f][I]) != s0) || continue
            xk = node_position(Val(f), I, X)
            c  = pair_crossing(lookup, fine_index(Val(f0), I0), fine_index(Val(f), I), (x0 + xk) / 2, J)
            Jc = J[c]
            δ  = xk - Jc.xΓ
            if !isnothing(Jexact)
                Je  = Jexact(tr[c].p)
                a   = f == :x ? 1 : 2
                val = if f == :p
                    Je.p + (order == 2 ? dot(Je.∇p, δ) : 0.0)
                else
                    Je.z[a] + dot(Je.∇z[a, :], δ) + (order == 2 ? dot(δ, Je.∇∇z[a, :, :] * δ) / 2 : 0.0)
                end
                b[row] -= s0 * coef * val
                continue
            end
            w  = if f == :p
                order == 2 ? [Jc.p[m] + dot(Jc.∇p[:, m], δ) for m in eachindex(Jc.p)] : Jc.p
            else
                a = f == :x ? 1 : 2
                [Jc.z[a, m] + dot(Jc.∇z[a, :, m], δ) + (order == 2 ? dot(δ, Jc.∇∇z[a, :, :, m] * δ) / 2 : 0.0)
                 for m in eachindex(Jc.p)]
            end
            for m in eachindex(w)
                add!(row, Jc.field[m], Jc.idx[m], s0 * coef * w[m])
            end
        end
    end

    dx, dy = Δ.x, Δ.y
    cd = comp ? 1 / 3 : 1.0       # ∇²z + cd ∇(∇·z) - ∇p = 0 (cd = 1: incompressible form)
    kc(f) = side(f) < 0 ? η.neg / ξ.neg : η.pos / ξ.pos
    # x-momentum
    for j in axes(num.x, 2), i in axes(num.x, 1)
        (row = num.x[i, j]) > 0 || continue
        equation!(off.x + row, :x, CartesianIndex(i, j), (
            (:x, CartesianIndex(i, j), -2(1 + cd) / dx^2 - 2 / dy^2),
            (:x, CartesianIndex(i - 1, j), (1 + cd) / dx^2), (:x, CartesianIndex(i + 1, j), (1 + cd) / dx^2),
            (:x, CartesianIndex(i, j - 1), 1 / dy^2), (:x, CartesianIndex(i, j + 1), 1 / dy^2),
            (:y, CartesianIndex(i + 1, j), cd / (dx * dy)), (:y, CartesianIndex(i, j), -cd / (dx * dy)),
            (:y, CartesianIndex(i + 1, j - 1), -cd / (dx * dy)), (:y, CartesianIndex(i, j - 1), cd / (dx * dy)),
            (:p, CartesianIndex(i, j - 1), -1 / dx), (:p, CartesianIndex(i - 1, j - 1), 1 / dx)))
    end
    # y-momentum
    for j in axes(num.y, 2), i in axes(num.y, 1)
        (row = num.y[i, j]) > 0 || continue
        equation!(off.y + row, :y, CartesianIndex(i, j), (
            (:y, CartesianIndex(i, j), -2 / dx^2 - 2(1 + cd) / dy^2),
            (:y, CartesianIndex(i - 1, j), 1 / dx^2), (:y, CartesianIndex(i + 1, j), 1 / dx^2),
            (:y, CartesianIndex(i, j - 1), (1 + cd) / dy^2), (:y, CartesianIndex(i, j + 1), (1 + cd) / dy^2),
            (:x, CartesianIndex(i, j + 1), cd / (dx * dy)), (:x, CartesianIndex(i - 1, j + 1), -cd / (dx * dy)),
            (:x, CartesianIndex(i, j), -cd / (dx * dy)), (:x, CartesianIndex(i - 1, j), cd / (dx * dy)),
            (:p, CartesianIndex(i - 1, j), -1 / dy), (:p, CartesianIndex(i - 1, j - 1), 1 / dy)))
    end
    # continuity ∇·z + k p = 0, k = η/ξ (+ Lagrange multiplier for the pressure level if incompressible)
    for j in axes(num.p, 2), i in axes(num.p, 1)
        (row = num.p[i, j]) > 0 || continue
        equation!(off.p + row, :p, CartesianIndex(i, j), (
            (:x, CartesianIndex(i + 1, j + 1), 1 / dx), (:x, CartesianIndex(i, j + 1), -1 / dx),
            (:y, CartesianIndex(i + 1, j + 1), 1 / dy), (:y, CartesianIndex(i + 1, j), -1 / dy),
            (comp ? ((:p, CartesianIndex(i, j), kc(φ.c[i, j])),) : ())...))
        comp || (push!(Is, off.p + row, ndof); push!(Js, ndof, off.p + row); push!(Vs, 1.0, 1.0))
    end

    K = sparse(Is, Js, Vs, ndof, ndof)
    return (K=K, b=b, num=num, off=off, Γ=Γ, tr=tr)
end

"""
    SolveIIM(φ, X, Δ, η, nc, zbc; ξ=nothing, order=1, fit=:linear, corrections=true, Jexact=nothing)

Assemble and solve with a sparse direct solver. Returns z (with the boundary values of `zbc`),
v = z/η and p on the full staggered arrays.
"""
function SolveIIM(φ, X, Δ, η, nc, zbc; ξ=nothing, order=1, fit=:linear, corrections=true, Jexact=nothing)
    S   = AssembleIIM(φ, X, Δ, η, nc, zbc; ξ, order, fit, corrections, Jexact)
    sol = S.K \ S.b
    z   = (x=copy(zbc.x), y=copy(zbc.y))
    p   = zeros(size(φ.c))
    for (f, a) in ((:x, z.x), (:y, z.y), (:p, p)), I in CartesianIndices(a)
        S.num[f][I] > 0 && (a[I] = sol[S.off[f] + S.num[f][I]])
    end
    ηof(φ) = side(φ) < 0 ? η.neg : η.pos
    v = (x=z.x ./ ηof.(φ.x), y=z.y ./ ηof.(φ.y))
    return (z=z, v=v, p=p, S=S)
end
