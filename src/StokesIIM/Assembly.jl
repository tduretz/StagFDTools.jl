# IIM operator for incompressible Stokes with piecewise constant viscosity, in z = ηv form.
#
# Regular equations (constant η on each side):
#   x-momentum: 2∂xx zx + ∂yy zx + ∂xy zy - ∂x p = 0
#   y-momentum: ∂xx zy + 2∂yy zy + ∂xy zx - ∂y p = 0
#   continuity: ∂x zx + ∂y zy = 0
# Correction (first order): in the equation of a node on side s, every stencil member k on the
# other side is replaced by the extension of the s-side field to x_k:
#   q_k  ->  q_k + s ([q] + [∇q]·(x_k - x_Γ))      ([p] only for pressure members)
# with [q] = q₊ - q₋ at a crossing x_Γ between the node and k. The jumps are linear in z through
# the interface traces, so the corrections are matrix coefficients on 12 nodes per crossing.

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
    JumpFunctionals(tr, η)

For every crossing, the jumps as linear functions of the 12 cloud values u = [zx(6); zy(6)]
of its traces: `z[a, m]`, `∇z[a, b, m]`, `p[m]` with [z_a] = Σ_m z[a, m] u_m, etc.
"""
function JumpFunctionals(tr, η)
    Δη = η.pos - η.neg
    return map(tr) do T
        Mv = vcat(hcat(T.x.v', zeros(1, 6)), hcat(zeros(1, 6), T.y.v'))   # v_Γ  = Mv u
        Mt = vcat(hcat(T.x.t', zeros(1, 6)), hcat(zeros(1, 6), T.y.t'))   # ∂ₜv_Γ = Mt u
        Jz, JG, Jp = zeros(2, 12), zeros(2, 2, 12), zeros(12)
        for m in 1:12
            J = InterfaceJumps(Δη, SVector(Mv[1, m], Mv[2, m]), SVector(Mt[1, m], Mt[2, m]), T.p.n)
            Jz[:, m] .= J.z
            JG[:, :, m] .= J.∇z
            Jp[m] = J.p
        end
        (xΓ=T.p.x, idx=vcat(Tuple(T.x.idx)..., Tuple(T.y.idx)...), z=Jz, ∇z=JG, p=Jp)
    end
end

# Crossing used for the stencil pair (node, member): the crossing on the fine-lattice edges
# between them that is closest to their midpoint
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
    best == 0 && error("no interface crossing between fine nodes $F0 and $F1")
    return best
end

"""
    AssembleIIM(φ, X, Δ, η, nc, zbc; corrections=true, Jexact=nothing)

Assemble the IIM system for z = ηv and p with Dirichlet velocities: `zbc = (x, y)` holds z on
the full Vx/Vy arrays and provides the values of the non-numbered (boundary and ghost) nodes.
The pressure null space is removed with a Lagrange multiplier (Σ p = 0).
Diagnostic: `Jexact(p)` returns given jumps (as `InterfaceJumps`) at crossing `p`; they then
enter the right-hand side instead of being reconstructed from z.
Returns the matrix, right-hand side, numbering and interface data.
"""
function AssembleIIM(φ, X, Δ, η, nc, zbc; corrections=true, Jexact=nothing)
    size_x, size_y, size_c = size(φ.x), size(φ.y), size(φ.c)
    type   = Fields(fill(:out, size_x...), fill(:out, size_y...), fill(:out, size_c...))
    number = Fields(zeros(Int, size_x...), zeros(Int, size_y...), zeros(Int, size_c...))
    set_boundaries_template!(type, :all_Dirichlet, nc)
    Numbering!(number, type, nc)
    nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
    off  = (x=0, y=nVx, p=nVx + nVy)
    ndof = nVx + nVy + nPt + 1
    num  = (x=number.Vx, y=number.Vy, p=number.Pt)
    φf   = (x=φ.x, y=φ.y, p=φ.c)

    Γ  = InterfaceCrossings(FineLattice(φ, X, Δ))
    tr = InterfaceTraces(Γ, φ, X, Δ, η, nc)
    J  = JumpFunctionals(tr, η)
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
            if !isnothing(Jexact)
                Je = Jexact(tr[c].p)
                a  = f == :x ? 1 : 2
                val = f == :p ? Je.p : Je.z[a] + dot(Je.∇z[a, :], xk - Jc.xΓ)
                b[row] -= s0 * coef * val
                continue
            end
            w  = if f == :p
                Jc.p
            else
                a = f == :x ? 1 : 2
                [Jc.z[a, m] + dot(Jc.∇z[a, :, m], xk - Jc.xΓ) for m in 1:12]
            end
            for m in 1:12
                add!(row, m <= 6 ? :x : :y, Jc.idx[m], s0 * coef * w[m])
            end
        end
    end

    dx, dy = Δ.x, Δ.y
    # x-momentum
    for j in axes(num.x, 2), i in axes(num.x, 1)
        (row = num.x[i, j]) > 0 || continue
        equation!(off.x + row, :x, CartesianIndex(i, j), (
            (:x, CartesianIndex(i, j), -4 / dx^2 - 2 / dy^2),
            (:x, CartesianIndex(i - 1, j), 2 / dx^2), (:x, CartesianIndex(i + 1, j), 2 / dx^2),
            (:x, CartesianIndex(i, j - 1), 1 / dy^2), (:x, CartesianIndex(i, j + 1), 1 / dy^2),
            (:y, CartesianIndex(i + 1, j), 1 / (dx * dy)), (:y, CartesianIndex(i, j), -1 / (dx * dy)),
            (:y, CartesianIndex(i + 1, j - 1), -1 / (dx * dy)), (:y, CartesianIndex(i, j - 1), 1 / (dx * dy)),
            (:p, CartesianIndex(i, j - 1), -1 / dx), (:p, CartesianIndex(i - 1, j - 1), 1 / dx)))
    end
    # y-momentum
    for j in axes(num.y, 2), i in axes(num.y, 1)
        (row = num.y[i, j]) > 0 || continue
        equation!(off.y + row, :y, CartesianIndex(i, j), (
            (:y, CartesianIndex(i, j), -2 / dx^2 - 4 / dy^2),
            (:y, CartesianIndex(i - 1, j), 1 / dx^2), (:y, CartesianIndex(i + 1, j), 1 / dx^2),
            (:y, CartesianIndex(i, j - 1), 2 / dy^2), (:y, CartesianIndex(i, j + 1), 2 / dy^2),
            (:x, CartesianIndex(i, j + 1), 1 / (dx * dy)), (:x, CartesianIndex(i - 1, j + 1), -1 / (dx * dy)),
            (:x, CartesianIndex(i, j), -1 / (dx * dy)), (:x, CartesianIndex(i - 1, j), 1 / (dx * dy)),
            (:p, CartesianIndex(i - 1, j), -1 / dy), (:p, CartesianIndex(i - 1, j - 1), 1 / dy)))
    end
    # continuity (+ Lagrange multiplier for the pressure level)
    for j in axes(num.p, 2), i in axes(num.p, 1)
        (row = num.p[i, j]) > 0 || continue
        equation!(off.p + row, :p, CartesianIndex(i, j), (
            (:x, CartesianIndex(i + 1, j + 1), 1 / dx), (:x, CartesianIndex(i, j + 1), -1 / dx),
            (:y, CartesianIndex(i + 1, j + 1), 1 / dy), (:y, CartesianIndex(i + 1, j), -1 / dy)))
        push!(Is, off.p + row, ndof); push!(Js, ndof, off.p + row); push!(Vs, 1.0, 1.0)
    end

    K = sparse(Is, Js, Vs, ndof, ndof)
    return (K=K, b=b, num=num, off=off, Γ=Γ, tr=tr)
end

"""
    SolveIIM(φ, X, Δ, η, nc, zbc; corrections=true, Jexact=nothing)

Assemble and solve with a sparse direct solver. Returns z (with the boundary values of `zbc`),
v = z/η and p on the full staggered arrays.
"""
function SolveIIM(φ, X, Δ, η, nc, zbc; corrections=true, Jexact=nothing)
    S   = AssembleIIM(φ, X, Δ, η, nc, zbc; corrections, Jexact)
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
