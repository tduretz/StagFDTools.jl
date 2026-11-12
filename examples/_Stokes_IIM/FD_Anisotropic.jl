# Standard staggered FD with an effective viscosity tensor in the cells cut by the interface.
#
# A cut control volume (centroid or vertex) with phase fraction f is treated as a layered composite
# parallel to Γ (normal n = ∇φ/|∇φ|). Across a material interface v, ε̇tt, ε̇nn (= -ε̇tt) and the
# traction (σnn, σnt) are continuous, hence:
#   - pure shear in the (n, t) frame: strain rate continuous, layers in parallel -> η_N = <η>
#   - shear on the interface plane:   stress σnt continuous, layers in series   -> η_S = <1/η>⁻¹
# giving the transversely isotropic tensor (as in Layered_Model_VEP.jl)
#   𝐷 = 2η_N I + 2(η_N - η_S) C_ANI(n)      acting on (ε̇xx', ε̇yy', ε̇xy)
# Uncut volumes are isotropic. Variants: :binary (phase of the node, "standard FD"), :arith,
# :harm (isotropic averages of sub-sampled fractions), :aniso (tensor above).
#
# Discretisation: the stress form of src/StokesIIM/AssemblyStress.jl without interface
# corrections (σxx, σyy at centroids, τxy at vertices, all four velocity gradient components
# available at both by 4-point averages as in the Stokes module), 𝐷 per stress point.
using StagFDTools, StagFDTools.StokesIIM, StaticArrays, LinearAlgebra, SparseArrays, Printf
import ForwardDiff
import StagFDTools.StokesIIM: velocity_gradient, deviatoric_stress, node_position, node_index, side
import StagFDTools.Stokes: set_boundaries_template!, Numbering!, Fields

# transversely isotropic viscosity tensor (normal n), on (ε̇xx', ε̇yy', ε̇xy)
function aniso_tensor(ηN, ηS, n)
    n  = n / norm(n)
    a0 = 2 * n[1]^2 * n[2]^2
    a1 = n[1] * n[2] * (n[1]^2 - n[2]^2)      # sign for n = normal (as Δ in Layered_Model_VEP.jl;
                                              # the old ViscosityTensor example takes the director along the layer)
    CA = @SMatrix [-a0 a0 2a1; a0 -a0 -2a1; a1 -a1 (-1 + 2a0)]
    return 2ηN * SMatrix{3,3}(1.0I) + 2(ηN - ηS) * CA
end

# effective tensor of the control volume centred at x (size Δ), φfun < 0: phase "neg"
function effective_tensor(φfun, x, Δ, η, avg; m=10)
    if avg == :binary
        return 2 * (φfun(x[1], x[2]) < 0 ? η.neg : η.pos) * SMatrix{3,3}(1.0I)
    end
    s = ((1:m) .- 0.5) ./ m .- 0.5
    f = count(φfun(x[1] + a * Δ.x, x[2] + b * Δ.y) < 0 for a in s, b in s) / m^2   # fraction of neg
    ηa = f * η.neg + (1 - f) * η.pos
    ηh = 1 / (f / η.neg + (1 - f) / η.pos)
    (avg == :arith || f == 0 || f == 1) && return 2 * (avg == :harm ? ηh : ηa) * SMatrix{3,3}(1.0I)
    avg == :harm && return 2ηh * SMatrix{3,3}(1.0I)
    avg == :geom && return 2 * sqrt(ηa * ηh) * SMatrix{3,3}(1.0I)
    h = 1e-6 * min(Δ.x, Δ.y)
    n = SVector(φfun(x[1] + h, x[2]) - φfun(x[1] - h, x[2]), φfun(x[1], x[2] + h) - φfun(x[1], x[2] - h))
    𝐷 = aniso_tensor(ηa, ηh, n)
    avg == :anisodiag && return SMatrix{3,3}(Diagonal(𝐷))   # no coupling to the averaged components
    return 𝐷
end

function resid(::Val{:x}, vx, vy, p, x0, Δ, Dc, Dv)
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    σxx(xc) = deviatoric_stress(velocity_gradient(vx, vy, xc, Δ, Val(:c)), Dc(xc))[1] - p(xc)
    τxy(xv) = deviatoric_stress(velocity_gradient(vx, vy, xv, Δ, Val(:v)), Dv(xv))[3]
    return (σxx(x0 + ex) - σxx(x0 - ex)) / Δ.x + (τxy(x0 + ey) - τxy(x0 - ey)) / Δ.y
end
function resid(::Val{:y}, vx, vy, p, x0, Δ, Dc, Dv)
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    σyy(xc) = deviatoric_stress(velocity_gradient(vx, vy, xc, Δ, Val(:c)), Dc(xc))[2] - p(xc)
    τxy(xv) = deviatoric_stress(velocity_gradient(vx, vy, xv, Δ, Val(:v)), Dv(xv))[3]
    return (τxy(x0 + ex) - τxy(x0 - ex)) / Δ.x + (σyy(x0 + ey) - σyy(x0 - ey)) / Δ.y
end
function resid(::Val{:p}, vx, vy, p, x0, Δ, Dc, Dv)
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    return (vx(x0 + ex) - vx(x0 - ex)) / Δ.x + (vy(x0 + ey) - vy(x0 - ey)) / Δ.y
end

"""
    SolveFDTensor(φfun, X, Δ, η, nc, vbc; avg=:aniso)

Incompressible Stokes, Dirichlet velocities `vbc = (x, y)` on the full Vx/Vy arrays, viscosity
`η = (neg, pos)` with φfun(x, y) < 0 on the `neg` side. Returns v, p (p on the centroid array).
"""
function SolveFDTensor(φfun, X, Δ, η, nc, vbc; avg=:aniso, avg_v=avg, m=10)
    size_x, size_y = size(vbc.x), size(vbc.y)
    size_c = (length(X.c_e.x), length(X.c_e.y))
    type   = Fields(fill(:out, size_x...), fill(:out, size_y...), fill(:out, size_c...))
    number = Fields(zeros(Int, size_x...), zeros(Int, size_y...), zeros(Int, size_c...))
    set_boundaries_template!(type, :all_Dirichlet, nc)
    Numbering!(number, type, nc)
    nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
    off  = (x=0, y=nVx, p=nVx + nVy)
    ndof = nVx + nVy + nPt + 1
    num  = (x=number.Vx, y=number.Vy, p=number.Pt)
    val0 = (x=vbc.x, y=vbc.y, p=zeros(size_c))
    # tensors per stress point, keyed by the position in half cells
    key(x) = (round(Int, 2x[1] / Δ.x), round(Int, 2x[2] / Δ.y))
    cc, cv = Dict{NTuple{2,Int},SMatrix{3,3,Float64,9}}(), Dict{NTuple{2,Int},SMatrix{3,3,Float64,9}}()
    Dc(x) = get!(() -> effective_tensor(φfun, x, Δ, η, avg; m), cc, key(x))      # centroids
    Dv(x) = get!(() -> effective_tensor(φfun, x, Δ, η, avg_v; m), cv, key(x))    # vertices
    Is, Js, Vs = Int[], Int[], Float64[]
    b = zeros(ndof)
    for f0 in (:x, :y, :p), I0 in CartesianIndices(num[f0])
        (row = num[f0][I0]) > 0 || continue
        x0 = node_position(Val(f0), I0, X)
        loc = Dict{Tuple{Symbol,CartesianIndex{2}},Int}(); nodes = Tuple{Symbol,CartesianIndex{2}}[]
        slot(f, x) = (k = (f, node_index(f, x, X, Δ)); get!(loc, k) do
            push!(nodes, k); length(nodes) end)
        kernel(u) = resid(Val(f0), x -> u[slot(:x, x)], x -> u[slot(:y, x)], x -> u[slot(:p, x)], x0, Δ, Dc, Dv)
        kernel(zeros(64))                                   # discover the stencil
        u0 = [num[f][I] > 0 ? 0.0 : val0[f][I] for (f, I) in nodes]
        g  = ForwardDiff.gradient(kernel, u0)
        r  = off[f0] + row
        b[r] -= kernel(u0)
        for (j, (f, I)) in enumerate(nodes)
            num[f][I] > 0 && g[j] != 0 && (push!(Is, r); push!(Js, off[f] + num[f][I]); push!(Vs, g[j]))
        end
        f0 == :p && (push!(Is, r, ndof); push!(Js, ndof, r); push!(Vs, 1.0, 1.0))
    end
    K   = sparse(Is, Js, Vs, ndof, ndof)
    sol = K \ b
    v   = (x=copy(vbc.x), y=copy(vbc.y))
    p   = zeros(size_c)
    for (f, a) in ((:x, v.x), (:y, v.y), (:p, p)), I in CartesianIndices(a)
        num[f][I] > 0 && (a[I] = sol[off[f] + num[f][I]])
    end
    return (v=v, p=p, K=K)
end
