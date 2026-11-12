# Layered_Model_V with the immersed interface method (stress form, StokesIIM) and the standard
# FD reference (Stokes module, phases sampled at centroids and vertices), for the same setup:
# unit box, free slip, pure shear D = [1 0; 0 -1], layers of η = 2 and η = 0.2, period 0.1,
# ratio 0.5, rotated by θ. The quantity of interest is the mean τII over |x|, |y| ≤ 0.3.
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, StagFDTools.StokesIIM
using ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, TimerOutputs
using GridGeometryUtils
import GridGeometryUtils: rotation_matrix
import Statistics: mean

# Level set of a Layering: φ < 0 in layer A (`inside`), distance to the nearest interface
function layering_levelset(lay)
    (; center, thickness, ratio, sinθ, cosθ) = lay
    R  = rotation_matrix(sinθ, cosθ)
    c  = SVector(center[1], center[2])
    hA = ratio * thickness / 2          # half thickness of layer A, centred at y' = hA
    hB = (1 - ratio) * thickness / 2
    return (x, y) -> begin
        y′ = (R*(SVector(x, y)-c))[2]
        u  = mod(y′ - hA, thickness)     # 0 at the centre of A, A is u ∈ [0, hA] ∪ [P - hA, P)
        d  = min(u, thickness - u)       # distance to the centre line of A
        d - hA                           # (needs hA, hB > a few cells: kink at the centre lines)
    end
end

τII_of(τxx, τyy, τxy) = sqrt.(0.5 .* (τxx .^ 2 .+ τyy .^ 2) .+ τxy .^ 2)   # as Layered_Model_V

# Physical centroids with |x|, |y| ≤ r (the same cells for both methods)
function inner_indices(nc; r=0.3)
    xc = LinRange(-0.5 + 0.5 / nc.x, 0.5 - 0.5 / nc.x, nc.x)
    yc = LinRange(-0.5 + 0.5 / nc.y, 0.5 - 0.5 / nc.y, nc.y)
    return findall(x -> abs(x) <= r + 1e-12, xc), findall(y -> abs(y) <= r + 1e-12, yc)
end

"""IIM solution; returns τII on the physical centroids, its inner mean, and the solution."""
function layered_IIM(nc, lay, D; η=(neg=0.2, pos=2.0), order=1, fit=:linear, corrections=true)
    L  = (x=1.0, y=1.0)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid((min=-L.x / 2, max=L.x / 2), (min=-L.y / 2, max=L.y / 2), Δ, nc)
    φ  = LevelSet(layering_levelset(lay), X)
    vbc = (x=[D[1, 1] * x + D[1, 2] * y for x in X.vx_e.x, y in X.vx_e.y],
           y=[D[2, 1] * x + D[2, 2] * y for x in X.vy_e.x, y in X.vy_e.y])
    sol = SolveIIMStress(φ, X, Δ, η, nc, vbc; order, fit, corrections, bc=:free_slip)
    τ   = CentroidStress(sol.S, sol.v, sol.p)
    ic, jc = 2:(nc.x+1), 2:(nc.y+1)
    τII = τII_of(τ.xx[ic, jc], τ.yy[ic, jc], τ.xy[ic, jc])
    ix, iy = inner_indices(nc)
    return (τII=τII, τmean=mean(τII[ix, iy]), x=X.c.x, y=X.c.y, φc=φ.c[ic, jc], p=sol.p[ic, jc],
            τ=map(a -> a[ic, jc], τ), sol=sol)
end

"""Standard FD (Stokes module), as Layered_Model_V.main, without plotting."""
@views function layered_FD(nc, lay, D; η=(neg=0.2, pos=2.0), config=:free_slip)
    nphases = 3
    materials = initialize_materials(nphases; compressible=false)
    materials.g .= [0., 0.]
    materials.ρ .= [1.0, 1.0, 1.0]
    materials.n .= [1.0, 1.0, 1.0]
    materials.η0 .= [η.pos, η.neg, 1e-1]
    materials.G .= [1e6, 1e6, 1e6]
    materials.β .= [1e-6, 1e-6, 1e-6]
    materials.B .= [0., 0., 0.]
    preprocess!(materials)
    Δt0 = 0.5
    iter_params = IterParams(niter=3, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    L = (x=1.0, y=1.0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    a = Allocs(nc, config, x, y, Δ, nphases)
    a.V.x .= D[1, 1] * a.X.vx_e.x .+ D[1, 2] * a.X.vx_e.y'
    a.V.y .= D[2, 1] * a.X.vy_e.x .+ D[2, 2] * a.X.vy_e.y'
    a.Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D[1, 1]
    BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D[1, 1]
    BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D[1, 1] * a.X.v.x .+ D[1, 2] * a.X.v.y[1])
    BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D[1, 1] * a.X.v.x .+ D[1, 2] * a.X.v.y[end])
    BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D[2, 2]
    BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D[2, 2]
    BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D[2, 1] * a.X.v.x[1] .+ D[2, 2] * a.X.v.y)
    BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D[2, 1] * a.X.v.x[end] .+ D[2, 2] * a.X.v.y)
    for i in inx_c, j in iny_c
        inside(@SVector([a.X.c.x[i-1], a.X.c.y[j-1]]), lay) && (a.phases.c[i, j] = 2)
    end
    for i in inx_v, j in iny_v
        inside(@SVector([a.X.v.x[i-1], a.X.v.y[j-1]]), lay) && (a.phases.v[i, j] = 2)
    end
    Set_PhaseRatios!(a)
    rvec = zeros(length(iter_params.α))
    err  = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    main_loop(a, 1, materials, BC, nc, Δ, TimerOutput(), nphases, iter_params, rvec, err)
    τxyc = av2D(a.τ.xy)
    τII  = τII_of(a.τ.xx[inx_c, iny_c], a.τ.yy[inx_c, iny_c], τxyc[inx_c, iny_c])
    ix, iy = inner_indices(nc)
    return (τII=τII, τmean=mean(τII[ix, iy]), x=a.X.c.x, y=a.X.c.y, p=a.Pt[inx_c, iny_c],
            τ=(xx=a.τ.xx[inx_c, iny_c], yy=a.τ.yy[inx_c, iny_c], xy=τxyc[inx_c, iny_c]))
end

make_layering(θ) = Layering((0 * 0.25, 0.025), 0.1, 0.5; θ, perturb_amp=0.0, perturb_width=1.0)
