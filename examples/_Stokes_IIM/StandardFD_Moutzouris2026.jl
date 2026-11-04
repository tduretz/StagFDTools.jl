using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology
using ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, TimerOutputs
import Statistics: mean
using ExactFieldSolutions

# Standard staggered FD (StagFDTools Stokes module, phase-ratio averaged η and ξ, continuity
# ∇·v + p/ξ = 0) for the compressible viscous inclusion of Moutzouris (2026), with Dirichlet
# velocities from the exact solution. Same setup as StandardFD_Schmid2003.jl.

@views function StandardFD_Moutzouris2026(n, params; shape=:circle, L=(x=1.0, y=1.0))
    nc = (x=n, y=n)
    config = :all_Dirichlet
    S(x) = (shape == :circle ? Stokes2D_Moutzouris_circle : Stokes2D_Moutzouris_ellipse)(x; params)
    a_, b_ = shape == :circle ? (params.ri, params.ri) : (params.t * params.ri, params.ri)

    nphases   = 2
    materials = initialize_materials(nphases; compressible=true)
    materials.η0 .= [params.ηm, params.ηi]
    materials.ξ0 .= [params.ξm, params.ξi]
    materials.G  .= [1e30, 1e30]          # purely viscous
    preprocess!(materials)

    iter_params = IterParams(niter=2, ϵ_nl=1e-10, α=LinRange(0.05, 1.0, 10))

    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=1.0)
    a = Allocs(nc, config, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Exact solution, also used as boundary values and initial guess
    BC     = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    Pt_ana = zero(a.Pt)
    for j in axes(a.Pt, 2), i in axes(a.Pt, 1)
        Pt_ana[i, j] = a.Pt[i, j] = S([a.X.c_e.x[i], a.X.c_e.y[j]]).p
    end
    for j in 2:(size_x[2]-1), i in 1:size_x[1]
        BC.Vx[i, j] = a.V.x[i, j] = S([a.X.v_e.x[i], a.X.c_e.y[j-1]]).V[1]
    end
    for j in 1:size_y[2], i in 2:(size_y[1]-1)
        BC.Vy[i, j] = a.V.y[i, j] = S([a.X.c_e.x[i-1], a.X.v_e.y[j]]).V[2]
    end

    # Phases (inclusion = 2)
    inside(x, y) = (x / a_)^2 + (y / b_)^2 <= 1 + 1e-13
    a.phases.c[inside.(a.X.c_e.x, a.X.c_e.y')] .= 2
    a.phases.v[inside.(a.X.v_e.x, a.X.v_e.y')] .= 2
    Set_PhaseRatios!(a)

    rvec = zeros(length(iter_params.α))
    err  = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    main_loop(a, 1, materials, BC, nc, Δ, TimerOutput(), nphases, iter_params, rvec, err)

    P  = a.Pt[inx_c, iny_c]
    Pa = Pt_ana[inx_c, iny_c]
    return (P=P, P_ana=Pa, X=a.X, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))))
end
