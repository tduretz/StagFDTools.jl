using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology
using ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, TimerOutputs
import Statistics: mean
using ExactFieldSolutions

# Standard staggered FD (StagFDTools Stokes module, phase-ratio averaged viscosity) for the
# viscous inclusion of Schmid & Podladchikov (2003), with Dirichlet velocities from the exact
# solution. Reference for the IIM comparison; setup follows
# examples/_Stokes_VEP_SSG/untested/ExampleADStokes_ViscousInclusionSchmid2003.jl.

@views function StandardFD_Schmid2003(n, params; L=(x=10.0, y=10.0))
    nc = (x=n, y=n)
    config = :all_Dirichlet

    nphases   = 2
    materials = initialize_materials(nphases; compressible=false)
    materials.η0 .= [params.mm, params.mc]
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
        Pt_ana[i, j] = Stokes2D_Schmid2003_circle([a.X.c_e.x[i], a.X.c_e.y[j]]; params).p
    end
    for j in 2:(size_x[2]-1), i in 1:size_x[1]
        BC.Vx[i, j] = a.V.x[i, j] = Stokes2D_Schmid2003_circle([a.X.v_e.x[i], a.X.c_e.y[j-1]]; params).V[1]
    end
    for j in 1:size_y[2], i in 2:(size_y[1]-1)
        BC.Vy[i, j] = a.V.y[i, j] = Stokes2D_Schmid2003_circle([a.X.c_e.x[i-1], a.X.v_e.y[j]]; params).V[2]
    end

    # Phases (inclusion = 2)
    rad = params.rc + 1e-13
    a.phases.c[(a.X.c_e.x .^ 2 .+ (a.X.c_e.y') .^ 2) .<= rad^2] .= 2
    a.phases.v[(a.X.v_e.x .^ 2 .+ (a.X.v_e.y') .^ 2) .<= rad^2] .= 2
    Set_PhaseRatios!(a)

    rvec = zeros(length(iter_params.α))
    err  = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    main_loop(a, 1, materials, BC, nc, Δ, TimerOutput(), nphases, iter_params, rvec, err)

    P  = a.Pt[inx_c, iny_c] .- mean(a.Pt[inx_c, iny_c])
    Pa = Pt_ana[inx_c, iny_c] .- mean(Pt_ana[inx_c, iny_c])
    return (P=P, P_ana=Pa, X=a.X, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))))
end
