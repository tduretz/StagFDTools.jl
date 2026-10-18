using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs

using ProfileCanvas, BenchmarkTools

@views function main(nc)
    #--------------------------------------------#

    # Resolution

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [ -1. 0.;
                          0  1 ])

    # Material parameters
    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; compressible=true)

    # Parameters
    params_bg = (η0=1e2, G=1e1, β=1e-2)
    params_in = (η0=1e-1, G=1e1, β=1e-2)

    materials.g .= [0., 0.]
    materials.η0 .= [params_bg.η0, params_in.η0]
    materials.G .= [params_bg.G, params_in.G]

    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Solver parameters
    iter_params = IterParams(niter=20, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c ]  .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Set material geometry
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= 0.1^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= 0.1^2] .= 2
    FillPhaseRatios!(a)

    #--------------------------------------------#

    compute_grid_fields!(a.G, a.β, a.ρ, a.ξ, materials, a.phase_ratios, nc, nphases)

    @info "Benchmark AssembleMomentum2D_x!"
    display( @benchmark AssembleMomentum2D_x!($(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, materials, a.number, a.pattern, a.type, BC, nc, Δ)...) )

    @info "Benchmark AssembleMomentum2D_y!"
    display( @benchmark AssembleMomentum2D_y!($(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, a.ρ, materials, a.number, a.pattern, a.type, BC, nc, Δ)...) )

    @info "Benchmark AssembleContinuity2D!"
    ProfileCanvas.@profview AssembleContinuity2D!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.β, a.ξ, materials, a.number, a.pattern, a.type, BC, nc, Δ)
    display( @benchmark AssembleMomentum2D_x!($(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, materials, a.number, a.pattern, a.type, BC, nc, Δ)...) )

end

let
    main((x = 100, y = 100))
end

