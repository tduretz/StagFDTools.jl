using CairoMakie
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 20, y = 20)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; compressible=true)
    materials.g   .= [0.0,  0.0]
    materials.ρ   .= [1.0,  1.0]
    materials.n   .= [1.0,  1.0]
    materials.η0  .= [1e2,  1e-1]
    materials.G   .= [1e1,  1e1]
    materials.β   .= [1e-2, 1e-2]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Solver parameters
    iter_params = IterParams(niter=3, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

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
    @views a.Pt[inx_c, iny_c]  .= 10.
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

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        #-----------
        fig = Figure(size=(600, 600))
        #-----------
        ax  = Axis(fig[1,1], aspect=DataAspect(), title="Vx", xlabel="x", ylabel="y")
        heatmap!(ax, a.X.v.x, a.X.c.y, (a.V.x[inx_Vx,iny_Vx]))
        ax  = Axis(fig[1,2], aspect=DataAspect(), title="Vy", xlabel="x", ylabel="y")
        heatmap!(ax, a.X.c.x, a.X.v.y, a.V.y[inx_Vy,iny_Vy])
        ax  = Axis(fig[2,1], aspect=DataAspect(), title="Pt", xlabel="x", ylabel="y")
        heatmap!(ax, a.X.c.x, a.X.c.y,  a.Pt[inx_c,iny_c])
        ax  = Axis(fig[2,2], aspect=DataAspect(), title="Convergence", xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error")
        scatter!(ax, 1:iter, log10.(err.x[1:iter]), label="Vx")
        scatter!(ax, 1:iter, log10.(err.y[1:iter]), label="Vy")
        scatter!(ax, 1:iter, log10.(err.p[1:iter]), label="Pt")
        #-----------
        display(fig)
        #-----------
    end

    display(to)

end

let
    # Boundary condition templates
    BCs = [
        :all_Dirichlet,
        :free_slip,
        :NS_Neumann,
        :EW_Neumann,
        :NS_periodic,
        :EW_periodic,
    ]

    # Boundary velocity gradient matrix
    D_BCs = [
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [0 0; 1  0] ),
        @SMatrix( [0 1; 0  0] ),
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end
