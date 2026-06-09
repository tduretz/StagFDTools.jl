using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs

@views function main(nc)

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [  1. 0.;
                          0  -1 ])

    # Material parameters
    nphases = 3
    materials = initialize_materials(nphases)
    materials.η0 .= [1e2,   1e-1,  1e2 ]
    materials.n  .= [3.0,   3.0,   1.0 ]
    materials.G  .= [1e20,  1e20,  1e20]
    preprocess!(materials)

    # Time steps
    Δt0 = 0.5
    nt  = 1

    # Solver parameters
    iter_params = IterParams() # default parameters

    # X
    L = (x=1.0, y=1.0)
    Δ = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    x = (min=-L.x/2, max=L.x/2)
    y = (min=-L.y,   max=0.0)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Grid coordinate arrays for phase initialisation and plotting
    xv = LinRange(-L.x/2,        L.x/2,        nc.x+1)
    yv = LinRange(-L.y/2,        L.y/2,        nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)

    # Initial velocity & pressure
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx]       .= (a.type.Vx[1, iny_Vx]   .== :Neumann_normal)   .* D_BC[1,1]
        BC.Vx[end-1, iny_Vx]   .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal)   .* D_BC[1,1]
        BC.Vx[inx_Vx, 2]       .= (a.type.Vx[inx_Vx, 2]   .== :Neumann_tangent)  .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, 2]   .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1])
        BC.Vx[inx_Vx, end-1]   .= (a.type.Vx[inx_Vx,end-1].== :Neumann_tangent)  .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,end-1].== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy, 2]       .= (a.type.Vy[inx_Vy, 1]   .== :Neumann_normal)   .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1]   .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal)   .* D_BC[2,2]
        BC.Vy[2, iny_Vy]        .= (a.type.Vy[2, iny_Vy]   .== :Neumann_tangent)  .* D_BC[2,1] .+ (a.type.Vy[2, iny_Vy]   .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[end-1, iny_Vy]   .= (a.type.Vy[end-1,iny_Vy].== :Neumann_tangent)  .* D_BC[2,1] .+ (a.type.Vy[end-1,iny_Vy].== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Phase geometry
    @views a.phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    @views a.phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2
    @views a.phases.v[[2,end-1], :] .= 3
    @views a.phases.v[:, [2,end-1]] .= 3
    @views a.phases.c[[2,end-1], :] .= 3
    @views a.phases.c[:, [2,end-1]] .= 3
    FillPhaseRatios!(a)

    rvec = zeros(length(α))
    err = (x=zeros(niter), y=zeros(niter), p=zeros(niter))
    to = TimerOutput()

    for it = 1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        fig = Figure(size=(900,700), fontsize=14)

        ax1 = Axis(fig[1,1], xlabel="Iterations @ step $(it)", ylabel="log₁₀ error", title="Convergence")
        scatter!(ax1, 1:iter, log10.(err.x[1:iter]), markersize=6, label="Vx")
        scatter!(ax1, 1:iter, log10.(err.y[1:iter]), markersize=6, label="Vy")
        axislegend(ax1, position=:rt)

        ax2 = Axis(fig[1,2], title="Vx", aspect=DataAspect())
        heatmap!(ax2, xv, yc, a.V.x[inx_Vx,iny_Vx]')
        xlims!(ax2, extrema(xv))

        ax3 = Axis(fig[2,1], title="ε̇II", aspect=DataAspect())
        hm3 = heatmap!(ax3, xc, yc, log10.(a.ε̇.II[inx_c,iny_c])'; colormap=:coolwarm, colorrange=(-0.4,0.4))
        xlims!(ax3, extrema(xc))
        Colorbar(fig[2,1, Right()], hm3, width=12)

        ax4 = Axis(fig[2,2], title="τxx", aspect=DataAspect())
        hm4 = heatmap!(ax4, xc, yc, a.τ.xx[inx_c,iny_c]'; colormap=:turbo)
        xlims!(ax4, extrema(xc))
        Colorbar(fig[2,2, Right()], hm4, width=12)

        display(fig)
    end
    display(to)
end


let
    main((x = 200, y = 200))
end
