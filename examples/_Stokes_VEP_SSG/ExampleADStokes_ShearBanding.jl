using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs

@views function main(nc)
    #--------------------------------------------#

    # Resolution

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [ -1. 0.;
                          0  1 ])

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; plasticity=DruckerPrager,compressible=true)
    materials.g .= [0. , 0.]
    materials.ρ   .= [1.0 ,   1.0  ]
    materials.n   .= [1.0 ,   1.0  ]
    materials.η0  .= [1e2 ,   1e-1 ]
    materials.ξ0  .= [1e50,   1e50 ]
    materials.G   .= [1e1 ,   1e1  ]
    materials.plasticity.C   .= [150 ,   150  ]
    materials.plasticity.ϕ   .= [30. ,   30.  ]
    materials.plasticity.ηvp .= [0.5 ,   0.5  ]
    materials.β   .= [1e-2,   1e-2 ]
    materials.plasticity.ψ   .= [0.0 ,   0.0  ]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    =  40

    # Solver parameters
    iter_params = IterParams(niter=10, γ=1e5, ϵ_l=1e-11, ϵ_nl=1e-8, inexact=false, solver_type=:GCR, α=LinRange(0.05, 1.0, 6))

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x   = (min=-L.x/2, max=L.x/2)
    y   = (min=-L.y/2,   max=L.x/2  )

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1, 1] * a.X.vx_e.x .+ D_BC[1, 2] * a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1] * a.X.vy_e.x .+ D_BC[2, 2] * a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 0.
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
    # @views a.phases.v[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    # @views a.phases.v[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries
    # @views a.phases.c[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    # @views a.phases.c[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries
    FillPhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    for it=1:nt #anim = @animate

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#
        fig = Figure(size=(900,700), fontsize=14)

        ax1 = Axis(fig[1,1], xlabel="Iterations @ step $(it)", ylabel="log₁₀ error", title="Convergence")
        scatter!(ax1, 1:iter, log10.(err.x[1:iter]), markersize=6, label="Vx")
        scatter!(ax1, 1:iter, log10.(err.y[1:iter]), markersize=6, label="Vy")
        axislegend(ax1, position=:rt)

        ax2 = Axis(fig[1,2], title="Vx", aspect=DataAspect())
        heatmap!(ax2, a.X.v.x, a.X.c.y, a.V.x[inx_Vx,iny_Vx]')
        xlims!(ax2, extrema(a.X.v.x))

        ax3 = Axis(fig[2,1], title="ε̇II", aspect=DataAspect())
        hm3 = heatmap!(ax3, a.X.c.x, a.X.c.y, log10.(a.ε̇.II[inx_c,iny_c])'; colormap=:coolwarm, colorrange=(-0.4,0.4))
        xlims!(ax3, extrema(a.X.c.x))
        Colorbar(fig[2,1, Right()], hm3, width=12)

        ax4 = Axis(fig[2,2], title="τxx", aspect=DataAspect())
        hm4 = heatmap!(ax4, a.X.c.x, a.X.c.y, a.τ.xx[inx_c,iny_c]'; colormap=:turbo)
        xlims!(ax4, extrema(a.X.c.x))
        Colorbar(fig[2,2, Right()], hm4, width=12)

        display(fig)

        @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))

    end
    # gif(anim, "./results/ShearBanding.gif", fps = 5)

    display(to)

end

let
    main((x = 100, y = 100))
end
