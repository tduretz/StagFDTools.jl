using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using DifferentiationInterface
using TimerOutputs
using GridGeometryUtils

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x=50, y=50) # number of cells
    nmpc = (x=4, y=4)  # markers per cell
    noise = false         # noise in marker distribution

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; compressible=false)

    # Parameters
    params_bg = (ρ=1.0, n=1.0, η0=1e0, G=1e6, β=0.5)
    params_in = (ρ=1.0, n=1.0, η0=1e5, G=1e6, β=0.5)

    materials.g .= [0., 0.]
    materials.ρ .= [params_bg.ρ, params_in.ρ]
    materials.n .= [params_bg.n, params_in.n]
    materials.η0 .= [params_bg.η0, params_in.η0]
    materials.G .= [params_bg.G, params_in.G]
    materials.β .= [params_bg.β, params_in.β]

    preprocess!(materials)

    # Time steps
    Δt0 = 0.5
    nt = 1

    # Solver parameters
    iter_params = IterParams() # default parameters

    # Intialise field
    L = (x=1.0, y=1.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases, nmpc, noise)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure
    @views a.V.x .= D_BC[1, 1] * a.X.vx_e.x .+ D_BC[1, 2] * a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1] * a.X.vy_e.x .+ D_BC[2, 2] * a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * a.X.v.x .+ D_BC[1, 2] * a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * a.X.v.x .+ D_BC[1, 2] * a.X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * a.X.v.x[1] .+ D_BC[2, 2] * a.X.v.y)
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * a.X.v.x[end] .+ D_BC[2, 2] * a.X.v.y)
    end

    # --------------------------------------------#
    # Set material geometry
    incl = Hexagon((0.0, 0.0), 0.2; θ=π / 10)
    for i in eachindex(a.m.phase)
        𝐱 = SVector(a.m.Xm[i], a.m.Ym[i])
        isin = inside(𝐱, incl)
        if isin
            a.m.phase[i] = 2
        end
    end

    # Set phase ratios on grid
    SetPhaseRatios!(a.phase_ratios, a.m, a.X.c_e.x, a.X.c_e.y, a.X.v_e.x, a.X.v_e.y, Δ, nphases)

    for I in CartesianIndices(a.phase_ratios.c)
        s = sum(a.phase_ratios.c[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.center at $I: sum = $s, values = $(phase_ratios.center[I])"
        end
    end

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#
        # Plot
        Fig = Figure(size=(1200, 900), fontsize=14)
        ax1 = Axis(Fig[1, 1], aspect=DataAspect(), title="Vx", xlabel="x", ylabel="y")
        ax2 = Axis(Fig[1, 3], aspect=DataAspect(), title="Vy", xlabel="x", ylabel="y")
        ax3 = Axis(Fig[2, 1], aspect=DataAspect(), title="Pt", xlabel="x", ylabel="y")
        # ax4 = Axis(Fig[2, 3], aspect=DataAspect(), title="Markers", xlabel="x", ylabel="y")
        ax4 = Axis(Fig[2, 3], xlabel="Iterations step $(it)", ylabel="log₁₀ error")

        hm1 = heatmap!(ax1, a.X.v.x, a.X.c.y, a.V.x[inx_Vx, iny_Vx]', colormap=:redsblues)
        hm2 = heatmap!(ax2, a.X.c.x, a.X.c.y, a.V.y[inx_Vy, iny_Vy]', colormap=:redsblues)
        hm3 = heatmap!(ax3, a.X.c.x, a.X.c.y, a.Pt[inx_c, iny_c]' .- mean(a.Pt[inx_c, iny_c]), colormap=:redsblues)
        Colorbar(Fig[1, 2], hm1)
        Colorbar(Fig[1, 4], hm2)
        Colorbar(Fig[2, 2], hm3)

        # hm4 = heatmap!(ax4, m.xm, m.ym, mphase, colormap=:viridis)

        scatter!(ax4, 1:iter_params.niter, log10.(err.x[1:iter_params.niter]), markersize=8, label="Vx")
        scatter!(ax4, 1:iter_params.niter, log10.(err.y[1:iter_params.niter]), markersize=8, label="Vy")
        scatter!(ax4, 1:iter_params.niter, log10.(err.p[1:iter_params.niter]), markersize=8, label="Pt")
        axislegend(ax4, position=:rt)
        display(Fig)
    end

    display(to)

end


let
    # # Boundary condition templates
    BCs = [
        :free_slip,
    ]

    # # Boundary deformation gradient matrix
    # D_BCs = [
    #     @SMatrix( [1 0; 0 -1] ),
    # ]

    # BCs = [
    #     # :EW_periodic,
    #     :all_Dirichlet,
    # ]

    # Boundary deformation gradient matrix
    er = -1
    # ∂𝐕∂𝐱 - velocity gradient tensor 
    D_BCs = [
        #  @SMatrix( [0 1; 0  0] ),
        @SMatrix([er 0;        #    ∂Vx∂x ∂Vx∂y
            0 -er]),    #    ∂Vy∂x ∂Vy∂y  div(V) = 0 = ∂Vx∂x + ∂Vy∂y --> ∂Vy∂y = - ∂Vx∂x
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end