using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, CairoMakie, MathTeXEngine
Makie.update_theme!(fonts=(regular=texfont(), bold=texfont(:bold), italic=texfont(:italic)))
import Statistics: mean
using JustPIC, JustPIC._2D
import JustPIC.@index
const backend = JustPIC.CPUBackend
using DifferentiationInterface
using TimerOutputs, GridGeometryUtils


function set_phases!(phases, particles, garnets, micas, layering)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                # quick escape
                @index(particles.index[ip, i, j]) == 0 && continue

                # Set material geometry 
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                𝐱 = @SVector([x, y])

                @index phases[ip, i, j] = 1.0

                if inside(𝐱, layering)
                    @index phases[ip, i, j] = 2.0
                end

                for igeom in eachindex(garnets) # Garnets: phase 2
                    if inside(𝐱, garnets[igeom])
                        @index phases[ip, i, j] = 3.0
                    end
                end

                # for igeom in eachindex(micas) # Micas: phase 3
                #     if inside(𝐱, micas[igeom])
                #         @index phases[ip, i, j] = 3.0
                #     end
                # end

            end
        end
    end
end

@views function main(nc, BC_template, D_template)
    #--------------------------------------------#

    # Boundary loading type
    config = :free_slip
    # config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 3
    materials = initialize_materials(nphases; compressible=true)
    materials.g .= [0., 0.]
    materials.η0 .= [1e0, 1e0, 1e3]
    materials.β .= [1e-5, 1e-5, 1e-5]
    preprocess!(materials)

    # Material geometries
    garnets = (
        Hexagon((-0.0, 0.0), 0.200; θ=π / 4),
    )

    micas = (
        Rectangle((0.1, -0.1), 0.03, 0.07; θ=-π / 4), #0.1, -0.1, 0.03, 0.07, -45
    )

    layering = Layering(
        (0., 0.5),
        0.1,
        0.5;
        θ=0.,
        perturb_amp=0 * 1.0,
        perturb_width=1.0
    )

    # Time steps
    Δt0 = 0.5
    nt = 100
    ALE = false
    C = 0.5

    # Solver parameters
    iter_params = IterParams(niter=2, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10)) # default parameters

    # X
    L = (x=1.0, y=1.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y, max=0.0)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure
    @views a.V.x .= D_BC[1, 1] * a.X.vx_e.x .+ D_BC[1, 2] * a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1] * a.X.vy_e.x .+ D_BC[2, 2] * a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 0.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
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

    # Initialize particles
    nxcell = 36 # initial number of particles per cell
    max_xcell = 36 * 2 # maximum number of particles per cell
    min_xcell = 1 # minimum number of particles per cell
    args = 1 # Fields to be advected (1=phase)
    adv = JustPICAdvection(backend, a, nxcell, max_xcell, min_xcell, nc, nphases, args)
    phases, = adv.particle_args
    # Set material geometry
    set_phases!(phases, adv.particles, garnets, micas, layering)
    update_phase_ratios!(adv.phase_ratios, adv.particles, adv.particle_args[1])
    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    fig = Figure(size=(500, 500))

    #--------------------------------------------#

    # for it=1:nt
    record(fig, "results/SimpleShearGarnets.mp4", 1:nt; framerate=15) do it

        Vmax = max(maximum(abs.(a.V.x)), maximum(abs.(a.V.y)))
        Δ = (x=Δ.x, y=Δ.y, t=C * min(Δ.x, Δ.y) / Vmax)
        main_loop(a, adv, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#
        # Visualise
        xc, yc = a.X.c.x, a.X.c.y
        xv, yv = a.X.v.x, a.X.v.y
        empty!(fig)
        ax = Axis(fig[1, 1], aspect=DataAspect(), title=L"$$Pressure", xlabel=L"$x$", ylabel=L"$y$")
        hm = heatmap!(ax, xc, yc, (a.Pt[inx_c, iny_c]), colormap=(:bluesreds), colorrange=(-3, 3))
        Colorbar(fig, hm, width=10,
            labelsize=10, ticklabelsize=10, bbox=ax.scene.viewport,
            alignmode=Outside(8), halign=:right, ticklabelcolor=:black, labelcolor=:black,
            tickcolor=:black)
        ax = Axis(fig[1, 2], aspect=DataAspect(), title=L"$$Materials", xlabel=L"$x$", ylabel=L"$y$")
        ppx, ppy = adv.particles.coords
        pxv = ppx.data[:]
        pyv = ppy.data[:]
        clr = adv.particle_args[1].data[:]
        idxv = adv.particles.index.data[:]
        scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=CairoMakie.Reverse(:roma), markersize=5)
        xlims!(ax, extrema(xv))
        ylims!(ax, extrema(yv))
        ax = Axis(fig[2, 1], aspect=DataAspect(), title=L"$\tau_{xx}$", xlabel=L"$x$", ylabel=L"$y$")
        hm = heatmap!(ax, xc, yc, a.τ.xx[inx_c, iny_c], colormap=(:bluesreds), colorrange=(-2, 2))
        Colorbar(fig, hm, width=10,
            labelsize=10, ticklabelsize=10, bbox=ax.scene.viewport,
            alignmode=Outside(8), halign=:right, ticklabelcolor=:black, labelcolor=:black,
            tickcolor=:black)
        ax = Axis(fig[2, 2], aspect=DataAspect(), title=L"$\tau_{xy}$", xlabel=L"$x$", ylabel=L"$y$")
        hm = heatmap!(ax, xv, yv, a.τ.xy[inx_v, iny_v], colormap=(:bluesreds), colorrange=(-0, 3.0))
        Colorbar(fig, hm, width=10,
            labelsize=10, ticklabelsize=10, bbox=ax.scene.viewport,
            alignmode=Outside(8), halign=:right, ticklabelcolor=:black, labelcolor=:black,
            tickcolor=:black)
        display(fig)
    end
    # display(to)
end



let

    # Resolution
    nc = (x=250, y=250)

    # # Boundary condition templates
    BCs = [
        # :free_slip,
        :EW_periodic,
    ]

    # Boundary velocity gradient matrix
    D_BCs = [
        # @SMatrix( [1 0; 0 -1] ),
        @SMatrix([0 1; 0 0]),
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(nc, BCs[iBC], D_BCs[iBC])
    end
end