using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, CairoMakie, MathTeXEngine
CairoMakie.activate!(type="png")
Makie.update_theme!(fonts=(regular=texfont(), italic=texfont(:italic)))
import Statistics: mean
using JustPIC
using CellArraysIndexing: @index
const backend = JustPIC.CPU
using DifferentiationInterface
using TimerOutputs, GridGeometryUtils

const year2sec = 3.154e7

function set_geometry!(phases, colors, particles, plate, cell_size)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                𝐱 = @SVector([x, y])
                cb = iseven(Int(fld(x, cell_size)) + Int(fld(y, cell_size))) ? 0. : 1.
                if inside(𝐱, plate)
                    @index phases[ip, i, j] = 1.
                    @index colors[ip, i, j] = 2. + cb
                else
                    @index phases[ip, i, j] = 2.
                    @index colors[ip, i, j] = cb
                end
            end
        end
    end
end

@views function main(nc, BC_template, D_template)
    #--------------------------------------------#

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    width = 1e6
    ρ_plate, ρ_matrix = 4000., 1.
    G_plate, G_matrix = 1e10, 1e20
    η_plate, η_matrix = 1e27, 1e21

    # Material parameters
    sc = (σ=1e10, t=1e10, L=1e5)
    nphases = 2
    materials = initialize_materials(nphases; compressible=false)
    # 1: block, 2: matrix, 3: thin layers (same rheology as the matrix - passive colouring only)
    materials.g .= [0.0, -9.81] ./ (sc.L/sc.t^2)
    materials.ρ .= [ρ_plate, ρ_matrix] ./ (sc.σ*sc.t^2/sc.L^2)
    materials.η0 .= [η_plate, η_matrix] ./ (sc.σ*sc.t)
    materials.G .= [G_plate, G_matrix] ./ sc.σ
    preprocess!(materials)

    # Setup
    L = (x=width/sc.L, y=width/sc.L)

    # Time steps
    nt = 200 #200
    Δt0 = 1e3*year2sec/sc.t

    # Solver parameters
    iter_params = IterParams(solver_type=:PH, niter=10, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    Δ = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    x = (min=0.0, max=L.x)
    y = (min=0.0, max=L.y)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field (D_BC is a negligible seed strain rate; flow is otherwise driven by buoyancy alone)
    @views a.V.x .= D_BC[1, 1]*a.X.vx_e.x .+ D_BC[1, 2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1]*a.X.vy_e.x .+ D_BC[2, 2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 0.0
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values (free-slip box, except left wall: all values are zero)
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    a.type.Vy[2, iny_Vy] .= :Dirichlet_tangent   # left wall: no-slip instead of free-slip
    @views begin
        BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* 0.0
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[end] .+ D_BC[2, 2]*a.X.v.y)
    end

    # Initialize particles
    nxcell = (4, 4)
    max_xcell = 25
    min_xcell = 10
    args = 2
    particles, tm = TensorMarkers(backend, a, nxcell, max_xcell, min_xcell)
    adv = Markers(backend, a, particles, tm, nc, nphases; args)
    phases, colors = adv.particle_args

    # Set material geometry
    n_layers = 12
    plate = Rectangle((4e5/sc.L, 5e5/sc.L), 8e5/sc.L, 6e5/sc.L)
    cell_size = plate.h / n_layers
    set_geometry!(phases, colors, adv.particles, plate, cell_size)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)
    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()
    time_tot = 0.

    # Visualise setup
    with_theme(theme_latexfonts()) do
        fig = Figure(size=(800, 600))
        width_km = width/1e3
        ax = Axis(fig[1, 1], aspect=DataAspect(), title="Initial setup", xlabel="x [km]", ylabel="y [km]", limits=(0, width_km, 0, width_km),
            titlesize=20, xlabelsize=20, ylabelsize=20, xticklabelsize=16, yticklabelsize=16)
        ppx, ppy = adv.particles.coords
        pxv = ppx.data[:]
        pyv = ppy.data[:]
        clr = colors.data[:]
        idxv = adv.particles.index.data[:]
        # 0/1=matrix checkerboard (blues), 2/3=plate checkerboard (greys)
        checker_colors = cgrad([:lightskyblue, :dodgerblue, :grey70, :grey20], 4, categorical=true)
        scatter!(ax, Array(pxv[idxv]) .* sc.L ./ 1e3, Array(pyv[idxv]) .* sc.L ./ 1e3, color=Array(clr[idxv]), colormap=checker_colors, colorrange=(-0.5, 3.5), marker=:rect, markerspace=:data, strokewidth=0)

        display(fig)
    end

    #--------------------------------------------#
    for it=1:nt

        if time_tot/year2sec*sc.t >= 9900
            materials.g .= [0., 0.]
        end

        # Adaptive time step
        Vmax = max(maximum(abs.(a.V.x)), maximum(abs.(a.V.y)))
        if Vmax > 0
            Δ = (x=Δ.x, y=Δ.y, t=min(Δt0, 0.5*min(Δ.x, Δ.y)/Vmax))
        end
        time_tot += Δ.t
        @time main_loop(a, adv, tm, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Visualise
        function visualisation()
            #-----------
            fig = Figure(size=(600, 600))
            ax = Axis(fig[1, 1], aspect=DataAspect(), title="time $(round(Int64, time_tot/year2sec*sc.t)) yr", xlabel="x [km]", ylabel="y [km]", limits=(0, 1e3, 0, 1e3),
                titlesize=20, xlabelsize=20, ylabelsize=20, xticklabelsize=16, yticklabelsize=16)
            ppx, ppy = adv.particles.coords
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            clr = colors.data[:]
            idxv = adv.particles.index.data[:]
            checker_colors = cgrad([:lightskyblue, :dodgerblue, :grey70, :grey20], 4, categorical=true)
            scatter!(ax, Array(pxv[idxv]) .* sc.L ./ 1e3, Array(pyv[idxv]) .* sc.L ./ 1e3, color=Array(clr[idxv]), colormap=checker_colors, colorrange=(-0.5, 3.5), markersize=4)

            display(fig)
        end
        with_theme(visualisation, theme_latexfonts())
    end
    display(to)
    return nothing
end


let
    nc = (x=51, y=51)

    # Boundary condition templates
    BCs = [
        :free_slip,
    ]

    D_BCs = [
        @SMatrix([1e-10 0.0; 0.0 -1e-10]),
    ]

    main(nc, BCs[1], D_BCs[1])
end
