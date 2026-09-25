using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, CairoMakie, MathTeXEngine
CairoMakie.activate!(type="png")
Makie.update_theme!(fonts=(regular=texfont(), italic=texfont(:italic)))
import Statistics: mean
using JustPIC
import CellArraysIndexing: @index
const backend = JustPIC.CPU
using DifferentiationInterface
using TimerOutputs, GridGeometryUtils

const year2sec = 3.154e7

function set_phases!(phases, particles, block, layering)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                # quick escape
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                𝐱 = SVector(x, y)
                if inside(𝐱, block)
                    @index phases[ip, i, j] = 1.0 # block
                elseif inside(𝐱, layering)
                    @index phases[ip, i, j] = 3.0 # thin (yellow) layers within the matrix
                else
                    @index phases[ip, i, j] = 2.0 # matrix (blue)
                end
            end
        end
    end
end

@views function main(nc, BC_template, D_template, η_block, η_matrix, it_export)
    #--------------------------------------------#

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    width = 5e5
    ρ_block, ρ_matrix = 3300.0, 3200.0

    # Material parameters
    sc = (σ=1e10, t=1e10, L=1e5)
    nphases = 3
    materials = initialize_materials(nphases; compressible=false)
    # 1: block, 2: matrix, 3: thin layers (same rheology as the matrix - passive colouring only)
    materials.g .= [0.0, -9.81] ./ (sc.L/sc.t^2)
    materials.ρ .= [ρ_block, ρ_matrix, ρ_matrix] ./ (sc.σ*sc.t^2/sc.L^2)
    materials.η0 .= [η_block, η_matrix, η_matrix] ./ (sc.σ*sc.t)
    materials.G .= [1e60, 1e60, 1e60] ./ sc.σ
    preprocess!(materials)

    # Setup
    L = (x=width/sc.L, y=width/sc.L)
    block = Rectangle((25e4/sc.L, 4e5/sc.L), 1e5/sc.L, 1e5/sc.L)

    n_layers = 10
    layering = Layering(
        (L.x/2, L.y/2), L.y/n_layers, 0.15;
        θ=0.0, perturb_amp=0.0, perturb_width=1.0,
    )

    # Time steps
    nt = 100
    Δt0 = 1e5*year2sec/sc.t

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

    # Boundary condition values (free-slip box: all values are zero)
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[1] .+ D_BC[2, 2]*a.X.v.y)
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[end] .+ D_BC[2, 2]*a.X.v.y)
    end

    # Initialize particles
    nxcell = (5, 5)
    max_xcell = 40
    min_xcell = 10
    args = 1
    adv = Markers(backend, a, nxcell, max_xcell, min_xcell, nc, nphases, args)
    phases, = adv.particle_args

    # Set material geometry
    set_phases!(phases, adv.particles, block, layering)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, adv.particle_args[1])

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    t_cur = 0.0
    Vy_block_t = zeros(nt) # time series of the phase-weighted average vertical block velocity [non-dim]
    markers_export = nothing
    for it=1:nt

        # Adaptive time step
        Vmax = max(maximum(abs.(a.V.x)), maximum(abs.(a.V.y)))
        if Vmax > 0
            Δ = (x=Δ.x, y=Δ.y, t=min(Δt0, 0.5*min(Δ.x, Δ.y)/Vmax))
        end
        t_cur += Δ.t

        block_frac = [a.phase_ratios.c[i, j][1] for i in inx_c, j in iny_c]

        @time main_loop(a, adv, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        Vy_c = 0.5 .* (a.V.y[inx_Vy, iny_Vy][:, 1:(end-1)] .+ a.V.y[inx_Vy, iny_Vy][:, 2:end])
        Vy_block_t[it] = sum(Vy_c .* block_frac) / sum(block_frac)

        if it == it_export
            idxv = adv.particles.index.data[:]
            ppx, ppy = adv.particles.coords
            markers_export = (
                it=it,
                t=t_cur * sc.t / year2sec, # [yr]
                x=Array(ppx.data[:][idxv]) .* sc.L ./ 1e3, # [km]
                y=Array(ppy.data[:][idxv]) .* sc.L ./ 1e3, # [km]
                phase=Array(phases.data[:][idxv]),
            )
        end

        #--------------------------------------------#

        # Visualise (converted back to physical units for readability)
        function visualisation()
            #-----------
            fig = Figure(size=(1000, 800))
            #-----------
            ax = Axis(fig[1, 1], aspect=DataAspect(), title="Dynamic pressure, step $(it) (t = $(round(t_cur*sc.t/year2sec, digits=1)) yr)", xlabel="x [km]", ylabel="y [km]")
            Pt_phys = a.Pt[inx_c, iny_c] .* sc.σ
            Pdyn = Pt_phys .- mean(Pt_phys, dims=1) # remove the depth-dependent lithostatic gradient, which dwarfs the block's signal and hides it
            hm = heatmap!(ax, a.X.c.x .* sc.L ./ 1e3, a.X.c.y .* sc.L ./ 1e3, Pdyn, colormap=:bluesreds)
            Colorbar(fig[1, 2], hm)

            ax = Axis(fig[1, 3], aspect=DataAspect(), title="Particles (phase)", xlabel="x [km]", ylabel="y [km]")
            ppx, ppy = adv.particles.coords
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            clr = phases.data[:]
            idxv = adv.particles.index.data[:]
            # 1=block (grey), 2=matrix (blue), 3=thin layers (yellow)
            phase_colors = cgrad([:grey40, :dodgerblue, :gold], 3, categorical=true)
            scatter!(ax, Array(pxv[idxv]) .* sc.L ./ 1e3, Array(pyv[idxv]) .* sc.L ./ 1e3, color=Array(clr[idxv]), colormap=phase_colors, colorrange=(0.5, 3.5), markersize=4)

            ax = Axis(fig[2, 1], aspect=DataAspect(), title=L"\tau_{II}", xlabel="x [km]", ylabel="y [km]")
            heatmap!(ax, a.X.c.x .* sc.L ./ 1e3, a.X.c.y .* sc.L ./ 1e3, a.τ.II[inx_c, iny_c] .* sc.σ, colormap=:bluesreds)
            ax = Axis(fig[2, 2], xlabel="Iterations step $(it)", ylabel="log₁₀ error")
            scatter!(ax, 1:iter_params.niter, log10.(err.x[1:iter_params.niter]), markersize=8, label="Vx")
            scatter!(ax, 1:iter_params.niter, log10.(err.y[1:iter_params.niter]), markersize=8, label="Vy")
            scatter!(ax, 1:iter_params.niter, log10.(err.p[1:iter_params.niter]), markersize=8, label="Pt")
            axislegend(ax, position=:rt)
            # display(fig)
        end
        with_theme(visualisation, theme_latexfonts())
    end
    display(to)
    v_avg = mean(Vy_block_t) * sc.L / sc.t # time-averaged vertical block velocity [m/s]
    return v_avg, markers_export
end


let
    nc = (x=51, y=51)

    # Boundary condition templates
    BCs = [
        :free_slip,
    ]

    D_BCs = [
        @SMatrix([1e-6 0.0; 0.0 -1e-6]),
    ]

    η_matrix = 1e21
    η_block = @SVector([1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24])
    v_avg = zeros(size(η_block))
    markers_all = Vector{Any}(undef, length(η_block))

    it_export = 70

    # Run them all
    for i in eachindex(η_block)
        @info "Running $(string(BCs[1])) and η_block = $(η_block[i])"
        v_avg[i], markers_all[i] = main(nc, BCs[1], D_BCs[1], η_block[i], η_matrix, it_export)
    end

    n_panels = length(η_block) + 1
    ncols = ceil(Int, sqrt(n_panels))
    nrows = ceil(Int, n_panels / ncols)

    function sweep_plot()
        fig = Figure(size=(320 * ncols, 300 * nrows))
        phase_colors = cgrad([:grey40, :dodgerblue, :gold], 3, categorical=true)

        for i in eachindex(η_block)
            row, col = divrem(i - 1, ncols) .+ (1, 1)
            m = markers_all[i]
            ax = Axis(fig[row, col], aspect=DataAspect(),
                title=L"\mathrm{time \ step} = 70, \ \eta_{block}/\eta_{matrix} = 10^{%$(round(log10(η_block[i]/η_matrix), digits=1))}",
                xlabel="x [km]", ylabel="y [km]")
            scatter!(ax, m.x, m.y, color=m.phase, colormap=phase_colors, colorrange=(0.5, 3.5), markersize=3)
        end

        row, col = divrem(n_panels - 1, ncols) .+ (1, 1)
        ax = Axis(fig[row, col:(col+1)], xlabel=L"\log_{10}(\eta_{block}/\eta_{matrix})", ylabel="Average block velocity [m/s]",
            title="Sinking velocity vs. viscosity ratio")
        scatterlines!(ax, Vector(log10.(η_block ./ η_matrix)), .-v_avg)

        display(fig)
    end
    with_theme(sweep_plot, theme_latexfonts())
end
