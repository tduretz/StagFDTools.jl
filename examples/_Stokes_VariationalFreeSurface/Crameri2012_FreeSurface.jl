# Variational free surface: the two free-surface benchmarks of Crameri et al. (2012, GJI)
#
#   Case 1: isostatic relaxation of a 7 km cosine topography (λ = 2800 km) on a 100 km thick lithosphere
#           (η = 1e23 Pa s) above a 600 km mantle (η = 1e21 Pa s), ρ = 3300 kg/m³ everywhere.
#           Analytical maximum topography: h(t) = 7 exp(-t / 14.825 ka) km (three-layer solution, Ramberg).
#   Case 2: plume (r = 50 km, ρ = 3200 kg/m³, η = 1e20 Pa s, centre 300 km above the bottom) rising below a
#           flat 100 km lithosphere. Reference: true free-surface MILAMIN_VEP, max. topography 398 m at 3 Ma.
#
# Box 2800 km wide, 700 km of rock; free-slip sides, no-slip bottom. The variational method needs no sticky
# air: the grid only extends a few cells above the surface (the air layer has no equations).
# Nondimensionalisation: L = 100 km, η = 1e21 Pa s, ρ = 3300 kg/m³, g = 10 m/s² → σ = ρgL, t = η/σ.
using StagFDTools, StagFDTools.Stokes, StagFDTools.StokesVariational, StagFDTools.Rheology
using StaticArrays, LinearAlgebra, Printf, CairoMakie
using JustPIC
using CellArraysIndexing: @index
const backend = JustPIC.CPU

const sc = let L = 1e5, η = 1e21, ρ = 3300.0, g = 10.0
    σ = ρ * g * L
    (L=L, η=η, ρ=ρ, g=g, σ=σ, t=η / σ)
end
const yr = 365.25 * 24 * 3600

# Phases
const mantle, lithosphere, plume, air = 1, 2, 3, 4

function set_initial_phases!(phases, particles, case)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                ph = y > -1.0 ? lithosphere : mantle
                case == 2 && (x - 14.0)^2 + (y + 4.0)^2 < 0.5^2 && (ph = plume)
                @index phases[ip, i, j] = Float64(ph)
            end
        end
    end
end

function main(case; nc=(x=280, y=146), Δt_yr=500.0, t_end_yr=100e3, nout=20, ytop=0.3, nxcell=12, tag="", θ_fs=0.0)

    # Materials (nondimensional): mantle, lithosphere, plume, air (air is removed from the phase ratios)
    nphases = 4
    materials = initialize_materials(nphases; compressible=false)
    materials.g  .= [0.0, -10.0 / sc.g]
    materials.ρ  .= [3300.0, 3300.0, 3200.0, 0.0] ./ sc.ρ
    materials.η0 .= [1e21, 1e23, 1e20, 1e21] ./ sc.η
    materials.n  .= 1.0
    materials.G  .= 1e30
    preprocess!(materials)

    # Grid: 2800 km × (700 km of rock + ytop of air)
    x  = (min=0.0, max=28.0)
    y  = (min=-7.0, max=ytop)
    L  = (x=x.max - x.min, y=y.max - y.min)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y, t=Δt_yr * yr / sc.t)
    a  = Allocs(nc, :free_slip, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    a.type.Vx[inx_Vx, 2] .= :Dirichlet_tangent    # no slip at the bottom (Vx = 0 on y = y.min)
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))

    # Free surface (topography in units of L)
    xv = collect(a.X.v.x)
    h0 = case == 1 ? @.(7e3 / sc.L * cos(2π * xv / 28.0)) : zero(xv)
    chain = init_markerchain(backend, 20, 10, 40, xv, h0)

    # Material markers
    adv = Markers(backend, a, nxcell, 2nxcell, nxcell ÷ 3, nc, nphases)
    set_initial_phases!(adv.particle_args[1], adv.particles, case)
    set_grid_phases_free_surface!(a, adv, chain; air_phase=air, fallback_phase=lithosphere)

    fs = VariationalFreeSurface(a, nc; θ_fs)
    update_free_surface!(fs, a, chain, nc, Δ)

    # Time loop
    nt   = round(Int, t_end_yr / Δt_yr)
    time = zeros(nt + 1)
    htop = zeros(nt + 1)                         # Case 1: topography at x = 0, Case 2: maximum topography
    htop[1] = case == 1 ? chain.h_vertices[1] : maximum(chain.h_vertices)
    tsolve = 0.0
    for it in 1:nt
        tsolve += @elapsed iter, err = solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter=2, ϵ_nl=1e-10, noisy=false)
        err.x[iter] > 1e-8 && @warn "step $it: residual $(max(err.x[iter], err.y[iter]))"
        advect_free_surface!(a, adv, chain, BC, Δ; air_phase=air, refill_phase=lithosphere, fallback_phase=lithosphere)
        update_free_surface!(fs, a, chain, nc, Δ)
        time[it+1] = it * Δ.t
        htop[it+1] = case == 1 ? chain.h_vertices[1] : maximum(chain.h_vertices)
        if it % nout == 0 || it == nt
            @printf("step %04d, t = %8.2f ka, h = %8.2f m, (%.2f s per Stokes solve)\n", it, time[it+1] * sc.t / yr / 1e3, htop[it+1] * sc.L, tsolve / it)
        end
    end

    t_ka = time .* sc.t ./ yr ./ 1e3
    h_m  = htop .* sc.L

    # Plot
    fig = Figure(size=(1200, 700), fontsize=14)
    ax1 = Axis(fig[1, 1], xlabel="t [ka]", ylabel=case == 1 ? "topography at x = 0 [m]" : "max. topography [m]",
               title=case == 1 ? "Crameri et al. (2012), Case 1" : "Crameri et al. (2012), Case 2")
    lines!(ax1, t_ka, h_m, label="variational")
    if case == 1
        lines!(ax1, t_ka, 7e3 .* exp.(-t_ka ./ 14.825), color=:black, linestyle=:dash, label="analytical")
        vlines!(ax1, [14.825], color=:gray, linestyle=:dot)
    else
        scatter!(ax1, [3e3], [398.0], color=:black, label="MILAMIN_VEP (3 Ma)")
    end
    axislegend(ax1, position=case == 1 ? :rt : :rb)
    ax2 = Axis(fig[2, 1], aspect=DataAspect(), xlabel="x [km]", ylabel="y [km]", title="t = $(round(t_ka[end], digits=1)) ka")
    η = [sum(a.phase_ratios.c[i+1, j+1] .* log10.(materials.η0 .* sc.η)) for i in 1:nc.x, j in 1:nc.y]
    η[fs.φ.c[2:end-1, 2:end-1] .== 0] .= NaN
    hm = heatmap!(ax2, a.X.c.x .* 100, a.X.c.y .* 100, η, colormap=:vik)
    lines!(ax2, xv .* 100, chain.h_vertices .* 100, color=:black)
    Colorbar(fig[2, 2], hm, label="log₁₀ η")
    save(joinpath(@__DIR__, "Crameri2012_Case$(case)$(tag).png"), fig)
    display(fig)

    return (t_ka=t_ka, h_m=h_m, a=a, fs=fs, chain=chain)
end

if abspath(PROGRAM_FILE) == @__FILE__
    case1 = main(1; Δt_yr=500.0, t_end_yr=100e3)
    # free surface stabilisation (θ_fs > 0): Δt ≫ isostatic relaxation time (~15 ka); unstable without it at Δt = 50 ka
    case2 = main(2; Δt_yr=20e3, t_end_yr=20e6, nout=100, θ_fs=1.0)
end
