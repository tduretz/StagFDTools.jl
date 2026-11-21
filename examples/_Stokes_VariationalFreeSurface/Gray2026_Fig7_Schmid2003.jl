# Variational free surface: inclusion benchmark of Gray et al. (2026, GMD), Fig. 7 (variational method) and
# the variational curves of Figs. 8-9 (convergence).
#
# Pure shear around a void (Schmid & Podladchikov, 2003, inclusion viscosity → 0): the void boundary is a
# free surface. The domain [-0.25, 0.25] × [-0.75, -0.25] cuts the bottom of the void (radius 0.5, centred at
# the origin), so the surface y = -√(r² - x²) is a height function, represented by a JustPIC marker chain.
# Dirichlet velocities from the analytical solution on all sides (as in Gray's FreeSurfaceAccuracyTest.jl).
using StagFDTools, StagFDTools.Stokes, StagFDTools.StokesVariational, StagFDTools.Rheology
using ExactFieldSolutions, StaticArrays, LinearAlgebra, Printf, CairoMakie
using JustPIC
const backend = JustPIC.CPU

function main(n; nxcell=16, plot=true, noisy=true)

    # Analytical solution: matrix viscosity mm, (vanishing) inclusion viscosity mc, radius rc, pure shear er
    params = (mm=1.0, mc=1e-10, rc=0.5, gr=0.0, er=-1.0)
    exact(x, y) = Stokes2D_Schmid2003_circle(SVector(x, y); params)

    # Materials: phase 1 = rock, phase 2 = air (removed from the grid phase ratios, the fluid fraction is in φ)
    nphases, rock, air = 2, 1, 2
    materials = initialize_materials(nphases; compressible=false)
    materials.g  .= [0.0, 0.0]
    materials.ρ  .= [0.0, 0.0]
    materials.n  .= [1.0, 1.0]
    materials.η0 .= [params.mm, params.mm]
    materials.G  .= [1e30, 1e30]   # viscous
    preprocess!(materials)

    # Grid
    nc = (x=n, y=n)
    L  = (x=0.5, y=0.5)
    x  = (min=-0.25, max=0.25)
    y  = (min=-0.75, max=-0.25)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y, t=1.0)
    a  = Allocs(nc, :all_Dirichlet, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial guess and Dirichlet values (boundary nodes) from the analytical solution
    for I in CartesianIndices(a.V.x)
        a.V.x[I] = exact(a.X.vx_e.x[I[1]], a.X.vx_e.y[I[2]]).V[1]
    end
    for I in CartesianIndices(a.V.y)
        a.V.y[I] = exact(a.X.vy_e.x[I[1]], a.X.vy_e.y[I[2]]).V[2]
    end
    # Boundary values of the ghost (Dirichlet_tangent) nodes
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    BC.Vx[inx_Vx, 2]     .= [exact(xv, y.min).V[1] for xv in a.X.v.x]
    BC.Vx[inx_Vx, end-1] .= [exact(xv, y.max).V[1] for xv in a.X.v.x]
    BC.Vy[2, iny_Vy]     .= [exact(x.min, yv).V[2] for yv in a.X.v.y]
    BC.Vy[end-1, iny_Vy] .= [exact(x.max, yv).V[2] for yv in a.X.v.y]

    # Free surface: marker chain at the grid vertices
    h     = @. -sqrt(params.rc^2 - a.X.v.x^2)
    chain = init_markerchain(backend, 20, 10, 40, collect(a.X.v.x), collect(h))

    # Material markers: air above the chain
    adv = Markers(backend, a, nxcell, 2nxcell, nxcell ÷ 4, nc, nphases)
    phases = adv.particle_args[1]
    set_phases_from_chain!(phases, adv.particles, chain, a.X.v.x; air_phase=Float64(air), rock_phase=Float64(rock))
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)
    remove_air_phase!(a.phase_ratios, air; fallback_phase=rock)

    # Fluid fractions, flags, numbering
    fs = VariationalFreeSurface(a, nc)
    update_free_surface!(fs, a, chain, nc, Δ)
    noisy && @printf("n = %d: %d Vx, %d Vy, %d Pt unknowns\n", n, maximum(a.number.Vx), maximum(a.number.Vy), maximum(a.number.Pt))

    # Solve
    iter, err = solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter=3, ϵ_nl=1e-11, noisy)

    # Errors: only in control volumes entirely within the fluid (as in Gray et al., 2026)
    full = 1.0 - 1e-9
    eP   = zeros(nc.x, nc.y)
    eVx  = zeros(nc.x + 1, nc.y)
    eVy  = zeros(nc.x, nc.y + 1)
    for I in CartesianIndices(eP)
        i, j = Tuple(I)
        fs.φ.c[i+1, j+1] ≥ full && (eP[I] = a.Pt[i+1, j+1] - exact(a.X.c.x[i], a.X.c.y[j]).p)
    end
    for I in CartesianIndices(eVx)
        i, j = Tuple(I)
        fs.φ.Vx[i+1, j+2] ≥ full && (eVx[I] = a.V.x[i+1, j+2] - exact(a.X.vx.x[i], a.X.vx.y[j]).V[1])
    end
    for I in CartesianIndices(eVy)
        i, j = Tuple(I)
        fs.φ.Vy[i+2, j+1] ≥ full && (eVy[I] = a.V.y[i+2, j+1] - exact(a.X.vy.x[i], a.X.vy.y[j]).V[2])
    end
    # L1 norms: Σ|e| Δx Δy / (Lx Ly) (definition of Gray's FreeSurfaceAccuracyTest.jl)
    L1(e) = sum(abs, e) * Δ.x * Δ.y / (L.x * L.y)
    errors = (P=L1(eP), Vx=L1(eVx), Vy=L1(eVy))
    noisy && @printf("h = %1.4e: L1 errors P = %1.4e, Vx = %1.4e, Vy = %1.4e\n", Δ.x, errors...)

    if plot
        fig = Figure(size=(1200, 450), fontsize=14)
        titles = ("Pressure error", "x velocity error", "y velocity error")
        fields = (eP, eVx, eVy)
        coords = ((a.X.c.x, a.X.c.y), (a.X.vx.x, a.X.vx.y), (a.X.vy.x, a.X.vy.y))
        clims  = ((-0.05, 0.05), (-5e-4, 5e-4), (-5e-4, 5e-4))
        for k in 1:3
            ax = Axis(fig[1, 2k-1], aspect=DataAspect(), title=titles[k], xlabel="x", ylabel="y")
            hm = heatmap!(ax, coords[k]..., fields[k], colormap=:RdBu, colorrange=clims[k])
            lines!(ax, a.X.v.x, h, color=:black, linewidth=2)
            xlims!(ax, x.min, x.max); ylims!(ax, y.min, y.max)
            Colorbar(fig[1, 2k], hm)
        end
        save(joinpath(@__DIR__, "Gray2026_Fig7_variational_$(n).png"), fig)
        display(fig)
    end

    return (h=Δ.x, errors..., iter=iter, err=err)
end

# Convergence (variational curves of Gray et al., 2026, Figs. 8-9)
function convergence(ns; kwargs...)
    res = [main(n; plot=false, kwargs...) for n in ns]
    h   = [r.h for r in res]
    fig = Figure(size=(1200, 400), fontsize=14)
    for (k, f) in enumerate((:P, :Vx, :Vy))
        e  = [getproperty(r, f) for r in res]
        ax = Axis(fig[1, k], xscale=log10, yscale=log10, xlabel="1/h", ylabel="L1 error $(f)", title="$(f)")
        scatterlines!(ax, 1 ./ h, e, label="variational")
        lines!(ax, 1 ./ h, e[1] .* (h ./ h[1]), color=:gray, linestyle=:dash, label="O(h)")
        axislegend(ax, position=:lb)
        rates = log.(e[1:end-1] ./ e[2:end]) ./ log.(h[1:end-1] ./ h[2:end])
        @printf("%s: rates = %s\n", f, join((@sprintf("%.2f", r) for r in rates), ", "))
    end
    save(joinpath(@__DIR__, "Gray2026_convergence_variational.png"), fig)
    display(fig)
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(100)                               # Fig. 7, variational column (100 × 100)
    # convergence([20, 40, 80, 160, 320])  # Figs. 8-9, variational curves
end
