# Stress invariant τII from the step_XXXX.jld2 files of one ShearApparatus run, to look at the
# apparent friction oscillations and the gouge resolution.
# The full centroid fields are recovered from the parents of the saved views (τII_rock, P_rock, λ̇_rock
# are views of a.τ.II, a.Pt, a.λ̇.c including ghost cells).
# For each step: τII map, zoom on the gouge centre (cells visible), τII profile across the gouge at
# mid-height (one marker per cell, gouge cells highlighted) and apparatus friction with the current step.
using CairoMakie, JLD2, Printf

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

# Interior centroid field from a saved view of a ghost-cell array
centroid_field(v, d) = reshape(parent(v), length(d.xc) + 2, length(d.yc) + 2)[2:end-1, 2:end-1]

shortening(d) = d.probes.t[1:d.it] * d.ε̇yy * d.Ly * d.sc.L * 1e3 # mm

function plot_stress_fields(d; crange=nothing, zoom=10.0)
    sc   = d.sc
    τII  = centroid_field(d.τII_rock, d) .* sc.σ ./ 1e6 # MPa
    x, y = d.xc .* sc.L * 1e3, d.yc .* sc.L * 1e3       # mm
    Δx, Δy = x[2] - x[1], y[2] - y[1]
    crange = crange === nothing ? extrema(τII) : crange
    jmid = argmin(abs.(y .- d.Ly * sc.L * 1e3 / 2))
    ymid = y[jmid]

    return with_theme(theme_latexfonts()) do
        fig = Figure(size=(1200, 900))
        Label(fig[0, 1:3], @sprintf("step %d, shortening = %.2f mm, nc = %d×%d, Δx = %.2f mm, Δy = %.2f mm",
              d.it, shortening(d)[end], length(x), length(y), Δx, Δy), fontsize=18)

        # Full domain
        ax = Axis(fig[1:2, 1], aspect=DataAspect(), title=L"$\tau_{II}$ [MPa]", xlabel="x [mm]", ylabel="y [mm]")
        hm = heatmap!(ax, x, y, τII, colormap=:turbo, colorrange=crange, highclip=:white)
        contour!(ax, x, y, d.phc, color=:white, levels=[1.5, 2.5])
        lines!(ax, [x[1], x[end]], [ymid, ymid], color=:white, linestyle=:dash)
        Colorbar(fig[3, 1], hm, vertical=false, flipaxis=false)

        # Zoom on the gouge centre: each cell is visible
        ax = Axis(fig[1, 2:3], aspect=DataAspect(), title="zoom on gouge centre", xlabel="x [mm]", ylabel="y [mm]")
        heatmap!(ax, x, y, τII, colormap=:turbo, colorrange=crange)
        contour!(ax, x, y, d.phc, color=:white, levels=[1.5, 2.5])
        lines!(ax, [x[1], x[end]], [ymid, ymid], color=:white, linestyle=:dash)
        limits!(ax, -zoom, zoom, ymid - zoom, ymid + zoom)

        # Profile across the gouge at mid-height
        ax = Axis(fig[2, 2], title=@sprintf("profile at y = %.1f mm", ymid), xlabel="x [mm]", ylabel=L"$\tau_{II}$ [MPa]")
        gouge = d.phc[:, jmid] .== 2
        lines!(ax, x, τII[:, jmid], color=:black)
        scatter!(ax, x[.!gouge], τII[.!gouge, jmid], color=:black, markersize=6, label="rock")
        scatter!(ax, x[gouge], τII[gouge, jmid], color=:red, markersize=9, label="gouge ($(count(gouge)) cells)")
        xlims!(ax, -zoom, zoom)
        axislegend(ax, position=:rb)

        # Apparatus friction with current step
        ax = Axis(fig[2, 3], title="apparent friction (apparatus)", xlabel="Axial shortening [mm]", ylabel=L"$-\tau/\sigma$")
        s = shortening(d)
        lines!(ax, s, d.probes.app_fric_lab[1:d.it], color=:darkorange)
        vlines!(ax, s[end], color=:gray, linestyle=:dash)
        fig
    end
end

let
    case    = "ShearApparatus_Tet15_H02"
    step    = 80 # file number: step_0700.jld2
    folder  = joinpath(@__DIR__, case)

    # Single step: colour range from this step, away from the plates (y in [Ly/8, 7Ly/8])
    d       = load_step(joinpath(folder, @sprintf("step_%04d.jld2", step)))
    jin     = findall(y -> d.Ly/8 < y < 7d.Ly/8, d.yc)
    crange  = (0.0, maximum(centroid_field(d.τII_rock, d)[:, jin]) * d.sc.σ / 1e6)
    display(plot_stress_fields(d; crange))

    # # All steps + movie
    # outdir  = joinpath(folder, "tauII")
    # mkpath(outdir)
    # files   = sort(filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder)))
    #
    # # Fixed colour range for all frames, from the last step, away from the plates (y in [Ly/8, 7Ly/8])
    # dlast   = load_step(joinpath(folder, last(files)))
    # jin     = findall(y -> dlast.Ly/8 < y < 7dlast.Ly/8, dlast.yc)
    # crange  = (0.0, maximum(centroid_field(dlast.τII_rock, dlast)[:, jin]) * dlast.sc.σ / 1e6)
    #
    # for f in files
    #     d = load_step(joinpath(folder, f))
    #     save(joinpath(outdir, @sprintf("figure_%04d.png", d.it)), plot_stress_fields(d; crange))
    # end
    #
    # # Movie of the frames (same settings as make_movies.jl)
    # run(`ffmpeg -hide_banner -loglevel error -y -framerate 10 -pattern_type glob -i $(joinpath(outdir, "figure_*.png"))
    #      -vf "scale=1600:-2:flags=lanczos,format=yuv420p" -c:v libx264 -preset slow -crf 20 -tune animation
    #      -movflags +faststart $(joinpath(outdir, case * "_tauII.mp4"))`)
end
