# Whole apparatus at the last saved step of each <prefix>_Tet*_H* run, side by side: plastic strain rate (white where λ̇ = 0)
# with phase contours (black) and σ1 orientation (grey). The dashed square is the zoom window of plot_snapshots.jl.
# All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Printf

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

let
    root    = dirname(@__DIR__) # run folders are next to postprocessing/
    prefix  = "ShearApparatus_v9"
    θsel    = [15, 20, 30, 40, 50, 60, 70, 80] # gouge angles (one panel each)
    h       = 3                     # gouge thickness [mm] (folder _H..)
    λ̇range  = (-7, -3)              # colour range of log10(plastic strain rate [1/s]); lower values are clipped
    height  = 760                   # height of the panels [px]
    nskip   = 20                    # σ1 ticks every nskip cells
    window  = 20.0                  # zoom window of plot_snapshots.jl, drawn as a dashed square (nothing: not drawn)
    centre  = (0.0, 100.0)          # centre of the zoom window (x, y) [mm]

    folder(θ) = joinpath(root, @sprintf("%s_Tet%02d_H%02d", prefix, θ, h))
    function lastfile(θ)
        fs = filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder(θ)))
        joinpath(folder(θ), last(sort(fs)))
    end

    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size=(1800, 1100), fontsize=22)
        hm  = nothing
        aspect = 1.0
        for (col, θ) in enumerate(θsel)
            d   = load_step(lastfile(θ))
            sc  = d.sc
            x, y = d.xc .* (sc.L * 1e3), d.yc .* (sc.L * 1e3)                # [mm]
            aspect = (x[end] - x[1] + x[2] - x[1]) / (y[end] - y[1] + y[2] - y[1]) # width / height of the domain
            λ̇   = [l > 0 ? log10(max(l / sc.t, 10.0^λ̇range[1])) : NaN for l in d.λ̇c] # not plastic (λ̇ = 0): NaN, drawn white
            δ   = d.probes.t[d.it] * d.ε̇yy * d.Ly * sc.L * 1e3              # axial shortening [mm]
            δs  = @sprintf("%.2f", δ)
            ax  = Axis(fig[1, col]; aspect=DataAspect(), title=L"\mathrm{(%$('a' + col - 1))}\ \ \theta = %$(θ)^\circ", subtitle=L"\delta = %$(δs)\ \mathrm{mm}",
                       titlefont=:regular, xticks=-20:20:20, yticks=0:50:200, yticklabelsvisible=col == 1,
                       xlabel=L"x\ \mathrm{[mm]}", ylabel=col == 1 ? L"y\ \mathrm{[mm]}" : "")
            hm  = heatmap!(ax, x, y, λ̇, colormap=:bluesreds, colorrange=λ̇range, nan_color=:white)
            contour!(ax, x, y, Float64.(d.phc), levels=[1.5, 2.5], color=:black, linewidth=1)
            # σ1 orientation: segments centred on every nskip-th cell
            Is, Js = 1+nskip÷2:nskip:length(x), 1+nskip÷2:nskip:length(y)
            ℓ   = 0.7 * nskip * (x[2] - x[1])
            seg = [Point2f(x[i] + sgn * ℓ/2 * cos(d.θc[i, j]), y[j] + sgn * ℓ/2 * sin(d.θc[i, j])) for i in Is for j in Js for sgn in (-1, 1)]
            linesegments!(ax, seg, color=:gray55, linewidth=1)
            if window !== nothing
                xw, yw = centre[1] .+ (-window/2, window/2), centre[2] .+ (-window/2, window/2)
                lines!(ax, [xw[1], xw[2], xw[2], xw[1], xw[1]], [yw[1], yw[1], yw[2], yw[2], yw[1]], color=:black, linestyle=:dash, linewidth=1.5)
            end
            limits!(ax, x[1] - (x[2] - x[1])/2, x[end] + (x[2] - x[1])/2, y[1] - (y[2] - y[1])/2, y[end] + (y[2] - y[1])/2)
        end
        Colorbar(fig[2, :], hm, vertical=false, flipaxis=false, height=14, width=Relative(0.5),
                 label=L"\log_{10}\ \dot{\lambda}\ \mathrm{[s^{-1}]}\ \mathrm{(white:\ not\ plastic;\ black\ lines:\ phase\ boundaries;\ grey:\ }\sigma_1\mathrm{\ orientation)}")
        rowsize!(fig.layout, 1, Fixed(height))
        for col in eachindex(θsel)
            colsize!(fig.layout, col, Aspect(1, aspect))
        end
        colgap!(fig.layout, 14)
        rowgap!(fig.layout, 24)
        resize_to_layout!(fig)
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), "final_states.png"), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), "final_states.pdf"), fig)
    display(fig)
end
