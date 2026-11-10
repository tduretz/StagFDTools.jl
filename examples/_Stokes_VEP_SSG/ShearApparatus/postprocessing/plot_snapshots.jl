# Evolution of selected <prefix>_Tet*_H* runs: one row per gouge angle, with snapshots of the plastic strain rate
# (white where λ̇ = 0, black phase contours, grey σ1 orientation, zoom on the centre of the gouge) at a few steps, and the apparatus friction
# vs axial shortening with markers at the snapshot steps. Same colours as plot_friction.jl.
# All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Printf

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

let
    root    = dirname(@__DIR__) # run folders are next to postprocessing/
    prefix  = "_ShearApparatus_v9"
    θsel    = [30, 50, 70]          # gouge angles (one row each)
    h       = 3                     # gouge thickness [mm] (folder _H..)
    steps   = [20, 100, 500]        # snapshot steps (must have a step_XXXX.jld2)
    window  = 20.0                  # side of the square visualisation window [mm]
    centre  = (0.0, 100.0)          # centre of the window (x, y) [mm]
    λ̇range  = (-7, -3)              # colour range of log10(plastic strain rate [1/s]); lower values are clipped
    nskip   = 5                     # σ1 ticks every nskip cells
    θ_all   = [15, 20, 30, 40, 50, 60, 70, 80, 85] # all gouge angles: fixes the colour of each angle as in plot_friction.jl

    cmap    = cgrad([colorant"#9ecae1", colorant"#4292c6", colorant"#08306b"], length(θ_all), categorical=true)
    colθ(θ) = cmap[findfirst(==(θ), θ_all)]
    ink     = colorant"#222222"
    xw      = centre[1] .+ (-window/2, window/2)
    yw      = centre[2] .+ (-window/2, window/2)

    folder(θ)  = joinpath(root, @sprintf("%s_Tet%02d_H%02d", prefix, θ, h))
    stepfile(θ, s) = joinpath(folder(θ), @sprintf("step_%04d.jld2", s))
    function lastfile(θ)
        fs = filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder(θ)))
        joinpath(folder(θ), last(sort(fs)))
    end
    shortening(d, it) = d.probes.t[1:it] * d.ε̇yy * d.Ly * d.sc.L * 1e3 # [mm]

    fig = with_theme(theme_latexfonts()) do
        fig  = Figure(size=(1450, 380 * length(θsel) + 120), fontsize=22)
        axkw = (xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08))
        ns   = length(steps)
        hm   = nothing
        axf  = Axis[]
        for (row, θ) in enumerate(θsel)
            dl = load_step(lastfile(θ))
            δ  = shortening(dl, dl.it)

            # Snapshots
            for (col, s) in enumerate(steps)
                d   = load_step(stepfile(θ, s))
                sc  = d.sc
                x, y = d.xc .* (sc.L * 1e3), d.yc .* (sc.L * 1e3)                 # [mm]
                I   = findall(xi -> xw[1] - 1 <= xi <= xw[2] + 1, x)
                J   = findall(yi -> yw[1] - 1 <= yi <= yw[2] + 1, y)
                λ̇   = [l > 0 ? log10(max(l / sc.t, 10.0^λ̇range[1])) : NaN for l in d.λ̇c[I, J]] # not plastic (λ̇ = 0): NaN, drawn white
                δs  = @sprintf("%.2f", δ[s])
                ax  = Axis(fig[row, col]; aspect=DataAspect(), axkw...,
                           title=row == 1 ? L"\mathbf{%$(col)}\quad \mathrm{step}\ %$(s),\ \delta = %$(δs)\ \mathrm{mm}" : "", titlefont=:regular, xticks=-8:4:8, yticks=ceil(Int, yw[1]):4:yw[2],
                           xticklabelsvisible=row == length(θsel), yticklabelsvisible=col == 1,
                           xlabel=row == length(θsel) ? L"x\ \mathrm{[mm]}" : "", ylabel=col == 1 ? L"y\ \mathrm{[mm]}" : "")
                hm  = heatmap!(ax, x[I], y[J], λ̇, colormap=:bluesreds, colorrange=λ̇range, nan_color=:white)
                contour!(ax, x[I], y[J], Float64.(d.phc[I, J]), levels=[1.5, 2.5], color=:black, linewidth=1.5)
                # σ1 orientation: segments centred on every nskip-th cell
                Is, Js = I[1:nskip:end], J[1:nskip:end]
                ℓ   = 0.4 * nskip * (x[2] - x[1])
                seg = [Point2f(x[i] + sgn * ℓ/2 * cos(d.θc[i, j]), y[j] + sgn * ℓ/2 * sin(d.θc[i, j])) for i in Is for j in Js for sgn in (-1, 1)]
                linesegments!(ax, seg, color=:gray55, linewidth=1)
                limits!(ax, xw..., yw...)
            end

            # Friction vs shortening, with the snapshot steps
            ax = Axis(fig[row, ns + 1]; axkw..., topspinevisible=false, rightspinevisible=false,
                      xlabel=row == length(θsel) ? L"\mathrm{Axial\ shortening}\ \delta\ \mathrm{[mm]}" : "", ylabel=L"\mu_\mathrm{app}",
                      xticklabelsvisible=row == length(θsel))
            push!(axf, ax)
            lines!(ax, δ, dl.probes.app_fric_lab[1:dl.it], color=colθ(θ), linewidth=2)
            for (k, s) in enumerate(steps)
                scatter!(ax, δ[s], dl.probes.app_fric_lab[s], color=:white, strokecolor=ink, strokewidth=1.5, markersize=34)
                text!(ax, δ[s], dl.probes.app_fric_lab[s], text=L"\mathbf{%$(k)}", color=ink, fontsize=20, align=(:center, :center))
            end
            Label(fig[row, 0], L"\theta = %$(θ)^\circ", rotation=π/2, fontsize=26, tellheight=false)
        end
        # Panel letters, row by row
        for row in eachindex(θsel), col in 1:ns+1
            Label(fig[row, col, TopLeft()], L"\mathrm{(%$('a' + (row - 1) * (ns + 1) + col - 1))}", fontsize=24, padding=(0, 6, 4, 0), halign=:right)
        end
        linkaxes!(axf...)
        ylims!(axf[1], 0, 0.85)
        xlims!(axf[1], 0, nothing)
        Colorbar(fig[length(θsel) + 1, 1:ns], hm, vertical=false, flipaxis=false, height=14, width=Relative(0.7),
                 label=L"\log_{10}\ \dot{\lambda}\ \mathrm{[s^{-1}]}\ \mathrm{(white:\ not\ plastic;\ black\ lines:\ gouge\ boundaries;\ grey:\ }\sigma_1\mathrm{\ orientation)}")
        colsize!(fig.layout, ns + 1, Relative(0.27))
        colgap!(fig.layout, 8)
        colgap!(fig.layout, ns, 30)
        rowgap!(fig.layout, 8)
        resize_to_layout!(fig)
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), "snapshots.png"), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), "snapshots.pdf"), fig)
    display(fig)
end
