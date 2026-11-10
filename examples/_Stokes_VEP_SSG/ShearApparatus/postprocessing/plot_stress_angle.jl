# Orientation of σ1 in the gouge for all <prefix>_Tet*_H* runs, read from the last step_XXXX.jld2 of each run.
# The probes (minθ, meanθ, maxθ) are taken over the gouge cells of the central box x ∈ [-1, 1] cm, y ∈ [7.5, 12.5] cm, away from the boundaries;
# θ is the angle of σ1 from the x axis (τ.θ). The gouge plane is at 90° + θgouge from the x axis, so the angle between σ1 and the gouge
# plane is β = 90° + θgouge - θ (the loading axis, y, is at β = θgouge).
# (a) β vs axial shortening, one small panel per run: mean (line) and min-max range (shaded), colour = gouge angle θgouge (same ramp as plot_friction.jl)
# (b) β and the σ1 angle from the loading axis at the end of the runs (mean over the last 20% of the axial shortening) vs θgouge
# All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Statistics

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

# Mean of a probe v over the last fraction `tail` of the run time t (= of the axial shortening), each step weighted by its Δt
# (adaptive time stepping: a plain mean over steps would favour the periods with small Δt); NaN entries are skipped
function tail_mean(v, t, tail)
    Δt = diff([0.0; t])
    w  = (t .>= (1 - tail) * t[end]) .& .!isnan.(v)
    return sum(v[w] .* Δt[w]) / sum(Δt[w])
end

let
    root   = dirname(@__DIR__) # run folders are next to postprocessing/
    prefix = "_ShearApparatus_v9"
    tail   = 0.2   # fraction of the axial shortening used for the end-of-run values
    θskip  = [80]  # gouge angles not shown (keeps panel (a) on a 2 x 4 grid)

    folders = filter(d -> isdir(d) && occursin(Regex("^$(prefix)_Tet\\d+_H\\d+\$"), basename(d)), readdir(root, join=true))
    runs    = map(folders) do folder
        m     = match(r"Tet(\d+)_H(\d+)$", basename(folder))
        steps = filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder))
        d     = load_step(joinpath(folder, last(sort(steps))))
        it    = d.it
        p     = d.probes
        θg    = parse(Int, m[1])
        β(θ)  = 90 + θg .- rad2deg.(θ[1:it])                       # angle between σ1 and the gouge plane [°]
        t     = p.t[1:it]
        (; θ=θg, h=parse(Int, m[2]), it,
           δ     = t * d.ε̇yy * d.Ly * d.sc.L * 1e3,                # axial shortening [mm]
           β     = β(p.meanθ), βmin = β(p.maxθ), βmax = β(p.minθ),
           β_end = tail_mean(β(p.meanθ), t, tail), βmin_end = tail_mean(β(p.maxθ), t, tail), βmax_end = tail_mean(β(p.minθ), t, tail))
    end
    sort!(runs, by=r -> (r.h, r.θ))
    θ_all  = sort(unique(r.θ for r in runs))      # all runs: fixes the colour of each angle as in plot_friction.jl
    filter!(r -> r.θ ∉ θskip, runs)
    θs, hs = sort(unique(r.θ for r in runs)), sort(unique(r.h for r in runs))
    println("Runs: ", [(r.θ, r.h, r.it) for r in runs])

    # Same one-hue sequential ramp as plot_friction.jl
    cmap   = cgrad([colorant"#9ecae1", colorant"#4292c6", colorant"#08306b"], length(θ_all), categorical=true)
    colθ(θ) = cmap[findfirst(==(θ), θ_all)]
    styles = [:solid, :dash, :dot, :dashdot]
    ink, muted = colorant"#222222", colorant"#777777"
    θ_yield = (60, 70)              # gouge angles between which the rock starts to yield (panel b background)
    c_yield = colorant"#d6404e"

    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size=(1300, 580), fontsize=22)
        axkw = (xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08), topspinevisible=false, rightspinevisible=false)

        # (a) angle between σ1 and the gouge plane vs shortening: one small panel per run (bands overlap too much on one axis)
        ga   = fig[1, 1] = GridLayout()
        ncol = 4
        axs  = map(enumerate(runs)) do (n, r)
            i, j = fldmod1(n, ncol)
            ax = Axis(ga[i, j]; axkw..., xticks=0:1:3, yticks=15:15:90,
                      xticklabelsvisible=i == cld(length(runs), ncol), yticklabelsvisible=j == 1)
            hlines!(ax, 45, color=muted, linestyle=:dash, linewidth=1.2)
            hlines!(ax, r.θ, color=muted, linestyle=:dot, linewidth=1.2)  # β if σ1 stayed along the loading axis
            band!(ax, r.δ, r.βmin, r.βmax, color=(colθ(r.θ), 0.25))
            lines!(ax, r.δ, r.β, color=colθ(r.θ), linewidth=2, linestyle=styles[findfirst(==(r.h), hs)])
            text!(ax, 0.04, 0.96, text=length(hs) > 1 ? L"	heta = %$(r.θ)^\circ,\ h = %$(r.h)\ \mathrm{mm}" : L"\theta = %$(r.θ)^\circ",
                  space=:relative, align=(:left, :top), color=ink, fontsize=18)
            ax
        end
        linkaxes!(axs...)
        xlims!(axs[1], 0, maximum(r.δ[end] for r in runs))
        ylims!(axs[1], 10, 95)
        Label(ga[0, :], L"\mathrm{(a)\ Angle\ }\beta\ \mathrm{between\ }\sigma_1\ \mathrm{and\ the\ gouge\ plane\ (mean\ and\ range)}", halign=:left, fontsize=22)
        Label(ga[end+1, :], L"\mathrm{Axial\ shortening}\ \delta\ \mathrm{[mm]}", fontsize=22)
        Label(ga[1:end-1, 0], L"\beta\ [^\circ]", rotation=π/2, fontsize=22)
        axislegend(axs[end], [LineElement(color=muted, linestyle=:dash), LineElement(color=muted, linestyle=:dot)],
                   [L"45^\circ", L"\sigma_1\parallel\mathrm{loading\ axis}"], position=:rb, framevisible=false, labelsize=15, patchsize=(18, 8), rowgap=0)
        colgap!(ga, 6); rowgap!(ga, 6)

        # (b) end-of-run angles vs θgouge
        ax2 = Axis(fig[1, 2]; xlabel=L"\mathrm{Gouge\ angle}\ \theta\ [^\circ]", ylabel=L"\mathrm{Angle\ (last\ %$(round(Int, 100tail))\%\ of\ shortening)}\ [^\circ]",
                   title=L"\mathrm{(b)\ Angles\ at\ the\ end\ of\ the\ runs}", titlealign=:left, xticks=10:10:90, axkw...)
        # Rock yields for large gouge angles: shaded background, fading in between θ_yield[1] and θ_yield[2]
        xr = (θs[1] - 4, θs[end] + 4)
        for k in 1:20
            vspan!(ax2, θ_yield[1] + (k - 1) / 20 * (θ_yield[2] - θ_yield[1]), θ_yield[1] + k / 20 * (θ_yield[2] - θ_yield[1]), color=(c_yield, 0.18 * (k - 0.5) / 20))
        end
        vspan!(ax2, θ_yield[2], xr[2], color=(c_yield, 0.18), label=L"\mathrm{rock\ yields}")
        xlims!(ax2, xr...)
        hlines!(ax2, 45, color=muted, linestyle=:dash, linewidth=1.5, label=L"45^\circ")
        for (k, h) in enumerate(hs)
            rh = filter(r -> r.h == h, runs)
            θh = [r.θ for r in rh]
            # σ1 to gouge plane, with the min-max range
            band!(ax2, θh, [r.βmin_end for r in rh], [r.βmax_end for r in rh], color=(ink, 0.08))
            lines!(ax2, θh, [r.β_end for r in rh], color=ink, linewidth=2, linestyle=styles[k])
            scatter!(ax2, θh, [r.β_end for r in rh], color=[colθ(θ) for θ in θh], markersize=12, strokecolor=ink, strokewidth=1,
                     label=L"\sigma_1\ \mathrm{to\ gouge\ plane}\ (\beta)")
            # σ1 to the loading axis: θgouge - β
            lines!(ax2, θh, [r.θ - r.β_end for r in rh], color=muted, linewidth=2, linestyle=styles[k])
            scatter!(ax2, θh, [r.θ - r.β_end for r in rh], color=:white, marker=:rect, markersize=10, strokecolor=muted, strokewidth=1.5,
                     label=L"\sigma_1\ \mathrm{to\ loading\ axis}")
        end
        axislegend(ax2, position=:lt, framevisible=false, unique=true, merge=true)

        colgap!(fig.layout, 1, 24)
        colsize!(fig.layout, 2, Relative(0.34))
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), "stress_angle.png"), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), "stress_angle.pdf"), fig)
    display(fig)
end
