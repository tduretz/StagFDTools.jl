# Apparent friction vs axial shortening for all <prefix>_Tet*_H* runs, read from the last step_XXXX.jld2 of each run.
# (a) apparatus friction vs shortening, colour = gouge angle θ (one-hue ramp, light = small θ), line style = gouge thickness;
#     numbered markers 1-3: stages shown in plot_snapshots.jl
# (b) friction at the end of the runs (mean over the last 20% of the axial shortening) vs θ
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
    root   = joinpath(dirname(@__DIR__), "_HR_hardening_2") # run folder
    prefix = "_ShearApparatus_v9"
    tail   = 0.2   # fraction of the axial shortening used for the end-of-run friction

    folders = filter(d -> isdir(d) && occursin(Regex("^$(prefix)_Tet\\d+_H\\d+\$"), basename(d)), readdir(root, join=true))
    runs    = map(folders) do folder
        m     = match(r"Tet(\d+)_H(\d+)$", basename(folder))
        steps = filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder))
        d     = load_step(joinpath(folder, last(sort(steps))))
        it    = d.it
        p     = d.probes
        t     = p.t[1:it]
        (; θ=parse(Int, m[1]), h=parse(Int, m[2]), it,
           δ       = t * d.ε̇yy * d.Ly * d.sc.L * 1e3,            # axial shortening [mm]
           μ_app   = p.app_fric_lab[1:it],                       # apparatus (lab) apparent friction
           μ_end   = tail_mean(p.app_fric_lab[1:it], t, tail),
           μ_true  = tail_mean(p.true_fric[1:it], t, 1.0))       # gouge material friction (plastic points), whole run
    end
    sort!(runs, by=r -> (r.h, r.θ))
    θs, hs = sort(unique(r.θ for r in runs)), sort(unique(r.h for r in runs))
    println("Runs: ", [(r.θ, r.h, r.it) for r in runs])

    # One-hue sequential ramp (light → dark with θ), skipping the lightest steps so every curve reads on white
    cmap   = cgrad([colorant"#9ecae1", colorant"#4292c6", colorant"#08306b"], length(θs), categorical=true)
    colθ(θ) = cmap[findfirst(==(θ), θs)]
    styles = [:solid, :dash, :dot, :dashdot]
    ink, muted = colorant"#222222", colorant"#777777"
    θ_yield = (60, 70)              # gouge angles between which the rock starts to yield (panel b background)
    # Lab end-of-run friction (apparatus mean over the last 20% of shortening) vs gouge angle, shown in panel (b).
    # Values read from the lab figure (approximate): replace with the exact numbers. Set to nothing to hide.
    lab     = (θ = [30, 40, 50, 60], μ = [0.82, 0.77, 0.68, 0.51])
    c_lab   = colorant"#e66101"
    c_yield = colorant"#d6404e"
    stage_θ, stage_h, stage_steps = (30, 50, 60), 3, (20, 100, 500) # snapshot stages of plot_snapshots.jl (θsel, h, steps)

    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size=(1150, 500), fontsize=22)
        axkw = (xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08), topspinevisible=false, rightspinevisible=false)

        # (a) friction vs shortening
        ax1 = Axis(fig[1, 1]; xlabel=L"\mathrm{Axial\ shortening}\ \delta\ \mathrm{[mm]}", ylabel=L"\mu_\mathrm{app} = -\tau/\sigma_n",
                   title=L"\mathrm{(a)\ Apparent\ friction\ vs.\ shortening}", titlealign=:left, yticks=0:0.1:0.9, axkw...)
        for r in runs
            lines!(ax1, r.δ, r.μ_app, color=colθ(r.θ), linewidth=2, linestyle=styles[findfirst(==(r.h), hs)])
        end
        # Stages shown in plot_snapshots.jl (keep in sync): numbered markers on the curves of the snapshot angles
        for r in runs
            (r.θ in stage_θ && r.h == stage_h) || continue
            for (k, s) in enumerate(stage_steps)
                s <= r.it || continue
                scatter!(ax1, r.δ[s], r.μ_app[s], color=:white, strokecolor=colθ(r.θ), strokewidth=2, markersize=24)
                text!(ax1, r.δ[s], r.μ_app[s], text=L"\mathbf{%$(k)}", color=ink, fontsize=15, align=(:center, :center))
            end
        end
        # Direct labels at the curve ends, pushed apart vertically where curves end close to each other
        ylab = [r.μ_app[end] for r in runs]
        dy   = 0.03
        order = sortperm(ylab)
        for (n, k) in enumerate(order), l in order[1:n-1] # bottom to top: stay dy above every lower label ending nearby
            abs(runs[l].δ[end] - runs[k].δ[end]) < 0.3 && (ylab[k] = max(ylab[k], ylab[l] + dy))
        end
        for (r, y) in zip(runs, ylab)
            text!(ax1, r.δ[end], y, text=L"%$(r.θ)^\circ", color=ink, fontsize=17, align=(:left, :center), offset=(16, 0))
        end
        xlims!(ax1, 0, 1.11 * maximum(r.δ[end] for r in runs))
        ylims!(ax1, 0, nothing)
        Colorbar(fig[1, 2], colormap=cmap, limits=(0.5, length(θs) + 0.5), ticks=(1:length(θs), [L"%$(θ)^\circ" for θ in θs]),
                 label=L"\mathrm{Gouge\ angle}\ \theta", width=12)
        if length(hs) > 1 # thickness legend only when several thicknesses are present
            axislegend(ax1, [LineElement(color=ink, linestyle=styles[k]) for k in eachindex(hs)], [L"h = %$(h)\ \mathrm{mm}" for h in hs],
                       position=:rb, framevisible=false)
        end

        # (b) end-of-run friction vs θ
        ax2 = Axis(fig[1, 3]; xlabel=L"\mathrm{Gouge\ angle}\ \theta\ [^\circ]", ylabel=L"\mu_\mathrm{app}\ \mathrm{(mean\ over\ last\ %$(round(Int, 100tail))\%\ of\ shortening)}",
                   title=L"\mathrm{(b)\ Apparent\ friction\ vs.\ gouge\ angle}", titlealign=:left, xticks=10:10:90, axkw...)
        # Rock yields for large gouge angles: shaded background, fading in between θ_yield[1] and θ_yield[2]
        xr = (θs[1] - 4, θs[end] + 4)
        for k in 1:20
            vspan!(ax2, θ_yield[1] + (k - 1) / 20 * (θ_yield[2] - θ_yield[1]), θ_yield[1] + k / 20 * (θ_yield[2] - θ_yield[1]), color=(c_yield, 0.18 * (k - 0.5) / 20))
        end
        vspan!(ax2, θ_yield[2], xr[2], color=(c_yield, 0.18), label=L"\mathrm{sample\ yields}")
        xlims!(ax2, xr...)
        if lab !== nothing # drawn first: behind the model results
            scatter!(ax2, lab.θ, lab.μ, color=c_lab, marker=:diamond, markersize=24, strokecolor=ink, strokewidth=1.5,
                     label=L"\mu_\mathrm{app}\ \mathrm{experiments}")
        end
        for (k, h) in enumerate(hs)
            rh = filter(r -> r.h == h, runs)
            θh = [r.θ for r in rh]
            lines!(ax2, θh, [r.μ_end for r in rh], color=ink, linewidth=2, linestyle=styles[k])
            scatter!(ax2, θh, [r.μ_end for r in rh], color=[colθ(θ) for θ in θh], markersize=12, strokecolor=ink, strokewidth=1,
                     label=L"\mu_\mathrm{app}\ \mathrm{models}")
        end
        μ_true = mean(filter(!isnan, [r.μ_true for r in runs]))
        hlines!(ax2, μ_true, color=muted, linestyle=:dash, linewidth=2.5, label=L"\mu_\mathrm{true}\ \mathrm{gouge}\ (\tan\phi)")
        axislegend(ax2, position=:lb, framevisible=false, unique=true, merge=true)

        colgap!(fig.layout, 1, 8)
        colsize!(fig.layout, 3, Relative(0.42))
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), "apparent_friction.png"), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), "apparent_friction.pdf"), fig)
    display(fig)
end
