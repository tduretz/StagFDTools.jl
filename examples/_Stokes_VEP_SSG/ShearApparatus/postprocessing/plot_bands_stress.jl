# Merged figure: σ1 orientation and shear band families (replaces the top row of bands_evolution.png and stress_angle.png).
# Left (θ = 30, 50, 60°, columns):
#   bottom: angle of σ1 from the vertical vs axial shortening δ (probes: mean (line) and min-max range (shaded) over the gouge cells
#           of the central box, as plot_stress_angle.jl), with the gouge plane (dashed) and the vertical (dotted)
#   top:    share of the plastic strain rate Σλ̇ of the analysed gouge carried by each band family (Fatbox time series,
#           fatbox/ts_analyse.jl): running mean over nav saves (thick) and every save (thin); grey dashed: no family + outside the bands
# Right (all runs):
#   bottom: σ1 angle at the end of the runs (mean over the last `tail` of the shortening) vs gouge angle θ, with its min-max range
#   top:    share of each family vs θ: average over the run (weighted by Δδ, line + markers)
# Usage: julia --project=<repo> plot_bands_stress.jl
using CairoMakie, JLD2, Printf, Statistics

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

# Mean of v over the last fraction `tail` of t, each entry weighted by its Δt (adaptive time stepping); NaN entries skipped
function tail_mean(v, t, tail)
    Δt = diff([0.0; t])
    w  = (t .>= (1 - tail) * t[end]) .& .!isnan.(v)
    return sum(v[w] .* Δt[w]) / sum(Δt[w])
end

let
    root     = joinpath(@__DIR__, "..", "_HR_hardening_2")
    prefix   = "_ShearApparatus_v9"
    data     = load(joinpath(@__DIR__, "fatbox", "timeseries", "bands_timeseries.jld2"), "data")
    θsel     = (30, 50, 60)                        # columns
    nav      = 5                                   # running mean over nav saves
    tail     = 0.2                                 # end-of-run σ1 angle: last 20% of the shortening
    θ_yield  = (60, 70)                            # gouge angles between which the sample starts to yield
    c_yield  = colorant"#d6404e"
    ink, muted = colorant"#222222", colorant"#777777"
    fams     = (:R1, :R2, :P, :Y)
    col      = Dict(:R1 => colorant"#1a8fa0", :R2 => colorant"#4d9a1f", :P => colorant"#c2427f", :Y => colorant"#3355cc", :rest => colorant"#8c8c8c")
    lab      = Dict(:R1 => L"R_1", :R2 => L"R_2", :P => L"P", :Y => L"Y", :rest => L"\mathrm{rest}")

    # ---- σ1 probes of every run (last step file)
    folders = filter(d -> isdir(d) && occursin(Regex("^$(prefix)_Tet\\d+_H03\$"), basename(d)), readdir(root, join=true))
    runs = map(folders) do folder
        θ     = parse(Int, match(r"Tet(\d+)_H", basename(folder))[1])
        steps = sort(filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder)))
        d     = load_step(joinpath(folder, last(steps)))
        p, it = d.probes, d.it
        fv(a) = rad2deg.(a[1:it]) .- 90                               # σ1 angle from the vertical [°]
        t     = p.t[1:it]
        (; θ, δ = t * d.ε̇yy * d.Ly * d.sc.L * 1e3, s1 = fv(p.meanθ), s1min = fv(p.minθ), s1max = fv(p.maxθ),
           s1_end = tail_mean(fv(p.meanθ), t, tail), s1min_end = tail_mean(fv(p.minθ), t, tail), s1max_end = tail_mean(fv(p.maxθ), t, tail))
    end
    sort!(runs, by=r -> r.θ)
    θall = [r.θ for r in runs]

    # ---- family shares of every run
    F(d, nm) = only(f.share for f in d.fams if f.name == nm)
    share(d, nm) = nm === :rest ? F(d, :none) + F(d, :outside) : F(d, nm)
    series(θ) = sort(filter(d -> d.θ == θ, data), by=d -> d.step)
    θfam = sort(unique(d.θ for d in data))
    avg  = Dict{Symbol, Vector{Float64}}(); fin = Dict{Symbol, Vector{Float64}}()
    for nm in (fams..., :rest)
        avg[nm] = map(θfam) do θ                                      # run average, weighted by Δδ (saves before plastic flow: skipped)
            D = series(θ); w = diff([0.0; [d.δ for d in D]]); ok = [!isempty(d.fams) for d in D]
            100 * sum(w[ok] .* [share(d, nm) for d in D[ok]]) / sum(w[ok])
        end
        fin[nm] = [100 * share(last(series(θ)), nm) for θ in θfam]
    end
    for (k, θ) in enumerate(θfam)
        @printf("θ = %2d°: run average %s | last save %s\n", θ, join([@sprintf("%s %4.1f%%", nm, avg[nm][k]) for nm in (fams..., :rest)], ", "),
                join([@sprintf("%s %4.1f%%", nm, fin[nm][k]) for nm in (fams..., :rest)], ", "))
    end

    function runmean(v)
        h = nav ÷ 2
        map(i -> mean(filter(!isnan, v[max(1, i - h):min(length(v), i + h)])), eachindex(v))
    end
    function endlabels!(ax, x, ends; gap, dx, lo=-Inf, hi=Inf)   # end labels pushed apart vertically, with a short leader
        sort!(ends, by=first); ys = [e[1] for e in ends]
        for _ in 1:200
            moved = false
            for i in 1:length(ys)-1
                dd = ys[i+1] - ys[i]
                dd < gap - 1e-9 && (ys[i] -= (gap - dd) / 2; ys[i+1] += (gap - dd) / 2; moved = true)
            end
            ys[1] < lo && (ys .+= lo - ys[1]; moved = true)
            ys[end] > hi && (ys .-= ys[end] - hi; moved = true)
            moved || break
        end
        for ((y0, l, cl), yl) in zip(ends, ys)
            lines!(ax, [x, x + 0.8dx], [y0, yl], color=(cl, 0.8), linewidth=1)
            text!(ax, x + dx, yl, text=l, color=cl, fontsize=28, align=(:left, :center))
        end
    end
    function yieldspan!(ax, xmax)
        for k in 1:20
            vspan!(ax, θ_yield[1] + (k - 1) / 20 * (θ_yield[2] - θ_yield[1]), θ_yield[1] + k / 20 * (θ_yield[2] - θ_yield[1]),
                   color=(c_yield, 0.18 * (k - 0.5) / 20))
        end
        vspan!(ax, θ_yield[2], xmax, color=(c_yield, 0.18))
    end

    fig = with_theme(theme_latexfonts()) do
        fig  = Figure(size=(2000, 1000), fontsize=31)
        axkw = (; xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08), topspinevisible=false, rightspinevisible=false)
        nc   = length(θsel)
        axt, axb = Axis[], Axis[]
        for (c, θ) in enumerate(θsel)
            # ---- bottom: σ1 angle vs shortening
            r  = runs[findfirst(==(θ), θall)]
            ax = Axis(fig[2, c]; title=L"\mathrm{(%$('a' + nc + 1 + c - 1))}", titlealign=:left, yticks=-30:30:90,
                      xlabel=L"\mathrm{Axial\ shortening}\ \delta\ \mathrm{[mm]}",
                      ylabel=c == 1 ? L"\sigma_1\ \mathrm{angle\ from\ vertical}\ [^\circ]" : "", axkw...)
            push!(axt, ax)
            hlines!(ax, 0, color=muted, linestyle=:dot, linewidth=1.5)
            hlines!(ax, θ, color=muted, linestyle=:dash, linewidth=1.5)
            band!(ax, r.δ, r.s1min, r.s1max, color=(ink, 0.12))
            lines!(ax, r.δ, r.s1, color=ink, linewidth=3)
            ylims!(ax, -40, 75)
            # ---- top: family shares vs shortening
            D  = filter(d -> !isempty(d.fams), series(θ)); δs = [d.δ for d in D]
            ax = Axis(fig[1, c]; title=L"\mathrm{(%$('a' + c - 1))}\ \theta = %$(θ)^\circ", titlealign=:left, xticklabelsvisible=false,
                      ylabel=c == 1 ? L"\mathrm{Share\ of}\ \Sigma\dot{\lambda}\ [\%]" : "", yticks=0:25:100, axkw...)
            push!(axb, ax)
            ends = Tuple{Float64, Any, Any}[]
            for nm in (fams..., :rest)
                v = [100 * share(d, nm) for d in D]
                lines!(ax, δs, v, color=(col[nm], 0.25), linewidth=1)
                m = runmean(v)
                lines!(ax, δs, m, color=col[nm], linewidth=3.5, linestyle=nm === :rest ? :dash : :solid)
                push!(ends, (m[end], lab[nm], col[nm]))
            end
            endlabels!(ax, δs[end], ends; gap=11, dx=0.12, lo=5, hi=96)
            ylims!(ax, 0, 100)
            linkxaxes!(axt[c], ax); xlims!(ax, 0, 3.85)
            c > 1 && (hideydecorations!(axt[c], grid=false, ticks=false); hideydecorations!(ax, grid=false, ticks=false))
        end
        axislegend(axt[end], [LineElement(color=muted, linestyle=:dash, linewidth=1.5), LineElement(color=muted, linestyle=:dot, linewidth=1.5),
                              [PolyElement(color=(ink, 0.12)), LineElement(color=ink, linewidth=3)]],
                   [L"\mathrm{gouge\ plane}", L"\mathrm{vertical}", L"\mathrm{mean\ and\ range}"],
                   position=:rb, framevisible=false, labelsize=24, patchsize=(32, 14), rowgap=0)

        # ---- right bottom: end-of-run σ1 angle vs gouge angle
        xr  = (θall[1] - 4, θall[end] + 4)
        ax  = Axis(fig[2, nc+1]; title=L"\mathrm{(%$('a' + 2nc + 1))}\ \sigma_1\ \mathrm{vs.\ gouge\ angle}", titlealign=:left,
                   xlabel=L"\mathrm{Gouge\ angle}\ \theta\ [^\circ]",
                   ylabel=L"\sigma_1\ \mathrm{angle\ from\ vertical}\ [^\circ]", xticks=10:10:90, axkw...)
        yieldspan!(ax, xr[2])
        hlines!(ax, 0, color=muted, linestyle=:dot, linewidth=1.5)
        lines!(ax, [xr...], [xr...], color=muted, linestyle=:dash, linewidth=1.5)
        band!(ax, θall, [r.s1min_end for r in runs], [r.s1max_end for r in runs], color=(ink, 0.12))
        scatterlines!(ax, θall, [r.s1_end for r in runs], color=ink, linewidth=2.5, markersize=12, markercolor=:white,
                      strokecolor=ink, strokewidth=1.5)
        for θ in θsel   # columns (e)-(g)
            k = findfirst(==(θ), θall)
            scatter!(ax, [θ], [runs[k].s1_end], color=ink, markersize=12)
        end
        xlims!(ax, xr...); ylims!(ax, -48, 95)
        text!(ax, 0.03, 0.97, text=L"\mathrm{mean\ over\ last\ %$(round(Int, 100tail))\%\ of}\ \delta", space=:relative,
              align=(:left, :top), fontsize=24, color=muted)
        # ---- right top: family shares vs gouge angle
        ax2 = Axis(fig[1, nc+1]; title=L"\mathrm{(%$('a' + nc))}\ \mathrm{Band\ families\ vs.\ gouge\ angle}", titlealign=:left,
                   ylabel=L"\mathrm{Share\ of}\ \Sigma\dot{\lambda}\ [\%]", xticklabelsvisible=false,
                   xticks=10:10:90, yticks=0:25:100, axkw...)
        yieldspan!(ax2, xr[2])
        for nm in (fams..., :rest)
            ls = nm === :rest ? :dash : :solid
            lines!(ax2, θfam, avg[nm], color=col[nm], linewidth=3, linestyle=ls)
            scatter!(ax2, θfam, avg[nm], color=col[nm], markersize=11)
        end
        linkxaxes!(ax, ax2); xlims!(ax2, xr...); ylims!(ax2, 0, 100)
        axislegend(ax2, [[LineElement(color=ink, linewidth=3), MarkerElement(marker=:circle, color=ink, markersize=11)],
                         PolyElement(color=(c_yield, 0.18))],
                   [L"\mathrm{run\ average}", L"\mathrm{sample\ yields}"],
                   position=:lt, framevisible=false, labelsize=24, patchsize=(32, 14), rowgap=0)

        Legend(fig[3, 1:nc+1], [LineElement(color=col[nm], linewidth=3.5, linestyle=nm === :rest ? :dash : :solid) for nm in (fams..., :rest)],
               [L"R_1", L"R_2", L"P", L"Y", L"\mathrm{rest:\ no\ family\ +\ outside\ the\ bands}"],
               orientation=:horizontal, framevisible=false, labelsize=30, colgap=30, patchsize=(34, 18))
        colgap!(fig.layout, 22); colgap!(fig.layout, nc, 70); rowgap!(fig.layout, 1, 18)
        colsize!(fig.layout, nc + 1, Relative(0.3))
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), "bands_stress.png"), fig, px_per_unit=2)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), "bands_stress.pdf"), fig)
    println(joinpath(@__DIR__, "png", "bands_stress.png"))
end
