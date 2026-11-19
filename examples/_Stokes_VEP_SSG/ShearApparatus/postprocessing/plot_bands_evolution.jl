# Evolution of the shear band families with shortening in the HR runs at θ = 30, 50, 60° (columns), from the Fatbox time series
# (fatbox/ts_export.jl → fatbox_extract.py 0.03 1.0 timeseries → fatbox/ts_analyse.jl: fatbox/timeseries/bands_timeseries.jld2).
# Segments: Fatbox skeleton segments (≥ 1 mm), classified by the nearest expected plane (R1, R2 at the Arthur angle, P, Y; within 20°).
# Top:    orientation of each family vs axial shortening δ: length-weighted mean angle of its segments (solid; shaded: ± length-weighted
#         rms spread), shown where the family carries ≥ minshare of the plastic strain rate; expected plane (dashed, same colour; it
#         moves as the mean σ1 orientation rotates); σ1 dotted grey. Running mean over nav saves (weighted by the family share).
# Bottom: share of the plastic strain rate Σλ̇ of the analysed gouge carried by each family: running mean over nav saves (thick) and
#         every save (thin); grey dashed: not in a family (segments of no family + outside the bands).
#         (λ̇ weighting: τII is only stored at the checkpoints; at step 500 the shares differ from the λ̇ τII shares by ≤ 2.6%.)
# Usage: julia --project=<repo> plot_bands_evolution.jl
using CairoMakie, JLD2, Printf

let
    data     = load(joinpath(@__DIR__, "fatbox", "timeseries", "bands_timeseries.jld2"), "data")
    θs       = (30, 50, 60)
    nav      = 5                                   # running mean over nav saves (Δδ ≈ 0.33 mm)
    minshare = 0.03                                # angles shown where the (averaged) family share is at least this
    ink      = colorant"#222222"
    fams     = (:R1, :R2, :P, :Y)
    col      = Dict(:R1 => colorant"#1a8fa0", :R2 => colorant"#4d9a1f", :P => colorant"#c2427f", :Y => colorant"#3355cc", :rest => colorant"#8c8c8c")
    lab      = Dict(:R1 => L"R_1", :R2 => L"R_2", :P => L"P", :Y => L"Y")
    # centred running mean over nav points, optionally weighted (NaN values ignored)
    function runmean(v, w=ones(length(v)))
        h = nav ÷ 2
        map(eachindex(v)) do i
            J = max(1, i - h):min(length(v), i + h)
            ok = [j for j in J if !isnan(v[j]) && w[j] > 0]
            isempty(ok) ? NaN : sum(v[j] * w[j] for j in ok) / sum(w[j] for j in ok)
        end
    end
    # end labels of the lines, pushed apart vertically (≥ gap, order kept), with a short leader
    function endlabels!(ax, x, ends; gap, dx, lo=-Inf, hi=Inf)
        sort!(ends, by=first); ys = [e[1] for e in ends]
        for _ in 1:200
            moved = false
            for i in 1:length(ys)-1
                dd = ys[i+1] - ys[i]
                dd < gap - 1e-9 && (ys[i] -= (gap - dd) / 2; ys[i+1] += (gap - dd) / 2; moved = true)
            end
            ys[1] < lo && (ys .+= lo - ys[1]; moved = true)                 # keep the labels inside the axis
            ys[end] > hi && (ys .-= ys[end] - hi; moved = true)
            moved || break
        end
        for ((y0, l, cl), yl) in zip(ends, ys)
            lines!(ax, [x, x + 0.8dx], [y0, yl], color=(cl, 0.8), linewidth=1)
            text!(ax, x + dx, yl, text=l, color=cl, fontsize=22, align=(:left, :center))
        end
    end

    fig = with_theme(theme_latexfonts()) do
        fig  = Figure(size=(1500, 1050), fontsize=24)
        axkw = (; xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08), topspinevisible=false, rightspinevisible=false)
        axt, axb = Axis[], Axis[]
        for (c, θ) in enumerate(θs)
            Dm = sort(filter(d -> d.θ == θ && !isempty(d.fams), data), by=d -> d.step)
            δs = [d.δ for d in Dm]
            F(d, nm, key) = only(getproperty(f, key) for f in d.fams if f.name == nm)
            # ---- top: family orientations
            ax = Axis(fig[1, c]; title=L"\mathrm{(%$('a' + c - 1))}\ \theta = %$(θ)^\circ", titlealign=:left, yticks=-60:30:60,
                      ylabel=c == 1 ? L"\mathrm{Band\ angle\ from\ vertical}\ [^\circ]" : "", xticklabelsvisible=false, axkw...)
            push!(axt, ax)
            ends = Tuple{Float64, Any, Any}[]
            lines!(ax, δs, [d.σ1 for d in Dm], color=:gray50, linewidth=1.5, linestyle=:dot)
            push!(ends, (Dm[end].σ1, L"\sigma_1", :gray40))
            for nm in fams
                ex = [F(d, nm, :expected) for d in Dm]
                lines!(ax, δs, ex, color=(col[nm], 0.9), linewidth=1.5, linestyle=:dash)
                push!(ends, (ex[end], lab[nm], col[nm]))
                sh = [F(d, nm, :share) for d in Dm]
                a  = runmean([F(d, nm, :angle) for d in Dm], sh)
                sp = runmean([F(d, nm, :spread) for d in Dm], sh)
                show = runmean(sh) .>= minshare
                a[.!show] .= NaN; sp[.!show] .= NaN
                ok = .!isnan.(a)
                any(ok) || continue
                band!(ax, δs, a .- sp, a .+ sp, color=(col[nm], 0.18))
                lines!(ax, δs, a, color=col[nm], linewidth=3.5)
            end
            endlabels!(ax, δs[end], ends; gap=10, dx=0.12, lo=-62, hi=84)
            ylims!(ax, -68, 88)
            # ---- bottom: shares
            ax = Axis(fig[2, c]; xlabel=L"\mathrm{Axial\ shortening}\ \delta\ \mathrm{[mm]}",
                      ylabel=c == 1 ? L"\mathrm{Share\ of}\ \Sigma\dot{\lambda}\ [\%]" : "", yticks=0:25:100, axkw...)
            push!(axb, ax)
            ends = Tuple{Float64, Any, Any}[]
            for nm in (fams..., :rest)
                v = nm === :rest ? [100 * (F(d, :none, :share) + F(d, :outside, :share)) for d in Dm] : [100 * F(d, nm, :share) for d in Dm]
                lines!(ax, δs, v, color=(col[nm], 0.25), linewidth=1)
                m = runmean(v)
                lines!(ax, δs, m, color=col[nm], linewidth=3.5, linestyle=nm === :rest ? :dash : :solid)
                push!(ends, (m[end], nm === :rest ? L"\mathrm{rest}" : lab[nm], col[nm]))
            end
            endlabels!(ax, δs[end], ends; gap=9, dx=0.12, lo=4, hi=97)
            ylims!(ax, 0, 100)
            linkxaxes!(axt[c], ax); xlims!(ax, 0, 3.75)
            c > 1 && (hideydecorations!(axt[c], grid=false, ticks=false); hideydecorations!(ax, grid=false, ticks=false))
        end
        Legend(fig[3, 1:length(θs)], [LineElement(color=ink, linewidth=3.5), [PolyElement(color=(ink, 0.18)), LineElement(color=ink, linewidth=3.5)],
                             LineElement(color=ink, linewidth=1.5, linestyle=:dash), LineElement(color=:gray50, linewidth=1.5, linestyle=:dot),
                             LineElement(color=(ink, 0.25), linewidth=1), LineElement(color=col[:rest], linewidth=3.5, linestyle=:dash)],
               [L"\mathrm{measured\ (running\ mean,\ %$(nav)\ saves)}", L"\pm\ \mathrm{spread\ of\ the\ segment\ angles}", L"\mathrm{expected\ plane}",
                L"\sigma_1", L"\mathrm{every\ save}", L"\mathrm{rest:\ no\ family\ +\ outside\ the\ bands}"],
               orientation=:horizontal, nbanks=2, framevisible=false, labelsize=22, colgap=26, patchsize=(34, 18))
        rowsize!(fig.layout, 2, Relative(0.36)); rowgap!(fig.layout, 1, 18); colgap!(fig.layout, 22)
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), "bands_evolution.png"), fig, px_per_unit=2)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), "bands_evolution.pdf"), fig)
    println(joinpath(@__DIR__, "png", "bands_evolution.png"))
end
