# Sensitivity of the reference model (v10, hardening P3, rate-and-state off, LR 70×250) to gouge dilatancy and hardening
#   batch 1: dilatancy ψ = 0 … ψ = ϕ = 34° (reference 15°);  batch 2: cohesion hardening ΔC = 0 … 5.6 MPa (reference 2.8)
# sensitivity.png:        apparent friction at the end of the runs vs gouge angle θ, one panel per batch: min/max envelope of the
#                         two end-members (▽ dashed: ψ = 0 / ΔC = 0, △ solid: ψ = 34° / ΔC = 5.6 MPa), reference, lab end values
# sensitivity_curves.png: apparent friction vs axial shortening at each θ: lab (shifted by b), reference, range spanned by each batch
# End-of-run friction: Δt-weighted mean over δ = 2.88-3.6 mm (last 20% of the complete runs), the same window for all runs.
# Missing or incomplete runs are skipped. Usage: julia --project=<repo> plot_sensitivity.jl
using CairoMakie, JLD2, Printf
const here = @__DIR__
load_nt(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(here, "..", "postprocessing", "lab_curves.jl")) # LAB: tag => (δ [mm], μ_app)

function readrun(folder)
    f = joinpath(folder, "probes.jld2"); isfile(f) || return nothing
    d = load_nt(f); it = d.it; t = d.probes.t[1:it]
    (; δ = t * d.ε̇yy * d.Ly * d.sc.L * 1e3, μ = d.probes.app_fric_lab[1:it], t)
end
# Δt-weighted mean over a fixed shortening window [δ1, δ2]; NaN if the run did not reach it (stopped early by nt)
function tail_mean(r; δ1=0.8 * 3.6, δ2=3.6)
    (r === nothing || r.δ[end] < 0.98δ2) && return NaN
    Δt = diff([0.0; r.t]); w = (r.δ .>= δ1) .& (r.δ .<= δ2) .& .!isnan.(r.μ)
    sum(r.μ[w] .* Δt[w]) / sum(Δt[w])
end
interp(x, xs, ys) = (x < xs[1] || x > xs[end]) ? NaN :
    (k = searchsortedlast(xs, x); k == length(xs) ? ys[end] : ys[k] + (ys[k+1] - ys[k]) * (x - xs[k]) / (xs[k+1] - xs[k]))

let
    b       = 2.05
    θs      = [30, 40, 50, 60]
    labtags = Dict(30 => ["30"], 40 => ["40"], 50 => ["50a", "50b"], 60 => ["60"])
    lab     = (θ = [30, 40, 50, 60], μ = [0.82, 0.77, 0.68, 0.51]) # lab end values, as in plot_friction.jl
    batches = [(name="dilatancy", lo="psi00", hi="psi34", col=colorant"#1b9e77",
                title="(a) Apparent friction vs. gouge angle: dilatancy", llo=L"\psi = 0", lhi=L"\psi = \phi = 34^\circ", lref=L"\mu_\mathrm{app}\ \mathrm{models}\ (\psi = 15^\circ)",
                lab=L"\mathrm{dilatancy}\ \psi = 0 \ldots 34^\circ\ (\mathrm{ref.}\ 15^\circ)"),
               (name="hardening", lo="dC0", hi="dC56", col=colorant"#2166ac",
                title="(b) Apparent friction vs. gouge angle: cohesion hardening", llo=L"\Delta C = 0", lhi=L"\Delta C = 5.6\ \mathrm{MPa}", lref=L"\mu_\mathrm{app}\ \mathrm{models}\ (\Delta C = 2.8\ \mathrm{MPa})",
                lab=L"\mathrm{hardening}\ \Delta C = 0 \ldots 5.6\ \mathrm{MPa}\ (\mathrm{ref.}\ 2.8)")]
    c_lab, ink, muted = colorant"#e66101", colorant"#222222", colorant"#8c8c8c"
    axkw    = (; xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08), topspinevisible=false, rightspinevisible=false)

    # Runs and end-of-run friction
    R    = Dict{Tuple{Int,String},Any}((θ, "ref") => readrun(joinpath(here, "..", "_resolution_tests", "LR_Tet$θ")) for θ in θs)
    for B in batches, θ in θs, p in (B.lo, B.hi)
        R[(θ, p)] = readrun(joinpath(here, "$(p)_Tet$θ"))
    end
    μend = Dict(k => tail_mean(r) for (k, r) in R)
    @printf("%-4s %-10s %8s %8s %8s\n", "θ", "batch", "low", "ref", "high")
    for B in batches, θ in θs
        @printf("%-4d %-10s %8.3f %8.3f %8.3f\n", θ, B.name, μend[(θ, B.lo)], μend[(θ, "ref")], μend[(θ, B.hi)])
    end

    # Figure 1: end-of-run friction vs θ, one panel per batch
    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size=(1400, 620), fontsize=20); axs = Axis[]
        for (k, B) in enumerate(batches)
            ax = Axis(fig[1, k]; title=B.title, titlealign=:left, titlefont=:regular, xticks=θs, xlabel=L"\mathrm{Gouge\ angle}\ \theta\ [^\circ]",
                      ylabel=k == 1 ? L"\mu_\mathrm{app}\ \mathrm{(mean\ over\ last\ 20\%\ of\ shortening)}" : "", axkw...); push!(axs, ax)
            lo = [μend[(θ, B.lo)] for θ in θs]; hi = [μend[(θ, B.hi)] for θ in θs]; rf = [μend[(θ, "ref")] for θ in θs]
            ok = .!isnan.(lo) .& .!isnan.(hi)
            count(ok) >= 2 && band!(ax, θs[ok], min.(lo, hi)[ok], max.(lo, hi)[ok], color=(B.col, 0.22))
            for (y, ls, mk) in ((lo, :dash, :dtriangle), (hi, :solid, :utriangle))
                m = .!isnan.(y)
                lines!(ax, θs[m], y[m], color=B.col, linestyle=ls, linewidth=2)
                scatter!(ax, θs[m], y[m], color=B.col, marker=mk, markersize=14, strokecolor=:white, strokewidth=1)
            end
            lines!(ax, θs, rf, color=ink, linewidth=2.5)
            scatter!(ax, θs, rf, color=ink, marker=:circle, markersize=11)
            scatter!(ax, lab.θ, lab.μ, color=c_lab, marker=:diamond, markersize=17, strokecolor=ink, strokewidth=1)
            xlims!(ax, 27, 63); ylims!(ax, 0.48, 1.02)
            # explicit legend: line + marker in one entry
            le(c, ls, mk, ms) = [LineElement(color=c, linestyle=ls, linewidth=2), MarkerElement(color=c, marker=mk, markersize=ms)]
            axislegend(ax, [PolyElement(color=(B.col, 0.22)), le(B.col, :dash, :dtriangle, 14), le(B.col, :solid, :utriangle, 14),
                            le(ink, :solid, :circle, 11), MarkerElement(color=c_lab, marker=:diamond, markersize=17, strokecolor=ink, strokewidth=1)],
                       [L"\mathrm{envelope}", B.llo, B.lhi, B.lref, L"\mu_\mathrm{app}\ \mathrm{experiments}"], position=:rt, labelsize=15, framevisible=false)
        end
        linkyaxes!(axs...); hideydecorations!(axs[2], grid=false, ticks=false)
        colgap!(fig.layout, 20)
        fig
    end
    save(joinpath(here, "sensitivity.png"), fig, px_per_unit=2)
    println(joinpath(here, "sensitivity.png"))

    # Figure 2: friction vs shortening at each θ, range spanned by each batch
    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size=(1400, 900), fontsize=19)
        for (k, θ) in enumerate(θs)
            i, j = fldmod1(k, 2)
            ax = Axis(fig[i, j]; title=L"\mathrm{(%$('a' + k - 1))}\ \theta = %$(θ)^\circ", titlealign=:left,
                      xlabel=i == 2 ? L"\mathrm{Axial\ shortening}\ \delta\ \mathrm{[mm]}" : "", ylabel=j == 1 ? L"\mu_\mathrm{app} = -\tau/\sigma_n" : "", axkw...)
            r0 = R[(θ, "ref")]
            for B in batches
                rl, rh = R[(θ, B.lo)], R[(θ, B.hi)]
                (rl === nothing || rh === nothing || r0 === nothing) && continue
                x  = r0.δ
                yl = interp.(x, Ref(rl.δ), Ref(rl.μ)); yh = interp.(x, Ref(rh.δ), Ref(rh.μ))
                ok = .!isnan.(yl) .& .!isnan.(yh)
                band!(ax, x[ok], min.(yl, yh)[ok], max.(yl, yh)[ok], color=(B.col, 0.22), label=B.lab)
                lines!(ax, rl.δ, rl.μ, color=B.col, linewidth=1.2, linestyle=:dash)
                lines!(ax, rh.δ, rh.μ, color=B.col, linewidth=1.2)
            end
            for t in labtags[θ]
                xs, ys = LAB[t]
                lines!(ax, xs .- b, ys, color=c_lab, linewidth=3, label=L"\mu_\mathrm{app}\ \mathrm{experiments}\ (\delta - %$(b)\ \mathrm{mm})")
            end
            r0 !== nothing && lines!(ax, r0.δ, r0.μ, color=ink, linewidth=2.5, label=L"\mu_\mathrm{app}\ \mathrm{models}")
            xlims!(ax, 0, 3.65)
            ylims!(ax, θ == 60 ? (0.3, 0.7) : (0.45, 1.05))
            if k == 1
                axislegend(ax, position=:rb, labelsize=13, framevisible=false, unique=true, padding=(0, 0, 0, 0))
                text!(ax, 0.02, 0.98, space=:relative, align=(:left, :top), fontsize=13, color=muted,
                      text="dashed: low end-member, solid: high")
            end
        end
        fig
    end
    save(joinpath(here, "sensitivity_curves.png"), fig, px_per_unit=2)
    println(joinpath(here, "sensitivity_curves.png"))
end
