# Evolution of strain localisation in the gouge of one run, from all saved steps (see band_analysis.jl for the method).
# (a) Localisation: A50 (fraction of the gouge carrying 50% of the plastic strain rate; uniform: 0.5) and plastic fraction vs shortening,
#     with the onset of localisation (first step with A50 < A50c), and the apparent friction for reference
# (b) Orientation spectrogram: ridge orientation density (log scale, 1 = isotropic) vs shortening and angle from the vertical,
#     with the expected planes (Y; R1, R2 at the Coulomb angle (dashed) and the Arthur angle (thin, superscript A) from the mean σ1
#     of each step; P; σ1) and the detected band families (dots, size ∝ density)
# The per-step results are cached in postprocessing/cache/bands_<run>.jld2 (delete the file to recompute).
# All angles in model coordinates, measured from the vertical. All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Printf, Statistics

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(@__DIR__, "band_analysis.jl"))

# Analyse all saved steps of one run (cached)
function band_timeseries(folder; recompute=false)
    cache = joinpath(mkpath(joinpath(@__DIR__, "cache")), "bands_" * basename(folder) * ".jld2")
    !recompute && isfile(cache) && return load(cache, "ts")
    files = sort(filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder)))
    rows  = map(files) do f
        d = load_step(joinpath(folder, f))
        r = analyse_bands(d)
        (; it = d.it, δ = d.probes.t[d.it] * d.ε̇yy * d.Ly * d.sc.L * 1e3, r.A50, r.PR, r.fpl, r.coh, r.dens,
           peaks = [(p.α, p.dens) for p in r.peaks], exp = r.exp, r.sense)
    end
    dl = load_step(joinpath(folder, last(files)))
    ts = (; rows, αb = collect(0.5:1:179.5),
          δall = dl.probes.t[1:dl.it] * dl.ε̇yy * dl.Ly * dl.sc.L * 1e3, μ = dl.probes.app_fric_lab[1:dl.it])
    jldsave(cache; ts)
    return ts
end

let
    root    = joinpath(dirname(@__DIR__), "_HR_hardening_2") # run folder
    prefix  = "_ShearApparatus_v9"
    θ       = 30                    # gouge angle of the run
    h       = 3                     # gouge thickness [mm]
    A50c    = 0.3                   # localisation onset: first step with A50 < A50c (uniform: 0.5) once the gouge has yielded
    fyield  = 0.9                   # the gouge has yielded at the first step with a plastic fraction ≥ fyield
    fmin    = 0.05                  # steps with a plastic fraction < fmin (onset of yield: a few cells) are not analysed
    folder  = joinpath(root, @sprintf("%s_Tet%02d_H%02d", prefix, θ, h))

    t0 = time()
    ts = band_timeseries(folder)
    @printf("%d steps analysed (%.0f s)\n", length(ts.rows), time() - t0)
    rows = [r for (k, r) in enumerate(ts.rows) if k == 1 || r.δ > ts.rows[k-1].δ]   # (a restart can repeat the last δ)
    rows = [r.fpl >= fmin ? r : merge(r, (; A50=NaN, PR=NaN, coh=NaN, dens=fill(NaN, 180), peaks=Tuple{Float64,Float64}[])) for r in rows]
    δ    = [r.δ for r in rows]
    A50  = [r.A50 for r in rows]
    ky   = findfirst(r -> r.fpl >= fyield, rows)
    kon  = ky === nothing ? nothing : findfirst(k -> k >= ky && A50[k] < A50c, eachindex(A50))
    δon  = kon === nothing ? NaN : δ[kon]
    println("shear sense: ", rows[end].sense)
    kon === nothing ? println("no localisation (A50 ≥ $A50c at all steps)") :
        @printf("localisation onset: step %d, δ = %.2f mm (A50 = %.2f)\n", rows[kon].it, δon, A50[kon])
    for r in rows
        @printf("step %4d  δ %.2f  A50 %.2f  fpl %.2f  coh %.2f  σ1 %+.1f  peaks %s\n", r.it, r.δ, r.A50, r.fpl, r.coh, fromvert(r.exp.σ1),
                join([@sprintf("%+.1f° (%.1f)", fromvert(a), w) for (a, w) in r.peaks], ", "))
    end

    ink, cyan = colorant"#222222", colorant"#35d0e0"
    names = [(:Y, L"Y", :solid), (:R1, L"R_1", :dash), (:R1A, L"R_1^A", :solid), (:R2, L"R_2", :dash), (:R2A, L"R_2^A", :solid),
             (:P, L"P", :dashdot), (:σ1, L"\sigma_1", :dot)]

    fig = with_theme(theme_latexfonts()) do
        fig  = Figure(size=(1300, 900), fontsize=22)
        axkw = (; xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08), topspinevisible=false, rightspinevisible=false)

        # (a) localisation
        ax1 = Axis(fig[1, 1]; title=L"\mathrm{(a)\ Localisation},\ \theta = %$(θ)^\circ", titlealign=:left, ylabel=L"A_{50},\ \mathrm{plastic\ fraction}",
                   xticklabelsvisible=false, axkw...)
        hlines!(ax1, 0.5, color=:gray60, linewidth=1)
        lines!(ax1, δ, A50, color=colorant"#2166ac", linewidth=2.5, label=L"A_{50}")
        scatter!(ax1, δ, A50, color=colorant"#2166ac", markersize=7)
        lines!(ax1, δ, [r.fpl for r in rows], color=:gray50, linewidth=2, linestyle=:dash, label=L"\mathrm{plastic\ fraction}")
        lines!(ax1, ts.δall, ts.μ, color=ink, linewidth=1.5, label=L"\mu_\mathrm{app}")
        isnan(δon) || vlines!(ax1, δon, color=colorant"#d6404e", linewidth=2, label=L"\mathrm{onset}\ (A_{50} < %$(A50c))")
        ylims!(ax1, 0, 1)
        axislegend(ax1, position=:rc, framevisible=false, labelsize=17)

        # (b) spectrogram
        ax2 = Axis(fig[2, 1]; title=L"\mathrm{(b)\ Band\ orientations}", titlealign=:left, xlabel=L"\mathrm{Axial\ shortening}\ \delta\ \mathrm{[mm]}",
                   ylabel=L"\mathrm{Angle\ from\ vertical}\ [^\circ]", yticks=-90:30:90, axkw...)
        av = fromvert.(ts.αb); o = sortperm(av)
        D  = hcat([r.dens[o] for r in rows]...)'                       # steps × angles
        hm = heatmap!(ax2, δ, av[o], log10.(max.(D, 1e-2)), colormap=:grays, colorrange=(-1, 1.5), nan_color=:white)
        for (k, lab, ls) in names
            a = [fromvert(getfield(r.exp, k)) for r in rows]
            lines!(ax2, δ, a, color=colorant"#e66101", linewidth=k in (:R1A, :R2A) ? 1 : 2, linestyle=ls)
            text!(ax2, δ[end], a[end], text=lab, color=colorant"#e66101", fontsize=18, align=(:left, :center), offset=(6, 0))
        end
        pk = [(r.δ, fromvert(a), w) for r in rows for (a, w) in r.peaks]
        isempty(pk) || scatter!(ax2, first.(pk), getindex.(pk, 2), markersize=[4 + 3 * log10(w) * 3 for (_, _, w) in pk],
                                color=cyan, strokecolor=ink, strokewidth=0.8)
        isnan(δon) || vlines!(ax2, δon, color=colorant"#d6404e", linewidth=2)
        ylims!(ax2, -90, 90)
        Colorbar(fig[2, 2], hm, label=L"\log_{10}\ \mathrm{ridge\ orientation\ density}", width=12)
        linkxaxes!(ax1, ax2)
        xlims!(ax2, 0, 1.06 * maximum(δ))
        rowsize!(fig.layout, 2, Relative(0.6))
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), @sprintf("bands_timeseries_Tet%02d.png", θ)), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), @sprintf("bands_timeseries_Tet%02d.pdf", θ)), fig)
    display(fig)
end
