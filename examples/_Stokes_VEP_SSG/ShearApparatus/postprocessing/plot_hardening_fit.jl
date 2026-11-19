# How the gouge strain hardening was fitted to the lab data.
# (a-d) Apparatus friction vs axial shortening for the four lab tests (30, 40, 50 (x2), 60°): digitised lab curves vs the runs
#       with hardening (_HR_hardening_2: ϕ = 34°, C(εp) below) and, if the folder still exists, without hardening
#       (HR_no_hardening: ϕ = 36.13°, C = 0.67 MPa).
#       The lab shortening axis is loosely constrained (corrections): the lab curves are shifted by a single offset b for all tests,
#       chosen (with C(εp)) to minimise the rms misfit over model δ ≥ 0.5 mm. The misfit of each run is printed in its panel.
# (e) The fitted law C(εp) = C0 + ΔC tanh(εp/εh) and the plastic strain reached in the gouge at the end of each run
#     (median and 10-90 percentile range of εp over the gouge cells).
# All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Printf, Statistics

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(@__DIR__, "lab_curves.jl")) # LAB: tag => (δ [mm], μ_app), digitised from the lab figure

# Linear interpolation (NaN outside the data)
interp(x, xs, ys) = (x < xs[1] || x > xs[end]) ? NaN :
    (k = searchsortedlast(xs, x); k == length(xs) ? ys[end] : ys[k] + (ys[k+1] - ys[k]) * (x - xs[k]) / (xs[k+1] - xs[k]))

let
    root    = dirname(@__DIR__)
    prefix  = "_ShearApparatus_v9"
    h       = 3                     # gouge thickness [mm]
    b       = 2.05                  # shift of the lab shortening axis [mm]: model δ = lab δ - b
    δmin    = 0.5                   # misfit computed for model δ ≥ δmin (after yield)
    tests   = [30 => ["30"], 40 => ["40"], 50 => ["50a", "50b"], 60 => ["60"]] # gouge angle => lab tests
    gouge   = 2                     # phase index of the gouge

    c_lab   = colorant"#e66101"     # same as plot_friction.jl
    c_hard  = colorant"#2166ac"
    muted   = colorant"#8c8c8c"
    ink     = colorant"#222222"

    function readrun(set, θ)
        folder = joinpath(root, set, @sprintf("%s_Tet%02d_H%02d", prefix, θ, h))
        fs = filter(f -> occursin(r"^step_\d+\.jld2$", f), readdir(folder))
        d  = load_step(joinpath(folder, last(sort(fs))))
        it = d.it
        (; δ = d.probes.t[1:it] * d.ε̇yy * d.Ly * d.sc.L * 1e3, μ = d.probes.app_fric_lab[1:it], d)
    end
    function misfit(r, tags)
        e = Float64[]
        for t in tags, (δ, μ) in zip(r.δ, r.μ)
            δ < δmin && continue
            μl = interp(δ + b, LAB[t]...)
            isnan(μl) || push!(e, μ - μl)
        end
        return isempty(e) ? NaN : sqrt(mean(abs2, e))
    end

    set0, set1 = "HR_no_hardening", "_HR_hardening_2"   # reference runs without hardening (optional), runs with hardening
    has0  = isdir(joinpath(root, set0))
    has0 || println("No $set0 folder: the runs without hardening are not shown")
    runs0 = has0 ? Dict(θ => readrun(set0, θ) for (θ, _) in tests) : nothing
    runs1 = Dict(θ => readrun(set1, θ) for (θ, _) in tests)

    # Hardening law, from the run parameters (scaled -> MPa)
    d1  = runs1[30].d
    pl  = d1.plasticity
    MPa = d1.sc.σ / 1e6
    C0, ΔC, εh, ϕ = pl.C[gouge] * MPa, pl.ΔC[gouge] * MPa, pl.εh[gouge], pl.ϕ[gouge]
    Claw(ε) = C0 + ΔC * tanh(ε / εh)
    if has0
        pl0 = runs0[30].d.plasticity
        C00, ϕ0 = pl0.C[gouge] * runs0[30].d.sc.σ / 1e6, pl0.ϕ[gouge]
    end

    fig = with_theme(theme_latexfonts()) do
        fig  = Figure(size=(2000, 480), fontsize=20)
        axkw = (; xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08), topspinevisible=false, rightspinevisible=false)
        δmax = maximum(r.δ[end] for r in values(runs1))
        for (k, (θ, tags)) in enumerate(tests)
            ax = Axis(fig[1, k]; title=L"\mathrm{(%$('a' + k - 1))}\ \theta = %$(θ)^\circ", titlealign=:left,
                      xlabel=L"\delta\ \mathrm{[mm]}", ylabel=k == 1 ? L"\mu_\mathrm{app}" : "", axkw...)
            for t in tags
                xs, ys = LAB[t]
                lines!(ax, xs .- b, ys, color=c_lab, linewidth=3.5, label=L"\mu_\mathrm{app}\ \mathrm{experiments}\ (\delta_\mathrm{lab} - %$(b)\ \mathrm{mm})")
            end
            r1 = runs1[θ]
            has0 && lines!(ax, runs0[θ].δ, runs0[θ].μ, color=muted, linewidth=2, linestyle=:dash,
                           label=L"\mathrm{no\ hardening}\ (\phi = %$(ϕ0)^\circ,\ C = %$(round(C00, digits=2))\ \mathrm{MPa})")
            lines!(ax, r1.δ, r1.μ, color=c_hard, linewidth=2.5, label=L"\mathrm{hardening}\ (\phi = %$(round(Int, ϕ))^\circ,\ C(\varepsilon_p))")
            m0, m1 = has0 ? misfit(runs0[θ], tags) : NaN, misfit(r1, tags)
            @printf("θ = %d°: rms misfit  no hardening %.3f  hardening %.3f\n", θ, m0, m1)
            s0, s1 = @sprintf("%.3f", m0), @sprintf("%.3f", m1)
            text!(ax, 0.97, 0.04, space=:relative, align=(:right, :bottom), fontsize=16, color=ink,
                  text=isnan(m1) ? L"\mathrm{lab\ test\ too\ short:\ not\ used\ in\ the\ fit}" :
                       has0 ? L"\mathrm{rms:}\ %$(s0)\ \rightarrow\ %$(s1)" : L"\mathrm{rms:}\ %$(s1)")
            xlims!(ax, 0, δmax)
            ylims!(ax, θ == 60 ? (0.35, 0.65) : (0.5, 0.9))
            k == 1 && axislegend(ax, position=:lt, labelsize=14, framevisible=false, unique=true, padding=(0, 0, 0, 0))
        end

        # (e) the law and the plastic strain reached in the gouge at the end of the runs
        ax = Axis(fig[1, length(tests) + 1]; title=L"\mathrm{(e)\ Gouge\ cohesion}", titlealign=:left,
                  xlabel=L"\varepsilon_p = \int \dot{\lambda}\, dt", ylabel=L"C\ \mathrm{[MPa]}", axkw...)
        ε = range(0, 3, length=200)
        hlines!(ax, [C0, C0 + ΔC], color=muted, linestyle=:dot, linewidth=1.5)
        lines!(ax, ε, Claw.(ε), color=c_hard, linewidth=3)
        text!(ax, 0.05, C0 + ΔC, text=L"C_0 + \Delta C", align=(:left, :bottom), fontsize=16, color=muted)
        text!(ax, 2.95, C0, text=L"C_0", align=(:right, :top), fontsize=16, color=muted)
        text!(ax, 0.97, 0.04, space=:relative, align=(:right, :bottom), fontsize=17, color=ink,
              text=L"C = %$(C0) + %$(ΔC)\,\tanh(\varepsilon_p / %$(εh))\ \mathrm{MPa}")
        cmap = cgrad([colorant"#9ecae1", colorant"#08306b"], length(tests), categorical=true)
        for (k, (θ, _)) in enumerate(tests)
            d   = runs1[θ].d
            εg  = d.εpc[d.phc .== gouge]
            q10, q50, q90 = quantile(εg, (0.1, 0.5, 0.9))
            yk  = Claw(q50)
            rangebars!(ax, [yk], [q10], [q90], direction=:x, color=cmap[k], linewidth=2, whiskerwidth=8)
            scatter!(ax, [q50], [yk], color=cmap[k], strokecolor=ink, strokewidth=1, markersize=12, label=L"%$(θ)^\circ")
            @printf("θ = %d°: gouge εp at the end  median %.2f  (10-90%%: %.2f-%.2f)\n", θ, q50, q10, q90)
        end
        xlims!(ax, 0, 3); ylims!(ax, 0, C0 + ΔC + 0.6)
        axislegend(ax, L"\mathrm{end\ of\ run}", position=:rc, labelsize=14, titlesize=15, framevisible=false)
        colgap!(fig.layout, 18)
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), "hardening_fit.png"), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), "hardening_fit.pdf"), fig)
end
