# Strain localisation bands in the gouge, for one saved step of each run (see band_analysis.jl, line_bands, for the method).
# (a) log10 λ̇ around the centre of the gouge (black: not plastic), with a Logan (1979)-style sketch in the gouge frame next to it:
#     boundaries B, Y, R1, R2, P and σ1 (white) and the detected band families (colours as in b and c), at their true orientation
# (b) band cells of each family (oriented line filter), over log10 λ̇ in grey
# (c) orientation histogram of the band cells vs angle from the vertical, with the expected planes: Y (gouge boundaries),
#     R1, R2 (Riedel shears: conjugate Coulomb planes about the mean σ1, dashed, shaded up to the Roscoe angle; thin solid,
#     superscript A: Arthur angle), P, σ1; for each family: orientation, number of bands and share of the plastic dissipation
#     (Σ λ̇ τII over the band cells / Σ λ̇ τII over the analysed gouge)
# The step must have a checkpoint (τII). All angles in model coordinates, measured from the vertical (positive: counter-clockwise,
# as the gouge angle θ). All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Printf, Statistics

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(@__DIR__, "band_analysis.jl"))

for θ in (30, 40, 50, 60)                               # gouge angles of the runs
    root    = joinpath(dirname(@__DIR__), "_HR_hardening_2") # run folder
    prefix  = "_ShearApparatus_v9"
    h       = 3                                    # gouge thickness [mm]
    step    = 500                                  # step to analyse (same for all runs; needs step_XXXX.jld2 and checkpoint_XXXX.jld2)
    window  = 24.0                                 # side of the map windows [mm], centred on the gouge centre
    λ̇range  = (-6, -3)                             # colour range of log10 λ̇ [1/s]
    sk      = (L = 11.0, W = 5.0, offset = 7.0)    # Logan-style sketch: length and width [mm], offset from the gouge centre line [mm]

    folder  = joinpath(root, @sprintf("%s_Tet%02d_H%02d", prefix, θ, h))
    d       = load_step(joinpath(folder, @sprintf("step_%04d.jld2", step)))
    τII     = load(joinpath(folder, @sprintf("checkpoint_%04d.jld2", step)), "state").grid.τ.II[2:end-1, 2:end-1] .* d.sc.σ # [Pa]
    r       = analyse_bands(d)
    lb      = line_bands(r, τII)
    δ       = d.probes.t[step] * d.ε̇yy * d.Ly * d.sc.L * 1e3

    # Report
    @printf("θ = %d°, step %d, δ = %.2f mm (%s shear)\n", θ, step, δ, r.sense)
    println("  expected planes, from the vertical: ", join([@sprintf("%s %.1f°", k, fromvert(v)) for (k, v) in pairs(r.exp)], ", "))
    for fm in lb.fams
        nm, m = nearest_family(fm.α, r.exp)
        @printf("  family %+.0f°: %.1f%% of the gouge cells, %d bands (mean length %.1f mm), %.1f%% of the dissipation; %+.0f° from Y, %+.0f° from σ1, closest to %s (%+.1f°)\n",
                fromvert(fm.α), 100fm.frac_cells, fm.nbands, fm.len, 100fm.diss, adiff(fm.α, r.exp.Y), adiff(fm.α, r.exp.σ1), nm, m)
    end

    ink    = colorant"#222222"
    fcol   = [colorant"#35d0e0", colorant"#9be15d", colorant"#f781bf", colorant"#ffd92f"]   # family colours (on dark and light)
    tcol   = [colorant"#1a8fa0", colorant"#4d9a1f", colorant"#c2427f", colorant"#b8960a"]   # darker versions for text on white
    sδ     = @sprintf("%.2f", δ)
    names  = [(:Y, L"Y"), (:R1, L"R_1"), (:R1A, L"R_1^A"), (:R2, L"R_2"), (:R2A, L"R_2^A"), (:P, L"P"), (:σ1, L"\sigma_1")]
    flab   = [(@sprintf("%+.0f", fromvert(fm.α)), string(fm.nbands), @sprintf("%.0f", 100fm.diss)) for fm in lb.fams]

    fig = with_theme(theme_latexfonts()) do
        fig  = Figure(size=(2200, 600), fontsize=22)
        axkw = (; xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08))
        xc = r.x0 + r.n0 * r.ng[1]; yc = r.y0 + r.n0 * r.ng[2]          # current gouge centre
        xw, yw = xc .+ (-window/2, window/2), yc .+ (-window/2, window/2)
        I  = findall(xi -> xw[1] - 1 <= xi <= xw[2] + 1, r.x)
        J  = findall(yi -> yw[1] - 1 <= yi <= yw[2] + 1, r.y)
        λ̇  = [l > 0 ? log10(max(l, 10.0^λ̇range[1])) : NaN for l in r.λ̇[I, J]]

        # (a) map and sketch
        ax = Axis(fig[1, 1]; aspect=DataAspect(), title=L"\mathrm{(a)}\ \theta = %$(θ)^\circ,\ \mathrm{step}\ %$(step),\ \delta = %$(sδ)\ \mathrm{mm}",
                  titlealign=:left, xlabel=L"x\ \mathrm{[mm]}", ylabel=L"y\ \mathrm{[mm]}")
        hm = heatmap!(ax, r.x[I], r.y[J], λ̇, colormap=:inferno, colorrange=λ̇range, nan_color=:black)
        contour!(ax, r.x[I], r.y[J], Float64.(r.M[I, J]), levels=[0.5], color=:gray60, linewidth=1)
        # Logan (1979)-style sketch in the rock next to the gouge, drawn in the gouge frame at its true orientation:
        # boundaries B and centre line Y, with shear-sense arrows; expected R1, R2 (white), P (white, dashed), σ1 (white, dotted),
        # detected families (family colours) through the centre of the sketch.
        # Labels sit outside the box (beyond the boundary or the end the line exits through), away from the lines and from each other.
        c   = (xc, yc) .+ sk.offset .* r.ng
        pt(a, b) = Point2f(c .+ a .* r.tg .+ b .* r.ng)          # a: along the gouge, b: across
        dir(α) = (cosd(α), sind(α))
        halflen(u) = min(sk.W / 2 / max(abs(sum(u .* r.ng)), 1e-9), sk.L / 2 / max(abs(sum(u .* r.tg)), 1e-9))
        function through!(α; kw...)                               # line at angle α through c, clipped to the sketch box
            u = dir(α); ℓh = halflen(u)
            lines!(ax, [Point2f(c .- ℓh .* u), Point2f(c .+ ℓh .* u)]; kw...)
        end
        taken = Point2f[]
        function label!(α, lab; pref=1, col=:white)
            u = dir(α); ℓh = halflen(u)
            cands = map(Iterators.product((pref, -pref), (0.0, 1.0))) do (e, extra)   # both ends, then further out along the line
                q  = e * (ℓh + extra) .* u
                qa, qb = sum(q .* r.tg), sum(q .* r.ng)
                abs(abs(e * ℓh * sum(u .* r.ng)) - sk.W / 2) < 1e-6 ? pt(qa, sign(qb) * (sk.W / 2 + 0.6) + sign(qb) * extra) :
                                                                    pt(sign(qa) * (sk.L / 2 + 0.8 + extra), qb)
            end |> vec
            free = filter(p -> all(hypot((p .- t)...) > 1.0 for t in taken), cands)
            p = isempty(free) ? cands[1] : free[1]
            push!(taken, p)
            text!(ax, p, text=lab, color=col, fontsize=16, align=(:center, :center))
        end
        sgn = r.sense == :dextral ? -1 : 1                        # +n side moves along sgn * t
        for b in (-1, 1)
            lines!(ax, [pt(-sk.L/2, b * sk.W/2), pt(sk.L/2, b * sk.W/2)], color=:white, linewidth=2)
            a0 = b * sgn * sk.L / 4                               # half-arrow on each boundary, outside the box
            arrows2d!(ax, [pt(-a0, b * (sk.W/2 + 1.7))], [Vec2f((2a0) .* r.tg)], color=:white, shaftwidth=1.5, tipwidth=8, tiplength=8)
            p = pt(b * (sk.L/2 + 0.7), b * sk.W/2); push!(taken, p)
            text!(ax, p, text=L"B", color=:white, fontsize=16, align=(:center, :center))
        end
        through!(r.exp.Y, color=:white, linewidth=1.5)
        for (k, ls) in ((:R1, :solid), (:R2, :solid), (:P, :dash), (:σ1, :dot))
            through!(getfield(r.exp, k), color=:white, linewidth=2, linestyle=ls)
        end
        for (k, fm) in enumerate(lb.fams)
            through!(fm.α, color=fcol[k], linewidth=2.5)
        end
        label!(r.exp.Y, L"Y"; pref=1)
        label!(r.exp.R1, L"R_1"; pref=-1)
        label!(r.exp.P, L"P"; pref=-1)
        label!(r.exp.σ1, L"\sigma_1"; pref=1)
        label!(r.exp.R2, L"R_2"; pref=1)
        limits!(ax, xw..., yw...)
        Colorbar(fig[1, 2], hm, label=L"\log_{10}\ \dot{\lambda}\ \mathrm{[s^{-1}]}", width=12)

        # (b) band cells of each family
        ax = Axis(fig[1, 3]; aspect=DataAspect(), title=L"\mathrm{(b)\ Band\ cells}", titlealign=:left,
                  xlabel=L"x\ \mathrm{[mm]}", yticklabelsvisible=false)
        heatmap!(ax, r.x[I], r.y[J], λ̇, colormap=:grays, colorrange=λ̇range, nan_color=:black)
        for (k, fm) in enumerate(lb.fams)
            F = [fm.mask[i, j] ? 1.0 : NaN for i in I, j in J]
            heatmap!(ax, r.x[I], r.y[J], F, colormap=[fcol[k], fcol[k]], colorrange=(0, 2), nan_color=:transparent)
        end
        contour!(ax, r.x[I], r.y[J], Float64.(r.M[I, J]), levels=[0.5], color=:gray60, linewidth=1)
        limits!(ax, xw..., yw...)

        # (c) orientation histogram of the band cells
        ax = Axis(fig[1, 4]; title=L"\mathrm{(c)\ Band\ orientations}", titlealign=:left, xticks=-90:30:90,
                  xlabel=L"\mathrm{Angle\ from\ vertical}\ [^\circ]", ylabel=L"\mathrm{Band\ cells\ [\%\ per\ degree]}",
                  topspinevisible=false, rightspinevisible=false, axkw...)
        av = fromvert.(0.5:1:179.5); o = sortperm(av)
        hp = 100 .* lb.hs
        ymax = 1.25 * maximum(hp)
        for (a, b) in ((r.exp.R1, r.exp.R1ψ), (r.exp.R2, r.exp.R2ψ)) # Coulomb → Roscoe range (may wrap around ±90°)
            lo, hi = fromvert(a), fromvert(a) + adiff(b, a)
            for (l, u) in (lo < hi ? ((lo, hi),) : ((hi, lo),)), sh in (-180, 0, 180)
                vspan!(ax, l + sh, u + sh, color=(:gray50, 0.12))
            end
        end
        placed = Tuple{Float64, Int}[]   # labels of close lines are staggered vertically; the lines stop below the labels
        for (k, lab) in sort(names, by=n -> fromvert(getfield(r.exp, n[1])))
            a = fromvert(getfield(r.exp, k))
            used = [l for (b, l) in placed if abs(a - b) < 12]
            lev  = findfirst(l -> l ∉ used, 0:10) - 1; push!(placed, (a, lev))
            lines!(ax, [a, a], [0, ymax], color=k == :σ1 ? :gray50 : ink, linestyle=k == :σ1 ? :dot : k in (:R1A, :R2A) ? :solid : :dash,
                   linewidth=k in (:R1A, :R2A) ? 1 : 1.5)
            text!(ax, a, ymax + lev * 0.09 * ymax, text=lab, align=(:center, :bottom), fontsize=18, color=ink)
        end
        lines!(ax, av[o], hp[o], color=ink, linewidth=2)
        for (k, fm) in enumerate(lb.fams)   # family: marker at the peak; orientation, bands and dissipation in the legend below
            a = fromvert(fm.α); yk = hp[findmin(abs.(av .- a))[2]]
            scatter!(ax, a, yk, color=fcol[k], strokecolor=ink, strokewidth=1, markersize=13)
        end
        isempty(lb.fams) || Legend(fig[1, 5], [MarkerElement(color=fcol[k], marker=:circle, strokecolor=ink, strokewidth=1, markersize=13) for k in eachindex(lb.fams)],
               [L"%$(f[1])^\circ:\ %$(f[2])\ \mathrm{bands},\ %$(f[3])\%\ \mathrm{of\ the\ plastic\ dissipation}" for f in flab],
               orientation=:vertical, framevisible=false, labelsize=18, tellheight=false, valign=:center, padding=(0, 0, 0, 0))
        xlims!(ax, -90, 90); ylims!(ax, 0, ymax * (1 + 0.09 * (maximum(last.(placed)) + 1)) + 0.02 * ymax)

        colgap!(fig.layout, 1, 8)
        colsize!(fig.layout, 4, Relative(0.30))
        resize_to_layout!(fig)
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), @sprintf("bands_Tet%02d_step%04d.png", θ, step)), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), @sprintf("bands_Tet%02d_step%04d.pdf", θ, step)), fig)
end
