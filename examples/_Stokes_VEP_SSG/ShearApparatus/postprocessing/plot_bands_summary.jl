# Strain localisation bands in the gouge at θ = 30, 50, 60° (one row each), for one saved step of each run (band_analysis.jl).
# Band families: oriented ridge filter (band_analysis.jl, ridge_bands): every cell responds at each orientation with a contrast
#        C(α) (elongated ridge kernel, matched to the band width across, σt = 1.5 mm along); a cell belongs to every family whose
#        orientation gives a local maximum of C above 0.3 (crossings: two families). Its plastic work rate λ̇ τII is split between its
#        families in proportion to C, so crossings are counted once. Families: peaks of the work-rate-weighted orientation histogram
#        carrying ≥ 2% of the work rate. Short segments linking neighbouring bands (P-like at 30-50°) carry < 0.5% and are not a family.
# Left:  log10 λ̇ around the centre of the gouge (black: not plastic); on the lower half of the gouge (beyond the dashed line), the
#        cells of each family in its colour (white: crossings), and a Logan (1979)-style sketch in the gouge frame:
#        boundaries B, Y, R1, R2, P, σ1 (white) and the detected families (family colours) at their true orientation.
# Right: orientation distributions vs angle from the vertical, in % per degree (both sum to 100%): family cells (line; a crossing
#        cell counts once per family) and plastic work rate λ̇ τII (filled; family cells at their own orientation with their share,
#        the other cells at the structure-tensor orientation), with the expected planes Y, R1, R2 (Coulomb, dashed; shaded up to the
#        Roscoe angle), R1^A, R2^A (Arthur, thin), P, σ1. Families listed top left with their number of bands and share of the work
#        rate, and the share outside the bands.
# Analysed gouge: ridge mode: gouge cells minus 2 cells at each boundary; fatbox mode: the whole gouge strip up to the contacts (Y shears
# localise there; contact rule of fatbox_bands); central 80% of the gouge length (not only the map window).
# The step must have a checkpoint (τII). Angles in model coordinates, from the vertical (positive: counter-clockwise, as θ).
using CairoMakie, JLD2, Printf, Statistics

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(@__DIR__, "band_analysis.jl"))

let
    root    = joinpath(dirname(@__DIR__), "_HR_hardening_2") # run folder
    prefix  = "_ShearApparatus_v9"
    θs      = (30, 50, 60)
    h       = 3                                    # gouge thickness [mm]
    method  = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :ridge   # :ridge (ridge_bands) or :fatbox (fatbox_bands: run fatbox/fatbox_export.jl and fatbox_extract.py first)
    step    = 500                                  # step to analyse (needs step_XXXX.jld2 and checkpoint_XXXX.jld2)
    window  = 36.0                                 # side of the map windows [mm], centred on the gouge centre
    λ̇range  = (-6, -3)                             # colour range of log10 λ̇ [1/s]
    sk      = (L = 13.0, W = 6.0, offset = 10.0)   # Logan-style sketch: length and width [mm], offset from the gouge centre line [mm]

    ink    = colorant"#222222"
    fcol   = [colorant"#35d0e0", colorant"#9be15d", colorant"#f781bf", colorant"#4d7bff"]   # family colours (on dark)
    tcol   = [colorant"#1a8fa0", colorant"#4d9a1f", colorant"#c2427f", colorant"#b8960a"]   # darker versions on white
    c_work = colorant"#8c6d31"
    fcolk(rb, k) = hasproperty(rb.fams[k], :cidx) ? fcol[rb.fams[k].cidx] : fcol[k]   # family colour (Fatbox planes: R1, R2, P, Y)
    pname   = Dict(:R1 => "R_1", :R2 => "R_2", :P => "P", :Y => "Y", :none => "")
    c_nofam = colorant"#ffffff"                     # Fatbox segments that belong to no family (white on the map)
    names  = [(:Y, L"Y"), (:R1, L"R_1"), (:R1A, L"R_1^A"), (:R2, L"R_2"), (:R2A, L"R_2^A"), (:P, L"P"), (:σ1, L"\sigma_1")]

    fig = with_theme(theme_latexfonts()) do
        fig  = Figure(size=(1500, 560 * length(θs)), fontsize=27)
        axkw = (; xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08))
        hm   = nothing
        for (row, θ) in enumerate(θs)
            folder = joinpath(root, @sprintf("%s_Tet%02d_H%02d", prefix, θ, h))
            d      = load_step(joinpath(folder, @sprintf("step_%04d.jld2", step)))
            τII    = load(joinpath(folder, @sprintf("checkpoint_%04d.jld2", step)), "state").grid.τ.II[2:end-1, 2:end-1] .* d.sc.σ # [Pa]
            r      = method === :fatbox ? fatbox_region(d) : analyse_bands(d)
            rb     = method === :fatbox ? fatbox_bands(r, τII, θ; planes=true) : ridge_bands(r, τII)   # band families (crossings shared)
            δ      = d.probes.t[step] * d.ε̇yy * d.Ly * d.sc.L * 1e3
            sδ     = @sprintf("%.2f", δ)

            @printf("θ = %d°, step %d, δ = %.2f mm (%s shear), ϕ = %.0f°, ψ = %.0f°\n", θ, step, δ, r.sense, r.ϕ, r.ψ)
            method === :fatbox && @printf("  %d segments, %.1f%% of the work rate in segments of no family\n", length(rb.segs), 100rb.other)
            for fm in rb.fams
                nm, m = hasproperty(fm, :name) && fm.name !== :none ? (fm.name, 0.0) : nearest_family(fm.α, r.exp)
                @printf("  family %+.0f° (%s %+.1f°): %d bands, %.1f%% of the cells, %.1f%% of the work rate\n",
                        fromvert(fm.α), nm, m, fm.nbands, 100fm.vol, 100fm.diss)
            end
            @printf("  crossings: %.1f%% of the cells; outside the bands (diffuse): %.1f%% of the cells, %.1f%% of the work rate\n",
                    100count(rb.cross[r.R]) / count(r.R), 100rb.diffuse.vol, 100rb.diffuse.diss)

            # ---- left: map with band volumes and sketch
            xc = r.x0 + r.n0 * r.ng[1]; yc = r.y0 + r.n0 * r.ng[2]          # current gouge centre
            xw, yw = xc .+ (-window/2, window/2), yc .+ (-window/2, window/2)
            I  = findall(xi -> xw[1] - 1 <= xi <= xw[2] + 1, r.x)
            J  = findall(yi -> yw[1] - 1 <= yi <= yw[2] + 1, r.y)
            λ̇  = [l > 0 ? log10(max(l, 10.0^λ̇range[1])) : NaN for l in r.λ̇[I, J]]
            ax = Axis(fig[row, 1]; aspect=DataAspect(), title=L"\mathrm{(%$('a' + 2row - 2))}\ \theta = %$(θ)^\circ,\ \delta = %$(sδ)\ \mathrm{mm}",
                      titlealign=:left, xlabel=row == length(θs) ? L"x\ \mathrm{[mm]}" : "", ylabel=L"y\ \mathrm{[mm]}")
            hm = heatmap!(ax, r.x[I], r.y[J], λ̇, colormap=:inferno, colorrange=λ̇range, nan_color=:black)
            if method === :fatbox
                # Fatbox segments (skeletons) over the whole map, in family colours; segments of no family in c_nofam
                for (q, sg) in enumerate(rb.segs)
                    P = [Point2f(r.x[p[1]], r.y[p[2]]) for p in sg.px if xw[1] <= r.x[p[1]] <= xw[2] && yw[1] <= r.y[p[2]] <= yw[2]]
                    isempty(P) || scatter!(ax, P, color=rb.segfam[q] > 0 ? fcolk(rb, rb.segfam[q]) : c_nofam, markersize=5, marker=:rect)
                end
            else
                # split view: band volumes only on the lower half of the gouge (along-gouge coordinate S < 0), λ̇ alone on the upper half
                for (k, fm) in enumerate(rb.fams)
                    F = [fm.mask[i, j] && !rb.cross[i, j] && r.S[i, j] < 0 ? 1.0 : NaN for i in I, j in J]
                    heatmap!(ax, r.x[I], r.y[J], F, colormap=[fcol[k], fcol[k]], colorrange=(0, 2), nan_color=:transparent)
                end
                F = [rb.cross[i, j] && r.S[i, j] < 0 ? 1.0 : NaN for i in I, j in J]                 # crossings (two families): white
                heatmap!(ax, r.x[I], r.y[J], F, colormap=[:white, :white], colorrange=(0, 2), nan_color=:transparent)
                hs = 5.0                                                                  # half length of the split line [mm]
                lines!(ax, [Point2f((xc, yc) .- hs .* r.ng), Point2f((xc, yc) .+ hs .* r.ng)], color=:white, linewidth=1.5, linestyle=:dash)
            end
            contour!(ax, r.x[I], r.y[J], Float64.(r.M[I, J]), levels=[0.5], color=:gray60, linewidth=1)
            # analysed region: ends of the central part of the gouge, if inside the window
            contour!(ax, r.x[I], r.y[J], Float64.(r.R[I, J]), levels=[0.5], color=:white, linewidth=0.8, linestyle=:dot)

            # Logan (1979)-style sketch in the rock next to the gouge, in the gouge frame (labels outside the box, staggered)
            c   = (xc, yc) .+ sk.offset .* r.ng
            pt(a, b) = Point2f(c .+ a .* r.tg .+ b .* r.ng)
            dir(α) = (cosd(α), sind(α))
            halflen(u) = min(sk.W / 2 / max(abs(sum(u .* r.ng)), 1e-9), sk.L / 2 / max(abs(sum(u .* r.tg)), 1e-9))
            through!(α; kw...) = (u = dir(α); ℓh = halflen(u); lines!(ax, [Point2f(c .- ℓh .* u), Point2f(c .+ ℓh .* u)]; kw...))
            taken = Point2f[]
            function label!(α, lab; pref=1)
                u = dir(α); ℓh = halflen(u)
                cands = map(Iterators.product((pref, -pref), (0.0, 1.3, 2.6))) do (e, extra)
                    q  = e * (ℓh + extra) .* u
                    qa, qb = sum(q .* r.tg), sum(q .* r.ng)
                    abs(abs(e * ℓh * sum(u .* r.ng)) - sk.W / 2) < 1e-6 ? pt(qa, sign(qb) * (sk.W / 2 + 0.8) + sign(qb) * extra) :
                                                                        pt(sign(qa) * (sk.L / 2 + 1.0 + extra), qb)
                end |> vec
                free = filter(p -> all(hypot((p .- t)...) > 2.2 for t in taken), cands)
                p = isempty(free) ? cands[1] : free[1]; push!(taken, p)
                text!(ax, p, text=lab, color=:white, fontsize=21, align=(:center, :center))
            end
            sgn = r.sense == :dextral ? -1 : 1
            for b in (-1, 1)
                lines!(ax, [pt(-sk.L/2, b * sk.W/2), pt(sk.L/2, b * sk.W/2)], color=:white, linewidth=2)
                a0 = b * sgn * sk.L / 4
                arrows2d!(ax, [pt(-a0, b * (sk.W/2 + 2.0))], [Vec2f((2a0) .* r.tg)], color=:white, shaftwidth=1.5, tipwidth=8, tiplength=8)
                p = pt(-a0 - sign(a0) * 1.6, b * (sk.W/2 + 2.0)); push!(taken, p)   # B: in line with the arrow, beyond its tail
                text!(ax, p, text=L"B", color=:white, fontsize=21, align=(:center, :center))
            end
            through!(r.exp.Y, color=:white, linewidth=1.5)
            for (k, ls) in ((:R1, :solid), (:R2, :solid), (:P, :dash), (:σ1, :dot))
                through!(getfield(r.exp, k), color=:white, linewidth=2, linestyle=ls)
            end
            for (k, fm) in enumerate(rb.fams)
                through!(fm.α, color=fcolk(rb, k), linewidth=3)
            end
            label!(r.exp.Y, L"Y"; pref=1); label!(r.exp.R1, L"R_1"; pref=-1); label!(r.exp.P, L"P"; pref=-1)
            label!(r.exp.σ1, L"\sigma_1"; pref=1); label!(r.exp.R2, L"R_2"; pref=1)
            limits!(ax, xw..., yw...)

            # ---- right: orientation distributions
            ax = Axis(fig[row, 2]; title=L"\mathrm{(%$('a' + 2row - 1))}\ \mathrm{Band\ orientations\ and\ plastic\ work\ rate}", titlealign=:left,
                      xticks=-90:30:90, xlabel=row == length(θs) ? L"\mathrm{Angle\ from\ vertical}\ [^\circ]" : "",
                      ylabel=L"\%\ \mathrm{per\ degree}", topspinevisible=false, rightspinevisible=false, axkw...)
            av = fromvert.(0.5:1:179.5); o = sortperm(av)
            hp = 100 .* rb.hs; hw = 100 .* rb.hw
            ymax = 1.3 * max(maximum(hp), maximum(hw))
            for (a, b) in ((r.exp.R1, r.exp.R1ψ), (r.exp.R2, r.exp.R2ψ))   # Coulomb → Roscoe range (may wrap around ±90°), below the labels
                lo, hi = fromvert(a), fromvert(a) + adiff(b, a)
                for (l, u) in (lo < hi ? ((lo, hi),) : ((hi, lo),)), sh in (-180, 0, 180)
                    band!(ax, [l + sh, u + sh], [0.0, 0.0], [ymax, ymax], color=(:gray50, 0.12))
                end
            end
            # expected planes: lines up to ymax; labels in one row above, pushed apart (≥ wlab degrees, order kept) and joined to
            # their line by a short leader, so that labels never overlap each other and lines never cross a label
            labs = sort([(fromvert(getfield(r.exp, k)), k, lab) for (k, lab) in names], by=first)
            xs   = [l[1] for l in labs]; wlab = 9.0
            for _ in 1:500
                moved = false
                for i in 1:length(xs)-1
                    dd = xs[i+1] - xs[i]
                    dd < wlab - 1e-9 && (xs[i] -= (wlab - dd) / 2; xs[i+1] += (wlab - dd) / 2; moved = true)
                end
                xs .= clamp.(xs, -86, 86)
                moved || break
            end
            ylead, ylab = 1.10ymax, 1.13ymax
            for ((a, k, lab), xl) in zip(labs, xs)
                lines!(ax, [a, a], [0, ymax], color=k == :σ1 ? :gray50 : ink, linestyle=k == :σ1 ? :dot : k in (:R1A, :R2A) ? :solid : :dash,
                       linewidth=k in (:R1A, :R2A) ? 1 : 1.5)
                lines!(ax, [a, xl], [ymax, ylead], color=:gray40, linewidth=0.8)
                text!(ax, xl, ylab, text=lab, align=(:center, :bottom), fontsize=22, color=ink)
            end
            band!(ax, av[o], zeros(180), hw[o], color=(c_work, 0.35))
            if method === :fatbox   # work rate of each family, stacked in the family colours (rest: no family + outside the bands)
                cum = zeros(180)
                for k in eachindex(rb.fams)
                    nxt = cum .+ 100 .* rb.hwf[k]
                    band!(ax, av[o], cum[o], nxt[o], color=(fcolk(rb, k), 0.75)); cum = nxt
                end
            end
            lines!(ax, av[o], hw[o], color=c_work, linewidth=1.5)
            lines!(ax, av[o], hp[o], color=ink, linewidth=2)
            # families: marker at the band-cell peak; list in a strip above the plane labels, with orientation, number of bands and
            # share of the work (band volume), then the share of the work outside the bands
            ytop  = 1.34ymax                                                    # top of the plane labels
            dy    = 0.13 * ymax; nl = length(rb.fams) + 1 + (method === :fatbox)
            yl(k) = ytop + dy * (nl - k + 0.5)                                 # line k of the list, from the top
            for (k, fm) in enumerate(rb.fams)
                a = fromvert(fm.α); yk = hp[findmin(abs.(av .- a))[2]]
                scatter!(ax, a, yk, color=fcolk(rb, k), strokecolor=ink, strokewidth=1, markersize=17)
                sa, sw = @sprintf("%+.0f", a), @sprintf("%.0f", 100fm.diss)
                sseg = fm.nbands == 1 ? "segment" : "segments"
                scatter!(ax, -86, yl(k), color=fcolk(rb, k), strokecolor=ink, strokewidth=1, markersize=17)
                text!(ax, -82, yl(k), align=(:left, :center), fontsize=22, color=ink,
                      text=method === :fatbox ? L"%$(pname[fm.name])\ (%$(sa)^\circ):\ %$(fm.nbands)\ \mathrm{%$(sseg)},\ %$(sw)\%\ \mathrm{of\ the\ work\ rate}" :
                                                L"%$(sa)^\circ:\ %$(fm.nbands)\ \mathrm{bands},\ %$(sw)\%\ \mathrm{of\ the\ work\ rate}")
            end
            if method === :fatbox   # segments of no family
                no = count(==(0), rb.segfam); so = @sprintf("%.0f", 100rb.other)
                sno = no == 1 ? "segment" : "segments"
                scatter!(ax, -86, yl(nl - 1), color=c_nofam, strokecolor=ink, strokewidth=1.5, markersize=17, marker=:rect)
                text!(ax, -82, yl(nl - 1), align=(:left, :center), fontsize=22, color=ink,
                      text=L"\mathrm{no\ family:}\ %$(no)\ \mathrm{%$(sno)},\ %$(so)\%\ \mathrm{of\ the\ work\ rate}")
            end
            sd = @sprintf("%.0f", 100rb.diffuse.diss)
            text!(ax, -82, yl(nl), align=(:left, :center), fontsize=22, color=ink,
                  text=L"\mathrm{outside\ the\ bands:}\ %$(sd)\%\ \mathrm{of\ the\ work\ rate}")
            xlims!(ax, -90, 90); ylims!(ax, 0, ytop + dy * nl + 0.02 * ymax)
        end
        Legend(fig[length(θs) + 1, 2], [LineElement(color=ink, linewidth=3), [PolyElement(color=(c_work, 0.35)), LineElement(color=c_work, linewidth=2)]],
               [method === :fatbox ? L"\mathrm{segment\ length\ (Fatbox)}" : L"\mathrm{band\ cells}",
                method === :fatbox ? L"\mathrm{plastic\ work\ rate}\ \dot{\lambda}\,\tau_{II}\ \ \mathrm{(coloured:\ per\ family)}" : L"\mathrm{plastic\ work\ rate}\ \dot{\lambda}\,\tau_{II}"], orientation=:horizontal, nbanks=2, framevisible=false,
               labelsize=25, patchsize=(36, 18), colgap=20, tellwidth=false, valign=:top)
        Colorbar(fig[length(θs) + 1, 1], hm, vertical=false, flipaxis=false, height=14, width=Relative(0.9),
                 label=method === :fatbox ? L"\log_{10}\ \dot{\lambda}\ \mathrm{[s^{-1}]}\ \mathrm{(Fatbox\ segments\ in\ family\ colours,\ white:\ no\ family)}" :
                       L"\log_{10}\ \dot{\lambda}\ \mathrm{[s^{-1}]}\ \mathrm{(lower\ half:\ bands\ in\ family\ colours,\ crossings\ white)}", labelsize=27)
        colsize!(fig.layout, 2, Relative(0.52)); colgap!(fig.layout, 30); rowgap!(fig.layout, 14)
        resize_to_layout!(fig)
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), @sprintf("bands_summary%s_step%04d.png", method === :fatbox ? "_fatbox" : "", step)), fig, px_per_unit=2)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), @sprintf("bands_summary%s_step%04d.pdf", method === :fatbox ? "_fatbox" : "", step)), fig)
end
