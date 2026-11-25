# Stress state of one <prefix>_Tet*_H* run at a saved step:
# (a) split heatmap around the gouge: pressure for x < 0, τII for x > 0, with phase contours and the Mohr-circle probes
# (b) Mohr circles of the probes and of the apparatus, with the rock and gouge yield envelopes (as in ShearApparatus_v9.jl)
# (c) stress points (P, τII) of the rock and gouge cells with the yield surfaces (as in ShearApparatus_v9.jl)
# The P and τII fields are only stored in the checkpoints: the step must have a checkpoint_XXXX.jld2 (by default the last one).
# All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Printf, Statistics

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

let
    root    = joinpath(dirname(@__DIR__), "_HR_hardening_2") # run folder
    prefix  = "_ShearApparatus_v9"
    θ       = 30                    # gouge angle of the run
    h       = 3                     # gouge thickness [mm] (folder _H..)
    step    = nothing               # step to show (nothing: last checkpoint)
    margin  = 8.0                   # y range of panel (a): gouge extent ± margin [mm]
    Prange  = nothing               # colour ranges [MPa] (nothing: percentiles `pclip` of the data shown, so that a few
    τrange  = nothing               #   stress concentrations at corners do not flatten the rest of the map)
    pclip   = (0.02, 0.98)
    nskip   = 10                    # σ1 ticks in (a) every nskip cells
    hmap    = 900                   # height of the map in (a) [px]...
    wmax    = 800                   # ...unless that makes it wider than wmax [px]
    wright  = 600                   # width of panels (b) and (c) [px]
    nrock   = 20_000                # rock points drawn in (c) (random subset; the gouge points are all drawn)

    folder = joinpath(root, @sprintf("%s_Tet%02d_H%02d", prefix, θ, h))
    if step === nothing
        cps  = filter(f -> occursin(r"^checkpoint_\d+\.jld2$", f), readdir(folder))
        step = parse(Int, match(r"\d+", last(sort(cps))).match)
    end
    d   = load_step(joinpath(folder, @sprintf("step_%04d.jld2", step)))
    g   = load(joinpath(folder, @sprintf("checkpoint_%04d.jld2", step)), "state").grid
    it, sc, pl = d.it, d.sc, d.plasticity
    MPa = sc.σ / 1e6
    mm  = sc.L * 1e3

    # Fields on the centroids (without ghost cells)
    x, y = d.xc .* mm, d.yc .* mm
    P    = g.Pt[2:end-1, 2:end-1] .* MPa
    τII  = g.τ.II[2:end-1, 2:end-1] .* MPa
    salt = d.phc .>= 3                                     # salt (and plates): not drawn (white)
    P[salt]   .= NaN
    τII[salt] .= NaN
    ygouge = extrema(y[j] for i in axes(d.phc, 1), j in axes(d.phc, 2) if d.phc[i, j] == 2)
    yw   = (ygouge[1] - margin, ygouge[2] + margin)
    J    = findall(yj -> yw[1] <= yj <= yw[2], y)
    IW, IE = findall(<(0), x), findall(>=(0), x)
    Prange = something(Prange, Tuple(quantile(filter(!isnan, vec(P[IW, J])), pclip)))
    τrange = something(τrange, Tuple(quantile(filter(!isnan, vec(τII[IE, J])), pclip)))
    δ    = d.probes.t[it] * d.ε̇yy * d.Ly * mm                                    # axial shortening [mm]
    δs   = @sprintf("%.2f", δ)

    # Yield parameters of phase ph at accumulated plastic strain εp, with strain hardening ϕ + Δϕ tanh(εp/εh), C + ΔC tanh(εp/εh)
    # (as cap_parameters in src/Rheology.jl). k = sinϕ, c = C cosϕ: Drucker-Prager τII = c + kP; cap: circle (centre py, radius Ry) for P < pd
    cap = get(pl, :cap, nothing)
    function yparams(ph, εp)
        h  = isempty(pl.Δϕ) ? 0.0 : tanh(εp / pl.εh[ph])
        ϕ  = pl.ϕ[ph] + (isempty(pl.Δϕ) ? 0.0 : pl.Δϕ[ph]) * h
        k  = sind(ϕ)
        c  = (pl.C[ph] + (isempty(pl.ΔC) ? 0.0 : pl.ΔC[ph]) * h) * cosd(ϕ)
        cap === nothing && return (; k, c)
        a  = sqrt(1 + k^2)
        Pt = cap.Pt[ph]
        py = (Pt + c / a) / (1 - k / a)
        Ry = py - Pt
        pd = py - Ry * k / a
        return (; k, c, a, Pt, py, Ry, pd, τd = k * pd + c)
    end
    τ_yield(P, y) = cap === nothing || P >= y.pd ? y.c + y.k * P : P >= y.Pt ? sqrt(max(y.Ry^2 - (P - y.py)^2, 0.0)) : NaN # not drawn below Pt
    # Stress state without the viscoplastic overstress ηvp λ̇ (the return mapping solves F = ηvp λ̇): along τII on the Drucker-Prager
    # branch, radially (by ηvp λ̇ / a) on the cap
    function on_yield(P, τ, λ̇, ph, y)
        ηλ = λ̇ * pl.ηvp[ph]
        (cap === nothing || τ * (y.py - y.pd) >= y.τd * (y.py - P)) && return P, τ - ηλ
        Rf = sqrt(τ^2 + (P - y.py)^2)
        f  = (Rf - ηλ / y.a) / Rf
        return y.py + f * (P - y.py), f * τ
    end
    # Rock and gouge cells (one cell away from any other phase, to skip mixed cells), with their εp at the start of the step
    # (hardening is explicit: εp is lagged by one step)
    Pc, τc, λ̇c = g.Pt[2:end-1, 2:end-1], g.τ.II[2:end-1, 2:end-1], g.λ̇.c[2:end-1, 2:end-1]
    εpc = g.εp.c[2:end-1, 2:end-1] .- d.probes.Δt[it] .* λ̇c
    nx, ny = size(d.phc)
    pure(i, j) = all(d.phc[ii, jj] == d.phc[i, j] for ii in max(i-1, 1):min(i+1, nx), jj in max(j-1, 1):min(j+1, ny))
    cells(ph) = [I for I in CartesianIndices(d.phc) if d.phc[I] == ph && pure(I[1], I[2])]
    Irock, Igouge = cells(1), cells(2)
    εg  = [εpc[I] for I in Igouge if λ̇c[I] > 0]                       # εp of the plastic gouge cells
    εq  = quantile(εg, (0.1, 0.5, 0.9))

    # Colours: rock probe, gouge probe, apparatus
    c_rock, c_gouge, c_app = colorant"#e66101", colorant"#5e3c99", colorant"#222222"
    ink, muted = colorant"#222222", colorant"#777777"

    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size=(1400, 1000), fontsize=28)

        # (a) split heatmap
        asp = (x[end] - x[1] + x[2] - x[1]) / (yw[2] - yw[1])  # width / height of the map
        wmap = min(hmap * asp, wmax)                           # tall maps: height hmap; wide maps: width wmax
        ga  = fig[1:2, 1] = GridLayout()
        ax  = Axis(ga[1, 2]; aspect=DataAspect(), height=wmap / asp, width=wmap, xlabel=L"x\ \mathrm{[mm]}", ylabel=L"y\ \mathrm{[mm]}", xticks=-20:10:20,
                   title=L"\mathrm{(a)}\ \ \theta = %$(θ)^\circ,\ \delta = %$(δs)\ \mathrm{mm}", titlealign=:left, titlefont=:regular)
        hmP = heatmap!(ax, x[IW], y[J], P[IW, J], colormap=:lapaz, colorrange=Prange, nan_color=:white)
        hmτ = heatmap!(ax, x[IE], y[J], τII[IE, J], colormap=Reverse(:lajolla), colorrange=τrange, nan_color=:white)
        contour!(ax, x, y[J], Float64.(d.phc[:, J]), levels=[1.5, 2.5], color=:black, linewidth=1.5)
        # σ1 orientation: segments centred on every nskip-th cell
        Is, Js = 1+nskip÷2:nskip:length(x), J[1+nskip÷2:nskip:end]
        ℓ   = 0.7 * nskip * (x[2] - x[1])
        seg = [Point2f(x[i] + sgn * ℓ/2 * cos(d.θc[i, j]), y[j] + sgn * ℓ/2 * sin(d.θc[i, j])) for i in Is for j in Js if !salt[i, j] for sgn in (-1, 1)]
        linesegments!(ax, seg, color=(:white, 0.85), linewidth=1.5)
        vlines!(ax, 0.0, color=:white, linewidth=3)
        # Field labels on small light boxes (marker sizes in pixels)
        scatter!(ax, [Point2f(0.25, 0.955), Point2f(0.75, 0.955)], space=:relative, marker=:rect, markersize=(64, 50), color=(:white, 0.8))
        text!(ax, 0.25, 0.955, text=L"P", space=:relative, align=(:center, :center), color=:black, fontsize=32)
        text!(ax, 0.75, 0.955, text=L"\tau_\mathrm{II}", space=:relative, align=(:center, :center), color=:black, fontsize=32)
        for (p, c, lab) in ((d.probe_MC1, c_rock, "1"), (d.probe_MC2, c_gouge, "2"))
            scatter!(ax, p.x[it] * mm, p.y[it] * mm, color=c, strokecolor=:white, strokewidth=2, markersize=30)
            text!(ax, p.x[it] * mm, p.y[it] * mm, text=L"\mathbf{%$(lab)}", color=:white, fontsize=18, align=(:center, :center))
        end
        limits!(ax, x[1] - (x[2] - x[1])/2, x[end] + (x[2] - x[1])/2, yw...)
        colgap!(ga, 8)
        Colorbar(ga[1, 1], hmP, label=L"P\ \mathrm{[MPa]}", flipaxis=false, width=14)
        Colorbar(ga[1, 3], hmτ, label=L"\tau_\mathrm{II}\ \mathrm{[MPa]}", width=14)

        # (b) Mohr circles (compression positive): centre -(σxx+σyy)/2, radius √(((σxx-σyy)/2)² + σxy²)
        ax  = Axis(fig[1, 2]; aspect=DataAspect(), width=wright, xlabel=L"\sigma_n\ \mathrm{[MPa]}", ylabel=L"\tau\ \mathrm{[MPa]}",
                   title=L"\mathrm{(b)\ Mohr\ circles}", titlealign=:left, titlefont=:regular, topspinevisible=false, rightspinevisible=false,
                   xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08))
        φ   = LinRange(0, π, 200)
        Mohr(c, R) = (c .+ R .* cos.(φ), R .* sin.(φ))
        # Probe circles without the viscoplastic overstress: radius reduced by ηvp λ̇ of the probe cell (Drucker-Prager branch)
        cell(p) = (argmin(abs.(d.xc .- p.x[it])), argmin(abs.(d.yc .- p.y[it])))
        circ(p, ph) = (I = cell(p); (-(p.σxx[it] + p.σyy[it])/2, sqrt(((p.σxx[it] - p.σyy[it])/2)^2 + p.σxy[it]^2) - pl.ηvp[ph] * λ̇c[I...]))
        c1, R1  = circ(d.probe_MC1, 1)
        c2, R2  = circ(d.probe_MC2, 2)
        y1, y2  = yparams(1, εpc[cell(d.probe_MC1)...]), yparams(2, εpc[cell(d.probe_MC2)...]) # yield parameters at the probes
        R3      = -(d.probe_plate.σyy[it] - d.probe_salt.σxx[it])/2
        c3      = -(d.probe_plate.σyy[it] + d.probe_salt.σxx[it])/2
        σn_lim  = (0.0, 1.1 * maximum((c1 + R1, c2 + R2, c3 + R3)))
        σn_ax   = LinRange(σn_lim..., 100)
        # Mohr-Coulomb lines τ = C + σn tanϕ (tangent to the circle of radius c + kP: plane-strain Drucker-Prager), at the probe εp
        MC(y) = (y.c .+ σn_ax .* y.k) ./ sqrt(1 - y.k^2)
        lines!(ax, σn_ax .* MPa, MC(y1) .* MPa, color=ink, linewidth=1.5, label=L"\mathrm{yield\ rock}")
        lines!(ax, σn_ax .* MPa, MC(yparams(2, 0.0)) .* MPa, color=muted, linewidth=1.2, linestyle=:dot, label=L"\mathrm{yield\ gouge},\ \varepsilon_p = 0")
        lines!(ax, σn_ax .* MPa, MC(y2) .* MPa, color=ink, linewidth=1.5, linestyle=:dash, label=L"\mathrm{yield\ gouge\ at\ probe\ 2}")
        for (c, R, col, lab) in ((c1, R1, c_rock, L"\mathrm{probe\ 1\ (rock)}"), (c2, R2, c_gouge, L"\mathrm{probe\ 2\ (gouge)}"), (c3, R3, c_app, L"\mathrm{apparatus}"))
            lines!(ax, (Mohr(c, R) .* MPa)..., color=col, linewidth=2.5, label=lab)
        end
        # Gouge-plane point on the apparatus circle
        scatter!(ax, (R3 * cos(2*(π/2 - d.θgouge)) + c3) * MPa, R3 * sin(2*(π/2 - d.θgouge)) * MPa, color=c_app, markersize=14,
                 label=L"\mathrm{apparatus,\ gouge\ plane}")
        limits!(ax, σn_lim .* MPa..., 0, 0.75 * σn_lim[2] * MPa)
        axislegend(ax, position=:lt, framevisible=true, framecolor=(:white, 0.0), backgroundcolor=(:white, 0.85), labelsize=20, rowgap=0)

        # (c) stress space with the yield surfaces: Drucker-Prager, or with DruckerPragerCap: Drucker-Prager for P ≥ pd, circle (centre py, radius Ry) down to Pt
        ax  = Axis(fig[2, 2]; width=wright, height=0.75wright, xlabel=L"P\ \mathrm{[MPa]}", ylabel=L"\tau_\mathrm{II}\ \mathrm{[MPa]}",
                   title=L"\mathrm{(c)\ Stress\ space}", titlealign=:left, titlefont=:regular, topspinevisible=false, rightspinevisible=false,
                   xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08))
        # Cells (viscoplastic overstress removed, each with its own hardening state)
        ir   = length(Irock) > nrock ? Irock[round.(Int, LinRange(1, length(Irock), nrock))] : Irock
        prock  = [on_yield(Pc[I], τc[I], λ̇c[I], 1, yparams(1, εpc[I])) for I in ir]
        pgouge = [on_yield(Pc[I], τc[I], λ̇c[I], 2, yparams(2, εpc[I])) for I in Igouge]
        Pall = [first.(prock); first.(pgouge)]
        Pmin = cap === nothing ? minimum(Pall) : min(minimum(Pall), cap.Pt[1], cap.Pt[2])
        P_ax = LinRange(Pmin, maximum(Pall), 300)
        # Yield surfaces: rock (no hardening), gouge at εp = 0 and over the εp range of the plastic gouge cells (10-90 %, line: median)
        lines!(ax, P_ax .* MPa, τ_yield.(P_ax, Ref(yparams(1, 0.0))) .* MPa, color=ink, linewidth=1.5, label=L"\mathrm{yield\ rock}")
        lines!(ax, P_ax .* MPa, τ_yield.(P_ax, Ref(yparams(2, 0.0))) .* MPa, color=muted, linewidth=1.2, linestyle=:dot, label=L"\mathrm{yield\ gouge},\ \varepsilon_p = 0")
        band!(ax, P_ax .* MPa, τ_yield.(P_ax, Ref(yparams(2, εq[1]))) .* MPa, τ_yield.(P_ax, Ref(yparams(2, εq[3]))) .* MPa, color=(c_gouge, 0.15))
        εs = @sprintf("%.2f", εq[2])
        lines!(ax, P_ax .* MPa, τ_yield.(P_ax, Ref(yparams(2, εq[2]))) .* MPa, color=ink, linewidth=1.5, linestyle=:dash,
               label=L"\mathrm{yield\ gouge},\ \varepsilon_p = %$(εs)")
        for (pts, col, lab) in ((prock, c_rock, L"\mathrm{rock\ cells}"), (pgouge, c_gouge, L"\mathrm{gouge\ cells}"))
            scatter!(ax, first.(pts) .* MPa, last.(pts) .* MPa, color=(col, 0.5), markersize=5, label=lab, rasterize=3)
        end
        # Check: largest distance above the local yield surface (should be ~0 after removing the overstress)
        for (I, ph, nm) in ((ir, 1, "rock"), (Igouge, 2, "gouge"))
            ex = [(q = on_yield(Pc[i], τc[i], λ̇c[i], ph, yparams(ph, εpc[i])); q[2] - τ_yield(q[1], yparams(ph, εpc[i]))) for i in I]
            @printf("%s cells: max τII above local yield %.3f MPa (before overstress removal: %.3f MPa)\n", nm, maximum(filter(!isnan, ex)) * MPa,
                    maximum(filter(!isnan, [τc[i] - τ_yield(Pc[i], yparams(ph, εpc[i])) for i in I])) * MPa)
        end
        axislegend(ax, position=:lt, framevisible=true, framecolor=(:white, 0.0), backgroundcolor=(:white, 0.85), labelsize=20, rowgap=0)

        colgap!(fig.layout, 12)
        resize_to_layout!(fig)
        fig
    end
    save(joinpath(mkpath(joinpath(@__DIR__, "png")), @sprintf("stress_state_Tet%02d.png", θ)), fig, px_per_unit=3)
    save(joinpath(mkpath(joinpath(@__DIR__, "pdf")), @sprintf("stress_state_Tet%02d.pdf", θ)), fig)
    display(fig)
end
