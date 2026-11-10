# Stress state of one <prefix>_Tet*_H* run at a saved step:
# (a) split heatmap around the gouge: pressure for x < 0, τII for x > 0, with phase contours and the Mohr-circle probes
# (b) Mohr circles of the probes and of the apparatus, with the rock and gouge yield envelopes (as in ShearApparatus_v9.jl)
# (c) stress points (P, τII) of the rock and gouge cells with the yield surfaces (as in ShearApparatus_v9.jl)
# The P and τII fields are only stored in the checkpoints: the step must have a checkpoint_XXXX.jld2 (by default the last one).
# All text is typeset with LaTeX fonts.
using CairoMakie, JLD2, Printf

load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

let
    root    = dirname(@__DIR__) # run folders are next to postprocessing/
    prefix  = "_ShearApparatus_v9"
    θ       = 30                    # gouge angle of the run
    h       = 3                     # gouge thickness [mm] (folder _H..)
    step    = nothing               # step to show (nothing: last checkpoint)
    margin  = 8.0                   # y range of panel (a): gouge extent ± margin [mm]
    Prange  = nothing               # colour ranges [MPa] (nothing: from the data shown)
    τrange  = nothing
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
    Prange = something(Prange, extrema(filter(!isnan, P[IW, J])))
    τrange = something(τrange, (0.0, maximum(filter(!isnan, τII[IE, J]))))
    δ    = d.probes.t[it] * d.ε̇yy * d.Ly * mm                                    # axial shortening [mm]
    δs   = @sprintf("%.2f", δ)

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
        circ(p) = (-(p.σxx[it] + p.σyy[it])/2, sqrt(((p.σxx[it] - p.σyy[it])/2)^2 + p.σxy[it]^2))
        c1, R1  = circ(d.probe_MC1)
        c2, R2  = circ(d.probe_MC2)
        R3      = -(d.probe_plate.σyy[it] - d.probe_salt.σxx[it])/2
        c3      = -(d.probe_plate.σyy[it] + d.probe_salt.σxx[it])/2
        σn_lim  = (0.0, 1.1 * maximum((c1 + R1, c2 + R2, c3 + R3)))
        σn_ax   = LinRange(σn_lim..., 100)
        lines!(ax, σn_ax .* MPa, (pl.C[1] .+ σn_ax .* pl.sinϕ[1] ./ pl.cosϕ[1]) .* MPa, color=ink, linewidth=1.5, label=L"\mathrm{yield\ rock}")
        lines!(ax, σn_ax .* MPa, (pl.C[2] .+ σn_ax .* pl.sinϕ[2] ./ pl.cosϕ[2]) .* MPa, color=ink, linewidth=1.5, linestyle=:dash, label=L"\mathrm{yield\ gouge}")
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
        cap = get(pl, :cap, nothing)
        incap(P, τ, ph) = cap !== nothing && τ * (cap.py[ph] - cap.pd[ph]) < cap.τd[ph] * (cap.py[ph] - P)
        τ_yield(P, ph) = cap === nothing || P >= cap.pd[ph] ? pl.C[ph] * pl.cosϕ[ph] + P * pl.sinϕ[ph] :
                         P >= cap.Pt[ph] ? sqrt(max(cap.Ry[ph]^2 - (P - cap.py[ph])^2, 0.0)) : NaN # not drawn below Pt
        # Stress state without the viscoplastic overstress (λ̇ηvp): along τ for Drucker-Prager, radial (by λ̇ηvp/a) on the cap
        function on_yield(P, τ, λ̇, ph)
            incap(P, τ, ph) || return P, τ - λ̇ * pl.ηvp[ph]
            Rf = sqrt(τ^2 + (P - cap.py[ph])^2)
            s  = (Rf - λ̇ * pl.ηvp[ph] / cap.a[ph]) / Rf
            return cap.py[ph] + s * (P - cap.py[ph]), s * τ
        end
        Pmin = cap === nothing ? minimum(d.P_rock) : min(minimum(d.P_rock), cap.Pt[1], cap.Pt[2])
        P_ax = LinRange(Pmin, maximum(d.P_rock), 300)
        lines!(ax, P_ax .* MPa, τ_yield.(P_ax, 1) .* MPa, color=ink, linewidth=1.5, label=L"\mathrm{yield\ rock}")
        lines!(ax, P_ax .* MPa, τ_yield.(P_ax, 2) .* MPa, color=ink, linewidth=1.5, linestyle=:dash, label=L"\mathrm{yield\ gouge}")
        ir  = length(d.P_rock) > nrock ? round.(Int, LinRange(1, length(d.P_rock), nrock)) : eachindex(d.P_rock)
        for (P_ph, τ_ph, λ̇_ph, ph, col, lab) in ((d.P_rock[ir], d.τII_rock[ir], d.λ̇_rock[ir], 1, c_rock, L"\mathrm{rock\ cells}"),
                                                 (d.P_gouge, d.τII_gouge, d.λ̇_gouge, 2, c_gouge, L"\mathrm{gouge\ cells}"))
            pts = on_yield.(P_ph, τ_ph, λ̇_ph, ph)
            scatter!(ax, first.(pts) .* MPa, last.(pts) .* MPa, color=(col, 0.5), markersize=5, label=lab, rasterize=3)
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
