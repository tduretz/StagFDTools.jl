# Viagra benchmark: grid averaging (phase fractions) vs. marker averaging, for iso-strain, iso-stress and iso-work
# (output of run_viagra_v3.jl). Tip deflection history, final plate shape, edge offsets at the tip, wall time.
# Legacy :default (each phase sees the cell strain rate, shared τ0) in black / 4th column.
# julia --project plot_viagra_v3.jl
using CairoMakie, JLD2, Printf, Statistics

let
    dir   = @__DIR__
    rules = (:strain, :stress, :work)
    grid  = Dict(:strain => :voigt, :stress => :reuss, :work => :isowork)
    mark  = Dict(:strain => :markers, :stress => :markers_stress, :work => :markers_work)
    names = Dict(:strain => "iso-strain", :stress => "iso-stress", :work => "iso-work")
    clr   = Dict(:strain => :crimson, :stress => :dodgerblue, :work => :darkorange)
    res = Dict{Symbol,Any}()
    for s in (:default, values(grid)..., values(mark)...)
        f = joinpath(dir, "ViagraMixing_$(s)_51_v3.jld2")
        isfile(f) && (res[s] = load(f))
    end
    timing = Dict{Symbol,Float64}()
    tf = joinpath(dir, "ViagraMixing_timing_v3.txt")
    isfile(tf) && for l in eachline(tf)
        s, t = split(l); timing[Symbol(s)] = parse(Float64, t)
    end

    # Plate edges at the tip (750 < x < 800 km) at the end: top (initially 800 km) and bottom (200 km) [km]
    function edges(d)
        m = d["snaps"]["final"]; L = d["sc_L"] / 1e3
        p = m.c .>= 2; x, y = m.x[p] .* L, m.y[p] .* L
        tip = 750 .< x .< 800
        return maximum(y[tip]) - 800, minimum(y[tip]) - 200
    end

    println("averaging  rule        max tip [km]  final tip [km]  recovered [%]  top edge [km]  bottom edge [km]  wall [s]")
    fig = Figure(size=(1900, 1100), fontsize=16)
    ax = Axis(fig[1, 1:4], xlabel="t [kyr]", ylabel="tip: mean y (x > 750 km) - 500 [km]",
              title="Plate tip (solid: grid averaging with phase fractions, dashed: marker averaging)")
    for r in rules, (lab, s, ls) in (("grid", grid[r], :solid), ("markers", mark[r], :dash))
        haskey(res, s) || continue
        ts = res[s]["ts"]
        lines!(ax, ts.t ./ 1e3, ts.tip, color=clr[r], linestyle=ls, linewidth=2, label="$(names[r]), $lab")
        dmax = ts.tip[argmax(abs.(ts.tip))]
        top, bot = edges(res[s])
        @printf("%-9s  %-10s  %12.2f  %14.2f  %13.1f  %13.2f  %16.2f  %8.0f\n", lab, names[r], dmax, ts.tip[end],
                100 * (1 - ts.tip[end] / dmax), top, bot, get(timing, s, NaN))
    end
    if haskey(res, :default)
        ts = res[:default]["ts"]
        lines!(ax, ts.t ./ 1e3, ts.tip, color=:black, linewidth=2, label="legacy (default)")
        dmax = ts.tip[argmax(abs.(ts.tip))]
        top, bot = edges(res[:default])
        @printf("%-9s  %-10s  %12.2f  %14.2f  %13.1f  %13.2f  %16.2f  %8.0f\n", "grid", "legacy", dmax, ts.tip[end],
                100 * (1 - ts.tip[end] / dmax), top, bot, get(timing, :default, NaN))
    end
    t_off = first(values(res))["t_off"]
    vlines!(ax, [t_off / 1e3], color=:grey, linestyle=:dot)
    axislegend(ax, position=:rb, nbanks=2, labelsize=13)

    checker = cgrad([:lightskyblue, :dodgerblue, :grey70, :grey20], 4, categorical=true)
    for (k, r) in enumerate(rules), (row, s) in enumerate((grid[r], mark[r]))
        haskey(res, s) || continue
        m = res[s]["snaps"]["final"]; L = res[s]["sc_L"] / 1e3
        ts = res[s]["ts"]
        a = Axis(fig[1+row, k], aspect=DataAspect(), limits=(0, 1e3, 0, 1e3), titlesize=13, xlabel=row == 2 ? "x [km]" : "",
                 title=@sprintf("%s, %s: tip %.1f km, %.0f s", names[r], row == 1 ? "grid" : "markers", ts.tip[end], get(timing, s, NaN)))
        scatter!(a, m.x .* L, m.y .* L, color=m.c, colormap=checker, colorrange=(-0.5, 3.5), markersize=2)
        lines!(a, [0, 800, 800, 0, 0], [200, 200, 800, 800, 200], color=:red, linewidth=1.2)
    end
    if haskey(res, :default)
        m = res[:default]["snaps"]["final"]; L = res[:default]["sc_L"] / 1e3
        a = Axis(fig[2, 4], aspect=DataAspect(), limits=(0, 1e3, 0, 1e3), titlesize=13,
                 title=@sprintf("legacy (default): tip %.1f km, %.0f s", res[:default]["ts"].tip[end], get(timing, :default, NaN)))
        scatter!(a, m.x .* L, m.y .* L, color=m.c, colormap=checker, colorrange=(-0.5, 3.5), markersize=2)
        lines!(a, [0, 800, 800, 0, 0], [200, 200, 800, 800, 200], color=:red, linewidth=1.2)
    end
    save(joinpath(dir, "ViagraMixing_grid_vs_markers_v3.png"), fig)
end
