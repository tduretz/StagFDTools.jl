# Grid convergence of the thinning-neck test at a FIXED neck width w (complements NeckThinning.jl,
# which sweeps w at fixed n). With w = 0.05, n = 20…160 gives w/Δ = 1…8: the neck goes from subgrid
# to resolved. Reference: FD arithmetic n = 1280 (w/Δ = 64). Also writes velocity (vy) field maps at n = 40.
# Writes NeckThinning_conv.csv, NeckThinning_conv.png, NeckThinning_vfields_eta*.png.
include(joinpath(@__DIR__, "NeckThinning.jl"))
using CairoMakie

function conv(; w=0.05, ηls=(100.0, 10.0), ns=(20, 40, 80, 160), nref=1280, nfield=40, tag="")
    rows = []; vf = Dict()
    for ηl in ηls
        ETA[] = ηl
        t   = @elapsed ref = run_ref(nref, w)
        rr  = neck_rate(ref, w)
        @printf("η = %g  reference n=%d: its %d, %.0fs, rate %.4f\n", ηl, nref, ref.its, t, rr)
        for n in ns, (name, f) in METHODS
            ok = true; local r, c
            try
                r = f(n, w); c = compare(r, ref, w)
            catch e
                ok = false; @printf("    n=%3d %-14s FAILED\n", n, name)
            end
            ok || (push!(rows, (eta=ηl, n=n, method=name, eP1=NaN, ePneck=NaN, evneck=NaN, evfar=NaN, rate=NaN)); continue)
            rate = neck_rate(r, w) / rr
            push!(rows, (eta=ηl, n=n, method=name, eP1=c.eP1, ePneck=c.ePneck, evneck=c.evneck, evfar=c.evfar, rate=rate))
            @printf("    n=%3d %-14s w/Δ %4.1f  L1P %.3e  L∞P neck %.3e  L∞v neck %.2e far %.2e  rate/ref %.3f\n",
                    n, name, w * n, c.eP1, c.ePneck, c.evneck, c.evfar, rate)
            n == nfield && (vf[(ηl, name)] = vfield(r, ref))
        end
        vf[(ηl, "reference")] = vfield(ref, ref)
        flush(stdout)
    end
    open(joinpath(@__DIR__, "NeckThinning_conv$tag.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end
    plot_conv(rows, ns, w, ηls, tag)
    for ηl in ηls
        plot_vfields(vf, ηl, w, nfield)
    end
    return rows
end

# vy on the vy nodes of r and the reference interpolated there (zoom window around the neck)
function vfield(r, ref; zx=0.25, zy=0.2)
    xs, ys = r.X.vy_e.x, r.X.vy_e.y
    ix = findall(x -> abs(x - NECK.x0) <= zx, xs); iy = findall(y -> abs(y) <= zy, ys)
    V  = r.v.y[ix, iy]
    Vr = [bilin(ref.v.y, ref.X.vy_e.x, ref.X.vy_e.y, xs[i], ys[j]) for i in ix, j in iy]
    return (x=xs[ix], y=ys[iy], V=V, E=V .- Vr)
end

function plot_conv(rows, ns, w, ηls, tag)
    fig = Figure(size=(1500, 400 * length(ηls)), fontsize=15)
    hs  = 1 ./ collect(ns)
    for (k, ηl) in enumerate(ηls)
        R = filter(r -> r.eta == ηl, rows)
        for (j, (key, ttl, logy)) in enumerate(((:eP1, "mean |P error|", true), (:evneck, "L∞(v) error, neck zone", true),
                                               (:evfar, "L∞(v) error, far zone", true), (:rate, "thinning rate / reference", false)))
            ax = Axis(fig[k, j]; title=@sprintf("%s, η_layer = %g", ttl, ηl), xlabel="w / Δ", xscale=log2,
                      yscale=logy ? log10 : identity, xticks=(w .* collect(ns), string.(round.(Int, w .* collect(ns)))))
            for (name, _) in METHODS
                S = filter(r -> r.method == name, R)
                y = [getfield(s, key) for s in S]
                logy && (y = max.(y, 1e-12))
                scatterlines!(ax, w .* [s.n for s in S], y; color=COLS[name], label=name)
            end
            if logy
                x = w .* collect(ns); y0 = minimum(filter(isfinite, [getfield(r, key) for r in R if r.n == last(ns)]))
                lines!(ax, x, y0 .* (x ./ x[end]) .^ -1; color=:gray60, linestyle=:dash, label="O(h)")
            else
                hlines!(ax, [1.0]; color=:black, linestyle=:dot); ylims!(ax, 0, 6)
            end
            k == 1 && j == 4 && axislegend(ax; position=:rt, framevisible=false)
        end
    end
    Label(fig[0, :], @sprintf("Neck of fixed width w = %g, grids n = %s (w/Δ = %s); reference %s; P errors > 2Δ from Γ", w, join(ns, ", "), join(round.(Int, w .* ns), ", "), REFNAME[]))
    save(joinpath(@__DIR__, "NeckThinning_conv$tag.png"), fig)
end

function plot_vfields(vf, ηl, w, n)
    names = ["reference", first.(METHODS)...]
    fig   = Figure(size=(380 * length(names), 650), fontsize=14)
    ref   = vf[(ηl, "reference")]
    cl    = 1.05 * maximum(abs, ref.V)
        φ     = neck_levelset(w)
    xs    = range(-0.25, 0.25, 300) .+ NECK.x0
    for (j, name) in enumerate(names)
        haskey(vf, (ηl, name)) || continue
        f = vf[(ηl, name)]
        for (i, (Z, cr, cm, lab)) in enumerate(((f.V, (-cl, cl), :RdBu, "vy"), (log10.(max.(abs.(f.E), 1e-6)), (-3.0, 0.0), :inferno, "log10 |vy − vy_ref|")))
            name == "reference" && i == 2 && continue
            ax = Axis(fig[i, j]; title=i == 1 ? name : @sprintf("error, L∞ %.2g", maximum(abs, f.E)), aspect=DataAspect(),
                      ylabel=j == 1 ? lab : "")
            hm = heatmap!(ax, f.x, f.y, Z; colormap=cm, colorrange=cr, lowclip=:black, highclip=:magenta)
            for s in (1, -1)
                yb = [find_zero_y(φ, x, s) for x in xs]
                lines!(ax, xs, yb; color=:black, linewidth=1)
            end
            j == length(names) && Colorbar(fig[i, j+1], hm; label=lab)
        end
    end
    Label(fig[0, :], @sprintf("vy around the neck, n = %d, w = %g (w/Δ = %g), η_layer/η_matrix = %g (top: vy, black/magenta beyond reference range; bottom: log10 of the error)", n, w, w * n, ηl))
    save(joinpath(@__DIR__, @sprintf("NeckThinning_vfields_eta%g.png", ηl)), fig)
end

# interface position y(x) of the neck (upper s = 1 / lower s = -1) by bisection on φ
function find_zero_y(φ, x, s)
    a, b = NECK.y0, NECK.y0 + s * 0.3
    for _ in 1:60
        m = (a + b) / 2
        (φ(x, m) < 0) ? (a = m) : (b = m)
    end
    return (a + b) / 2
end

if abspath(PROGRAM_FILE) == @__FILE__
    conv()
end
