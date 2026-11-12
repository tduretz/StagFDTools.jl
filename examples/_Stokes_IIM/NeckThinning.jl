# Thinning neck: how do the sharp-interface methods behave when a stiff layer necks down below
# the grid scale? Stiff layer (η = 100) in a weak matrix (η = 1), unit box, Dirichlet pure shear
# with layer-parallel extension (vx = x, vy = -y). Layer of thickness H with a Gaussian neck of
# minimum width w; the neck width is swept from w = H (no neck) to w = Δ/2 on the coarse grids.
#
# Methods: standard FD (Stokes module, cell fractions as with markers, arithmetic or harmonic
# averaging), IIM stress form order 1 and order 2, xGFM. Reference: xGFM on a fine grid (the
# thinnest neck spans ≥ 8 reference cells). Writes NeckThinning_eta*.csv, NeckThinning_eta*.png
# and NeckThinning_fields_eta*.png next to this file (layer/matrix contrast 100 and 10).
include(joinpath(@__DIR__, "xGFM_Stokes.jl"))
using StagFDTools.Stokes, StagFDTools.Rheology, TimerOutputs
import CairoMakie as cm

const NECK = (H=0.2, σ=0.1, x0=0.0043, y0=0.0071, ηm=1.0)
const ETA  = Ref(100.0)           # layer viscosity (set by main)
# Reference solution. FD arithmetic is used: it converges in the thinning rate (n = 160…1280 agree
# to 1 %, and harmonic FD converges to the same value), whereas xGFM n = 640 is still far from
# converged at high contrast (rate +6 % at contrast 100, +57 % at 1000 for w = 0.05).
const REFNAME = Ref("FD arithmetic n = 1280")
run_ref(n, w) = run_fd(n, w; avg=:arithmetic)

# Level set: φ < 0 in the layer; half thickness h(x), normalised by √(1 + h'²) (≈ distance near Γ)
function neck_levelset(w; H=NECK.H, σ=NECK.σ, x0=NECK.x0, y0=NECK.y0)
    A = (H - w) / 2
    return (x, y) -> begin
        g  = exp(-((x - x0) / σ)^2)
        h  = H / 2 - A * g
        dh = A * g * 2 * (x - x0) / σ^2
        (abs(y - y0) - h) / sqrt(1 + dh^2)
    end
end

grid(n) = (Δ = (x=1.0 / n, y=1.0 / n); nc = (x=n, y=n);
           (nc=nc, Δ=Δ, X=GenerateGrid((min=-0.5, max=0.5), (min=-0.5, max=0.5), Δ, nc)))
pure_shear(X) = (x=[x for x in X.vx_e.x, y in X.vx_e.y], y=[-y for x in X.vy_e.x, y in X.vy_e.y])
ηpair() = (neg=ETA[], pos=NECK.ηm)

function run_xgfm(n, w)
    (; nc, Δ, X) = grid(n)
    φ   = LevelSet(neck_levelset(w), X)
    sol = SolveXGFM(φ, X, Δ, ηpair(), nc, pure_shear(X); ϵ=1e-11, restart=100, maxit=3000)
    return (v=sol.v, p=sol.p, X=X, its=sol.its)
end

function run_iim(n, w; order=1)
    (; nc, Δ, X) = grid(n)
    φ   = LevelSet(neck_levelset(w), X)
    fit = order == 1 ? :linear : :quadratic
    sol = SolveIIMStress(φ, X, Δ, ηpair(), nc, pure_shear(X); order, fit)
    return (v=sol.v, p=sol.p, X=X, its=0)
end

# Standard FD: each centroid / vertex gets the layer fraction of its control volume (sub-sampled,
# as markers would), then arithmetic or harmonic viscosity. The Stokes module averages
# arithmetically; harmonic averaging is obtained exactly with the fraction f' that gives the
# harmonic mean under arithmetic averaging (linear viscosity).
@views function run_fd(n, w; avg=:arithmetic, avg_v=avg, nsub=10)   # avg_v: averaging at vertices
    nc = (x=n, y=n)
    materials = initialize_materials(2; compressible=false)
    materials.η0 .= [NECK.ηm, ETA[]]
    materials.G  .= [1e30, 1e30]
    preprocess!(materials)
    iter_params = IterParams(niter=2, ϵ_nl=1e-10, α=LinRange(0.05, 1.0, 10))
    Δ = (x=1.0 / n, y=1.0 / n, t=1.0)
    a = redirect_stdout(devnull) do
        Allocs(nc, :all_Dirichlet, (min=-0.5, max=0.5), (min=-0.5, max=0.5), Δ, 2)
    end
    BC = (Vx=zeros(size(a.V.x)), Vy=zeros(size(a.V.y)))
    BC.Vx .= a.V.x .= [x for x in a.X.vx_e.x, y in a.X.vx_e.y]
    BC.Vy .= a.V.y .= [-y for x in a.X.vy_e.x, y in a.X.vy_e.y]
    Set_PhaseRatios!(a)
    φf = neck_levelset(w)
    s  = ((1:nsub) .- 0.5) ./ nsub .- 0.5
    function frac(xc, yc, avg)
        f = count(φf(xc + sx * Δ.x, yc + sy * Δ.y) < 0 for sx in s, sy in s) / nsub^2
        avg == :harmonic || return f
        ηh = 1 / (f / ETA[] + (1 - f) / NECK.ηm)
        return (ηh - NECK.ηm) / (ETA[] - NECK.ηm)
    end
    for (pr, xs, ys, av) in ((a.phase_ratios.c, a.X.c_e.x, a.X.c_e.y, avg), (a.phase_ratios.v, a.X.v_e.x, a.X.v_e.y, avg_v))
        @assert size(pr) == (length(xs), length(ys))
        for j in axes(pr, 2), i in axes(pr, 1)
            f = frac(xs[i], ys[j], av)
            pr[i, j] .= (1 - f, f)
        end
    end
    rvec = zeros(length(iter_params.α))
    err  = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    redirect_stdout(devnull) do
        main_loop(a, 1, materials, BC, nc, Δ, TimerOutput(), 2, iter_params, rvec, err)
    end
    return (v=(x=copy(a.V.x), y=copy(a.V.y)), p=copy(a.Pt), X=a.X, its=0)
end

# Bilinear interpolation of f (on the tensor grid xs × ys) at (x, y)
function bilin(f, xs, ys, x, y)
    i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    j = clamp(searchsortedlast(ys, y), 1, length(ys) - 1)
    tx = (x - xs[i]) / (xs[i+1] - xs[i])
    ty = (y - ys[j]) / (ys[j+1] - ys[j])
    return (1 - tx) * (1 - ty) * f[i, j] + tx * (1 - ty) * f[i+1, j] + (1 - tx) * ty * f[i, j+1] + tx * ty * f[i+1, j+1]
end

# Thinning rate of the neck, -(vy(top) - vy(bottom))/w at x0 (far field: 1), from bilinear
# interpolation of vy (what marker advection would see)
function neck_rate(r, w)
    vy(y) = bilin(r.v.y, r.X.vy_e.x, r.X.vy_e.y, NECK.x0, y)
    return -(vy(NECK.y0 + w / 2) - vy(NECK.y0 - w / 2)) / w
end

# Pressure on the physical centroids (mean removed) and reference pressure at the same points:
# mean of the reference centroids of the 2×2 block around the point that are on the same side
function compare(r, ref, w)
    n   = length(r.X.c_e.x) - 2
    ic  = 2:(n+1)
    xc, yc = r.X.c_e.x[ic], r.X.c_e.y[ic]
    φf  = neck_levelset(w)
    P   = r.p[ic, ic]; P = P .- mean(P)
    xr, yr = ref.X.c_e.x, ref.X.c_e.y
    Pr  = similar(P)
    for j in eachindex(yc), i in eachindex(xc)
        s  = φf(xc[i], yc[j]) < 0
        i0 = searchsortedlast(xr, xc[i]); j0 = searchsortedlast(yr, yc[j])
        vals = [ref.p[ii, jj] for ii in i0:i0+1, jj in j0:j0+1 if (φf(xr[ii], yr[jj]) < 0) == s]
        Pr[i, j] = isempty(vals) ? bilin(ref.p, xr, yr, xc[i], yc[j]) : mean(vals)
    end
    Pr  = Pr .- mean(Pr)
    neck = [abs(x - NECK.x0) <= 0.15 for x in xc, y in yc]
    far  = [abs(x - NECK.x0) > 0.15 && abs(x) <= 0.35 for x in xc, y in yc]   # away from the wall/interface corners
    eP  = abs.(P .- Pr)
    # pressure errors only > 2 coarse cells from Γ: the FD reference has O(1) staircase errors at Γ
    away = [abs(φf(x, y)) > 2 / n for x in xc, y in yc]
    neck .&= away; far .&= away
    # velocities on all nodes inside the box
    ev  = (neck=0.0, far=0.0)
    for (f, xs, ys, fr, xsr, ysr) in ((r.v.x, r.X.vx_e.x, r.X.vx_e.y, ref.v.x, ref.X.vx_e.x, ref.X.vx_e.y),
                                       (r.v.y, r.X.vy_e.x, r.X.vy_e.y, ref.v.y, ref.X.vy_e.x, ref.X.vy_e.y))
        for j in eachindex(ys), i in eachindex(xs)
            (abs(xs[i]) <= 0.5 + 1e-12 && abs(ys[j]) <= 0.5 + 1e-12) || continue
            e = abs(f[i, j] - bilin(fr, xsr, ysr, xs[i], ys[j]))
            ev = abs(xs[i] - NECK.x0) <= 0.15 ? (neck=max(ev.neck, e), far=ev.far) : (neck=ev.neck, far=max(ev.far, e))
        end
    end
    return (ePneck=maximum(eP[neck]), ePfar=maximum(eP[far]), eP1=mean(eP[away]), evneck=ev.neck, evfar=ev.far,
            P=P, Pr=Pr, xc=xc, yc=yc)
end

const METHODS = [("FD arithmetic", (n, w) -> run_fd(n, w; avg=:arithmetic)),
                 ("FD harmonic",   (n, w) -> run_fd(n, w; avg=:harmonic)),
                 ("IIM1",          (n, w) -> run_iim(n, w; order=1)),
                 ("IIM2",          (n, w) -> run_iim(n, w; order=2)),
                 ("xGFM",          run_xgfm)]

function main(; ηl=100.0, nref=1280, ns=(40, 80), wratios=(8, 6, 4, 3, 2, 1.5, 1, 0.75, 0.5), fieldratios=(4, 2, 1, 0.5))
    ETA[]  = ηl
    tag    = @sprintf("_eta%g", ηl)
    rows   = []
    fields = Dict()
    Δ40    = 1.0 / first(ns)
    for wr in wratios
        w   = min(wr * Δ40, NECK.H)
        t   = @elapsed ref = run_ref(nref, w)
        rr  = neck_rate(ref, w)
        @printf("w/Δ%d = %-5g  reference n=%d: its %d, %.0fs, thinning rate %.4f\n", first(ns), wr, nref, ref.its, t, rr)
        push!(rows, (wr=wr, w=w, n=nref, method="reference", ePneck=0.0, ePfar=0.0, eP1=0.0, evneck=0.0, evfar=0.0, rate=rr, its=ref.its, ok=true))
        for n in ns, (name, f) in METHODS
            local r, c
            ok = true
            t = @elapsed try
                r = f(n, w)
                c = compare(r, ref, w)
            catch e
                ok = false
                @printf("    n=%3d %-14s FAILED: %s\n", n, name, sprint(showerror, e)[1:min(end, 120)])
            end
            ok || (push!(rows, (wr=wr, w=w, n=n, method=name, ePneck=NaN, ePfar=NaN, eP1=NaN, evneck=NaN, evfar=NaN, rate=NaN, its=0, ok=false)); continue)
            rate = neck_rate(r, w)
            push!(rows, (wr=wr, w=w, n=n, method=name, ePneck=c.ePneck, ePfar=c.ePfar, eP1=c.eP1, evneck=c.evneck, evfar=c.evfar, rate=rate, its=r.its, ok=true))
            @printf("    n=%3d %-14s w/Δ %5.2f  L∞P neck %.3e far %.3e  L1P %.3e  L∞v neck %.2e far %.2e  rate %.4f (ref %.4f)  its %d  %.1fs\n",
                    n, name, w * n, c.ePneck, c.ePfar, c.eP1, c.evneck, c.evfar, rate, rr, r.its, t)
            n == first(ns) && wr in fieldratios && (fields[(wr, name)] = c)
            n == first(ns) && wr in fieldratios && name == "xGFM" && (fields[(wr, "reference")] = c)
        end
        flush(stdout)
    end
    open(joinpath(@__DIR__, "NeckThinning$tag.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end
    plot_summary(rows, ns, tag)
    plot_fields(fields, fieldratios, tag)
    return rows, fields
end

const COLS = Dict("FD arithmetic" => :gray, "FD harmonic" => :black, "IIM1" => :steelblue, "IIM2" => :navy, "xGFM" => :darkorange)

function plot_summary(rows, ns, tag)
    cm.set_theme!(fontsize=14)
    fig = cm.Figure(size=(1400, 820))
    panels = ((:ePneck, "L∞(P) error, neck zone |x - x₀| ≤ 0.15", true), (:ePfar, "L∞(P) error, 0.15 < |x - x₀|, |x| ≤ 0.35", true),
              (:evneck, "L∞(v) error, neck zone", true), (:rate, "neck thinning rate / reference (clipped to [0, 8])", false))
    for (k, (key, ttl, logy)) in enumerate(panels)
        ax = cm.Axis(fig[(k-1)÷2+1, (k-1)%2+1], xscale=log2, yscale=logy ? log10 : identity, xlabel="w / Δ (minimum neck width in cells)",
                     title=ttl, xticks=[0.5, 1, 2, 4, 8, 16], xreversed=true)
        for (name, _) in METHODS, (s, n) in enumerate(ns)
            sel = filter(r -> r.method == name && r.n == n && r.ok && r.w < NECK.H, rows)   # the straight layer (no neck) is exact
            isempty(sel) && continue
            x = [r.w * n for r in sel]
            y = if key == :rate
                [r.rate / only(filter(q -> q.method == "reference" && q.wr == r.wr, rows)).rate for r in sel]
            else
                [max(getfield(r, key), 1e-6) for r in sel]
            end
            cm.scatterlines!(ax, x, y, color=COLS[name], marker=s == 1 ? :circle : :utriangle, linestyle=s == 1 ? :solid : :dash,
                             label="$name, n = $n")
        end
        key == :rate && (cm.hlines!(ax, [1.0], color=:black, linestyle=:dot); cm.ylims!(ax, 0, 8))
        k == 1 && cm.Legend(fig[1:2, 3], ax, framevisible=false, labelsize=12)
    end
    cm.Label(fig[0, 1:3], "Thinning neck: stiff layer (η = $(ETA[])) in matrix (η = 1), layer-parallel pure shear; reference $(REFNAME[]); P errors > 2Δ from Γ")
    cm.save(joinpath(@__DIR__, "NeckThinning$tag.png"), fig)
end

function plot_fields(fields, fieldratios, tag; zoom=(x=0.25, y=0.2))
    names = ["reference", "FD arithmetic", "FD harmonic", "IIM1", "IIM2", "xGFM"]
    cm.set_theme!(fontsize=13)
    fig = cm.Figure(size=(300 * length(names) + 120, 260 * length(fieldratios) + 60))
    for (r, wr) in enumerate(fieldratios)
        ref = fields[(wr, "reference")]
        pm  = maximum(abs.(ref.Pr[[abs(x - NECK.x0) <= zoom.x for x in ref.xc], [abs(y - NECK.y0) <= zoom.y for y in ref.yc]]))
        φf  = neck_levelset(wr / 40)
        xx  = range(NECK.x0 - zoom.x, NECK.x0 + zoom.x, 400)
        hm  = nothing
        for (c, name) in enumerate(names)
            haskey(fields, (wr, name)) || continue
            d  = fields[(wr, name)]
            P  = name == "reference" ? d.Pr : d.P
            ttl = name == "reference" ? "reference ($(REFNAME[]))" : @sprintf("%s  (L∞ %.2g)", name, maximum(abs.(d.P .- d.Pr)))
            ax = cm.Axis(fig[r, c], aspect=cm.DataAspect(), title=ttl, ylabel=c == 1 ? "w = $(wr)Δ\ny" : "")
            hm = cm.heatmap!(ax, d.xc, d.yc, P, colormap=:RdBu, colorrange=(-pm, pm), highclip=:black, lowclip=:magenta)
            hl = [NECK.H / 2 - (NECK.H - wr / 40) / 2 * exp(-((x - NECK.x0) / NECK.σ)^2) for x in xx]
            cm.lines!(ax, xx, NECK.y0 .+ hl, color=:black, linewidth=1)
            cm.lines!(ax, xx, NECK.y0 .- hl, color=:black, linewidth=1)
            cm.limits!(ax, NECK.x0 - zoom.x, NECK.x0 + zoom.x, NECK.y0 - zoom.y, NECK.y0 + zoom.y)
            cm.hidexdecorations!(ax, ticks=false); c > 1 && cm.hideydecorations!(ax, ticks=false)
        end
        cm.Colorbar(fig[r, length(names)+1], hm, label="P")
    end
    cm.Label(fig[0, 1:length(names)], "Pressure around the neck, n = 40, η_layer/η_matrix = $(ETA[]) (black/magenta: above/below the reference range)")
    cm.save(joinpath(@__DIR__, "NeckThinning_fields$tag.png"), fig)
end

# Redraw the summary figure from a CSV written by main
function replot(ηl; ns=(40, 80))
    ETA[] = ηl
    tag   = @sprintf("_eta%g", ηl)
    lines = readlines(joinpath(@__DIR__, "NeckThinning$tag.csv"))
    hdr   = Symbol.(split(lines[1], ","))
    parse_field(k, v) = k == :method ? String(v) : k == :ok ? v == "true" : k in (:n, :its) ? parse(Int, v) : parse(Float64, v)
    rows  = [NamedTuple{Tuple(hdr)}(Tuple(parse_field(k, v) for (k, v) in zip(hdr, split(l, ",")))) for l in lines[2:end]]
    plot_summary(rows, ns, tag)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ηl=100.0)
    main(ηl=10.0)
end
