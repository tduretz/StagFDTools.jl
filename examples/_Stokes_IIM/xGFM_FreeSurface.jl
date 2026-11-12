# Free surface with the xGFM, (a) void: the η_air = 0 limit of the closures with no unknowns in the
# void (AssembleXGFM(...; void)), (b) sticky air: two-phase xGFM with small η_air. Two tests of
# docs/IIM_notes (sections 9-10), against the IIM free surface (ghost extension, no air):
#   1. Schmid & Podladchikov (2003) inviscid inclusion seen through [-0.25, 0.25] × [-0.75, -0.25]
#      (exact solution; pure shear, Dirichlet velocities on all sides, top in the air);
#   2. relaxation of a Gaussian topography under gravity (η = ρ = 1, g = -1, no slip on all sides,
#      air above the surface up to y = 1.5, ρ_air = 0, pressure level: zero mean in the air).
# Writes xGFM_FreeSurface.csv and the figures xGFM_FreeSurface_Schmid.png and
# xGFM_FreeSurface_Gaussian.png next to this file.
include(joinpath(@__DIR__, "xGFM_Stokes.jl"))
include(joinpath(@__DIR__, "IIM_FreeSurface_Schmid.jl"))      # free_surface, sticky_fd, sticky_iim, FreeSurface
import CairoMakie as cm

# ---------------------------------------------------------------- 1. inviscid inclusion
# ηair = 0: void
function sticky_xgfm(n, ηair; ϵ=1e-11)
    nc = (x=n, y=n); Δ = (x=0.5 / n, y=0.5 / n)
    X  = GenerateGrid((min=XLIM[1], max=XLIM[2]), (min=YLIM[1], max=YLIM[2]), Δ, nc)
    φ  = LevelSet((x, y) -> sqrt(x^2 + y^2) - P0.rc, X)
    Va = (x=[va(SVector(X.vx_e.x[i], X.vx_e.y[j]))[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
          y=[va(SVector(X.vy_e.x[i], X.vy_e.y[j]))[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    void = ηair == 0 ? :neg : nothing
    t  = @elapsed sol = SolveXGFM(φ, X, Δ, (neg=ηair == 0 ? 1.0 : ηair, pos=1.0), nc, Va; void, ϵ, maxit=2000)
    ic = 2:(n+1)
    ep = [φ.c[i, j] > 0 && sol.A.num.p[i, j] > 0 ? sol.p[i, j] - pa(SVector(X.c_e.x[i], X.c_e.y[j])) : NaN for i in ic, j in ic]
    isnothing(void) && (ep .-= mean(filter(!isnan, ep)))      # sticky air: pressure up to a constant
    ev = vcat([sol.v.x[I] - Va.x[I] for I in CartesianIndices(φ.x) if sol.A.num.x[I] > 0 && φ.x[I] > 0],
              [sol.v.y[I] - Va.y[I] for I in CartesianIndices(φ.y) if sol.A.num.y[I] > 0 && φ.y[I] > 0])
    return (p=norms(filter(!isnan, ep)), v=norms(ev), map=ep, x=X.c_e.x[ic], y=X.c_e.y[ic], its=sol.its, time=t)
end

# ---------------------------------------------------------------- 2. Gaussian topography
const TOPO = Gaussian(1.0, 0.5, 0.0)

function gaussian_xgfm(ncx, ηair; ϵ=1e-11)     # ηair = 0: void
    Δ  = 6.0 / ncx; ncy = round(Int, 6.5 / Δ)
    nc = (x=ncx, y=ncy); D = (x=Δ, y=Δ)
    X  = GenerateGrid((min=-3.0, max=3.0), (min=-5.0, max=-5.0 + ncy * Δ), D, nc)
    φ  = LevelSet((x, y) -> y - TOPO.h(x), X)
    v0 = (x=zeros(size(φ.x)), y=zeros(size(φ.y)))
    t  = @elapsed sol = SolveXGFM(φ, X, D, (neg=1.0, pos=ηair == 0 ? 1.0 : ηair), nc, v0; ρ=(neg=1.0, pos=0.0), g=SVector(0.0, -1.0),
                                  void=ηair == 0 ? :pos : nothing, pref=:pos, ϵ, maxit=2000)
    ic, jc = 2:(ncx+1), 2:(ncy+1)
    xc, yc = X.c_e.x[ic], X.c_e.y[jc]
    fl  = [y < TOPO.h(x) for x in xc, y in yc]
    vxc = (sol.v.x[2:(ncx+1), 3:(ncy+2)] .+ sol.v.x[3:(ncx+2), 3:(ncy+2)]) ./ 2   # (Vx: x index 2:ncx+2, y 3:ncy+2)
    vyc = (sol.v.y[3:(ncx+2), 2:(ncy+1)] .+ sol.v.y[3:(ncx+2), 3:(ncy+2)]) ./ 2
    m(a) = ifelse.(fl, a, NaN)
    return (x=m(vxc), y=m(vyc), p=m(sol.p[ic, jc]), xc=xc, yc=yc, its=sol.its, time=t)
end

summary(C) = (vx=maximum(abs, filter(!isnan, C.x)), vy=maximum(filter(!isnan, C.y)), vymin=minimum(filter(!isnan, C.y)), p=maximum(filter(!isnan, C.p)))
coarsen(F) = (F[1:2:end, 1:2:end] .+ F[2:2:end, 1:2:end] .+ F[1:2:end, 2:2:end] .+ F[2:2:end, 2:2:end]) ./ 4
# max difference to the reference (finer grid, 4-cell averages, factor 2^k), over common fluid cells
function diff_to(c, f)
    while size(f.p, 1) > size(c.p, 1)
        f = (x=coarsen(f.x), y=coarsen(f.y), p=coarsen(f.p))
    end
    return map(q -> maximum(filter(!isnan, abs.(c[q] .- f[q]))), (x=:x, y=:y, p=:p))
end

function main()
    rows = []
    # 1. inviscid inclusion
    R = Dict{Any,Any}()
    ns, ns_iim = (20, 40, 80, 160, 320), (20, 40, 80, 160)
    for ηa in (0.0, 1e-2, 1e-4, 1e-6), n in ns
        r = R[(:xgfm, ηa, n)] = sticky_xgfm(n, ηa)
        @printf("Schmid xGFM ηa=%.0e n=%4d  P L1 %.2e Linf %.2e | V L1 %.2e Linf %.2e  its %4d  %.1fs\n", ηa, n, r.p.L1, r.p.Linf, r.v.L1, r.v.Linf, r.its, r.time)
        push!(rows, (test="Schmid", method=ηa == 0 ? "xGFM void" : "xGFM sticky", ηair=ηa, n=n, pL1=r.p.L1, pLinf=r.p.Linf, vL1=r.v.L1, vLinf=r.v.Linf, its=r.its))
        flush(stdout)
    end
    for n in ns
        r = R[(:fs, n)] = free_surface(n)
        push!(rows, (test="Schmid", method="IIM free surface", ηair=0.0, n=n, pL1=r.p.L1, pLinf=r.p.Linf, vL1=r.v.L1, vLinf=r.v.Linf, its=0))
        r = R[(:fd, n)] = sticky_fd(n, 1e-4)
        push!(rows, (test="Schmid", method="FD sticky", ηair=1e-4, n=n, pL1=r.p.L1, pLinf=r.p.Linf, vL1=NaN, vLinf=NaN, its=0))
    end
    for n in ns_iim
        r = R[(:iim, n)] = sticky_iim(n, 1e-6)
        push!(rows, (test="Schmid", method="IIM sticky", ηair=1e-6, n=n, pL1=r.p.L1, pLinf=r.p.Linf, vL1=r.v.L1, vLinf=r.v.Linf, its=0))
    end
    for r in rows
        startswith(r.method, "xGFM") || @printf("Schmid %-16s ηa=%.0e n=%4d  P L1 %.2e Linf %.2e\n", r.method, r.ηair, r.n, r.pL1, r.pLinf)
    end

    # 2. Gaussian topography: xGFM against the IIM free surface (no slip)
    G = Dict{Any,Any}()
    ncs = (48, 96, 192, 384)
    for n in ncs
        G[(:iim, n)] = centroid_fields(FreeSurface(n; bc=:no_slip))
    end
    for ηa in (0.0, 1e-3, 1e-6), n in ncs
        G[(:xgfm, ηa, n)] = gaussian_xgfm(n, ηa)
    end
    ref = G[(:iim, 384)]
    for n in ncs
        for (lab, key) in (("IIM free surface", (:iim, n)), ("xGFM void", (:xgfm, 0.0, n)), ("xGFM ηa=1e-3", (:xgfm, 1e-3, n)), ("xGFM ηa=1e-6", (:xgfm, 1e-6, n)))
            C = G[key]; s = summary(C); d = n < 384 ? diff_to(C, ref) : (x=NaN, y=NaN, p=NaN)
            @printf("Gaussian %-17s ncx=%3d  max|Vx| %.5f  max Vy %.5f  min Vy %.5f  max P %.4f | vs IIM 384: ΔVx %.1e ΔVy %.1e ΔP %.1e  its %s\n",
                    lab, n, s.vx, s.vy, s.vymin, s.p, d.x, d.y, d.p, haskey(C, :its) ? string(C.its) : "-")
            push!(rows, (test="Gaussian", method=lab, ηair=key[1] == :iim ? 0.0 : key[2], n=n, pL1=d.p, pLinf=s.p, vL1=d.y, vLinf=s.vymin, its=haskey(C, :its) ? C.its : 0))
        end
        flush(stdout)
    end
    open(joinpath(@__DIR__, "xGFM_FreeSurface.csv"), "w") do io
        println(io, "# Schmid: pL1, pLinf, vL1, vLinf = errors; Gaussian: pL1 = max|ΔP|, vL1 = max|ΔVy| vs IIM free surface ncx=384, pLinf = max P, vLinf = min Vy")
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end
    figures(R, G, ns, ns_iim, ncs)
    return R, G, rows
end

function figures(R, G, ns, ns_iim, ncs)
    cm.set_theme!(fontsize=15)
    # Schmid: convergence and error maps
    fig = cm.Figure(size=(1500, 950))
    h(n) = 0.5 ./ collect(n)
    mk  = Dict(1e-2 => :rect, 1e-4 => :diamond, 1e-6 => :star5)
    for (c, (nrm, lab)) in enumerate(((:L1, "L1"), (:Linf, "L∞")))
        ax = cm.Axis(fig[1, c], xscale=log10, yscale=log10, xlabel="h", ylabel="pressure error ($lab)", title="pressure, $lab")
        cm.scatterlines!(ax, h(ns), [getfield(R[(:fs, n)].p, nrm) for n in ns], color=:black, marker=:circle, label="IIM free surface, cubic fits")
        cm.scatterlines!(ax, h(ns_iim), [getfield(R[(:iim, n)].p, nrm) for n in ns_iim], color=:steelblue, marker=:star5, label="IIM sticky air, η_air = 1e-6")
        cm.scatterlines!(ax, h(ns), [getfield(R[(:fd, n)].p, nrm) for n in ns], color=:gray, marker=:diamond, label="FD sticky air, η_air = 1e-4")
        cm.scatterlines!(ax, h(ns), [getfield(R[(:xgfm, 0.0, n)].p, nrm) for n in ns], color=:red, marker=:circle, linewidth=3,
                         label="xGFM free surface (void)")
        for ηa in (1e-2, 1e-4, 1e-6)
            cm.scatterlines!(ax, h(ns), [getfield(R[(:xgfm, ηa, n)].p, nrm) for n in ns], color=:darkorange, marker=mk[ηa],
                             label=@sprintf("xGFM sticky air, η_air = %.0e", ηa))
        end
        hh = h(ns)
        cm.lines!(ax, hh, (nrm == :L1 ? 0.05 : 1.0) .* (hh ./ hh[1]), color=:black, linestyle=:dash, label="O(h)")
        cm.lines!(ax, hh, (nrm == :L1 ? 0.005 : 0.1) .* (hh ./ hh[1]) .^ 2, color=:black, linestyle=:dot, label="O(h²)")
        c == 1 && cm.Legend(fig[1, 4], ax, framevisible=false, labelsize=12)
    end
    ax = cm.Axis(fig[1, 3], xscale=log10, yscale=log10, xlabel="h", ylabel="velocity error", title="velocity (L1 solid, L∞ dashed)")
    for (nrm, ls) in ((:L1, :solid), (:Linf, :dash))
        cm.scatterlines!(ax, h(ns), [getfield(R[(:fs, n)].v, nrm) for n in ns], color=:black, marker=:circle, linestyle=ls)
        cm.scatterlines!(ax, h(ns), [getfield(R[(:xgfm, 0.0, n)].v, nrm) for n in ns], color=:red, marker=:circle, linestyle=ls)
        cm.scatterlines!(ax, h(ns), [getfield(R[(:xgfm, 1e-6, n)].v, nrm) for n in ns], color=:darkorange, marker=:star5, linestyle=ls)
    end
    for (c, (key, lab)) in enumerate((((:fs, 160), "IIM free surface"), ((:xgfm, 0.0, 160), "xGFM free surface (void)"), ((:xgfm, 1e-6, 160), "xGFM sticky, η_air = 1e-6")))
        r  = R[key]
        ax = cm.Axis(fig[2, c], aspect=cm.DataAspect(), title="log₁₀|p error|, $lab", limits=(XLIM..., YLIM...))
        hm = cm.heatmap!(ax, r.x, r.y, log10.(abs.(r.map)), colormap=:turbo, colorrange=(-7, 0))
        xs = range(-0.25, 0.25, length=200); cm.lines!(ax, xs, -sqrt.(0.25 .- xs .^ 2), color=:black)
        c == 3 && cm.Colorbar(fig[2, 4][1, 1], hm, width=20, height=cm.Relative(0.8), label="n = 160")
    end
    cm.save(joinpath(@__DIR__, "xGFM_FreeSurface_Schmid.png"), fig)

    # Gaussian: fields (ncx = 192) and differences to the IIM free surface
    fig = cm.Figure(size=(1500, 1000))
    xs  = range(-3, 3, length=400)
    for (row, (key, lab)) in enumerate((((:iim, 192), "IIM free surface"), ((:xgfm, 0.0, 192), "xGFM free surface (void)")))
        C = G[key]; xc, yc = G[(:xgfm, 0.0, 192)].xc, G[(:xgfm, 0.0, 192)].yc
        for (k, (q, nm, cr)) in enumerate(((:x, "Vx", (-0.055, 0.055)), (:y, "Vy", (-0.2, 0.1)), (:p, "P", (0.0, 5.0))))
            ax = cm.Axis(fig[row, 2k-1], aspect=cm.DataAspect(), title="$nm, $lab", limits=(-3.2, 3.2, -5.2, 1.3))
            hm = cm.heatmap!(ax, xc, yc, C[q], colormap=:turbo, colorrange=cr)
            cm.lines!(ax, xs, TOPO.h.(xs), color=:black)
            cm.Colorbar(fig[row, 2k], hm)
        end
    end
    C, I = G[(:xgfm, 0.0, 192)], G[(:iim, 192)]
    for (k, (q, nm)) in enumerate(((:x, "Vx"), (:y, "Vy"), (:p, "P")))
        d  = C[q] .- I[q]; m = maximum(filter(!isnan, abs.(d)))
        ax = cm.Axis(fig[3, 2k-1], aspect=cm.DataAspect(), title="$nm: xGFM - IIM (ncx = 192)", limits=(-3.2, 3.2, -5.2, 1.3))
        hm = cm.heatmap!(ax, C.xc, C.yc, d, colormap=:vik, colorrange=(-m, m))
        cm.lines!(ax, xs, TOPO.h.(xs), color=:black)
        cm.Colorbar(fig[3, 2k], hm)
    end
    cm.save(joinpath(@__DIR__, "xGFM_FreeSurface_Gaussian.png"), fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
