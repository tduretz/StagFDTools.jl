# Free-surface accuracy test of Duretz et al. (2016, GJI 204, Fig. 3) and of the GMD (2026, 19,
# 8839, Figs 7-9) variational/staircase study: the inviscid circular inclusion of Schmid &
# Podladchikov (2003), radius 0.5 centred at the origin, seen through the box
# [-0.25, 0.25] × [-0.75, -0.25], so that the top of the box is void and the matrix (η = 1) has a
# curved free surface. Velocities on the sides and the bottom from the exact solution; pure
# shear (er = -1, gr = 0), no gravity.
# - free surface: IIM ghost extension (IIM_FreeSurface.jl), Circle geometry, Dirichlet sides,
#                 cubic (default) or quadratic one-sided fits
# - sticky air:   inclusion of viscosity ηair, (a) two-phase IIM (z-form, AssembleIIM, order 2,
#                 cubic or quadratic fits),
#                 (b) standard FD with phase-averaged viscosity (StandardFD_Schmid2003.jl);
#                 Dirichlet velocities of the inviscid solution on all sides (top in the air)
# Errors against the inviscid solution at the centroids (pressure) and faces (velocity) in the
# matrix: L1 = mean |e| (= (1/V)∫|e|dV for equal cells), L∞ = max |e|. With sticky air the
# pressure is defined up to a constant: the mean difference over the matrix is removed.
using StagFDTools, StagFDTools.StokesIIM
import ExactFieldSolutions: Stokes2D_Schmid2003_V, Stokes2D_Schmid2003_p
import Statistics: mean
include(joinpath(@__DIR__, "IIM_FreeSurface.jl"))
include(joinpath(@__DIR__, "StandardFD_Schmid2003.jl"))

const P0   = (mm=1.0, mc=0.0, rc=0.5, gr=0.0, er=-1.0)      # inviscid inclusion
const XLIM = (-0.25, 0.25)
const YLIM = (-0.75, -0.25)
va(x) = Stokes2D_Schmid2003_V(x, P0)
pa(x) = Stokes2D_Schmid2003_p(x, P0)
norms(e) = (L1=mean(abs.(e)), Linf=maximum(abs.(e)))

function free_surface(n; kw...)
    r = FreeSurface(n; topo=Circle(SVector(0.0, 0.0), P0.rc), xmin=XLIM[1], xmax=XLIM[2], ymin=YLIM[1], ytop=YLIM[2],
                    bc=:dirichlet, vbc=va, force=x -> zero(SVector{2,Float64}), kw...)
    err(f, a) = [r.V[f][I] - a(SVector(r.X[f].x[I[1]], r.X[f].y[I[2]])) for I in CartesianIndices(r.num[f]) if r.num[f][I] > 0 && r.φ[f][I] <= 0]
    ep = [r.num.p[I] > 0 ? r.V.p[I] - pa(SVector(r.X.p.x[I[1]], r.X.p.y[I[2]])) : NaN for I in CartesianIndices(r.num.p)]
    return (p=norms(filter(!isnan, ep)), v=norms(vcat(err(:x, x -> va(x)[1]), err(:y, x -> va(x)[2]))), map=ep, x=r.X.p.x, y=r.X.p.y)
end

function sticky_iim(n, ηair; fit=:cubic)
    nc = (x=n, y=n); Δ = (x=0.5 / n, y=0.5 / n)
    X  = GenerateGrid((min=XLIM[1], max=XLIM[2]), (min=YLIM[1], max=YLIM[2]), Δ, nc)
    φ  = LevelSet((x, y) -> sqrt(x^2 + y^2) - P0.rc, X)
    η  = (neg=ηair, pos=1.0)
    ηof(f) = side(f) < 0 ? η.neg : η.pos
    Va = (x=[va(SVector(X.vx_e.x[i], X.vx_e.y[j]))[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
          y=[va(SVector(X.vy_e.x[i], X.vy_e.y[j]))[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    sol = SolveIIM(φ, X, Δ, η, nc, (x=ηof.(φ.x) .* Va.x, y=ηof.(φ.y) .* Va.y); order=2, fit)
    ic = 2:(n+1)
    ep = [φ.c[i, j] > 0 ? sol.p[i, j] - pa(SVector(X.c_e.x[i], X.c_e.y[j])) : NaN for i in ic, j in ic]
    ep .-= mean(filter(!isnan, ep))
    ev = vcat([sol.v.x[I] - Va.x[I] for I in CartesianIndices(φ.x) if sol.S.num.x[I] > 0 && φ.x[I] > 0],
              [sol.v.y[I] - Va.y[I] for I in CartesianIndices(φ.y) if sol.S.num.y[I] > 0 && φ.y[I] > 0])
    return (p=norms(filter(!isnan, ep)), v=norms(ev), map=ep, x=X.c_e.x[ic], y=X.c_e.y[ic])
end

function sticky_fd(n, ηair)
    r  = StandardFD_Schmid2003(n, merge(P0, (mc=ηair,)); xlim=XLIM, ylim=YLIM, bcparams=P0)
    xc, yc = r.X.c_e.x[2:end-1], r.X.c_e.y[2:end-1]
    ep = [sqrt(x^2 + y^2) > P0.rc ? r.Pt[i, j] - r.Pt_ana[i, j] : NaN for (i, x) in enumerate(xc), (j, y) in enumerate(yc)]
    ep .-= mean(filter(!isnan, ep))
    return (p=norms(filter(!isnan, ep)), map=ep, x=xc, y=yc)
end

rate(e, k) = k == 1 ? "    " : @sprintf("%4.2f", log2(e[k-1] / e[k]))

function main()
    ns_fs, ns_iim, ns_fd = (20, 40, 80, 160, 320, 640), (20, 40, 80, 160), (20, 40, 80, 160, 320, 640)
    # variants: free surface with cubic (default) or quadratic fits; sticky air IIM with cubic fits
    # (three air viscosities) or quadratic fits; sticky air FD
    fs  = ((:cubic, (;)), (:quadratic, (degree=2, npts=16)))
    iim = ((:cubic, 1e-2), (:cubic, 1e-4), (:cubic, 1e-6), (:quadratic, 1e-6))
    ηs_fd = (1e-2, 1e-4)
    R = Dict{Any,Any}()
    for (fit, kw) in fs, n in ns_fs
        R[(:fs, fit, n)] = free_surface(n; kw...)
    end
    for (fit, ηa) in iim, n in ns_iim
        R[(:iim, fit, ηa, n)] = sticky_iim(n, ηa; fit)
    end
    for ηa in ηs_fd, n in ns_fd
        R[(:fd, ηa, n)] = sticky_fd(n, ηa)
    end

    # Table (rates between successive resolutions)
    function table(lab, keys, ns)
        println(lab)
        e = [R[k] for k in keys]
        for (k, n) in enumerate(ns)
            @printf("  n = %4d  P L1 %.2e (%s) Linf %.2e (%s)", n, e[k].p.L1, rate([q.p.L1 for q in e], k), e[k].p.Linf, rate([q.p.Linf for q in e], k))
            haskey(e[k], :v) && @printf(" | V L1 %.2e (%s) Linf %.2e (%s)", e[k].v.L1, rate([q.v.L1 for q in e], k), e[k].v.Linf, rate([q.v.Linf for q in e], k))
            println()
        end
    end
    for (fit, _) in fs
        table("free surface IIM, $fit fits", [(:fs, fit, n) for n in ns_fs], ns_fs)
    end
    for (fit, ηa) in iim
        table(@sprintf("sticky air IIM, %s fits, ηair = %.0e", fit, ηa), [(:iim, fit, ηa, n) for n in ns_iim], ns_iim)
    end
    for ηa in ηs_fd
        table(@sprintf("sticky air FD, ηair = %.0e", ηa), [(:fd, ηa, n) for n in ns_fd], ns_fd)
    end

    # Figure
    fig = Figure(size=(1500, 950), fontsize=16)
    h(ns) = 0.5 ./ collect(ns)
    curve!(ax, key, ns, nrm, field; kw...) = scatterlines!(ax, h(ns), [getfield(getfield(R[(key..., n)], field), nrm) for n in ns]; kw...)
    mk = Dict(1e-2 => :rect, 1e-4 => :diamond, 1e-6 => :star5)
    for (c, (nrm, lab)) in enumerate(((:L1, "L1"), (:Linf, "L∞")))
        ax = Axis(fig[1, c], xscale=log10, yscale=log10, xlabel="h", ylabel="pressure error ($lab)", title="pressure, $lab")
        curve!(ax, (:fs, :cubic), ns_fs, nrm, :p; color=:black, marker=:circle, label="IIM free surface, cubic fits")
        curve!(ax, (:fs, :quadratic), ns_fs, nrm, :p; color=:black, marker=:utriangle, linestyle=:dash, label="IIM free surface, quadratic fits")
        for (fit, ηa) in iim
            curve!(ax, (:iim, fit, ηa), ns_iim, nrm, :p; color=:steelblue, marker=mk[ηa], linestyle=fit == :cubic ? :solid : :dash,
                   label=@sprintf("IIM sticky air, %s fits, η_air = %.0e", fit, ηa))
        end
        for ηa in ηs_fd
            curve!(ax, (:fd, ηa), ns_fd, nrm, :p; color=:orange, marker=mk[ηa], label=@sprintf("FD sticky air, η_air = %.0e", ηa))
        end
        hh = h(ns_fs)
        lines!(ax, hh, 0.5 * (hh ./ hh[1]) .* (nrm == :L1 ? 0.1 : 3.0), color=:gray, label="O(h)")
        lines!(ax, hh, 0.5 * (hh ./ hh[1]) .^ 2 .* (nrm == :L1 ? 0.01 : 0.1), color=:gray, linestyle=:dash, label="O(h²)")
        lines!(ax, hh, 0.5 * (hh ./ hh[1]) .^ 3 .* (nrm == :L1 ? 0.01 : 0.1), color=:gray, linestyle=:dot, label="O(h³)")
        c == 1 && Legend(fig[1, 4], ax, framevisible=false, labelsize=13)
    end
    ax = Axis(fig[1, 3], xscale=log10, yscale=log10, xlabel="h", ylabel="velocity error", title="velocity (L1 solid, L∞ dashed)")
    for (nrm, ls) in ((:L1, :solid), (:Linf, :dash))
        curve!(ax, (:fs, :cubic), ns_fs, nrm, :v; color=:black, marker=:circle, linestyle=ls)
        curve!(ax, (:fs, :quadratic), ns_fs, nrm, :v; color=:gray, marker=:utriangle, linestyle=ls)
        curve!(ax, (:iim, :cubic, 1e-6), ns_iim, nrm, :v; color=:steelblue, marker=:star5, linestyle=ls)
    end
    # Error maps at n = 160
    for (c, (key, lab)) in enumerate((((:fs, :cubic, 160), "IIM free surface"), ((:iim, :cubic, 1e-6, 160), "IIM sticky, η_air = 1e-6"),
                                      ((:fd, 1e-4, 160), "FD sticky, η_air = 1e-4")))
        r  = R[key]
        ax = Axis(fig[2, c], aspect=DataAspect(), title="log₁₀|p error|, $lab", limits=(XLIM..., YLIM...))
        hm = heatmap!(ax, r.x, r.y, log10.(abs.(r.map)), colormap=:turbo, colorrange=(-7, 0))
        xs = range(-0.25, 0.25, length=200); lines!(ax, xs, -sqrt.(0.25 .- xs .^ 2), color=:black)
        c == 3 && Colorbar(fig[2, 4][1, 1], hm, width=20, height=Relative(0.8), label="n = 160")
    end
    save(joinpath(@__DIR__, "..", "..", "docs", "IIM_notes", "figures", "IIM_FreeSurface_Schmid.png"), fig)
    return R
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
