# Compressible viscous inclusion of Moutzouris (2026), circle and ellipse (aspect ratio 2, far
# field rotated by π/6): xGFM (GCR + Cholesky of the symmetric part) against standard FD and IIM
# (z-form order 1, stress form orders 1 and 2). Same parameters as IIM_Step6_Compressible.jl.
# Writes xGFM_Moutzouris2026.csv and xGFM_Moutzouris2026.png next to this file.
include(joinpath(@__DIR__, "xGFM_Stokes.jl"))
include(joinpath(@__DIR__, "IIM_Moutzouris2026.jl"))
include(joinpath(@__DIR__, "StandardFD_Moutzouris2026.jl"))
import CairoMakie as cm

function xGFM_Moutzouris2026(n, params; shape=:circle, L=1.0, solver=:gcr)
    nc = (x=n, y=n)
    Δ  = (x=L / n, y=L / n)
    X  = GenerateGrid((min=-L / 2, max=L / 2), (min=-L / 2, max=L / 2), Δ, nc)
    a, b = semi_axes(shape, params)
    φ  = LevelSet((x, y) -> b * (sqrt((x / a)^2 + (y / b)^2) - 1), X)
    η  = (neg=params.ηi, pos=params.ηm)
    ξ  = (neg=params.ξi, pos=params.ξm)
    S(x) = Moutzouris(shape)(x; params)
    Va = (x=[S(SVector(X.vx_e.x[i], X.vx_e.y[j])).V[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
          y=[S(SVector(X.vy_e.x[i], X.vy_e.y[j])).V[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    t  = @elapsed sol = SolveXGFM(φ, X, Δ, η, nc, Va; ξ, solver, ϵ=1e-12, ηb=1e3, restart=100, maxit=2000)
    ic = 2:(n+1)
    P  = sol.p[ic, ic]
    Pa = [S(SVector(X.c_e.x[i], X.c_e.y[j])).p for i in ic, j in ic]
    rows = maximum(diff(sparse(sol.A.M')[:, 1:sol.A.off.p].colptr))
    return (P=P, P_ana=Pa, X=X, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))), its=sol.its, maxrow=rows, time=t)
end

const MOUTZ_BASE  = (ηm=1.0, ξm=2.0, ri=0.2, t=2.0, α=π / 6, ε̇=1.0, γ̇=0.0, ζ̇=0.5, ε̇zz=0.0)
const MOUTZ_CASES = ((label="stiff", p=(; MOUTZ_BASE..., ηi=1e3, ξi=1e5)), (label="weak", p=(; MOUTZ_BASE..., ηi=1e-3, ξi=5e-4)))

function main()
    rows = []
    for shape in (:circle, :ellipse), case in MOUTZ_CASES, n in (20, 40, 80, 160, 320)
        r = xGFM_Moutzouris2026(n, case.p; shape)
        push!(rows, (shape=shape, case=case.label, n=n, method="xGFM", L1=r.ϵP.L1, Linf=r.ϵP.Linf, its=r.its, maxrow=r.maxrow))
        @printf("%-7s %-5s n=%4d %-14s L1 %.3e  Linf %.3e  its %3d  nnz/row %2d  %.1fs\n", shape, case.label, n, "xGFM", r.ϵP.L1, r.ϵP.Linf, r.its, r.maxrow, r.time)
        r = redirect_stdout(() -> StandardFD_Moutzouris2026(n, case.p; shape), devnull)
        push!(rows, (shape=shape, case=case.label, n=n, method="standard FD", L1=r.ϵP.L1, Linf=r.ϵP.Linf, its=0, maxrow=0))
        @printf("%-7s %-5s n=%4d %-14s L1 %.3e  Linf %.3e\n", shape, case.label, n, "standard FD", r.ϵP.L1, r.ϵP.Linf)
        n > 160 && continue                       # direct IIM solves at 320 need ~10 GB
        for (lab, kw) in (("IIM z-form 1", (form=:z,)), ("IIM stress 1", (form=:stress,)),
                          ("IIM stress 2", (form=:stress, order=2, fit=:quadratic)))
            r = IIM_Moutzouris2026(n, case.p; shape, kw...)
            push!(rows, (shape=shape, case=case.label, n=n, method=lab, L1=r.ϵP.L1, Linf=r.ϵP.Linf, its=0, maxrow=0))
            @printf("%-7s %-5s n=%4d %-14s L1 %.3e  Linf %.3e\n", shape, case.label, n, lab, r.ϵP.L1, r.ϵP.Linf)
        end
        flush(stdout)
    end
    open(joinpath(@__DIR__, "xGFM_Moutzouris2026.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end
    plot(rows)
    return rows
end

function plot(rows)
    cm.set_theme!(fontsize=15)
    fig  = cm.Figure(size=(1300, 950))
    cols = Dict("standard FD" => :gray, "xGFM" => :darkorange, "IIM z-form 1" => :mediumpurple,
                "IIM stress 1" => :steelblue, "IIM stress 2" => :navy)
    for (c, (shape, case)) in enumerate(((:circle, "stiff"), (:circle, "weak"), (:ellipse, "stiff"), (:ellipse, "weak")))
        for (r, key) in enumerate((:L1, :Linf))
            ax = cm.Axis(fig[r, c], xscale=log10, yscale=log10, xlabel="n", xticks=[20, 80, 320],
                         ylabel=c == 1 ? (key == :L1 ? "L1(P)" : "L∞(P)") : "",
                         title=r == 1 ? "$shape, $(case == "stiff" ? "η_i/η_m = 10³" : "η_i/η_m = 10⁻³")" : "")
            for m in ("standard FD", "xGFM", "IIM z-form 1", "IIM stress 1", "IIM stress 2")
                sel = filter(x -> x.shape == shape && x.case == case && x.method == m, rows)
                isempty(sel) || cm.scatterlines!(ax, [x.n for x in sel], [getfield(x, key) for x in sel], color=cols[m], label=m)
            end
            h  = [20, 320]
            e0 = minimum(getfield(x, key) for x in rows if x.shape == shape && x.case == case && x.n == 20)
            cm.lines!(ax, h, e0 .* (h ./ 20) .^ -1, color=:black, linestyle=:dash, label="O(h)")
            (r, c) == (2, 1) && cm.axislegend(ax, position=:lb, labelsize=11)
        end
    end
    cm.save(joinpath(@__DIR__, "xGFM_Moutzouris2026.png"), fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
