# Viscous inclusion of Schmid & Podladchikov (2003): xGFM (GCR + Cholesky of the symmetric part)
# against standard FD (phase-averaged viscosity) and the IIM variants (z-form, stress form).
# Writes xGFM_Schmid2003.csv and xGFM_Schmid2003.png next to this file.
include(joinpath(@__DIR__, "xGFM_Stokes.jl"))
include(joinpath(@__DIR__, "IIM_Schmid2003.jl"))
include(joinpath(@__DIR__, "StandardFD_Schmid2003.jl"))
import CairoMakie as cm

function xGFM_Schmid2003(n, params; L=10.0, solver=:gcr)
    nc = (x=n, y=n)
    Δ  = (x=L / n, y=L / n)
    X  = GenerateGrid((min=-L / 2, max=L / 2), (min=-L / 2, max=L / 2), Δ, nc)
    φ  = LevelSet((x, y) -> sqrt(x^2 + y^2) - params.rc, X)
    η  = (neg=params.mc, pos=params.mm)
    Va = (x=[Stokes2D_Schmid2003_V(SVector(X.vx_e.x[i], X.vx_e.y[j]), params)[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
          y=[Stokes2D_Schmid2003_V(SVector(X.vy_e.x[i], X.vy_e.y[j]), params)[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    t  = @elapsed sol = SolveXGFM(φ, X, Δ, η, nc, Va; solver, ϵ=1e-12)
    ic = 2:(n+1)
    P  = sol.p[ic, ic] .- mean(sol.p[ic, ic])
    Pa = [Stokes2D_Schmid2003_p(SVector(X.c_e.x[i], X.c_e.y[j]), params) for i in ic, j in ic]
    Pa = Pa .- mean(Pa)
    ev = maximum(abs.(sol.v.x .- Va.x))
    rows = maximum(diff(sparse(sol.A.M')[:, 1:sol.A.off.p].colptr))
    return (P=P, P_ana=Pa, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))), ev=ev, its=sol.its, maxrow=rows, time=t, X=X)
end

function main()
    rows = []
    for mc in (1e3, 1e-3)
        ps = (mm=1.0, mc=mc, rc=2.0, gr=0.0, er=1.0)
        for n in (20, 40, 80, 160, 320)
            r = xGFM_Schmid2003(n, ps)
            push!(rows, (mc=mc, n=n, method="xGFM", L1=r.ϵP.L1, Linf=r.ϵP.Linf, its=r.its, maxrow=r.maxrow))
            @printf("mc=%-6g n=%4d %-14s L1 %.3e  Linf %.3e  its %3d  nnz/row %2d  %.1fs\n", mc, n, "xGFM", r.ϵP.L1, r.ϵP.Linf, r.its, r.maxrow, r.time)
            r = StandardFD_Schmid2003(n, ps)
            push!(rows, (mc=mc, n=n, method="standard FD", L1=r.ϵP.L1, Linf=r.ϵP.Linf, its=0, maxrow=0))
            @printf("mc=%-6g n=%4d %-14s L1 %.3e  Linf %.3e\n", mc, n, "standard FD", r.ϵP.L1, r.ϵP.Linf)
            n > 160 && continue                       # direct IIM solves at 320 need ~10 GB
            for (lab, kw) in (("IIM z-form 1", (form=:z,)), ("IIM stress 1", (form=:stress,)),
                              ("IIM stress 2", (form=:stress, order=2, fit=:quadratic)))
                r = IIM_Schmid2003(n, ps; kw...)
                K = r.sol.S.K
                nu = size(K, 1) - length(r.P) - 1            # velocity rows (the last row is the Lagrange multiplier)
                push!(rows, (mc=mc, n=n, method=lab, L1=r.ϵP.L1, Linf=r.ϵP.Linf, its=0, maxrow=maximum(diff(sparse(K')[:, 1:nu].colptr))))
                @printf("mc=%-6g n=%4d %-14s L1 %.3e  Linf %.3e  nnz/row %2d\n", mc, n, lab, r.ϵP.L1, r.ϵP.Linf, rows[end].maxrow)
            end
            flush(stdout)
        end
    end
    open(joinpath(@__DIR__, "xGFM_Schmid2003.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end
    cm.set_theme!(fontsize=15)
    fig = cm.Figure(size=(1200, 900))
    cols = Dict("standard FD" => :gray, "xGFM" => :darkorange, "IIM z-form 1" => :mediumpurple,
                "IIM stress 1" => :steelblue, "IIM stress 2" => :navy)
    for (c, mc) in enumerate((1e3, 1e-3)), (r, key) in enumerate((:L1, :Linf))
        ax = cm.Axis(fig[r, c], xscale=log10, yscale=log10, xlabel="n", xticks=[20, 40, 80, 160, 320],
                     ylabel=key == :L1 ? "L1(P)" : "L∞(P)", title=r == 1 ? (mc > 1 ? "stiff inclusion, η_inc/η_mat = 10³" : "soft inclusion, η_inc/η_mat = 10⁻³") : "")
        for m in ("standard FD", "xGFM", "IIM z-form 1", "IIM stress 1", "IIM stress 2")
            sel = filter(x -> x.mc == mc && x.method == m, rows)
            cm.scatterlines!(ax, [x.n for x in sel], [getfield(x, key) for x in sel], color=cols[m], label=m)
        end
        h = [20, 320]
        cm.lines!(ax, h, (key == :L1 ? 0.05 : 0.5) .* (h ./ 20) .^ -1, color=:black, linestyle=:dash, label="O(h)")
        (r, c) == (1, 1) && cm.axislegend(ax, position=:lb, labelsize=12)
    end
    cm.save(joinpath(@__DIR__, "xGFM_Schmid2003.png"), fig)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
