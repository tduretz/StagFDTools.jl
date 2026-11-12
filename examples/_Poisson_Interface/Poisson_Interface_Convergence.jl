# Convergence of GFM, xGFM and IIM (orders 1 and 2) for the Poisson interface problem
# (see Poisson_Interface.jl). Writes results/convergence.csv and results/convergence.png.
include(joinpath(@__DIR__, "Poisson_Interface.jl"))
import CairoMakie as cm

OUT = mkpath(joinpath(@__DIR__, "results"))
Ns  = (40, 80, 160, 320, 640)
cases = [(T1, 1.0, 1e3), (T1, 1e3, 1.0), (T2, 1.0, 1e3), (T2, 1e3, 1.0)]
methods = (:standard, :GFM, :xGFM, :IIM1, :IIM2, :IIMz1, :IIMz2)
rows = []
for (T, βm, βp) in cases
    prob = T(βm, βp)
    for N in Ns
        G = setup(prob, N)
        for m in methods
            (m == :standard && T === T2) && continue          # jumps a, b ≠ 0 not representable
            t = @elapsed s = solve(G, m)
            e = errors(G, s, m)
            push!(rows, (test=prob.name[1:2], βm=βm, βp=βp, N=N, method=m, eu=e.u, ef=e.flux, it=s.it,
                         maxrow=s.stats.maxrow, nnz=s.stats.nnz, asym=s.stats.asym, time=t))
            @printf("%s β⁻=%-6g β⁺=%-6g N=%4d %-8s u %.2e  flux %.2e  it %3d  nnz/row %2d  asym %.1e  %.1fs\n",
                    prob.name[1:2], βm, βp, N, m, e.u, e.flux, s.it, s.stats.maxrow, s.stats.asym, t)
        end
    end
end
open(joinpath(OUT, "convergence.csv"), "w") do io
    println(io, join(keys(rows[1]), ","))
    foreach(r -> println(io, join(values(r), ",")), rows)
end

cm.set_theme!(fontsize=15)
fig = cm.Figure(size=(1400, 700))
cols = Dict(:standard => :gray, :GFM => :tomato, :xGFM => :darkorange, :IIM1 => :lightblue, :IIM2 => :steelblue, :IIMz1 => :mediumpurple, :IIMz2 => :purple4)
for (c, (T, βm, βp)) in enumerate(cases)
    name = T(βm, βp).name[1:2] * (βm > βp ? ", stiff inclusion" : ", soft inclusion")
    for (r, (key, lab)) in enumerate(((:eu, "L∞(u) / max|u|  (worse side)"), (:ef, "L∞(flux) / max|flux|  (worse side)")))
        ax = cm.Axis(fig[r, c], xscale=log10, yscale=log10, xlabel=r == 2 ? "N" : "", ylabel=c == 1 ? lab : "",
                     title=r == 1 ? name : "", xticks=collect(Ns))
        for m in methods
            sel = filter(x -> x.test == T(βm, βp).name[1:2] && x.βm == βm && x.method == m, rows)
            isempty(sel) && continue
            cm.scatterlines!(ax, [x.N for x in sel], [getfield(x, key) for x in sel], color=cols[m], label=string(m))
        end
        h = collect(Ns)
        cm.lines!(ax, h, 0.5 .* (h ./ h[1]) .^ -1, color=:black, linestyle=:dash, label="O(h)")
        cm.lines!(ax, h, 0.05 .* (h ./ h[1]) .^ -2, color=:black, linestyle=:dot, label="O(h²)")
        (r, c) == (1, 1) && cm.axislegend(ax, position=:lb, labelsize=11)
    end
end
cm.save(joinpath(OUT, "convergence.png"), fig)
