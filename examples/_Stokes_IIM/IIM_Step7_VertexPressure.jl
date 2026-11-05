using CairoMakie, Printf
import Statistics: mean
include(joinpath(@__DIR__, "IIM_Schmid2003.jl"))
include(joinpath(@__DIR__, "IIM_Moutzouris2026.jl"))

# Pressure at vertices, as needed by the pressure column of 𝐷 (pressure-dependent rheology):
# plain 4-point average of the centroid pressures (Stokes module, Pt_v = av(Pt)) against the
# one-sided average (VertexPressure), on the converged stress-form IIM solutions (quadratic
# fits, order 2) of the Schmid circle and the compressible Moutzouris ellipse, η_inc/η_mat = 10³.

function vertex_errors(r, pexact; meanshift=false)
    X, φ, sol = r.X, r.φ, r.sol
    ic  = 2:(size(φ.c, 1) - 1)
    shift = meanshift ? mean(sol.p[ic, ic]) - mean([pexact(SVector(X.c_e.x[i], X.c_e.y[j]), side(φ.c[i, j])) for i in ic, j in ic]) : 0.0
    Pex = [pexact(SVector(X.v_e.x[i], X.v_e.y[j]), side(φ.v[i, j])) for i in axes(φ.v, 1), j in axes(φ.v, 2)]
    e   = Dict(o => abs.(VertexPressure(sol.S, sol.v, sol.p, φ, X; onesided=o) .- shift .- Pex) for o in (false, true))
    # cut vertices: not all 4 surrounding centroids on the vertex's side
    cut = [3 <= i <= size(φ.v, 1) - 2 && 3 <= j <= size(φ.v, 2) - 2 &&
           any(side(φ.c[i + di, j + dj]) != side(φ.v[i, j]) for di in -1:0, dj in -1:0) for i in axes(φ.v, 1), j in axes(φ.v, 2)]
    ok  = .!isnan.(e[true])
    return (naive=maximum(e[false][cut]), onesided=maximum(e[true][cut]), all=maximum(e[true][ok]), e=e, cut=cut, X=X)
end

function main()
    ncs = [40, 80, 160]
    ps  = (mm=1.0, mc=1e3, rc=2.0, gr=0.0, er=1.0)
    pm  = (ηm=1.0, ηi=1e3, ξm=2.0, ξi=1e5, ri=0.2, t=2.0, α=π / 6, ε̇=1.0, γ̇=0.0, ζ̇=0.5, ε̇zz=0.0)
    a, b = semi_axes(:ellipse, pm)
    # exact pressure on the side of the node (nodes on Γ are snapped to the + side)
    pS(x, s) = Stokes2D_Schmid2003_p(x * (1 + 1e-9s), ps)
    pM(x, s) = Moutzouris(:ellipse)(x * (1 + 1e-9s); params=pm).p
    cases = (
        (title="Schmid circle", L=10.0, run=n -> IIM_Schmid2003(n, ps; order=2, fit=:quadratic, form=:stress), pex=pS, shift=true),
        (title="Moutzouris ellipse", L=1.0, run=n -> IIM_Moutzouris2026(n, pm; shape=:ellipse, order=2, fit=:quadratic, form=:stress), pex=pM, shift=false),
    )
    res = Dict()
    for c in cases, n in ncs
        r = c.run(n)
        E = vertex_errors(r, c.pex; meanshift=c.shift)
        res[(c.title, n)] = E
        @printf("%-18s nc = %3d: max |P_v - P_ex| at cut vertices: plain average %.3e, one-sided %.3e (all vertices %.3e); centroids %.3e\n",
                c.title, n, E.naive, E.onesided, E.all, r.ϵP.Linf)
    end

    Fig = Figure(size=(1500, 520), fontsize=15)
    ax  = Axis(Fig[1, 1], xscale=log10, yscale=log10, xlabel="h / L", ylabel="max |P_v - P_ex| at cut vertices",
               title="Vertex pressure next to Γ")
    for (c, col) in zip(cases, Makie.wong_colors()[1:2])
        h = 1 ./ ncs
        scatterlines!(ax, h, [res[(c.title, n)].naive for n in ncs], color=col, linestyle=:dash, marker=:xcross, label="$(c.title): plain average")
        scatterlines!(ax, h, [res[(c.title, n)].onesided for n in ncs], color=col, label="$(c.title): one-sided")
    end
    Legend(Fig[2, 1], ax, framevisible=false, labelsize=12, nbanks=2)
    E = res[("Moutzouris ellipse", 80)]
    θ = range(0, 2π, 300)
    local hm
    for (j, (o, name)) in enumerate(((false, "plain average"), (true, "one-sided")))
        ax = Axis(Fig[1, j + 1], aspect=DataAspect(), title="log₁₀|P_v - P_ex|, ellipse, nc = 80, $name")
        hm = heatmap!(ax, E.X.v_e.x, E.X.v_e.y, log10.(E.e[o] .+ 1e-12), colormap=:viridis, colorrange=(-4, 1))
        lines!(ax, a .* cos.(θ), b .* sin.(θ), color=:white, linewidth=0.8)
        limits!(ax, -0.48, 0.48, -0.48, 0.48)
        hidedecorations!(ax)
    end
    Colorbar(Fig[1, 4], hm)
    save(joinpath(@__DIR__, "IIM_Step7_VertexPressure.png"), Fig, px_per_unit=2)
    return nothing
end

main()
