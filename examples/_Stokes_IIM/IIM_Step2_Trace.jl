using StagFDTools, StagFDTools.StokesIIM, CairoMakie, LinearAlgebra, StaticArrays, Printf
import ExactFieldSolutions: Stokes2D_Schmid2003_V, Stokes2D_Schmid2003_L

# Step 2 of the IIM plan (docs/IIM_plan.md): interface velocity v_Γ and tangential derivative
# ∂ₜv_Γ from one-sided linear fits of z = ηv, tested on the exact solution of
# Schmid & Podladchikov (2003).

function setup(nc, params; L=(x=10.0, y=10.0))
    x   = (min=-L.x / 2, max=L.x / 2)
    y   = (min=-L.y / 2, max=L.y / 2)
    Δ   = (x=L.x / nc.x, y=L.y / nc.y)
    X   = GenerateGrid(x, y, Δ, nc)
    φ   = LevelSet((x, y) -> sqrt(x^2 + y^2) - params.rc, X)
    Γ   = InterfaceCrossings(FineLattice(φ, X, Δ))
    η   = (neg=params.mc, pos=params.mm)
    tr  = InterfaceTraces(Γ, φ, X, Δ, η, nc)
    return (; X, Δ, φ, η, tr)
end

# z = ηv sampled from the exact solution, η taken from the phase of the node
function exact_z(g, params)
    ηof(φ) = side(φ) < 0 ? g.η.neg : g.η.pos
    Vx = [ηof(g.φ.x[i, j]) * Stokes2D_Schmid2003_V(SVector(g.X.vx_e.x[i], g.X.vx_e.y[j]), params)[1] for i in axes(g.φ.x, 1), j in axes(g.φ.x, 2)]
    Vy = [ηof(g.φ.y[i, j]) * Stokes2D_Schmid2003_V(SVector(g.X.vy_e.x[i], g.X.vy_e.y[j]), params)[2] for i in axes(g.φ.y, 1), j in axes(g.φ.y, 2)]
    return (x=Vx, y=Vy)
end

# Max errors of v_Γ and ∂ₜv_Γ over all crossings and both components
function trace_errors(g, z, params)
    ev, et = 0.0, 0.0
    Lin = Stokes2D_Schmid2003_L(SVector(0.0, 0.0), params)   # inclusion: ∇v is uniform
    for T in g.tr
        V = Stokes2D_Schmid2003_V(T.p.x, params)
        t = tangent(T.p.n)
        for (k, comp) in enumerate((:x, :y))
            vΓ, ∂tvΓ = apply_trace(T[comp], z[comp])
            ev = max(ev, abs(vΓ - V[k]))
            et = max(et, abs(∂tvΓ - dot(Lin[k, :], t)))       # ∂ₜv is continuous across Γ
        end
    end
    return ev, et
end

function main()

    contrasts = [1e-3, 1e3, 1e6]                      # η_inclusion / η_matrix
    ncs       = [20, 40, 80, 160, 320, 640, 1280]
    h         = 10.0 ./ ncs

    #--------------------------------------------#
    # Convergence
    err = Dict(m => (v=zeros(length(ncs)), t=zeros(length(ncs))) for m in contrasts)
    for m in contrasts, (k, n) in enumerate(ncs)
        params = (mm=1.0, mc=m, rc=2.0, gr=0.0, er=1.0)
        g = setup((x=n, y=n), params)
        err[m].v[k], err[m].t[k] = trace_errors(g, exact_z(g, params), params)
    end
    for m in contrasts
        @printf("η_inc/η_mat = %.0e\n", m)
        for k in eachindex(ncs)
            ov = k == 1 ? NaN : log2(err[m].v[k-1] / err[m].v[k])
            ot = k == 1 ? NaN : log2(err[m].t[k-1] / err[m].t[k])
            @printf("  nc = %4d  e_v = %.3e (%5.2f)  e_t = %.3e (%5.2f)\n", ncs[k], err[m].v[k], ov, err[m].t[k], ot)
        end
    end

    #--------------------------------------------#
    # Visualise the selected fits at nc = 20
    nc     = (x=20, y=20)
    params = (mm=1.0, mc=1e3, rc=2.0, gr=0.0, er=1.0)
    g      = setup(nc, params)
    X      = g.X

    cin, cout = RGBf(0.85, 0.33, 0.10), RGBf(0.00, 0.45, 0.70)
    col(φ) = side(φ) < 0 ? cin : cout
    θ    = range(0, 2π, 400)
    circ = Point2f.(params.rc .* cos.(θ), params.rc .* sin.(θ))

    Fig = Figure(size=(1600, 560), fontsize=16)
    ax1 = Axis(Fig[1, 1], aspect=DataAspect(), title="One-sided fits at three crossings, nc = 20")
    for xv in X.v.x
        lines!(ax1, [xv, xv], [X.v.y[1], X.v.y[end]], color=:gray85, linewidth=0.8)
    end
    for yv in X.v.y
        lines!(ax1, [X.v.x[1], X.v.x[end]], [yv, yv], color=:gray85, linewidth=0.8)
    end
    poly!(ax1, circ, color=(cin, 0.08))
    lines!(ax1, circ, color=:black, linewidth=2)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy = 2:(nc.x+2), 3:(nc.y+2), 3:(nc.x+2), 2:(nc.y+2)
    for (φc, Xc, rx, ry, mk) in ((g.φ.x, X.vx_e, inx_Vx, iny_Vx, :rtriangle), (g.φ.y, X.vy_e, inx_Vy, iny_Vy, :utriangle))
        P = [Point2f(Xc.x[i], Xc.y[j]) for i in rx, j in ry]
        C = [(col(φc[i, j]), 0.35) for i in rx, j in ry]
        scatter!(ax1, vec(P), marker=mk, color=vec(C), markersize=9)
    end

    # Three crossings with different normal orientations
    angles = [12, 45, 80]
    sel    = [argmin(T -> abs(atand(T.p.x[2], T.p.x[1]) - a), filter(T -> T.p.x[1] > 0 && T.p.x[2] > 0, g.tr)) for a in angles]
    for T in sel
        for (comp, Xc, ls, mk) in ((:x, X.vx_e, :solid, :rtriangle), (:y, X.vy_e, :dash, :utriangle))
            tr = T[comp]
            pos(I) = Point2f(Xc.x[I[1]], Xc.y[I[2]])
            for (r, c) in ((1:3, cin), (4:6, cout))
                tri = collect(pos.(tr.idx[r]))
                poly!(ax1, tri, color=(c, 0.10), strokecolor=c, strokewidth=1.5, linestyle=ls)
                scatter!(ax1, tri, marker=mk, color=c, markersize=14, strokecolor=:black, strokewidth=1)
            end
            lines!(ax1, Point2f[tr.xo[1], tr.xo[2]], color=:black, linewidth=1)
            scatter!(ax1, Point2f.(collect(tr.xo)), marker=:cross, color=:black, markersize=12)
        end
        scatter!(ax1, Point2f(T.p.x), marker=:star5, color=:gold, markersize=20, strokecolor=:black, strokewidth=1)
    end
    limits!(ax1, -0.2, 3.2, -0.2, 3.2)

    ax2 = Axis(Fig[1, 2], xscale=log10, yscale=log10, xlabel=L"h", ylabel=L"\max |v_\Gamma - v_\Gamma^{ex}|",
        title=L"\text{Interface velocity } v_\Gamma", aspect=1)
    ax3 = Axis(Fig[1, 3], xscale=log10, yscale=log10, xlabel=L"h", ylabel=L"\max |\partial_t v_\Gamma - \partial_t v_\Gamma^{ex}|",
        title=L"\text{Tangential derivative } \partial_t v_\Gamma", aspect=1)
    for m in contrasts
        lab = L"\eta_{inc}/\eta_{mat} = 10^{%$(round(Int, log10(m)))}"
        scatterlines!(ax2, h, err[m].v, label=lab)
        scatterlines!(ax3, h, err[m].t, label=lab)
    end
    m0 = contrasts[1]
    lines!(ax2, h, 0.5 * err[m0].v[1] .* (h ./ h[1]) .^ 2, color=:gray50, linestyle=:dash, label=L"O(h^2)")
    lines!(ax3, h, 0.5 * err[m0].t[1] .* (h ./ h[1]) .^ 1, color=:gray50, linestyle=:dash, label=L"O(h)")
    axislegend(ax2, position=:rb, framevisible=false, labelsize=13)
    ylims!(ax3, 1e-10, 10)
    axislegend(ax3, position=:rb, framevisible=false, labelsize=13)

    colsize!(Fig.layout, 1, Relative(0.36))
    display(Fig)
    save(joinpath(@__DIR__, "IIM_Step2_Trace.png"), Fig, px_per_unit=2)
    return nothing
end

main()
