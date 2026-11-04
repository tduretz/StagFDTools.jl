using StagFDTools, StagFDTools.StokesIIM, CairoMakie, LinearAlgebra, Printf

# Step 1 of the IIM plan (docs/IIM_plan.md): interface geometry for the viscous inclusion
# test of Schmid & Podladchikov (2003): crossings of Γ with the staggered grids,
# normals and irregular equations.

function geometry(nc, φfun; L=(x=10.0, y=10.0))
    x  = (min=-L.x / 2, max=L.x / 2)
    y  = (min=-L.y / 2, max=L.y / 2)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid(x, y, Δ, nc)
    φ  = LevelSet(φfun, X)
    fl = FineLattice(φ, X, Δ)
    Γ  = InterfaceCrossings(fl)
    irr = IrregularNodes(φ, nc)
    return (; X, Δ, φ, fl, Γ, irr)
end

# Keep only interface points inside the physical domain
inside_domain(p, X) = X.v.x[1] <= p.x[1] <= X.v.x[end] && X.v.y[1] <= p.x[2] <= X.v.y[end]

function main()

    rc = 2.0
    φ_dist(x, y) = sqrt(x^2 + y^2) - rc        # signed distance
    φ_tanh(x, y) = tanh(sqrt(x^2 + y^2) - rc)  # same zero level, not a distance function

    #--------------------------------------------#
    # Geometry at a resolution where nodes can be seen
    nc = (x=20, y=20)
    g  = geometry(nc, φ_dist)
    X  = g.X
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c = 2:(nc.x+2), 3:(nc.y+2), 3:(nc.x+2), 2:(nc.y+2), 2:(nc.x+1), 2:(nc.y+1)

    pts = filter(p -> inside_domain(p, X), g.Γ.points)
    @printf("nc = %d: %d interface points, irregular equations: Vx = %d, Vy = %d, Pt = %d, multiple crossings = %d\n",
        nc.x, length(pts), count(g.irr.x), count(g.irr.y), count(g.irr.c), MultipleCrossings(g.Γ))

    #--------------------------------------------#
    # Convergence of crossing positions and normals
    ncs = [10, 20, 40, 80, 160, 320]
    err = (dist=(x=zeros(length(ncs)), n=zeros(length(ncs))), tanh=(x=zeros(length(ncs)), n=zeros(length(ncs))))
    for (k, n) in enumerate(ncs)
        for (key, φfun) in ((:dist, φ_dist), (:tanh, φ_tanh))
            gk = geometry((x=n, y=n), φfun)
            pk = filter(p -> inside_domain(p, gk.X), gk.Γ.points)
            err[key].x[k] = maximum(abs(norm(p.x) - rc) for p in pk)
            err[key].n[k] = maximum(norm(p.n - p.x / norm(p.x)) for p in pk)
        end
    end
    h = 10.0 ./ ncs
    for key in (:dist, :tanh)
        @printf("%s: order position = %.2f, order normal = %.2f\n", key,
            log(err[key].x[end-1] / err[key].x[end]) / log(2), log(err[key].n[end-1] / err[key].n[end]) / log(2))
    end

    #--------------------------------------------#
    # Visualise
    cin, cout = RGBf(0.85, 0.33, 0.10), RGBf(0.00, 0.45, 0.70)
    col(φ) = side(φ) < 0 ? cin : cout
    θ = range(0, 2π, 400)
    circ = Point2f.(rc .* cos.(θ), rc .* sin.(θ))

    Fig = Figure(size=(1600, 600), fontsize=16)

    function plot_grid!(ax; ms=1.0)
        for xv in X.v.x
            lines!(ax, [xv, xv], [X.v.y[1], X.v.y[end]], color=:gray80, linewidth=0.8)
        end
        for yv in X.v.y
            lines!(ax, [X.v.x[1], X.v.x[end]], [yv, yv], color=:gray80, linewidth=0.8)
        end
        poly!(ax, circ, color=(cin, 0.08))
        lines!(ax, circ, color=:black, linewidth=2)
        for (φa, irra, xs, ys, mk) in ((g.φ.c, g.irr.c, X.c_e, (inx_c, iny_c), :circle),
                                         (g.φ.x, g.irr.x, X.vx_e, (inx_Vx, iny_Vx), :rtriangle),
                                         (g.φ.y, g.irr.y, X.vy_e, (inx_Vy, iny_Vy), :utriangle))
            I = CartesianIndices((ys[1], ys[2]))
            P = [Point2f(xs.x[c[1]], xs.y[c[2]]) for c in I]
            C = [col(φa[c]) for c in I]
            R = [irra[c] for c in I]
            scatter!(ax, P[.!R], marker=mk, color=C[.!R], markersize=7ms)
            scatter!(ax, P[R], marker=mk, color=C[R], markersize=11ms, strokecolor=:black, strokewidth=1.5)
        end
        scatter!(ax, [Point2f(p.x) for p in pts], marker=:diamond, color=:gold, markersize=7ms,
            strokecolor=:black, strokewidth=0.8)
    end

    ax1 = Axis(Fig[1, 1], aspect=DataAspect(), title=L"$$\text{Phases, irregular equations (outlined) and } \Gamma \text{ crossings}, n_c = %$(nc.x)")
    plot_grid!(ax1)

    ax2 = Axis(Fig[1, 2], aspect=DataAspect(), title=L"\text{Zoom with normals } n")
    plot_grid!(ax2; ms=2.0)
    arrows2d!(ax2, [Point2f(p.x) for p in pts], [Vec2f(0.35 .* p.n) for p in pts], color=:black, shaftwidth=1.5, tipwidth=7, tiplength=7)
    limits!(ax2, 0.0, 3.0, 0.0, 3.0)

    ax3 = Axis(Fig[1, 3], xscale=log10, yscale=log10, xlabel=L"h", ylabel=L"\text{max error}",
        title=L"\text{Convergence of } x_\Gamma \text{ and } n", aspect=1)
    scatterlines!(ax3, h, err.dist.x, label=L"|x_\Gamma| - r_c,\ \phi = r - r_c")
    scatterlines!(ax3, h, err.tanh.x, label=L"|x_\Gamma| - r_c,\ \phi = \tanh(r - r_c)")
    scatterlines!(ax3, h, err.dist.n, label=L"|n - n_{ex}|,\ \phi = r - r_c")
    scatterlines!(ax3, h, err.tanh.n, label=L"|n - n_{ex}|,\ \phi = \tanh(r - r_c)")
    lines!(ax3, h, 0.5 * err.dist.n[1] .* (h ./ h[1]) .^ 2, color=:gray50, linestyle=:dash, label=L"O(h^2)")
    lines!(ax3, h, 0.5 * err.tanh.x[1] .* (h ./ h[1]) .^ 4, color=:gray50, linestyle=:dot, label=L"O(h^4)")
    axislegend(ax3, position=:rb, framevisible=false, labelsize=12)

    colsize!(Fig.layout, 1, Relative(0.36))
    colsize!(Fig.layout, 2, Relative(0.36))
    display(Fig)
    save(joinpath(@__DIR__, "IIM_Step1_Geometry.png"), Fig, px_per_unit=2)
    return nothing
end

main()
