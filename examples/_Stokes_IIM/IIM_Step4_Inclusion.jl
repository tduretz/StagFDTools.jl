using StagFDTools, StagFDTools.StokesIIM, CairoMakie, LinearAlgebra, StaticArrays, Printf
import Statistics: mean
import ExactFieldSolutions: Stokes2D_Schmid2003_V, Stokes2D_Schmid2003_p, Stokes2D_Schmid2003_L
include(joinpath(@__DIR__, "StandardFD_Schmid2003.jl"))

# Step 4 of the IIM plan (docs/IIM_plan.md): viscous inclusion of Schmid & Podladchikov (2003)
# solved with the first-order IIM, compared with standard staggered FD (StagFDTools Stokes).

function IIM_Schmid2003(n, params; exact_jumps=false, L=10.0)
    nc = (x=n, y=n)
    Δ  = (x=L / n, y=L / n)
    X  = GenerateGrid((min=-L / 2, max=L / 2), (min=-L / 2, max=L / 2), Δ, nc)
    φ  = LevelSet((x, y) -> sqrt(x^2 + y^2) - params.rc, X)
    η  = (neg=params.mc, pos=params.mm)
    ηof(f) = side(f) < 0 ? η.neg : η.pos
    Va  = (x=[Stokes2D_Schmid2003_V(SVector(X.vx_e.x[i], X.vx_e.y[j]), params)[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
           y=[Stokes2D_Schmid2003_V(SVector(X.vy_e.x[i], X.vy_e.y[j]), params)[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    zbc = (x=ηof.(φ.x) .* Va.x, y=ηof.(φ.y) .* Va.y)

    # Diagnostic: exact jumps instead of jumps reconstructed from z. The exact solution is
    # evaluated at x_Γ projected on the circle: x_Γ lies O(h⁴) off Γ, where v has a kink,
    # and Δη would amplify that offset at high contrast.
    Lin = Stokes2D_Schmid2003_L(SVector(0.0, 0.0), params)
    Je(p) = InterfaceJumps(η.pos - η.neg, Stokes2D_Schmid2003_V(params.rc * normalize(p.x), params), Lin * tangent(p.n), p.n)
    sol = SolveIIM(φ, X, Δ, η, nc, zbc; Jexact=exact_jumps ? Je : nothing)

    ic = 2:(n+1)
    P  = sol.p[ic, ic] .- mean(sol.p[ic, ic])
    Pa = [Stokes2D_Schmid2003_p(SVector(X.c_e.x[i], X.c_e.y[j]), params) for i in ic, j in ic]
    Pa = Pa .- mean(Pa)
    return (P=P, P_ana=Pa, X=X, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))))
end

function main()
    contrasts = [1e-3, 1e3, 1e6]
    ncs       = [20, 40, 80, 160, 320]
    h         = 10.0 ./ ncs
    methods   = (:FD, :IIM, :IIM_exact)

    #--------------------------------------------#
    err = Dict((m, c) => (L1=zeros(length(ncs)), Linf=zeros(length(ncs))) for m in methods, c in contrasts)
    for c in contrasts, (k, n) in enumerate(ncs)
        params = (mm=1.0, mc=c, rc=2.0, gr=0.0, er=1.0)
        r = (FD        = redirect_stdout(() -> StandardFD_Schmid2003(n, params), devnull),
             IIM       = IIM_Schmid2003(n, params),
             IIM_exact = IIM_Schmid2003(n, params; exact_jumps=true))
        for m in methods
            err[(m, c)].L1[k], err[(m, c)].Linf[k] = r[m].ϵP.L1, r[m].ϵP.Linf
        end
        @printf("η_inc/η_mat = %.0e  nc = %4d   Linf(P): FD %.3e  IIM %.3e  IIM exact jumps %.3e   L1(P): FD %.3e  IIM %.3e  IIM exact jumps %.3e\n",
            c, n, (err[(m, c)].Linf[k] for m in methods)..., (err[(m, c)].L1[k] for m in methods)...)
    end

    #--------------------------------------------#
    # Pressure profile and error maps at nc = 80, η_inc/η_mat = 1e3
    n, params = 80, (mm=1.0, mc=1e3, rc=2.0, gr=0.0, er=1.0)
    fd  = redirect_stdout(() -> StandardFD_Schmid2003(n, params), devnull)
    iim = IIM_Schmid2003(n, params)
    xc  = iim.X.c.x
    jrow = n ÷ 2 + 1                                   # cells at y = Δy/2

    Fig = Figure(size=(1500, 1050), fontsize=16)
    cols = Makie.wong_colors()
    ls   = Dict(:FD => :solid, :IIM => :solid, :IIM_exact => :dash)
    mk   = Dict(:FD => :rect, :IIM => :circle, :IIM_exact => :utriangle)
    name = Dict(:FD => "standard FD", :IIM => "IIM", :IIM_exact => "IIM, exact jumps")
    for (col, norm_) in enumerate((:Linf, :L1))
        ax = Axis(Fig[1, col], xscale=log10, yscale=log10, xlabel=L"h",
            ylabel=norm_ == :Linf ? L"\max|P - P^{ex}|" : L"\text{mean}|P - P^{ex}|",
            title=norm_ == :Linf ? L"L_\infty \text{ pressure error}" : L"L_1 \text{ pressure error}")
        for (ic, c) in enumerate(contrasts), m in methods
            scatterlines!(ax, h, getfield(err[(m, c)], norm_), color=cols[ic], linestyle=ls[m], marker=mk[m],
                label="$(name[m]), η_inc/η_mat = 1e$(round(Int, log10(c)))")
        end
        lines!(ax, h, 0.3 .* (h ./ h[1]) .^ 1, color=:gray40, linestyle=:dot, label="O(h)")
        col == 2 && Legend(Fig[1, 3], ax, framevisible=false, labelsize=12, nbanks=1)
    end

    ax = Axis(Fig[2, 1:2], xlabel=L"x", ylabel=L"P", title=L"\text{Pressure along } y = \Delta y/2, \; n_c = 80, \; \eta_{inc}/\eta_{mat} = 10^3")
    lines!(ax, xc, iim.P_ana[:, jrow], color=:black, linewidth=2, label="exact")
    scatterlines!(ax, xc, fd.P[:, jrow], color=cols[1], marker=:rect, markersize=7, label="standard FD")
    scatterlines!(ax, xc, iim.P[:, jrow], color=cols[2], marker=:circle, markersize=7, label="IIM")
    vlines!(ax, [-params.rc, params.rc], color=:gray50, linestyle=:dash)
    xlims!(ax, -4, 4)
    axislegend(ax, position=:rt, framevisible=false)

    θ = range(0, 2π, 200)
    for (k, (r, ttl)) in enumerate(((fd, "standard FD"), (iim, "IIM")))
        axe = Axis(Fig[2, 3][k, 1], aspect=DataAspect(), title="|P − P_ex|, $ttl")
        hm  = heatmap!(axe, xc, xc, log10.(abs.(r.P .- r.P_ana) .+ 1e-12), colormap=:viridis, colorrange=(-4, 1))
        lines!(axe, params.rc .* cos.(θ), params.rc .* sin.(θ), color=:white, linewidth=1)
        limits!(axe, -3, 3, -3, 3)
        hidedecorations!(axe)
        k == 2 && Colorbar(Fig[2, 3][1:2, 2], hm, label=L"\log_{10}|P - P^{ex}|")
    end
    rowsize!(Fig.layout, 1, Relative(0.5))
    save(joinpath(@__DIR__, "IIM_Step4_Inclusion.png"), Fig, px_per_unit=2)
    return nothing
end

main()
