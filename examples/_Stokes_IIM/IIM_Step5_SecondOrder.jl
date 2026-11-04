using StagFDTools, StagFDTools.StokesIIM, CairoMakie, LinearAlgebra, StaticArrays, Printf
import Statistics: mean
import ExactFieldSolutions: Stokes2D_Schmid2003_V, Stokes2D_Schmid2003_p, Stokes2D_Schmid2003_L
include(joinpath(@__DIR__, "StandardFD_Schmid2003.jl"))
include(joinpath(@__DIR__, "IIM_Schmid2003.jl"))

# Step 5 (docs/IIM_plan.md): quadratic least-squares fits for the interface traces and
# second-order IIM corrections, on the viscous inclusion of Schmid & Podladchikov (2003).

function main()
    ncs = [20, 40, 80, 160, 320]
    h   = 10.0 ./ ncs
    variants = (
        FD = (label="standard FD",                       run=(n, p) -> redirect_stdout(() -> StandardFD_Schmid2003(n, p), devnull)),
        A  = (label="IIM: linear fits, order 1",         run=(n, p) -> IIM_Schmid2003(n, p)),
        B  = (label="IIM: quadratic fits, order 1",      run=(n, p) -> IIM_Schmid2003(n, p; fit=:quadratic)),
        C  = (label="IIM: quadratic fits, order 2",      run=(n, p) -> IIM_Schmid2003(n, p; order=2, fit=:quadratic)),
        D  = (label="IIM: order 2, exact jumps",         run=(n, p) -> IIM_Schmid2003(n, p; order=2, fit=:quadratic, exact_jumps=true)),
    )
    cases = [(1e3, (:FD, :A, :B, :C, :D)), (1e-3, (:FD, :A, :C)), (1e6, (:FD, :A, :C))]

    #--------------------------------------------#
    err = Dict{Tuple{Symbol,Float64},NamedTuple}()
    for (c, keys) in cases, k in keys
        params = (mm=1.0, mc=c, rc=2.0, gr=0.0, er=1.0)
        e = (L1=zeros(length(ncs)), Linf=zeros(length(ncs)))
        for (i, n) in enumerate(ncs)
            r = variants[k].run(n, params)
            e.L1[i], e.Linf[i] = r.ϵP.L1, r.ϵP.Linf
        end
        err[(k, c)] = e
        @printf("η_inc/η_mat = %.0e  %-30s Linf(P) = %s\n", c, variants[k].label, join((@sprintf("%.3e", x) for x in e.Linf), "  "))
        @printf("%46s L1(P)   = %s\n", "", join((@sprintf("%.3e", x) for x in e.L1), "  "))
    end

    #--------------------------------------------#
    n, params = 80, (mm=1.0, mc=1e3, rc=2.0, gr=0.0, er=1.0)
    prof = (FD=variants.FD.run(n, params), A=variants.A.run(n, params), C=variants.C.run(n, params))
    xc   = prof.C.X.c.x
    jrow = n ÷ 2 + 1

    Fig  = Figure(size=(1500, 1050), fontsize=16)
    cols = Dict(:FD => Makie.wong_colors()[1], :A => Makie.wong_colors()[2], :B => Makie.wong_colors()[4],
                :C => Makie.wong_colors()[3], :D => :gray30)
    mk   = Dict(1e3 => :circle, 1e-3 => :utriangle, 1e6 => :rect)
    for (col, nrm) in enumerate((:Linf, :L1))
        ax = Axis(Fig[1, col], xscale=log10, yscale=log10, xlabel=L"h",
            ylabel=nrm == :Linf ? L"\max|P - P^{ex}|" : L"\text{mean}|P - P^{ex}|",
            title=nrm == :Linf ? L"L_\infty \text{ pressure error}" : L"L_1 \text{ pressure error}")
        for (c, keys) in cases, k in keys
            scatterlines!(ax, h, getfield(err[(k, c)], nrm), color=cols[k], marker=mk[c],
                linestyle=k == :D ? :dash : :solid,
                label="$(variants[k].label), η_inc/η_mat = 1e$(round(Int, log10(c)))")
        end
        lines!(ax, h, 0.5 .* (h ./ h[1]) .^ 1, color=:black, linestyle=:dot, label="O(h)")
        lines!(ax, h, 0.05 .* (h ./ h[1]) .^ 2, color=:black, linestyle=:dashdot, label="O(h²)")
        col == 2 && Legend(Fig[1, 3], ax, framevisible=false, labelsize=11)
    end

    ax = Axis(Fig[2, 1:2], xlabel=L"x", ylabel=L"P", title=L"\text{Pressure along } y = \Delta y/2, \; n_c = 80, \; \eta_{inc}/\eta_{mat} = 10^3")
    lines!(ax, xc, prof.C.P_ana[:, jrow], color=:black, linewidth=2, label="exact")
    for k in (:FD, :A, :C)
        scatterlines!(ax, xc, prof[k].P[:, jrow], color=cols[k], markersize=6, label=variants[k].label)
    end
    vlines!(ax, [-params.rc, params.rc], color=:gray50, linestyle=:dash)
    xlims!(ax, -4, 4)
    axislegend(ax, position=:rt, framevisible=false, labelsize=13)

    θ = range(0, 2π, 200)
    for (i, k) in enumerate((:A, :C))
        axe = Axis(Fig[2, 3][i, 1], aspect=DataAspect(), title="|P − P_ex|, $(variants[k].label)", titlesize=13)
        hm  = heatmap!(axe, xc, xc, log10.(abs.(prof[k].P .- prof[k].P_ana) .+ 1e-12), colormap=:viridis, colorrange=(-4, 1))
        lines!(axe, params.rc .* cos.(θ), params.rc .* sin.(θ), color=:white, linewidth=1)
        limits!(axe, -3, 3, -3, 3)
        hidedecorations!(axe)
        i == 2 && Colorbar(Fig[2, 3][1:2, 2], hm, label=L"\log_{10}|P - P^{ex}|")
    end
    rowsize!(Fig.layout, 1, Relative(0.5))
    save(joinpath(@__DIR__, "IIM_Step5_SecondOrder.png"), Fig, px_per_unit=2)
    return nothing
end

main()
