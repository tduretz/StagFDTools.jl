# Pressure fields for the Schmid & Podladchikov (2003) inclusion: analytical solution next to
# standard FD, IIM (stress form, order 2) and xGFM, for a stiff and a soft inclusion, plus a
# profile along y ≈ 0. Writes xGFM_Schmid2003_fields.png next to this file.
include(joinpath(@__DIR__, "xGFM_Schmid2003.jl"))

function fields_figure(; n=80, xlim=4.0)
    cm.set_theme!(fontsize=14)
    fig  = cm.Figure(size=(1650, 720))
    θ    = range(0, 2π, 200)
    for (r, mc) in enumerate((1e3, 1e-3))
        ps  = (mm=1.0, mc=mc, rc=2.0, gr=0.0, er=1.0)
        xg  = xGFM_Schmid2003(n, ps)
        fd  = StandardFD_Schmid2003(n, ps)
        iim = IIM_Schmid2003(n, ps; form=:stress, order=2, fit=:quadratic)
        xc  = xg.X.c_e.x[2:n+1]
        pm  = maximum(abs.(xg.P_ana))
        lab = mc > 1 ? "stiff, η_inc/η_mat = 10³" : "soft, η_inc/η_mat = 10⁻³"
        sets = (("analytical", xg.P_ana, nothing), ("standard FD", fd.P, fd.ϵP),
                ("IIM stress 2", iim.P, iim.ϵP), ("xGFM", xg.P, xg.ϵP))
        hm = nothing
        for (c, (name, P, ϵ)) in enumerate(sets)
            ttl = isnothing(ϵ) ? name : @sprintf("%s  (L∞ %.2f)", name, ϵ.Linf)
            ax = cm.Axis(fig[r, c], aspect=cm.DataAspect(), title=ttl,
                         xlabel=r == 2 ? "x" : "", ylabel=c == 1 ? "$lab\ny" : "")
            hm = cm.heatmap!(ax, xc, xc, P, colormap=:RdBu, colorrange=(-pm, pm), highclip=:black, lowclip=:magenta)
            cm.lines!(ax, ps.rc .* cos.(θ), ps.rc .* sin.(θ), color=:black, linewidth=1)
            cm.limits!(ax, -xlim, xlim, -xlim, xlim)
            r == 1 && cm.hidexdecorations!(ax, ticks=false)
            c > 1 && cm.hideydecorations!(ax, ticks=false)
        end
        cm.Colorbar(fig[r, 5], hm, label="P")
        j  = n ÷ 2 + 1                                  # cell row just above y = 0
        ax = cm.Axis(fig[r, 6], title=@sprintf("profile y = %.3f", xc[j]), xlabel=r == 2 ? "x" : "", ylabel="P")
        cm.lines!(ax, xc, xg.P_ana[:, j], color=:black, linewidth=2.5, label="analytical")
        cm.scatterlines!(ax, xc, fd.P[:, j], color=:gray, markersize=5, label="standard FD")
        cm.scatterlines!(ax, xc, iim.P[:, j], color=:navy, markersize=5, label="IIM stress 2")
        cm.scatterlines!(ax, xc, xg.P[:, j], color=:darkorange, markersize=5, label="xGFM")
        cm.vlines!(ax, [-ps.rc, ps.rc], color=:black, linestyle=:dot)
        cm.xlims!(ax, -xlim, xlim)
        cm.ylims!(ax, -1.5pm, 1.5pm)
        r == 1 && cm.axislegend(ax, position=:rt, labelsize=11)
    end
    cm.colsize!(fig.layout, 6, cm.Relative(0.26))
    cm.Label(fig[0, 1:6], "Schmid & Podladchikov (2003) inclusion, pressure at n = $n (zoom |x|,|y| ≤ $xlim; black/magenta = above/below the analytical range)")
    cm.save(joinpath(@__DIR__, "xGFM_Schmid2003_fields.png"), fig)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    fields_figure()
end
