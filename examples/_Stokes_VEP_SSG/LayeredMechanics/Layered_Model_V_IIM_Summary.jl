# Summary figure: τII fields at oblique layer angles (standard FD vs IIM) and mean τII vs θ
# (read from IIM_results/sweep_nc200.csv, written by Layered_Model_V_IIM.jl).
include(joinpath(@__DIR__, "Layered_IIM_lib.jl"))
import CairoMakie as cm

OUT = joinpath(@__DIR__, "IIM_results")
D   = @SMatrix([1.0 0; 0 -1])
η   = (neg=0.2, pos=2.0)
nc  = (x=200, y=200)
θs  = LinRange(0, π, 30)[[4, 6, 11]]          # oblique angles of the sweep (19°, 31°, 62°)

res = [(θ, layered_FD(nc, make_layering(θ), D; η), layered_IIM(nc, make_layering(θ), D; η, order=2, fit=:quadratic)) for θ in θs]

lines = split.(readlines(joinpath(OUT, "sweep_nc200.csv"))[2:end], ",")
θd, τF, τI = (parse.(Float64, getindex.(lines, k)) for k in 1:3)
ε̇bg = sqrt(sum(1 / 2 .* D[:] .^ 2))
τs  = 2 * (0.5 * η.neg + 0.5 * η.pos) * ε̇bg
τw  = 2 * (0.5 / η.neg + 0.5 / η.pos)^(-1) * ε̇bg

cm.set_theme!(fontsize=16)
fig = cm.Figure(size=(1300, 1450))
box = ([-0.3, 0.3, 0.3, -0.3, -0.3], [-0.3, -0.3, 0.3, 0.3, -0.3])
for (c, (θ, rF, rI)) in enumerate(res)
    for (r, (m, name)) in enumerate(((rF, "FD"), (rI, "IIM")))
        ax = cm.Axis(fig[r, c], aspect=cm.DataAspect(), title=@sprintf("%s, θ = %.0f°", name, θ * 180 / π),
                     xticks=[-0.4, 0, 0.4], yticks=[-0.4, 0, 0.4], xticklabelsvisible=r == 2, yticklabelsvisible=c == 1)
        cm.heatmap!(ax, m.x, m.y, m.τII, colormap=:bluesreds, colorrange=(0, 4))
        cm.lines!(ax, box..., color=:black, linestyle=:dash, linewidth=1)
        # zoom on the centre: interface oscillations
        ins = cm.Axis(fig[r, c], width=cm.Relative(0.38), height=cm.Relative(0.38), halign=0.98, valign=0.02,
                      aspect=cm.DataAspect(), limits=((-0.06, 0.06), (-0.06, 0.06)),
                      xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        cm.heatmap!(ins, m.x, m.y, m.τII, colormap=:bluesreds, colorrange=(0, 4))
        cm.translate!(ins.blockscene, 0, 0, 100)
    end
end
cm.Colorbar(fig[1:2, 4], colormap=:bluesreds, limits=(0, 4), label="τII")
ax = cm.Axis(fig[3, 1:3], xlabel="layer angle θ [°]", ylabel="mean τII  (|x|, |y| ≤ 0.3)", xticks=0:30:180)
cm.hlines!(ax, [τs], color=:gray, linestyle=:dash)
cm.hlines!(ax, [τw], color=:gray, linestyle=:dot)
cm.text!(ax, 45, τs, text="Voigt bound (strong)", align=(:center, :top), color=:gray, fontsize=13)
cm.text!(ax, 90, τw, text="Reuss bound (weak)", align=(:center, :bottom), color=:gray, fontsize=13)
cm.scatterlines!(ax, θd, τF, label="standard FD", markersize=8)
cm.scatterlines!(ax, θd, τI, label="IIM", markersize=8)
cm.vlines!(ax, θs .* 180 ./ π, color=(:black, 0.3), linewidth=1)
cm.axislegend(ax, position=(0.5, 0.45))
cm.xlims!(ax, -5, 185)
ax2 = cm.Axis(fig[4, 1:3], xlabel="layer angle θ [°]", ylabel="IIM / FD − 1 [%]", xticks=0:30:180)
cm.hlines!(ax2, [0.0], color=:gray)
cm.scatterlines!(ax2, θd, 100 .* (τI ./ τF .- 1), color=:black, markersize=7)
cm.xlims!(ax2, -5, 185)
cm.rowsize!(fig.layout, 3, cm.Relative(0.25))
cm.rowsize!(fig.layout, 4, cm.Relative(0.12))
cm.Label(fig[0, 1:3], "Layered pure shear, η = 2 / 0.2, nc = 200: standard FD vs immersed interface method", fontsize=18, font=:bold)
cm.save(joinpath(OUT, "summary_FD_vs_IIM.png"), fig, px_per_unit=2)
