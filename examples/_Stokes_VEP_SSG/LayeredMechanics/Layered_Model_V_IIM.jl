# Layered_Model_V repeated with the immersed interface method (IIM, stress form) next to the
# standard FD discretisation: mean τII over |x|, |y| ≤ 0.3 as a function of the layer angle θ,
# fields at a few angles, and a resolution check at two angles.
# Results (CSV + PNG) go to ./IIM_results. Runs serially (≈ 1 min per IIM solve at nc = 200).
include(joinpath(@__DIR__, "Layered_IIM_lib.jl"))
import CairoMakie as cm

const OUT = mkpath(joinpath(@__DIR__, "IIM_results"))
const D   = @SMatrix([1.0 0; 0 -1])
const η   = (neg=0.2, pos=2.0)          # layer A (inside) weak, layer B strong
writecsv(file, header, cols) = open(joinpath(OUT, file), "w") do io
    println(io, join(header, ","))
    for i in eachindex(cols[1])
        println(io, join((c[i] for c in cols), ","))
    end
end

function sweep(; n=200, nθ=30, iim=(order=2, fit=:quadratic), keep=(30, 45, 60, 90))
    nc = (x=n, y=n)
    θ  = LinRange(0, π, nθ)
    τF, τI = zeros(nθ), zeros(nθ)
    fields = Dict{Int,Any}()
    keepi  = [argmin(abs.(θ .* 180 ./ π .- d)) for d in keep]
    for (i, θi) in enumerate(θ)
        lay = make_layering(θi)
        rF  = layered_FD(nc, lay, D; η)
        rI  = layered_IIM(nc, lay, D; η, iim...)
        τF[i], τI[i] = rF.τmean, rI.τmean
        @printf("θ = %6.2f°  FD %.5f  IIM %.5f  (%+.2f %%)\n", θi * 180 / π, τF[i], τI[i], 100 * (τI[i] / τF[i] - 1))
        i in keepi && (fields[i] = (FD=(τII=rF.τII, p=rF.p, x=rF.x, y=rF.y), IIM=(τII=rI.τII, p=rI.p, x=rI.x, y=rI.y)))
        writecsv("sweep_nc$(n).csv", ("theta_deg", "tau_FD", "tau_IIM"), (θ[1:i] .* 180 ./ π, τF[1:i], τI[1:i]))
    end
    return θ, τF, τI, fields
end

function bounds()
    ε̇bg = sqrt(sum(1 / 2 .* D[:] .^ 2))
    α   = 0.5
    return 2 * (α * η.neg + (1 - α) * η.pos) * ε̇bg, 2 * (α / η.neg + (1 - α) / η.pos)^(-1) * ε̇bg
end

function plot_sweep(θ, τF, τI)
    τs, τw = bounds()
    fig = cm.Figure(size=(800, 500))
    ax  = cm.Axis(fig[1, 1], xlabel="θ [°]", ylabel="mean τII (|x|,|y| ≤ 0.3)")
    d   = θ .* 180 ./ π
    cm.hlines!(ax, [τs, τw], color=:gray, linestyle=:dash)
    cm.scatterlines!(ax, d, τF, label="standard FD")
    cm.scatterlines!(ax, d, τI, label="IIM")
    cm.axislegend(ax, position=:ct)
    ax2 = cm.Axis(fig[2, 1], xlabel="θ [°]", ylabel="IIM / FD − 1 [%]")
    cm.scatterlines!(ax2, d, 100 .* (τI ./ τF .- 1), color=:black)
    cm.rowsize!(fig.layout, 2, cm.Relative(0.3))
    cm.save(joinpath(OUT, "tau_vs_theta.png"), fig)
end

function plot_fields(θ, fields)
    ks  = sort(collect(keys(fields)))
    fig = cm.Figure(size=(380 * length(ks), 1450))
    for (c, k) in enumerate(ks)
        f = fields[k]
        t = @sprintf("%.0f°", θ[k] * 180 / π)
        for (r, (m, name)) in enumerate(((f.FD, "FD"), (f.IIM, "IIM")))
            ax = cm.Axis(fig[r, c], aspect=cm.DataAspect(), title="τII $name, θ = $t")
            cm.heatmap!(ax, m.x, m.y, m.τII, colormap=:bluesreds, colorrange=(0, 4))
            ax = cm.Axis(fig[r+2, c], aspect=cm.DataAspect(), title="P $name (zoom)", limits=((-0.1, 0.1), (-0.1, 0.1)))
            cm.heatmap!(ax, m.x, m.y, m.p .- mean(m.p), colormap=:vik, colorrange=(-3, 3))
        end
        ax = cm.Axis(fig[5, c], aspect=cm.DataAspect(), title="τII IIM − FD")
        cm.heatmap!(ax, f.FD.x, f.FD.y, f.IIM.τII .- f.FD.τII, colormap=:vik, colorrange=(-1, 1))
    end
    cm.save(joinpath(OUT, "fields.png"), fig)
end

function resolution(; angles=(30, 45), nFD=(100, 200, 400), nIIM=(100, 200, 300))
    rows = []
    for a in angles
        lay = make_layering(a * π / 180)
        for n in nFD
            r = layered_FD((x=n, y=n), lay, D; η); push!(rows, (a, "FD", n, r.τmean))
            @printf("θ=%d FD   nc=%d  %.5f\n", a, n, r.τmean)
        end
        for n in nIIM, (lab, kw) in (("IIM1", (order=1, fit=:linear)), ("IIM2", (order=2, fit=:quadratic)))
            r = layered_IIM((x=n, y=n), lay, D; η, kw...); push!(rows, (a, lab, n, r.τmean))
            @printf("θ=%d %s nc=%d  %.5f\n", a, lab, n, r.τmean)
        end
        writecsv("resolution.csv", ("theta_deg", "method", "nc", "tau"), Tuple([r[k] for r in rows] for k in 1:4))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    θ, τF, τI, fields = sweep()
    plot_sweep(θ, τF, τI)
    plot_fields(θ, fields)
    resolution()
end
