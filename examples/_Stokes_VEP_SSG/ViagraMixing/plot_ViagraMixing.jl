# Compare the phase mixing schemes on the Viagra test (output of ViagraTest_mixing.jl)
# julia --project plot_ViagraMixing.jl [ncx]
using CairoMakie, JLD2, Printf

function plot_ViagraMixing(; ncx=51, advection=:standard, dir=@__DIR__, schemes=(:default, :voigt, :reuss))
    suffix = "$(ncx)" * (advection === :standard ? "" : "_$(advection)")
    res = Dict{Symbol,Any}()
    for s in schemes
        f = joinpath(dir, "ViagraMixing_$(s)_$(suffix).jld2")
        isfile(f) && (res[s] = load(f))
    end
    isempty(res) && error("no results in $dir")
    clr = Dict(:default => :black, :voigt => :crimson, :reuss => :dodgerblue)
    t_off = first(values(res))["t_off"]

    fig = Figure(size=(1300, 1000), fontsize=16)
    ax1 = Axis(fig[1, 1], xlabel="t [kyr]", ylabel="tip: mean y (x > 750 km) - 500 [km]", title="Plate tip")
    ax2 = Axis(fig[1, 2], xlabel="t [kyr]", ylabel="rms plate displacement [km]", title="Plate markers vs. initial position (approx.)")
    for s in schemes
        haskey(res, s) || continue
        ts = res[s]["ts"]
        lines!(ax1, ts.t ./ 1e3, ts.tip, color=clr[s], label=string(s))
        lines!(ax2, ts.t ./ 1e3, ts.rms, color=clr[s], label=string(s))
    end
    for ax in (ax1, ax2)
        vlines!(ax, [t_off / 1e3], color=:grey, linestyle=:dash)
    end
    axislegend(ax1, position=:rb)

    # Final marker snapshots (plate colours only)
    checker = cgrad([:lightskyblue, :dodgerblue, :grey70, :grey20], 4, categorical=true)
    for (k, s) in enumerate(schemes)
        haskey(res, s) || continue
        m = res[s]["snaps"]["final"]
        L = res[s]["sc_L"] / 1e3
        ts = res[s]["ts"]
        imax = argmax(abs.(ts.tip))
        ax = Axis(fig[2, 1:2][1, k], aspect=DataAspect(), limits=(0, 1e3, 0, 1e3), titlesize=14,
            title=@sprintf("%s (final): tip %.1f km, max %.1f km", s, ts.tip[end], ts.tip[imax]), xlabel="x [km]", ylabel=k == 1 ? "y [km]" : "")
        scatter!(ax, m.x .* L, m.y .* L, color=m.c, colormap=checker, colorrange=(-0.5, 3.5), markersize=2)
        lines!(ax, [0, 800, 800, 0, 0], [200, 200, 800, 800, 200], color=:red, linewidth=1.5) # initial plate
    end

    # Summary table
    println("scheme    max tip defl. [km]   final tip [km]   recovered [%]   final rms [km]")
    for s in schemes
        haskey(res, s) || continue
        ts = res[s]["ts"]
        dmax = ts.tip[argmax(abs.(ts.tip))]
        @printf("%-8s  %16.3f   %14.3f   %12.1f   %14.3f\n", s, dmax, ts.tip[end], 100 * (1 - ts.tip[end] / dmax), ts.rms[end])
    end
    Label(fig[0, 1:2], "advection: $(advection), $(ncx)×$(ncx)", fontsize=18)
    save(joinpath(dir, "ViagraMixing_$(suffix).png"), fig)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_ViagraMixing(; ncx=length(ARGS) > 0 ? parse(Int, ARGS[1]) : 51, advection=length(ARGS) > 1 ? Symbol(ARGS[2]) : :standard)
end
