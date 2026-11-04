using CairoMakie, Printf
include(joinpath(@__DIR__, "StandardFD_Schmid2003.jl"))
include(joinpath(@__DIR__, "IIM_Schmid2003.jl"))

# Pressure fields of the viscous inclusion (Schmid & Podladchikov, 2003) at three resolutions:
# standard FD (left), the IIM variants, and the exact solution (right).

function main()
    params = (mm=1.0, mc=1e3, rc=2.0, gr=0.0, er=1.0)
    ncs    = [40, 80, 160]
    methods = [
        ("standard FD",             (n, p) -> redirect_stdout(() -> StandardFD_Schmid2003(n, p), devnull)),
        ("IIM: linear, order 1",    (n, p) -> IIM_Schmid2003(n, p)),
        ("IIM: quadratic, order 1", (n, p) -> IIM_Schmid2003(n, p; fit=:quadratic)),
        ("IIM: quadratic, order 2", (n, p) -> IIM_Schmid2003(n, p; order=2, fit=:quadratic)),
    ]
    zoom = 3.5
    θ    = range(0, 2π, 300)

    Fig = Figure(size=(1750, 1080), fontsize=15)
    local hm
    for (i, n) in enumerate(ncs)
        res = [m[2](n, params) for m in methods]
        X   = res[end].X
        Pa  = res[end].P_ana
        pmax = maximum(abs, Pa[(abs.(X.c.x) .<= zoom), (abs.(X.c.y) .<= zoom)])
        fields = [(m[1], r.P, @sprintf("L∞ = %.3f", r.ϵP.Linf)) for (m, r) in zip(methods, res)]
        push!(fields, ("exact", Pa, ""))
        for (j, (name, P, info)) in enumerate(fields)
            ax = Axis(Fig[i, j], aspect=DataAspect(), title=i == 1 ? name : "", subtitle=info,
                      ylabel=j == 1 ? "nc = $n" : "", ylabelsize=18)
            hm = heatmap!(ax, X.c.x, X.c.y, P, colormap=:vik, colorrange=(-1.1pmax, 1.1pmax),
                          lowclip=:lime, highclip=:magenta)
            lines!(ax, params.rc .* cos.(θ), params.rc .* sin.(θ), color=:black, linewidth=0.8)
            limits!(ax, -zoom, zoom, -zoom, zoom)
            hidexdecorations!(ax)
            hideydecorations!(ax, label=false)
        end
    end
    Colorbar(Fig[:, length(methods) + 2], hm, label="P (mean removed); lime / magenta: more than 10% beyond the exact range")
    Label(Fig[0, :], "Pressure, η_inc/η_mat = 10³ (zoom on the inclusion)", fontsize=20)
    save(joinpath(@__DIR__, "IIM_PressureFields.png"), Fig, px_per_unit=2)
    return nothing
end

main()
