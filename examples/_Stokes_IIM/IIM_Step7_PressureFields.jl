using CairoMakie, Printf
include(joinpath(@__DIR__, "StandardFD_Schmid2003.jl"))
include(joinpath(@__DIR__, "StandardFD_Moutzouris2026.jl"))
include(joinpath(@__DIR__, "IIM_Schmid2003.jl"))
include(joinpath(@__DIR__, "IIM_Moutzouris2026.jl"))

# Pressure fields of the z-form IIM (Landry) and of the stress-form IIM (StagFDTools kernels),
# quadratic fits, order 2, next to standard FD and the exact solution, at three resolutions,
# for the Schmid (2003) circle and the compressible Moutzouris (2026) ellipse. Last column:
# difference between the two IIM forms.

function pressure_fields(title, file, fd, iim, outline; zoom)
    θ   = range(0, 2π, 300)
    Fig = Figure(size=(2000, 1080), fontsize=15)
    local hm, hd
    for (i, n) in enumerate([40, 80, 160])
        res = (fd(n), iim(n, :z), iim(n, :stress))
        X, Pa = res[2].X, res[2].P_ana
        xc, yc = X.c.x, X.c.y
        pmax = maximum(abs, Pa[(abs.(xc) .<= zoom), (abs.(yc) .<= zoom)])
        fields = [("standard FD", res[1].P, @sprintf("L∞ = %.3f", res[1].ϵP.Linf)),
                  ("IIM z-form", res[2].P, @sprintf("L∞ = %.4f", res[2].ϵP.Linf)),
                  ("IIM stress form", res[3].P, @sprintf("L∞ = %.4f", res[3].ϵP.Linf)),
                  ("exact", Pa, "")]
        for (j, (name, P, info)) in enumerate(fields)
            ax = Axis(Fig[i, j], aspect=DataAspect(), title=i == 1 ? name : "", subtitle=info,
                      ylabel=j == 1 ? "nc = $n" : "", ylabelsize=18)
            hm = heatmap!(ax, xc, yc, P, colormap=:vik, colorrange=(-1.1pmax, 1.1pmax), lowclip=:lime, highclip=:magenta)
            lines!(ax, outline[1] .* cos.(θ), outline[2] .* sin.(θ), color=:black, linewidth=0.8)
            limits!(ax, -zoom, zoom, -zoom, zoom)
            hidexdecorations!(ax); hideydecorations!(ax, label=false)
        end
        dP = abs.(res[2].P .- res[3].P)
        ax = Axis(Fig[i, 6], aspect=DataAspect(), title=i == 1 ? "|P_z − P_stress|" : "",
                  subtitle=@sprintf("max = %.4f", maximum(dP)))
        hd = heatmap!(ax, xc, yc, log10.(dP .+ 1e-12), colormap=:viridis, colorrange=(-5, 0))
        lines!(ax, outline[1] .* cos.(θ), outline[2] .* sin.(θ), color=:white, linewidth=0.8)
        limits!(ax, -zoom, zoom, -zoom, zoom)
        hidedecorations!(ax)
    end
    Colorbar(Fig[:, 5], hm, label="P; lime / magenta: more than 10% beyond the exact range")
    Colorbar(Fig[:, 7], hd, label="log₁₀ |P_z − P_stress|")
    Label(Fig[0, :], title, fontsize=20)
    save(joinpath(@__DIR__, file), Fig, px_per_unit=2)
end

function main()
    ps = (mm=1.0, mc=1e3, rc=2.0, gr=0.0, er=1.0)
    pressure_fields("Schmid (2003), incompressible circle, η_inc/η_mat = 10³ (zoom); IIM: quadratic fits, order 2",
        "IIM_Step7_PressureFields_Schmid.png",
        n -> redirect_stdout(() -> StandardFD_Schmid2003(n, ps), devnull),
        (n, form) -> IIM_Schmid2003(n, ps; order=2, fit=:quadratic, form),
        (ps.rc, ps.rc); zoom=3.5)
    pm = (ηm=1.0, ηi=1e3, ξm=2.0, ξi=1e5, ri=0.2, t=2.0, α=π / 6, ε̇=1.0, γ̇=0.0, ζ̇=0.5, ε̇zz=0.0)
    pressure_fields("Moutzouris (2026), compressible ellipse, η_inc/η_mat = 10³ (zoom); IIM: quadratic fits, order 2",
        "IIM_Step7_PressureFields_Ellipse.png",
        n -> redirect_stdout(() -> StandardFD_Moutzouris2026(n, pm; shape=:ellipse), devnull),
        (n, form) -> IIM_Moutzouris2026(n, pm; shape=:ellipse, order=2, fit=:quadratic, form),
        semi_axes(:ellipse, pm); zoom=0.48)
    return nothing
end

main()
