using CairoMakie, Printf
include(joinpath(@__DIR__, "StandardFD_Moutzouris2026.jl"))
include(joinpath(@__DIR__, "IIM_Moutzouris2026.jl"))

# Step 6 (docs/IIM_plan.md): compressible viscous inclusion of Moutzouris (2026), circle and
# ellipse (aspect ratio 2, far field rotated by π/6), standard FD against IIM variants.

function main()
    ncs = [20, 40, 80, 160, 320]
    h   = 1.0 ./ ncs
    variants = (
        FD = (label="standard FD",                  run=(n, p, s) -> redirect_stdout(() -> StandardFD_Moutzouris2026(n, p; shape=s), devnull)),
        A  = (label="IIM: linear fits, order 1",    run=(n, p, s) -> IIM_Moutzouris2026(n, p; shape=s)),
        C  = (label="IIM: quadratic fits, order 2", run=(n, p, s) -> IIM_Moutzouris2026(n, p; shape=s, order=2, fit=:quadratic)),
        D  = (label="IIM: order 2, exact jumps",    run=(n, p, s) -> IIM_Moutzouris2026(n, p; shape=s, order=2, fit=:quadratic, exact_jumps=true)),
    )
    base   = (ηm=1.0, ξm=2.0, ri=0.2, t=2.0, α=π / 6, ε̇=1.0, γ̇=0.0, ζ̇=0.5, ε̇zz=0.0)
    stiff  = (; base..., ηi=1e3, ξi=1e5)
    weak   = (; base..., ηi=1e-3, ξi=5e-4)
    cases  = [(stiff, (:FD, :A, :C, :D)), (weak, (:FD, :C))]

    #--------------------------------------------#
    err = Dict{Tuple{Symbol,Symbol,Float64},Vector{Float64}}()
    for shape in (:circle, :ellipse), (prm, keys) in cases, k in keys
        e = [variants[k].run(n, prm, shape).ϵP.Linf for n in ncs]
        err[(shape, k, prm.ηi)] = e
        @printf("%-7s η_inc = %.0e  %-30s Linf(P) = %s\n", shape, prm.ηi, variants[k].label, join((@sprintf("%.3e", x) for x in e), "  "))
    end

    #--------------------------------------------#
    Fig  = Figure(size=(1400, 600), fontsize=16)
    cols = Dict(:FD => Makie.wong_colors()[1], :A => Makie.wong_colors()[2], :C => Makie.wong_colors()[3], :D => :gray30)
    local ax
    for (j, shape) in enumerate((:circle, :ellipse))
        ax = Axis(Fig[1, j], xscale=log10, yscale=log10, xlabel=L"h", ylabel=L"\max|P - P^{ex}|",
                  title=shape == :circle ? "circle" : "ellipse (aspect ratio 2)")
        for (prm, keys) in cases, k in keys
            scatterlines!(ax, h, err[(shape, k, prm.ηi)], color=cols[k], marker=prm.ηi > 1 ? :circle : :utriangle,
                          linestyle=k == :D ? :dash : :solid,
                          label="$(variants[k].label), η_inc/η_mat = 1e$(round(Int, log10(prm.ηi)))")
        end
        lines!(ax, h, 0.5 .* (h ./ h[1]) .^ 1, color=:black, linestyle=:dot, label="O(h)")
        lines!(ax, h, 0.05 .* (h ./ h[1]) .^ 2, color=:black, linestyle=:dashdot, label="O(h²)")
    end
    Legend(Fig[1, 3], ax, framevisible=false, labelsize=12)
    save(joinpath(@__DIR__, "IIM_Step6_Compressible.png"), Fig, px_per_unit=2)

    #--------------------------------------------#
    # Pressure maps, ellipse, stiff inclusion
    a, b = semi_axes(:ellipse, stiff)
    θ    = range(0, 2π, 300)
    zoom = 0.48
    Fig  = Figure(size=(1300, 1080), fontsize=15)
    local hm
    for (i, n) in enumerate([40, 80, 160])
        res  = [variants[k].run(n, stiff, :ellipse) for k in (:FD, :A, :C)]
        X, Pa = res[end].X, res[end].P_ana
        pmax = maximum(abs, Pa[(abs.(X.c.x) .<= zoom), (abs.(X.c.y) .<= zoom)])
        fields = [(variants[k].label, r.P, @sprintf("L∞ = %.3f", r.ϵP.Linf)) for (k, r) in zip((:FD, :A, :C), res)]
        push!(fields, ("exact", Pa, ""))
        for (j, (name, P, info)) in enumerate(fields)
            ax = Axis(Fig[i, j], aspect=DataAspect(), title=i == 1 ? name : "", subtitle=info,
                      ylabel=j == 1 ? "nc = $n" : "", ylabelsize=18)
            hm = heatmap!(ax, X.c.x, X.c.y, P, colormap=:vik, colorrange=(-1.1pmax, 1.1pmax),
                          lowclip=:lime, highclip=:magenta)
            lines!(ax, a .* cos.(θ), b .* sin.(θ), color=:black, linewidth=0.8)
            limits!(ax, -zoom, zoom, -zoom, zoom)
            hidexdecorations!(ax)
            hideydecorations!(ax, label=false)
        end
    end
    Colorbar(Fig[:, 5], hm, label="P; lime / magenta: more than 10% beyond the exact range")
    Label(Fig[0, :], "Pressure, compressible elliptical inclusion, η_inc/η_mat = 10³ (zoom)", fontsize=20)
    save(joinpath(@__DIR__, "IIM_Step6_EllipseFields.png"), Fig, px_per_unit=2)
    return nothing
end

main()
