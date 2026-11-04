using CairoMakie, Printf
include(joinpath(@__DIR__, "IIM_Schmid2003.jl"))
include(joinpath(@__DIR__, "IIM_Moutzouris2026.jl"))

# Step 7 (docs/IIM_plan.md): cross-comparison of the z-form IIM (AssembleIIM, Landry) with the
# stress form (AssembleIIMStress: StagFDTools-style Vx↔Vy interpolation, stresses at centroids
# and vertices, one-sided stencils), on the three inclusion tests of steps 5 and 6.

function main()
    ncs = [20, 40, 80, 160]          # nc = 320 needs about 10 GB with the direct solver
    ps  = (mm=1.0, mc=1e3, rc=2.0, gr=0.0, er=1.0)
    pm  = (ηm=1.0, ηi=1e3, ξm=2.0, ξi=1e5, ri=0.2, t=2.0, α=π / 6, ε̇=1.0, γ̇=0.0, ζ̇=0.5, ε̇zz=0.0)
    cases = (
        (title="Schmid (2003), incompressible",     L=10.0, run=(n; kw...) -> IIM_Schmid2003(n, ps; kw...)),
        (title="Moutzouris (2026), circle",         L=1.0,  run=(n; kw...) -> IIM_Moutzouris2026(n, pm; kw...)),
        (title="Moutzouris (2026), ellipse",        L=1.0,  run=(n; kw...) -> IIM_Moutzouris2026(n, pm; shape=:ellipse, kw...)),
    )
    variants = (
        (key=:A, label="linear fits, order 1",    kw=(;)),
        (key=:C, label="quadratic fits, order 2", kw=(order=2, fit=:quadratic)),
        (key=:D, label="order 2, exact jumps",    kw=(order=2, fit=:quadratic, exact_jumps=true)),
    )
    cols = Dict(:A => Makie.wong_colors()[2], :C => Makie.wong_colors()[3], :D => :gray30)

    Fig = Figure(size=(1600, 560), fontsize=16)
    local ax
    for (j, c) in enumerate(cases)
        h  = c.L ./ ncs
        ax = Axis(Fig[1, j], xscale=log10, yscale=log10, xlabel=L"h", ylabel=j == 1 ? L"\max|P - P^{ex}|" : "",
                  title=c.title, titlesize=15)
        for v in variants, form in (:z, :stress)
            e = [c.run(n; form, v.kw...).ϵP.Linf for n in ncs]
            @printf("%-30s %-24s %-7s Linf(P) = %s\n", c.title, v.label, form, join((@sprintf("%.3e", x) for x in e), "  "))
            flush(stdout)
            scatterlines!(ax, h, e, color=cols[v.key], marker=form == :z ? :circle : :xcross, markersize=form == :z ? 9 : 14,
                          linestyle=form == :z ? :solid : :dash, label="$(form == :z ? "z-form" : "stress form"): $(v.label)")
        end
        # stress form with the v-extension (contrast-amplified Taylor remainder)
        e = [c.run(n; form=:stress, extend=:v, order=2, fit=:quadratic).ϵP.Linf for n in ncs]
        @printf("%-30s %-24s %-7s Linf(P) = %s\n", c.title, "quadratic, order 2", "v-ext", join((@sprintf("%.3e", x) for x in e), "  "))
        scatterlines!(ax, h, e, color=:firebrick, marker=:utriangle, linestyle=:dot,
                      label="stress form, v-extension: quadratic fits, order 2")
        lines!(ax, h, 0.05 .* (h ./ h[1]) .^ 2 .* (j == 1 ? 1 : 2), color=:black, linestyle=:dashdot, label="O(h²)")
    end
    Legend(Fig[2, :], ax, framevisible=false, labelsize=13, nbanks=3, orientation=:horizontal)
    Label(Fig[0, :], "IIM: z-form (Landry) against stress form (StagFDTools kernels), η_inc/η_mat = 10³", fontsize=19)
    save(joinpath(@__DIR__, "IIM_Step7_StressForm.png"), Fig, px_per_unit=2)
    return nothing
end

main()
