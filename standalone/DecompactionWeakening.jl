using CairoMakie

function DecompactionWeakening()

    ϕ0 = 1e-3
    ϕ  = ϕ0
    C  = 1.
    ηs = 1e15
    ηc = ηs/C/ϕ0
    λp = 1e6
    pe = LinRange(-2e7, 2e7, 100)
    R  = 1e6

    ηϕ(R) = @. ηc * ϕ0/ϕ *(1 + 1/2*(1/R-1)*(1 + tanh(-pe/λp)))

    ηϕηs_1 = ηϕ(1.)  ./ηs
    ηϕηs_3 = ηϕ(1e-3)./ηs
    ηϕηs_6 = ηϕ(1e-6)./ηs 

    ftsz = 16
    f = Figure( fontsize=ftsz, size=(550,300))

    ax1 = Axis(f[1, 1], title = L"$\Delta P_\text{t}^*$", xlabel = L"$P_\mathrm{e}$", ylabel = L"$\log_\text{10}$ $\Omega_\eta$")
    lines!(ax1, pe, log10.(ηϕηs_1), label=L"$$R=1")
    lines!(ax1, pe, log10.(ηϕηs_3), label=L"$$R=1e3")
    lines!(ax1, pe, log10.(ηϕηs_6), label=L"$$R=1e6")
    axislegend(framevisible = false, position = :rt, labelsize=15)

    display(f)

end

DecompactionWeakening()