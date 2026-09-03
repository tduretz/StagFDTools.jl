using JLD2, Printf, StaticArrays, LinearAlgebra, UnPack
import CairoMakie as cm
import Statistics: mean

function read_data(filename)
    File = joinpath(@__DIR__, filename)
    data = load(File)
    return data
end

function plot_stress_paths_n(data, nrange, θvec)
    fig = cm.Figure(size=(600, 400))
    colors=cm.cgrad(:roma, length(nrange), categorical=true)
    ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\tau_{xx}' \ [-]$", ylabel=cm.L"$\tau_{xy}' \ [-]$", aspect=cm.DataAspect())
    τ_plot_θ = zeros(last(nrange), 2)
    for i in nrange
        τ_plot_θ = zeros(2, length(θvec))
        n = nrange[i]
        g = "n$(n)"
        for j in eachindex(θvec)
            θ = θvec[j]
            𝐐_plot = @SMatrix([cos(θ) sin(θ);
                -sin(θ) cos(θ)])
            τ_plot = @SMatrix([data["$g/τ_xx"][j] data["$g/τ_xy"][j]; data["$g/τ_xy"][j] data["$g/τ_yy"][j]])
            τ′_plot = 𝐐_plot * τ_plot * 𝐐_plot'
            τ_plot_θ[1, j] = τ′_plot[1, 1]
            τ_plot_θ[2, j] = τ′_plot[1, 2]
        end
        cm.scatterlines!(ax, τ_plot_θ, color=colors[i])
        cm.scatterlines!(ax, -τ_plot_θ, color=colors[i])
    end
    cm.Colorbar(fig[1, 2], colormap=colors, limits=(first(nrange), last(nrange)), label="n")
    return fig
end

function plot_stress_components(data, nvec, θ)
    figures = Vector{cm.Figure}(undef, length(nvec))
    for i in eachindex(nvec)
        fig = cm.Figure(size=(600, 800))
        n = nvec[i]
        g = "n$(n)"
        ax = cm.Axis(fig[1, 1], title="m = $(data["m"]), n = $(n)", xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16, aspect=2)
        cm.scatterlines!(ax, θ * 180 / π, data["$g/τ_xx"], label=cm.L"\tau_{xx}")
        cm.scatterlines!(ax, θ * 180 / π, data["$g/τ_xy"], label=cm.L"\tau_{xy}")
        cm.scatterlines!(ax, θ * 180 / π, data["$g/τ_II_mean"], label=cm.L"\tau_{II} \ \mathrm{(averaged)}")
        cm.scatterlines!(ax, θ * 180 / π, data["$g/τ_II"], label=cm.L"\tau_{II} \ \mathrm{(components)}")
        cm.Legend(fig[1, 2], ax, labelsize=14, titlesize=13)
        figures[i] = fig
    end
    return figures
end

function fit_factor(δ, n, r)
    # if n = 1, α = 0
    # if n = 2, α = 1 ---> the form is (n-1)^(something)
    α = (n-1)^(1/(2*r))
    return δ * α
end

function plot_anisotropic_factor(data, nrange, θvec)
    fig = cm.Figure(size=(1000, 1000))
    # ax1 = cm.Axis(fig[1, 1], xlabel="n", ylabel=cm.L"\frac{\delta_{\mathrm{eff}}}{\delta}", limits=((first(nrange), last(nrange)), nothing))
    # ax2 = cm.Axis(fig[2, 1], xlabel=cm.L"n \ [\mathrm{log10}]", ylabel=cm.L"\frac{\delta_{\mathrm{eff}}}{\delta} \ [\mathrm{log10}]", xscale=log10, yscale=log10)

    # plot of δs
    ax11 = cm.Axis(fig[1, 1], xlabel="n", ylabel=cm.L"\delta_{\mathrm{eff}}", limits=((first(nrange), last(nrange)), nothing))
    ax12 = cm.Axis(fig[2, 1], xlabel="n", ylabel=cm.L"\delta_{\mathrm{eff}}", limits=((first(nrange), last(nrange)), nothing))
    # ax13 = cm.Axis(fig[3, 1], xlabel="n", ylabel=cm.L"\delta_{\mathrm{eff}}", limits=((first(nrange), last(nrange)), nothing))

    # δ linear from analytical formula:
    δ = (data["α1"] + data["α2"] * data["m"]) * (data["α1"] + data["α2"] / data["m"])
    istrong = 1
    iweak = length(θvec) ÷ 2 + 1

    #      Allocate δs ---------------
    δ_comp = zeros(length(nrange)) # computed from  runs withb superellipse expression
    δ_II = zeros(length(nrange)) # computed as ratio between weak and strong second invariants
    δ_fit = zeros(length(nrange)) # computed with expression depoendent on n
    # normalised δs
    δnorm1 = zeros(length(nrange))

    # δeff fitted manually
    r = 3
    δeff_man = [δ*0, δ,
        δ*1.19,
        δ*1.28,
        δ*1.35,
        δ*1.4,
        δ*1.44,
        δ*1.46,
        δ*1.49,
        δ*1.51,
        δ*1.53,
        δ*1.54,
        δ*1.56,
        δ*1.57,
        δ*1.57,
        δ*1.59,
        δ*1.61,
        δ*1.61,
        δ*1.62,
        δ*1.63]

    for (i, n) in enumerate(nrange)
        g = "n$(n)"

        # δ_eff of the superellipse computed from models 
        τxx = data["$g/τ_xx"][iweak]
        τxy = data["$g/τ_xy"][iweak]
        τy = τxx
        r = 3
        δeff_comp = ((τxx/τxy)^(2*r) - δ^(2*r))^(1/(2*r))
        δ_comp[i] = δeff_comp

        # δ_eff from expression
        δ_eff_fit = fit_factor(δ, n, r)
        δ_fit[i] = δ_eff_fit

        # δeff as ratio between invariants:
        τII_strong = data["$g/τ_II"][istrong]
        τII_weak = data["$g/τ_II"][iweak]
        δeff_II = τII_strong / τII_weak

        println("******* n = $n")
        @show δeff_II, δ_eff_fit
        # δeff_cp  = abs(τ_strong) / abs(τ_weak)
        δnorm1[i] = δeff_man[i] / δ
        # δnorm2[i] = δeff_cp / δ
        δ_II[i] = δeff_II
        # δ_cp[i]     = δeff_cp

        # Compare δs
        println("******* n = $n")
        @show δeff_II, δ_eff_fit
    end
    labels = [@sprintf("%.2f", v) for v in δ_comp]

    # Plot δs
    cm.scatterlines!(ax11, collect(nrange), δ_comp, color=:green, label="computed (from runs)")
    # cm.scatterlines!(ax11, collect(nrange), δ_II, color=:red, label="ratio between invariants")
    # cm.scatterlines!(ax12, collect(nrange), δnorm1, color=:blue, label="α")
    # cm.scatterlines!(ax11, collect(nrange), δeff_man, color=:green, label="computed (from runs)")
    # cm.scatterlines!(ax11, collect(nrange), δ_fit, color=:blue, label="function of n")
    # cm.axislegend(ax11, position=:rb)
    # cm.axislegend(ax12, position=:rb)
    # cm.axislegend(ax13, position=:rb)

    # # Plot normalised δ (from II invariant)
    # cm.scatterlines!(ax1, collect(nrange), δ_II, color=:red, label = "from experiment")
    # cm.scatterlines!(ax1, collect(nrange), δ_fit, color=:blue, label="expression for δ")
    # cm.text!(ax1, collect(nrange), δ_II; text=labels, align=(:left, :bottom), fontsize=10)
    # cm.scatterlines!(ax2, collect(nrange), δ_II, color=:red)
    # cm.text!(ax2, collect(nrange), δ_II; text=labels, align=(:left, :bottom), fontsize=10)

    return fig
end

# Anisotropic yield function z
f(τxx, τxy, δ, r, τy, n) = ((τxx ^ 2 + δ^2 * τxy ^ 2) ^ r + ((δ*(n-1)^(1/(2*r)))^2 * τxy ^ 2) ^ r) ^ (1/2/r) - τy

function aniso_pwl_yield_function(τxx, τxy, data_Anna, data_Filippo; nrange=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], αs=0.5, αw=0.5, τy=10.)

    @unpack δ, m, α2, ε̇ref, τref2, npwl, θ, τ_pwl, τ_pwl_shift, ηpwl1, ηpwl2, ηani, ηlin2, τs, τw, τ_min = data_Anna

    # Ellipse plot
    fig = cm.Figure(size=(800, 800))
    # Components plot
    fig2 = cm.Figure(size=(600, 600))
    # Compare outputs
    fig3 = cm.Figure(size=(600, 600))

    ax1 = cm.Axis(fig[1, 1], xlabel=cm.L"\tau_{xx} \prime", ylabel=cm.L"\tau_{xy} \prime", aspect=cm.DataAspect())
    ax2 = cm.Axis(fig2[1, 1], title="Anna vs Filippo", xlabel=cm.L"\tau_{xx} \prime", ylabel=cm.L"\tau_{xy} \prime", aspect=cm.DataAspect())
    # ax3 = cm.Axis(fig3[1,1], title=cm.L"\delta_{\mathrm{eff}} (\mathrm{from ratio or normal and shear stress})", xlabel="n", ylabel=cm.L"\delta_{\mathrm{}}")
    colors = cm.cgrad(:roma, length(nrange), categorical=true)

    stress = 1
    τy = τs[stress] # Strong end member (from Biot 1965) (used as yield stress)
    τweak = τw[stress]

    # Extract data for all values of n
    τxx_pow = τ_pwl.xx[stress, :, :]
    τyy_pow = τ_pwl.yy[stress, :, :]
    τxy_pow = τ_pwl.xy[stress, :, :]
    outp_Anna = (xx=τxx_pow, yy=τyy_pow, xy=τxy_pow)

    # Choose between 11 (Anna output) and 15 angles
    θ = data_Filippo["n1/θ"]
    δeff_n = zeros(length(nrange))
    for i in eachindex(nrange)
        n = nrange[i]
        @show n
        g = "n$(n)"

        τxx′_vec1 = zeros(length(θ))
        τxy′_vec1 = zeros(length(θ))
        τxx′_vec2 = zeros(length(θ))
        τxy′_vec2 = zeros(length(θ))
        τxx_vec2 = zeros(length(θ))
        τxy_vec2 = zeros(length(θ))
        τII_vec2 = zeros(length(θ))
        δeff_n[i] = data_Filippo["$g/τ_xx"][1] / data_Filippo["$g/τ_xy"][1]

        # # Number of sampled angles between outputs is different!!!!!!!
        # @show length(data_Filippo["$g/θ"])
        # @show length(θ)
        # error()

        for j in eachindex(θ)
            𝐐_plot = @SMatrix([cos(θ[j]) sin(θ[j]);
                -sin(θ[j]) cos(θ[j])])

            # # Save τ components from outp 1 (Anna)
            # τxx_1 = outp_Anna.xx[i, j]
            # τyy_1 = outp_Anna.yy[i, j]
            # τxy_1 = outp_Anna.xy[i, j]

            # Save τ components from outp 2 (Filippo)
            τxx_2 = data_Filippo["$g/τ_xx"][j]
            τyy_2 = data_Filippo["$g/τ_yy"][j]
            τxy_2 = data_Filippo["$g/τ_xy"][j]
            τII_2 = data_Filippo["$g/τ_II"][j]

            # Rotate in material frame
            # τ_plot1 = @SMatrix([τxx_1 τxy_1;
            #     τxy_1 τyy_1])
            # τ′_plot1 = 𝐐_plot * τ_plot1 * 𝐐_plot'

            τ_plot2 = @SMatrix([τxx_2 τxy_2;
                τxy_2 τyy_2])
            τ′_plot2 = 𝐐_plot * τ_plot2 * 𝐐_plot'

            # τxx′_vec1[j] = τ′_plot1[1, 1]
            # τxy′_vec1[j] = τ′_plot1[2, 1]
            τxx_vec2[j] = τxx_2
            τxy_vec2[j] = τxy_2
            τII_vec2[j] = τII_2
        end

        # Output
        # cm.scatterlines!(ax1, τxx′_vec1, τxy′_vec1, color=colors[i], label="n = $(n), Anna")
        # cm.scatterlines!(ax1, -τxx′_vec1, -τxy′_vec1, color=colors[i])
        # cm.scatter!(ax1, τxx′_vec2, τxy′_vec2, color=colors[i], markersize=10, label="n = $(n)")
        # cm.scatter!(ax1, -τxx′_vec2, -τxy′_vec2, color=colors[i], markersize=10)
        # cm.Colorbar(fig[1, 2], colormap=colors, limits=(first(nrange), last(nrange)), label="n")
        # cm.axislegend(ax1)

        τy = τs[1]
        δ = 1.5625
        r = 20 # 2 + 2n/n^2 # probably r=3 fits all
        𝑓 = f.(τxx, τxy', δ, r, τy, 1000000000)
        cm.contour!(ax1, τxx, τxy, 𝑓; levels=[0], color=colors[i], linewidth=3, linestyle=:dash)

        if n === 2
            # 1) plot components
            ax2 = cm.Axis(fig2[1, 1], title="n=$(n)", xlabel="θ", ylabel="τ")
            cm.scatterlines!(τxx_2, θ, label=cm.L"\tau_{xx} \mathrrm{(composite)}")
            cm.scatterlines!(τxx_2, θ, label=cm.L"\tau_{xy} \mathrrm{(composite)}")
            cm.scatterlines!(data_Filippo["$g/τ_II"], θ, label=cm.L"\tau_{II} \mathrrm{(composite)}")
            cm.scatterlines!(τxx_2, θ, label=cm.L"\tau_{xx} \mathrrm{(analytical)}")
            cm.scatterlines!(τxy_2, θ, label=cm.L"\tau_{xy} \mathrrm{(analytical)}")


            # 2) plot superellipse
        elseif n === 3
            break
        elseif n === 4
        elseif n === 5
        elseif n === 10
        elseif n === 20
        end
    end
    return fig

end

# Values of r 
# n = 2 --> r ∼ 2.
# n = 3 --> r ∼ 2.45
# n = 4 --> r ∼ 2.45
# n = 5 --> r ∼ 2.5
# n = 6 --> r ∼ 2.56
# n = 7 -->  r ∼ 2.6
# n = 8 -->  r ∼ 2.64
# n = 9 -->  r ∼ 2.7
# n = 10 --> r ∼ 2.74
# n = 11 --> r ∼ 2.78
# n = 12 --> r ∼ 2.82
# n = 13 --> r ∼ 2.86
# n = 14 --> r ∼ 2.89
# n = 15 --> r ∼ 2.91
# n = 16 --> r ∼ 2.92
# n = 17 --> r ∼ 2.95
# n = 18 --> r ∼ 2.98
# n = 19 --> r ∼ 2.98
# n = 20 --> r = 3

function fit_power_law(data_Anna, data_Filippo)
    τxx = LinRange(-15, 15, 200)
    τxy = LinRange(-10, 10, 200)

    ηw, ηs = 1, 1
    fig = aniso_pwl_yield_function(τxx, τxy, data_Anna, data_Filippo)
    display(fig)
end

let

    # Load output
    file_Filippo=@sprintf("Layered_pwl.jld2")
    file_Anna = jldopen("$(@__DIR__)/PowerLaw_multilayer_tau20.jld2", "r")
    data_Fil = read_data(file_Filippo)
    nrange = 1:20
    nvec = [1, 5, 10, 15, 20]
    nθ = 15
    θ = LinRange(0, π/2, nθ)

    cm.with_theme(cm.theme_latexfonts()) do
        # Plots
        # --- 1) grid fields
        # [...]

        # --- 2) stress components vs time ----------------------
        # fig2 = plot_stress_components(data, nvec, θ)
        # for i in eachindex(fig2)
        #     display(fig2[i])
        # end

        # # --- 3) stress paths vs time for each n ----------------
        # fig3 = plot_stress_paths_n(data, nrange, θ)
        # display(fig3)

        # --- 4) δeff/δ vs n plot -------------------------------
        # fig4 = plot_anisotropic_factor(data_Fil, nrange, θ)
        # display(fig4)

        # 5) ========
        fig = fit_power_law(file_Anna, data_Fil)
        display(fig)
    end

end