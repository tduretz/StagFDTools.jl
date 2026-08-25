using JLD2, Printf, StaticArrays, LinearAlgebra
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

function plot_anisotropic_factor(data, nrange, θvec)
    fig = cm.Figure(size=(600, 800))
    ax1 = cm.Axis(fig[1, 1], xlabel="n", ylabel=cm.L"\frac{\delta_{\mathrm{eff}}}{\delta}", limits=((first(nrange), last(nrange)), nothing))
    ax2 = cm.Axis(fig[2, 1], xlabel=cm.L"n \ [\mathrm{log10}]", ylabel=cm.L"\frac{\delta_{\mathrm{eff}}}{\delta} \ [\mathrm{log10}]", xscale=log10, yscale=log10)
    # δ linear from analytical formula:
    δ = (data["α1"] + data["α2"] * data["m"]) * (data["α1"] + data["α2"] / data["m"])
    istrong = 1
    iweak = length(θvec) ÷ 2 + 1
    δnorm = zeros(length(nrange))
    for (i, n) in enumerate(nrange)
        g = "n$(n)"
        τII_strong = data["$g/τ_II"][istrong]
        τII_weak = data["$g/τ_II"][iweak]
        δeff = τII_strong / τII_weak
        δnorm[i] = δeff / δ
    end
    labels = [@sprintf("%.2f", v) for v in δnorm]
    cm.scatterlines!(ax1, collect(nrange), δnorm, color=:red)
    cm.text!(ax1, collect(nrange), δnorm; text=labels, align=(:left, :bottom), fontsize=10)
    cm.scatterlines!(ax2, collect(nrange), δnorm, color=:red)
    cm.text!(ax2, collect(nrange), δnorm; text=labels, align=(:left, :bottom), fontsize=10)
    return fig
end

f(τxx, τxy, δ, δeff, r, τy) = ((τxx ^ 2 + δ^2 * τxy ^ 2) ^ r + (δeff^2 * τxy ^ 2) ^ r) ^ (1/2/r) - τy

function aniso_pwl_yield_function(τxx, τxy, data_Anna, data_Filippo; nrange=[1, 2, 5, 10], αs=0.5, αw=0.5, τy=10.)

    @unpack δ, m, α2, ε̇ref, τref2, npwl, θ, τ_pwl, τ_pwl_shift, ηpwl1, ηpwl2, ηani, ηlin2, τs, τw, τ_min = data_Anna

    # Ellipse plot
    fig = Figure(size=(600, 600))
    # Components plot
    fig2 = Figure(size=(600, 600))
    # Compare outputs
    fig3 = Figure(size=(600, 600))

    ax1 = Axis(fig[1, 1], xlabel=L"\tau_{xx} \prime", ylabel=L"\tau_{xy} \prime", aspect=DataAspect())
    ax3 = Axis(fig2[1, 1], title="Anna vs Filippo", xlabel=L"\tau_{xx} \prime", ylabel=L"\tau_{xy} \prime", aspect=DataAspect())
    colors = cgrad(:roma, length(nrange), categorical=true)

    stress = 1
    τy = τs[stress] # Strong end member (from Biot 1965) (used as yield stress)
    τweak = τw[stress]

    # Extract data for all values of n
    τxx_pow = τ_pwl.xx[stress, :, :]
    τyy_pow = τ_pwl.yy[stress, :, :]
    τxy_pow = τ_pwl.xy[stress, :, :]
    outp_Anna = (xx=τxx_pow, yy=τyy_pow, xy=τxy_pow)

    for i in eachindex(nrange)
        n = nrange[i]
        g = "n$(n)"

        τxx′_vec1 = zeros(length(θ))
        τxy′_vec1 = zeros(length(θ))
        τxx′_vec2 = zeros(length(θ))
        τxy′_vec2 = zeros(length(θ))

        # # Number of sampled angles is different!!!!!!!
        # @show length(data_Filippo["$g/θ"])
        # @show length(θ)
        # error()

        for j in eachindex(θ)
            𝐐_plot = @SMatrix([cos(θ[j]) sin(θ[j]);
                -sin(θ[j]) cos(θ[j])])

            # Save τ components from outp 1 (Anna)
            τxx_1 = outp_Anna.xx[i, j]
            τyy_1 = outp_Anna.yy[i, j]
            τxy_1 = outp_Anna.xy[i, j]

            # Save τ components from outp 2 (Filippo)
            τxx_2 = data_Filippo["$g/τ_xx"]
            τyy_2 = data_Filippo["$g/τ_yy"]
            τxy_2 = data_Filippo["$g/τ_xy"]

            # Rotate in material frame
            τ_plot1 = @SMatrix([τxx_1 τxy_1;
                τxy_1 τyy_1])
            τ′_plot1 = 𝐐_plot * τ_plot1 * 𝐐_plot'

            τ_plot2 = @SMatrix([τxx_2 τxy_2;
                τxy_2 τyy_2])
            τ′_plot2 = 𝐐_plot * τ_plot2 * 𝐐_plot'

            τxx′_vec1[j] = τ′_plot1[1, 1]
            τxy′_vec1[j] = τ′_plot1[2, 1]
            τxx′_vec2[j] = τ′_plot2[1, 1]
            τxy′_vec2[j] = τ′_plot2[2, 1]

        end
        scatter!(ax1, τxx′_vec1, τxy′_vec1, color=colors[ipwl], label="n = $(npwl1), Anna")
        scatter!(ax2, -τxx′_vec1, -τxy′_vec1, color=colors[ipwl])
        scatter!(ax1, τxx′_vec2, τxy′_vec2, color=colors[ipwl], label="n = $(npwl1), Filippo")
        scatter!(ax2, -τxx′_vec2, -τxy′_vec2, color=colors[ipwl])

        if n === 1
            # 1) plot components
            ax2 = Axis(fig2[1, 1], title="n=$(n)")

            # 2) plot superellipse


        elseif n === 2

        elseif n === 5
        elseif n === 10
        elseif n === 20

        end
        τy = τs[1]
        δ = 1.5625
        δeff = δ * 1.5
        n = 2
        r = 1.5
        𝑓 = f.(τxx, τxy', δ, δeff, r, τy)
        contour!(ax, τxx, τxy, 𝑓; levels=[0], color=:blue, linewidth=3)
    end
    return fig

end

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

        # # --- 4) δeff/δ vs n plot -------------------------------
        # fig4 = plot_anisotropic_factor(data, nrange, θ)
        # display(fig4)

        # 5) ========
        fig = fit_power_law(file_Anna, data_Fil)
        display(fig)
    end

end