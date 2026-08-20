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

function plot_anisotropic_factor(data, nrange, θ)

end

let

    # Load output
    filename=@sprintf("Layered_pwl.jld2")
    data = read_data(filename)
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

        # --- 3) stress paths vs time for each n ----------------
        fig3 = plot_stress_paths_n(data, nrange, θ)
        display(fig3)

    end

end