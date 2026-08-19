using JLD2, Printf, StaticArrays, LinearAlgebra
import CairoMakie as cm
import Statistics: mean

# Post-processing on Layered VEVP_Layered_Model ==================================================

function inner_indices(x_c, y_c, half_width)
    imin_x = argmin(abs.(x_c .+ half_width))
    imax_x = argmin(abs.(x_c .- half_width))
    imin_y = argmin(abs.(y_c .+ half_width))
    imax_y = argmin(abs.(y_c .- half_width))
    return imin_x:imax_x, imin_y:imax_y
end

# --- path helpers (domain-size aware) ------------------------------------------------

domain_dir(datapath, Lwidth) = joinpath(datapath, @sprintf("domainL%.1f", Lwidth))

function evolution_dir(datapath, angle_deg; Lwidth=nothing)
    base = Lwidth === nothing ? datapath : domain_dir(datapath, Lwidth)
    joinpath(base, @sprintf("fabric%03ddeg_evolution", angle_deg))
end

function step_path(datapath, nc_x, angle_deg, it; Lwidth=nothing)
    joinpath(evolution_dir(datapath, angle_deg; Lwidth=Lwidth),
        @sprintf("fields_res%d_fabric%03ddeg_step%03d.jld2", nc_x, angle_deg, it))
end

function final_path(datapath, nc_x, angle_deg; Lwidth=nothing)
    base = Lwidth === nothing ? datapath : domain_dir(datapath, Lwidth)
    joinpath(base, @sprintf("fields_res%d_fabric%03ddeg_final.jld2", nc_x, angle_deg))
end

function load_final(datapath, nc_x, angle_deg, nt; Lwidth=nothing)
    sp = step_path(datapath, nc_x, angle_deg, nt; Lwidth=Lwidth)
    fp = final_path(datapath, nc_x, angle_deg; Lwidth=Lwidth)
    isfile(sp) && return load(sp)
    isfile(fp) && return load(fp)
    suffix = Lwidth === nothing ? "" : @sprintf(", L = %.1f", Lwidth)
    error("No saved data found for angle = $(angle_deg)°, resolution = $(nc_x)" * suffix)
end

function τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=0.6, Lwidth=nothing)
    τIIev = fill(NaN, nt)
    for it in 1:nt
        fp = step_path(datapath, nc_x, angle_deg, it; Lwidth=Lwidth)
        isfile(fp) || continue
        data = load(fp)
        inner_x, inner_y = inner_indices(data["x_c"], data["y_c"], half_width)
        τIIev[it] = mean(data["τII"][inner_x, inner_y])
    end
    return τIIev
end

function stresstensor_time(datapath, nc_x, angle_deg, nt; half_width=0.6, Lwidth=nothing)

    inner_x = inner_y = nothing
    for it in 1:nt
        fp = step_path(datapath, nc_x, angle_deg, it; Lwidth=Lwidth)
        isfile(fp) || continue
        data = load(fp)
        inner_x, inner_y = inner_indices(data["x_c"], data["y_c"], half_width)
        break
    end
    inner_x === nothing && error("No saved data found for angle = $(angle_deg)°")

    nix, niy = length(inner_x), length(inner_y)
    τxx = fill(NaN, nt, nix, niy)
    τyy = fill(NaN, nt, nix, niy)
    τxy = fill(NaN, nt, nix, niy)
    Pt = fill(NaN, nt, nix, niy)

    for it in 1:nt
        fp = step_path(datapath, nc_x, angle_deg, it; Lwidth=Lwidth)
        isfile(fp) || continue
        data = load(fp)
        inner_x, inner_y = inner_indices(data["x_c"], data["y_c"], half_width)

        τxx[it, :, :] .= data["τxx"][inner_x, inner_y]
        τyy[it, :, :] .= data["τyy"][inner_x, inner_y]
        τxy[it, :, :] .= [0.25 * (data["τxy"][i, j] + data["τxy"][i+1, j] + data["τxy"][i, j+1] + data["τxy"][i+1, j+1])
                          for i in inner_x, j in inner_y]
        Pt[it, :, :] .= data["Pt"][inner_x, inner_y]
    end
    return (xx=τxx, yy=τyy, xy=τxy, Pt=Pt)
end

has_evolution(datapath, angle_deg; Lwidth=nothing) = isdir(evolution_dir(datapath, angle_deg; Lwidth=Lwidth))

# 1) GRID / FIELD PLOT --------------------------------------------------------------------------

function plot_grid_fields(datapath, nc_x, angle_deg, nt, half_width::Float64; save_fig=false, figpath=datapath, Lwidth=nothing)

    data = load_final(datapath, nc_x, angle_deg, nt; Lwidth=Lwidth)
    x_c, y_c = data["x_c"], data["y_c"]
    inner_x, inner_y = inner_indices(x_c, y_c, half_width)

    τIIev = has_evolution(datapath, angle_deg; Lwidth=Lwidth) ?
            τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=half_width, Lwidth=Lwidth) :
            fill(NaN, nt)
    τIIev[nt] = mean(data["τII"][inner_x, inner_y])  # always have the last point

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(700, 600), px_per_unit=2)

        ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5], xlabelsize=26, ylabelsize=26, titlesize=20, title=@sprintf("θ = %d°", angle_deg))
        hm = cm.heatmap!(ax, x_c, y_c, data["τII"], colormap=cm.cgrad(:roma, rev=true))
        cm.poly!(ax, cm.Rect(x_c[inner_x[1]], y_c[inner_y[1]], x_c[inner_x[end]] - x_c[inner_x[1]], y_c[inner_y[end]] - y_c[inner_y[1]]), strokecolor=:white, strokewidth=2, color=:transparent)
        st = max(1, length(x_c) ÷ 25)
        # cm.arrows2d!(ax, x_c[1:st:end], y_c[1:st:end], data["σ1x"][1:st:end, 1:st:end], data["σ1y"][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
        cm.Colorbar(fig[1, 2], hm, label=cm.L"$\tau_{II} \ [-]$", labelsize=18)

        ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
        hm2 = cm.heatmap!(ax2, x_c, y_c, data["ε̇II"], colormap=cm.cgrad(:roma, rev=true), colorrange=(0, 10))
        cm.poly!(ax2, cm.Rect(x_c[inner_x[1]], y_c[inner_y[1]],
                x_c[inner_x[end]] - x_c[inner_x[1]],
                y_c[inner_y[end]] - y_c[inner_y[1]]),
            strokecolor=:white, strokewidth=2, color=:transparent)
        cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\dot\varepsilon_{II} \ [-]$", labelsize=18)

        ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect())
        hm3 = cm.heatmap!(ax3, x_c, y_c, data["Vx"], colormap=:vik)
        cm.Colorbar(fig[2, 2], hm3, label=cm.L"$v_x \ [-]$", labelsize=18)

        ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect())
        hm4 = cm.heatmap!(ax4, x_c, y_c, data["Vy"], colormap=:vik)
        cm.Colorbar(fig[2, 4], hm4, label=cm.L"$v_y \ [-]$", labelsize=18)

        ax5 = cm.Axis(fig[3, 1:4], xlabel="time step", ylabel=cm.L"$\tau_{II} \ [-]$",
            xlabelsize=18, ylabelsize=18)
        cm.xlims!(ax5, 0, nt)
        valid = .!isnan.(τIIev)
        if count(valid) > 1
            cm.lines!(ax5, (1:nt)[valid], τIIev[valid])
        else
            cm.scatter!(ax5, [nt], [τIIev[nt]])
        end

        display(fig)
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_fabric%03ddeg_box%.2f.png", nc_x, angle_deg, half_width)
        cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
    end

    return τIIev
end

# -----------------------------------------------------------------------------------
function plot_grid_fields(datapath, nc_x, angle_deg, nt, half_width::Vector{Float64}; save_fig=false, figpath=datapath, Lwidth=nothing)

    data = load_final(datapath, nc_x, angle_deg, nt; Lwidth=Lwidth)
    x_c, y_c = data["x_c"], data["y_c"]

    # Use the largest box for the stress-time inset / final value, just for reference
    half_width_ref = maximum(half_width)
    inner_x_ref, inner_y_ref = inner_indices(x_c, y_c, half_width_ref)

    τIIev = has_evolution(datapath, angle_deg; Lwidth=Lwidth) ?
            τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=half_width_ref, Lwidth=Lwidth) :
            fill(NaN, nt)
    τIIev[nt] = mean(data["τII"][inner_x_ref, inner_y_ref])

    nb = length(half_width)
    box_colors = cm.cgrad(:viridis, nb, categorical=true)

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(700, 600), px_per_unit=2)

        ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5],
            xlabelsize=26, ylabelsize=26, titlesize=26,
            title=@sprintf("θ = %d°", angle_deg))
        hm = cm.heatmap!(ax, x_c, y_c, data["τII"], colormap=cm.cgrad(:roma, rev=true))
        for (ib, hw) in enumerate(half_width)
            inner_x, inner_y = inner_indices(x_c, y_c, hw)
            cm.poly!(ax, cm.Rect(x_c[inner_x[1]], y_c[inner_y[1]],
                    x_c[inner_x[end]] - x_c[inner_x[1]],
                    y_c[inner_y[end]] - y_c[inner_y[1]]),
                strokecolor=box_colors[ib], strokewidth=2, color=:transparent,
                label=@sprintf("box = %.2f", hw))
        end
        st = max(1, length(x_c) ÷ 25)
        cm.arrows2d!(ax, x_c[1:st:end], y_c[1:st:end],
            data["σ1x"][1:st:end, 1:st:end], data["σ1y"][1:st:end, 1:st:end],
            tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
        cm.Colorbar(fig[1, 2], hm, label=cm.L"$\tau_{II} \ [-]$", labelsize=18)

        ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
        hm2 = cm.heatmap!(ax2, x_c, y_c, data["ε̇II"], colormap=cm.cgrad(:roma, rev=true))
        for (ib, hw) in enumerate(half_width)
            inner_x, inner_y = inner_indices(x_c, y_c, hw)
            cm.poly!(ax2, cm.Rect(x_c[inner_x[1]], y_c[inner_y[1]],
                    x_c[inner_x[end]] - x_c[inner_x[1]],
                    y_c[inner_y[end]] - y_c[inner_y[1]]),
                strokecolor=box_colors[ib], strokewidth=2, color=:transparent)
        end
        cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\dot\varepsilon_{II} \ [-]$", labelsize=18)

        ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect())
        hm3 = cm.heatmap!(ax3, x_c, y_c, data["Vx"], colormap=:vik)
        cm.Colorbar(fig[2, 2], hm3, label=cm.L"$v_x \ [-]$", labelsize=18)

        ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect())
        hm4 = cm.heatmap!(ax4, x_c, y_c, data["Vy"], colormap=:vik)
        cm.Colorbar(fig[2, 4], hm4, label=cm.L"$v_y \ [-]$", labelsize=18)

        ax5 = cm.Axis(fig[3, 1:3], xlabel="time step", ylabel=cm.L"$\tau_{II} \ [-]$",
            xlabelsize=18, ylabelsize=18)
        cm.xlims!(ax5, 0, nt)
        valid = .!isnan.(τIIev)
        if count(valid) > 1
            cm.lines!(ax5, (1:nt)[valid], τIIev[valid])
        else
            cm.scatter!(ax5, [nt], [τIIev[nt]])
        end

        cm.Legend(fig[3, 4], ax, "box half-width", labelsize=12, titlesize=13)

        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_fabric%03ddeg_boxsizes.png", nc_x, angle_deg)
        cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
    end

    return τIIev
end

# 2) STRESS-TIME EVOLUTION -------------------------------------------------------------------------

function plot_stress_time(datapath, nc_x, angles_deg, nt, half_widths::Float64; m=nothing, save_fig=false, figpath=datapath, Lwidth=nothing)
    nθ = length(angles_deg)
    colors = cm.cgrad(:roma, nθ, categorical=true)

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(700, 500), px_per_unit=2)
        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16, aspect=3, title=m === nothing ? "" : "m = $(m)")
        for (iθ, angle_deg) in enumerate(angles_deg)
            if has_evolution(datapath, angle_deg; Lwidth=Lwidth)
                τIIev = τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=half_widths, Lwidth=Lwidth)
                valid = .!isnan.(τIIev)
                cm.lines!(ax, (1:nt)[valid], τIIev[valid], color=colors[iθ], label=@sprintf("θ = %d°", angle_deg))
            end
        end
        cm.Colorbar(fig[2, 1], colormap=cm.cgrad(:roma, nθ, categorical=true), limits=(0, 90), ticks=([0, 90], ["0° (strong)", "45° (weak)"]), label="layer orientation, m = $(m)", labelsize=16, vertical=false, width=cm.Relative(0.4))
        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_stresstime_box%.2f.pdf", nc_x, half_widths)
        cm.save(joinpath(figpath, figname), fig)
    end

    return fig
end

# -----------------------------------------------------------------------------------
function plot_stress_time(datapath, nc_x, angles_deg, nt, half_widths::Vector{Float64}; m=nothing, save_fig=true, figpath=datapath, Lwidth=nothing)

    @assert length(half_widths) <= 4 "Maximum of 4 half widths supported"

    nθ = length(angles_deg)
    colors = cm.cgrad(:roma, nθ, categorical=true)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(900, 700), px_per_unit=2)

        local ax
        for (ib, hw) in enumerate(half_widths)
            row, col = positions[ib]
            ax = cm.Axis(fig[row, col], xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$",
                xlabelsize=16, ylabelsize=16, titlesize=16, aspect=2,
                title=@sprintf("box = %.2f", 2 * hw))

            for (iθ, angle_deg) in enumerate(angles_deg)
                if has_evolution(datapath, angle_deg; Lwidth=Lwidth)
                    τIIev = τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=hw, Lwidth=Lwidth)
                    valid = .!isnan.(τIIev)
                    cm.lines!(ax, (1:nt)[valid], τIIev[valid], color=colors[iθ], label=@sprintf("θ = %d°", angle_deg))
                end
            end
        end
        cm.Colorbar(fig[3, 1:2], colormap=cm.cgrad(:roma, nθ, categorical=true),
            limits=(0, 90), ticks=([0, 90], ["0° (strong)", "45° (weak)"]),
            label="layer orientation, m = $(m)", labelsize=16, vertical=false, width=cm.Relative(0.4))

        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_stresstime_boxsizes.pdf", nc_x)
        cm.save(joinpath(figpath, figname), fig)
    end

    return fig
end

# 2b) STRESS COMPONENT TIME SERIES (single orientation)

function plot_stress_components_time(datapath, nc_x, angle_deg, nt, half_width::Float64; Lwidth=nothing, save_fig=false, figpath=datapath)

    Τ = stresstensor_time(datapath, nc_x, angle_deg, nt; half_width=half_width, Lwidth=Lwidth)

    τxx = dropdims(mean(Τ.xx, dims=(2, 3)), dims=(2, 3))
    τyy = dropdims(mean(Τ.yy, dims=(2, 3)), dims=(2, 3))
    τxy = dropdims(mean(Τ.xy, dims=(2, 3)), dims=(2, 3))
    τII = sqrt.(0.5 .* (τxx .^ 2 .+ τyy .^ 2 .+ (.-τxx .- τyy) .^ 2) .+ τxy .^ 2)

    valid = .!isnan.(τxx)

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(700, 450), px_per_unit=2)

        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16, title=@sprintf("θ = %d°, box = %.2f", angle_deg, half_width))
        cm.xlims!(ax, 0, nt)

        cm.lines!(ax, (1:nt)[valid], τxx[valid], color=:red, label=cm.L"$\tau_{xx}$")
        cm.lines!(ax, (1:nt)[valid], τyy[valid], color=:green, label=cm.L"$\tau_{yy}$")
        cm.lines!(ax, (1:nt)[valid], τxy[valid], color=:blue, label=cm.L"$\tau_{xy}$")
        cm.lines!(ax, (1:nt)[valid], τII[valid], color=:black, linewidth=2, label=cm.L"$\tau_{II}$")

        cm.Legend(fig[1, 2], ax, "component", labelsize=14, titlesize=15)

        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_fabric%03ddeg_components_box%.2f.pdf", nc_x, angle_deg, half_width)
        cm.save(joinpath(figpath, figname), fig)
    end

    return (xx=τxx, yy=τyy, xy=τxy, II=τII)
end

# 3) ORIENTATION vs τII (final step) ----------------------------------------------------------

function plot_orientation_strength(datapath, nc_x, angles_deg, nt, half_width::Float64; m=nothing, save_fig=false, figpath=datapath, Lwidth=nothing)

    nθ = length(angles_deg)
    τ_final = zeros(nθ)
    colors = cm.cgrad(:roma, nθ, categorical=true)

    for (iθ, angle_deg) in enumerate(angles_deg)
        data = load_final(datapath, nc_x, angle_deg, nt; Lwidth=Lwidth)
        inner_x, inner_y = inner_indices(data["x_c"], data["y_c"], half_width)
        τ_final[iθ] = mean(data["τII"][inner_x, inner_y])
    end

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(700, 350), px_per_unit=2)
        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16, title=m === nothing ? "" : "m = $(m)")
        cm.scatterlines!(ax, angles_deg, τ_final, color=colors[1:nθ], label=cm.L"$\tau_{II}$ final step")
        cm.Colorbar(fig[2, 1], colormap=cm.cgrad(:roma, nθ, categorical=true), limits=(0, 90), ticks=([0, 45, 90], ["strong", "weak", "strong"]), label="layer orientation", labelsize=16, vertical=false, width=cm.Relative(0.6))
        cm.rowgap!(fig.layout, 5)
        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_orientation_box%.2f.pdf", nc_x, half_width)
        cm.save(joinpath(figpath, figname), fig)
    end

    return τ_final
end

# -----------------------------------------------------------------------------------
function plot_orientation_strength_evolution(datapath, nc_x, angles_deg, nt, half_width::Float64; m=nothing, save_fig=false, figpath=datapath, Lwidth=nothing)
    nθ = length(angles_deg)
    τ_evo = fill(NaN, nθ, nt)
    colors = cm.cgrad(:roma, nt, categorical=true)

    for (iθ, angle_deg) in enumerate(angles_deg)
        has_evolution(datapath, angle_deg; Lwidth=Lwidth) || continue
        τ_evo[iθ, :] .= τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=half_width, Lwidth=Lwidth)
    end

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(700, 450), px_per_unit=2)

        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16, title=m === nothing ? "" : "m = $(m)")

        for it in 1:nt
            valid = .!isnan.(τ_evo[:, it])
            any(valid) || continue
            cm.lines!(ax, angles_deg[valid], τ_evo[valid, it], color=colors[it])
        end

        cm.Colorbar(fig[1, 2], colormap=cm.cgrad(:roma, nt, categorical=true),
            limits=(1, nt), label="time step", labelsize=16)

        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_orientation_evolution_box%.2f.pdf", nc_x, half_width)
        cm.save(joinpath(figpath, figname), fig)
    end

    return τ_evo
end
# -----------------------------------------------------------------------------------
function plot_orientation_strength(datapath, nc_x, angles_deg, nt, half_widths::Vector{Float64}; m=nothing, save_fig=false, figpath=datapath, Lwidth=nothing)

    nθ = length(angles_deg)
    nb = length(half_widths)
    τ_final_box = zeros(nθ, nb)  # rows = angle, cols = box size
    box_colors = cm.cgrad(:viridis, nb, categorical=true)

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(700, 350), px_per_unit=2)

        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$",
            xlabelsize=18, ylabelsize=18, titlesize=16,
            title=m === nothing ? "" : "m = $(m)")

        for (ib, hw) in enumerate(half_widths)
            for (iθ, angle_deg) in enumerate(angles_deg)
                data = load_final(datapath, nc_x, angle_deg, nt; Lwidth=Lwidth)
                inner_x, inner_y = inner_indices(data["x_c"], data["y_c"], hw)
                τ_final_box[iθ, ib] = mean(data["τII"][inner_x, inner_y])
            end
            cm.scatterlines!(ax, angles_deg, τ_final_box[:, ib],
                color=box_colors[ib], label=@sprintf("box = %.2f", 2 * hw))
        end

        cm.Legend(fig[1, 2], ax, "box width", labelsize=14, titlesize=15)

        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_orientation_boxsizes.pdf", nc_x)
        cm.save(joinpath(figpath, figname), fig)
    end

    return τ_final_box
end

# 4) DOMAIN-SIZE COMPARISON -----------------------------------------------------
function plot_domain_comparison(datapath, nc_x, Lwidths::Vector{Float64}, angles_deg, nt;
    half_width=0.6, m=nothing, save_fig=false, figpath=datapath)

    nL = length(Lwidths)
    nθ = length(angles_deg)
    colors = cm.cgrad(:roma, nL, categorical=true)  # color = domain width

    iθ_first, iθ_mid, iθ_last = 1, (nθ + 1) ÷ 2, nθ
    linestyles = Dict(iθ_first => :solid, iθ_mid => :dash, iθ_last => :dot)
    orientation_label = Dict(iθ_first => "Strong (hor)", iθ_mid => "Weak (45°)", iθ_last => "Strong (vert)")

    τ_final = zeros(nL, nθ)

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(800, 650), px_per_unit=2)

        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$",
            xlabelsize=18, ylabelsize=18, titlesize=16, title=m === nothing ? "" : "m = $(m)")

        for (iL, Lwidth) in enumerate(Lwidths)
            for iθ in (iθ_first, iθ_mid, iθ_last)
                angle_deg = angles_deg[iθ]
                has_evolution(datapath, angle_deg; Lwidth=Lwidth) || continue
                τIIev = τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=half_width * Lwidth, Lwidth=Lwidth)
                valid = .!isnan.(τIIev)
                label = iL == 1 ? orientation_label[iθ] : nothing  # only label once
                cm.lines!(ax, (1:nt)[valid], τIIev[valid], color=colors[iL], linestyle=linestyles[iθ], label=label)
            end
        end
        cm.Legend(fig[1, 2], ax, "orientation\n(color = domain width, see below)", labelsize=14, titlesize=13)

        ax2 = cm.Axis(fig[2, 1], xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$",
            xlabelsize=18, ylabelsize=18, titlesize=16, title=m === nothing ? "" : "m = $(m)")

        for (iL, Lwidth) in enumerate(Lwidths)
            for (iθ, angle_deg) in enumerate(angles_deg)
                data = load_final(datapath, nc_x, angle_deg, nt; Lwidth=Lwidth)
                inner_x, inner_y = inner_indices(data["x_c"], data["y_c"], half_width * Lwidth)
                τ_final[iL, iθ] = mean(data["τII"][inner_x, inner_y])
            end
            cm.scatterlines!(ax2, angles_deg, τ_final[iL, :], color=colors[iL], label=@sprintf("L = %.1f", Lwidth))
        end
        cm.Legend(fig[2, 2], ax2, "domain width", labelsize=14, titlesize=13)

        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_domainsweep.pdf", nc_x)
        cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
    end

    return τ_final
end

# 5) stress-time paths, yield function --------------------------------------------------------------------------

function yieldSurface(θ; D_BC=@SMatrix([1 0; 0 -1]), C1=10., C2=10/4, α1=0.5, α2=0.5)
    # if Dbg = 1:
    # τ = sqrt((Dl * cos(2*θ))^2*(α1*C1 + α2*C2)^2 + (Dn * sin(2*θ))^2 * C2^2)

    𝐷 = @SMatrix([D_BC[1, 1] D_BC[1, 2] 0.; D_BC[2, 1] D_BC[2, 2] 0.; 0. 0. 0.])
    𝑛 = @SVector([-cos(θ), sin(θ), 0.])
    # strain rate invariant normal to layering (3D)
    Dn = sqrt(dot(𝐷 * 𝑛, 𝐷 * 𝑛) - (dot(𝑛, 𝐷 * 𝑛))^2)
    # Deviatoric strain rate projection 
    Dd = 1/3*tr(𝐷) - dot(𝑛, 𝐷 * 𝑛)
    # Second invariant of dev strain rate
    De = sqrt(0.5*(𝐷[1, 1]^2 + 𝐷[2, 2]^2 + 𝐷[3, 3]^2) + 𝐷[1, 2]^2)
    # strain rate parallel to layers 
    Dp = De^2 - Dd^2 - Dn^2
    Dl = sqrt(Dp + Dd^2)

    # Macroscopic stress as function of invariants
    Τxx = Dl*(α1*C1 + α2*C2)
    Τyy = Dn*C2
    ΤII = sqrt(Dl^2*(α1*C1 + α2*C2)^2 + Dn^2 * C2^2)
    return (ΤII, Τxx, Τyy)
end

function flat_yieldsurface()

end

function compute_rotated_stress_time(datapath, nc_x, angles_deg, nt, half_width::Float64; Lwidth=nothing, save_fig=false, figpath=datapath)
    outp = Dict{Int,NamedTuple}()

    for angle_deg in angles_deg
        has_evolution(datapath, angle_deg; Lwidth=Lwidth) || continue

        Τ = stresstensor_time(datapath, nc_x, angle_deg, nt; half_width=half_width, Lwidth=Lwidth)

        Τxx_mean = dropdims(mean(Τ.xx, dims=(2, 3)), dims=(2, 3))
        Τyy_mean = dropdims(mean(Τ.yy, dims=(2, 3)), dims=(2, 3))
        Τxy_mean = dropdims(mean(Τ.xy, dims=(2, 3)), dims=(2, 3))

        outp[angle_deg] = (xx=Τxx_mean, yy=Τyy_mean, xy=Τxy_mean)
    end

    function plot_stress_time_paths()
        fig = cm.with_theme(cm.theme_latexfonts()) do
            fig = cm.Figure(size=(800, 400), px_per_unit=2)
            ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\tau_{xx} \ [-]$", ylabel=cm.L"$\tau_{xy} \ [-]$", aspect=cm.DataAspect())

            local sc
            angles_sorted = sort(collect(keys(outp)))
            Τxx_ana = zeros(length(angles_sorted))
            Τyy_ana = zeros(length(angles_sorted))
            Τxx_initial = fill(NaN, length(angles_sorted))
            Τxy_initial = fill(NaN, length(angles_sorted))
            Τxx_final = fill(NaN, length(angles_sorted))
            Τxy_final = fill(NaN, length(angles_sorted))
            # Categorical color per orientation (one solid color per stress-time path)
            colors = cm.cgrad(:roma, length(angles_sorted), categorical=true)

            for (i, angle_deg) in enumerate(angles_sorted)
                τxx = outp[angle_deg].xx
                τyy = outp[angle_deg].yy
                τxy = outp[angle_deg].xy
                valid = .!isnan.(τxx) .& .!isnan.(τxy)

                θ = deg2rad(angle_deg)
                𝐐 = @SMatrix([cos(θ) sin(θ); -sin(θ) cos(θ)])
                _, Τxx_ana[i], Τyy_ana[i] = yieldSurface(θ)

                if any(valid)
                    itlast = findlast(valid)
                    Τxx_final[i] = τxx[itlast]
                    Τxy_final[i] = τxy[itlast]
                end

                Τxx_initial =
                    τxx_rot = fill(NaN, nt)
                τxy_rot = fill(NaN, nt)
                for it in 1:nt
                    valid[it] || continue
                    τ_loc = @SMatrix([τxx[it] τxy[it]; τxy[it] τyy[it]])
                    τ_rot = 𝐐 * τ_loc * 𝐐'
                    τxx_rot[it] = τ_rot[1, 1]
                    τxy_rot[it] = τ_rot[1, 2]
                end

                # Color by time step within each path (original behaviour):
                sc = cm.scatter!(ax, τxx_rot[valid], τxy_rot[valid], color=(1:nt)[valid], colormap=:viridis, colorrange=(1, nt))
                sc = cm.scatter!(ax, -τxx_rot[valid], -τxy_rot[valid], color=(1:nt)[valid], colormap=:viridis, colorrange=(1, nt))


                # Color by orientation, one solid color per path:
                # sc = cm.scatter!(ax, τxx_rot[valid], τxy_rot[valid], color=colors[i])
            end
            cm.Colorbar(fig[1, 2], sc, label="time step")
            # cm.Colorbar(fig[1, 2], colormap=cm.cgrad(:roma, length(angles_sorted), categorical=true), limits=(minimum(angles_sorted), maximum(angles_sorted)), ticks=([minimum(angles_sorted), maximum(angles_sorted)], ["strong", "weak"]), label="layer orientation")
            cm.lines!(ax, Τxx_ana, -Τyy_ana, color=:red, linewidth=2, label="analytical yield surface")
            cm.lines!(ax, -Τxx_ana, Τyy_ana, color=:red, linewidth=2)
            cm.lines!(ax, -Τxx_ana, -Τyy_ana, color=:red, linewidth=2)
            cm.lines!(ax, Τxx_ana, Τyy_ana, color=:red, linewidth=2)
            cm.axislegend(ax, labelsize=14, titlesize=13)
            display(fig)
            fig
        end

        if save_fig
            mkpath(figpath)
            figname = @sprintf("LayeredVEVP_res%d_stresstimepaths_box%.2f.pdf", nc_x, half_width)
            cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
        end

        return fig
    end

    plot_stress_time_paths()

    return outp
end


# 6) Compare stress metrics plot ----------------------------------------------------------------------------
function compare_stress_metrics(datapath, nc_x, angles_deg, nt, half_width::Float64; m=nothing, save_fig=false, figpath=datapath, Lwidth=nothing)

    nθ = length(angles_deg)
    τII_macro = fill(NaN, nθ)
    τII_mean = fill(NaN, nθ)
    Τxx_final = fill(NaN, nθ)
    Τyy_final = fill(NaN, nθ)
    ana_ΤII = zeros(nθ)
    ana_Τxx = zeros(nθ)
    ana_Τyy = zeros(nθ)

    for (iθ, angle_deg) in enumerate(angles_deg)
        θ = deg2rad(angle_deg)
        ana_ΤII[iθ], ana_Τxx[iθ], ana_Τyy[iθ] = yieldSurface(θ)

        has_evolution(datapath, angle_deg; Lwidth=Lwidth) || continue

        Τ = stresstensor_time(datapath, nc_x, angle_deg, nt; half_width=half_width, Lwidth=Lwidth)
        τxx_t = dropdims(mean(Τ.xx, dims=(2, 3)), dims=(2, 3))
        τyy_t = dropdims(mean(Τ.yy, dims=(2, 3)), dims=(2, 3))
        τxy_t = dropdims(mean(Τ.xy, dims=(2, 3)), dims=(2, 3))
        τII_ev = τII_timeseries(datapath, nc_x, angle_deg, nt; half_width=half_width, Lwidth=Lwidth)

        valid = .!isnan.(τxx_t)
        any(valid) || continue
        itlast = findlast(valid)

        Τxx_final[iθ] = τxx_t[itlast]
        Τyy_final[iθ] = τyy_t[itlast]
        τxy_last = τxy_t[itlast]

        τII_macro[iθ] = sqrt(0.5 * (Τxx_final[iθ]^2 + Τyy_final[iθ]^2 + (-Τxx_final[iθ] - Τyy_final[iθ])^2) + τxy_last^2)
        τII_mean[iθ] = τII_ev[itlast]
    end

    fig = cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(800, 650), px_per_unit=2)
        ax = cm.Axis(fig[1, 1], title=m === nothing ? "" : "m = $(m)", xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16)
        cm.lines!(ax, angles_deg, ana_ΤII, color=:blue, label="τII (analytical)", linestyle=:dash)
        cm.lines!(ax, angles_deg, ana_Τxx, color=:red, label="Τxx (analytical)", linestyle=:dash)
        cm.lines!(ax, angles_deg, ana_Τyy, color=:green, label="Τyy (analytical)", linestyle=:dash)
        cm.scatterlines!(ax, angles_deg, τII_macro, color=:orange, label="τII from macrocomponents")
        cm.scatterlines!(ax, angles_deg, τII_mean, color=:gray, label="τII mean")
        cm.scatterlines!(ax, angles_deg, Τxx_final, color=:red, label="Τxx from model", linestyle=:solid)
        cm.scatterlines!(ax, angles_deg, τxy_last, color=:green, label="Τxy from model", linestyle=:solid)
        cm.Legend(fig[1, 2], ax, "metric", labelsize=14, titlesize=13)
        display(fig)
        fig
    end

    if save_fig
        mkpath(figpath)
        figname = @sprintf("LayeredVEVP_res%d_stressmetrics.pdf", nc_x)
        cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
    end

    return (τII_macro=τII_macro, τII_mean=τII_mean, ana_ΤII=ana_ΤII, ana_Τxx=ana_Τxx, ana_Τyy=ana_Τyy)
end


let
    # datapath = "/Users/filippozarabara/Documents/PHD/MEDIA/VEVP_Layered_Model/high_res_test_2/domainL3.0/"
    datapath = "/Users/filippozarabara/Documents/PHD/MEDIA/VEVP_Layered_Model/plastic_test_3/"

    nc_x = 200
    nt = 150
    angles_deg = round.(Int, rad2deg.(LinRange(0, π / 2, 13)))
    half_widths = 0.8

    # --- 1) grid/field plots --------------------------------------------------
    # single box size
    # plot_grid_fields(datapath, nc_x, angles_deg[11], nt, half_widths; save_fig=false)

    # several box sizes drawn on the same fields
    # plot_grid_fields(datapath, nc_x, angles_deg[16], nt; half_widths=[0.3, 0.6, 0.9], save_fig=true)

    # --- 2) stress-time evolution ----------------------------------------------
    # plot_stress_time(datapath, nc_x, angles_deg, nt, half_widths; m=4, save_fig=false)

    # --- 2b) stress component time series for a single orientation --------------
    # plot_stress_components_time(datapath, nc_x, 34, nt, half_widths)

    # --- 3) orientation vs final τII --------------------------------------------
    # single box size, strong/weak/strong colorbar
    # plot_orientation_strength(datapath, nc_x, angles_deg, nt, 0.6; m=4, save_fig=true)
    # plot_orientation_strength_evolution(datapath, nc_x, angles_deg, nt, half_widths)

    # several box sizes overlaid, legend by box size
    # plot_orientation_strength(datapath, nc_x, angles_deg, nt, half_widths; m=4, save_fig=true)

    # --- 4) domain-size comparison (NEW) ----------------------------------------
    # reads datapath/domainL<Lwidth>/fabric<angle>deg_evolution/... for each Lwidth
    # Lwidths = [1.0, 2.0, 3.0, 4.0, 5.0]
    # plot_domain_comparison(datapath, nc_x, Lwidths, angles_deg, nt; half_width=0.4, m=4, save_fig=true)

    # --- 5) stress-time paths in normal-shear stress space aligned with fabric
    outp = compute_rotated_stress_time(datapath, nc_x, angles_deg, nt, half_widths)

    # --- 6) Visualize stress matrics at each orientation
    # outp = compare_stress_metrics(datapath, nc_x, angles_deg, nt, half_widths)
end