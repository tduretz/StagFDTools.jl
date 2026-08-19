using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils
import CairoMakie as cm
const save = true

@views function main(nc, nt, layering, BC_template, D_template, factorization, η1, η2, G1, G2, C1, C2; fabric_angle=nothing)
    #--------------------------------------------#   

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; plasticity=VonMises, compressible=false)
    materials.η0 .= [η1, η2]
    materials.G .= [G1, G2]
    materials.plasticity.C .= [C1, C2]
    materials.plasticity.ηvp .= [1e-3, 1e-3]
    preprocess!(materials)

    nmpc = (x=4, y=4)
    noise = false

    # Time steps
    Δt0 = 0.5

    # Newton solver
    iter_params = IterParams(niter=3, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    L = (x=1.0, y=1.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases, nmpc, noise)
    τIIev = ones(nt)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Only account for the subdomain
    imin_x = argmin(abs.(a.X.c_e.x .+ 0.3))
    imax_x = argmin(abs.(a.X.c_e.x .- 0.3))
    imin_y = argmin(abs.(a.X.c_e.y .+ 0.3))
    imax_y = argmin(abs.(a.X.c_e.y .- 0.3))
    inner_x = imin_x:imax_x
    inner_y = imin_y:imax_y

    # Initial velocity & pressure field
    a.V.x[inx_Vx, iny_Vx] .= D_BC[1, 1] * a.X.v.x .+ D_BC[1, 2] * a.X.c.y'
    a.V.y[inx_Vy, iny_Vy] .= D_BC[2, 1] * a.X.c.x .+ D_BC[2, 2] * a.X.v.y'
    a.Pt[inx_c, iny_c] .= 0.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * a.X.v.x .+ D_BC[1, 2] * a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * a.X.v.x .+ D_BC[1, 2] * a.X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * a.X.v.x[1] .+ D_BC[2, 2] * a.X.v.y)
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * a.X.v.x[end] .+ D_BC[2, 2] * a.X.v.y)
    end

    # MARKERS ------------------------------------------------------------
    # Assign marker phases from layering geometry (1 or 2) #           |
    for I in CartesianIndices(a.m.phase) #                                |
        xm = a.m.Xm[I]
        ym = a.m.Ym[I]
        isin = inside(@SVector([xm, ym]), layering)
        a.m.phase[I] = isin ? 2 : 1
    end

    # Build extended vertex arrays (with ghost vertices) and accumulate marker contributions
    SetPhaseRatios!(a.phase_ratios, a.m, a.X.c_e.x, a.X.c_e.y, a.X.v_e.x, a.X.v_e.y, Δ, nphases)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Principal stress
        σ1 = (x=zeros(size(a.Pt)), y=zeros(size(a.Pt)), v=zeros(size(a.Pt)))

        for i in inx_c, j in iny_c
            τxyc = 1 / 4 * (a.τ.xy[i, j] + a.τ.xy[i+1, j] + a.τ.xy[i, j+1] + a.τ.xy[i+1, j+1])
            σ = @SMatrix[-a.Pt[i, j]+a.τ.xx[i, j] τxyc 0.; τxyc -a.Pt[i, j]+a.τ.yy[i, j] 0.; 0. 0. -a.Pt[i, j]+(-a.τ.xx[i, j]-a.τ.yy[i, j])]
            v = eigvecs(σ)
            σp = eigvals(σ)
            scale = sqrt(v[1, 1]^2 + v[2, 1]^2)
            σ1.x[i, j] = v[1, 1] / scale
            σ1.y[i, j] = v[2, 1] / scale
            σ1.v[i] = σp[1]
        end
        τIIev[it] = mean(a.τ.II[inner_x, inner_y])

        if it == nt
            cm.with_theme(cm.theme_latexfonts()) do
                fig = cm.Figure(size=(700, 600), px_per_unit=2)
                ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5], xlabelsize=26, ylabelsize=26, titlesize=26)
                hm = cm.heatmap!(ax, a.X.c.x, a.X.c.y, a.τ.II[inx_c, iny_c], colormap=cgrad(:roma, rev=true))
                cm.poly!(ax, cm.Rect(a.X.c_e.x[imin_x], a.X.c_e.y[imin_y], a.X.c_e.x[imax_x] - a.X.c_e.x[imin_x], a.X.c_e.y[imax_y] - a.X.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
                st = 15
                # cm.arrows2d!(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
                cm.Colorbar(fig[1, 2], hm, label=cm.L"$\tau_{II} \ [-]$", labelsize=18)

                ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                # hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.η.c[inx_c, iny_c], colormap=:roma)
                hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.ε̇.II[inx_c, iny_c], colormap=cgrad(:roma, rev=true))
                cm.poly!(ax2, cm.Rect(a.X.c_e.x[imin_x], a.X.c_e.y[imin_y], a.X.c_e.x[imax_x] - a.X.c_e.x[imin_x], a.X.c_e.y[imax_y] - a.X.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
                # cm.Colorbar(fig[1, 4], hm2, label="η")
                cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\dot\varepsilon_{II} \ [-]$", labelsize=18)

                ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect(), xticks=([-0.25, 0.25]))
                hm3 = cm.heatmap!(ax3, a.X.v.x, a.X.c.y, a.V.x[inx_Vx, iny_Vx], colormap=:vik)
                cm.Colorbar(fig[2, 2], hm3, label=cm.L"$v_x \ [-]$", labelsize=18)

                ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect(), xticks=([-0.25, 0.25]))
                hm4 = cm.heatmap!(ax4, a.X.c.x, a.X.v.y, a.V.y[inx_Vy, iny_Vy], colormap=:vik)
                cm.Colorbar(fig[2, 4], hm4, label=cm.L"$v_y \ [-]$", labelsize=18)

                ax5_title = fabric_angle === nothing ? "Fabric inclination" : @sprintf("Fabric inclination = %.1f°", rad2deg(fabric_angle))
                ax5 = cm.Axis(fig[3, 1:4], xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$", title=ax5_title, xlabelsize=18, ylabelsize=18)
                cm.xlims!(ax5, 0, nt)
                cm.lines!(ax5, 1:it, τIIev[1:it])
                display(fig)

                # Save 
                if save
                    angle_deg = fabric_angle === nothing ? 0 : round(Int, rad2deg(fabric_angle))
                    figpath = "/Users/filippozarabara/Documents/PHD/MEDIA/VEVP_Layered_Model/Resolution test/"
                    mkpath(figpath)
                    figname = @sprintf("LayeredVEVP_res%d_cohesionshearm_fabric%03ddeg.png", nc.x, angle_deg)
                    cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
                end
            end
        end
    end

    display(to)

    return mean(a.τ.II[inner_x, inner_y]), τIIev

end

let
    # Boundary condition templates
    BCs = [
        # :EW_periodic,
        # :all_Dirichlet,
        :free_slip,
    ]

    # Boundary deformation gradient matrix
    D_BCs = [
        @SMatrix([1 0; 0 -1]),
    ]

    nc = [50, 100, 200, 400]
    nt = 50

    # Discretise angle of layer 
    # nθ = 1
    nθ = 7
    θ = LinRange(0, π / 2, nθ)
    # θ = LinRange(π / 8, π / 8, nθ)
    # τ_cart = zeros(nθ)
    τ_cart_lay = zeros(nθ, length(nc))
    # τ_cart_ana = zeros(nθ)
    τ_time = zeros(nθ, nt, length(nc))

    #  Viscosity
    m = 4
    η2 = 1e10
    η1 = η2

    G2 = 1.
    G1 = G2

    C2 = 10.
    C1 = C2 / m

    α2 = 0.5
    α1 = 1 - α2

    ηn = α1 * η1 + α2 * η2
    δ = (α1 + α2 * m) * (α1 + α2 / m)

    # elasticity
    tmax = 1.0

    cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(1100, 700), px_per_unit=2)

        # Run them all
        for (res, val) in enumerate(nc)
            for iθ in eachindex(θ)

                layering = Layering(
                    (0 * 0.25, 0.025),
                    0.15,
                    α2;
                    θ=θ[iθ],
                    perturb_amp=0. * 1.0,
                    perturb_width=1.0
                )

                τ_cart_lay[iθ, res], τ_time[iθ, :, res] = main((x=val, y=val), nt, layering, BCs[1], D_BCs[1], :lu, η1, η2, G1, G2, C1, C2; fabric_angle=θ[iθ])
            end
        end

        colors = Makie.wong_colors()[1:length(nc)]
        linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :dash, :dot][1:4]

        # Plot 1: stress-time paths
        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$",
            xlabelsize=24, ylabelsize=24, xticklabelsize=20, yticklabelsize=20)
        for (res, nc_val) in enumerate(nc)
            for iθ in 1:4
                cm.lines!(ax, 1:nt, τ_time[iθ, :, res],
                    color=colors[res],
                    linestyle=linestyles[iθ])
            end
        end

        # Plot 2: τ_II vs θ
        ax2 = cm.Axis(fig[2, 1], xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$",
            xlabelsize=24, ylabelsize=24, xticklabelsize=20, yticklabelsize=20)
        for (res, nc_val) in enumerate(nc)
            cm.lines!(ax2, θ * 180 / π, τ_cart_lay[:, res],
                color=colors[res],
                linestyle=:solid,
                label="$(nc_val)²")
        end

        res_elements = [cm.LineElement(color=colors[res], linestyle=:solid) for res in eachindex(nc)]
        angle_elements = [cm.LineElement(color=:black, linestyle=linestyles[iθ]) for iθ in 1:4]
        res_labels = ["$(nc_val)²" for nc_val in nc]
        angle_labels = ["$(round(Int, rad2deg(θ[iθ])))°" for iθ in 1:4]

        # cm.Legend(fig[1:2, 2],
        #     [angle_elements, res_elements],
        #     [angle_labels, res_labels],
        #     ["Fabric orientation", "Resolution"],
        #     framevisible=true, titlesize=20, labelsize=20,
        #     patchsize=(50, 20), rowgap=6, gridshalign=:center,
        #     padding=(12, 12, 12, 12), nbanks=2)

        cm.Legend(fig[1, 2],
            angle_elements, angle_labels, "Fabric orientation",
            framevisible=true, titlesize=21, labelsize=23,
            patchsize=(50, 20), rowgap=6, gridshalign=:center,
            padding=(12, 12, 12, 12), nbanks=2)

        cm.Legend(fig[2, 2],
            res_elements, res_labels, "Resolution",
            framevisible=true, titlesize=21, labelsize=23,
            patchsize=(50, 20), rowgap=6, gridshalign=:center,
            padding=(12, 12, 12, 12), nbanks=2)

        display(fig)
        # Save 
        if save
            figpath = "/Users/filippozarabara/Documents/PHD/MEDIA/VEVP_Layered_Model/Resolution test/"
            mkpath(figpath)
            figname = @sprintf("LayeredVEVP_restest.png")
            cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
        end
    end

end