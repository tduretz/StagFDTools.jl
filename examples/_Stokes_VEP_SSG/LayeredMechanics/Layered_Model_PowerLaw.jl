using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils, JLD2
import CairoMakie as cm
using DifferentiationInterface
using ForwardDiff: ForwardDiff
const save = true
const backend = AutoForwardDiff()
const figpath = "$(@__DIR__)/../../../"

function Analytical(θ, η, δ, D_BC)
    #= define velocity gradient components and resulting deviatoric strain rate components
    pure shear   ε̇ = [ε̇xx  0 ;  0  -ε̇xx]
    simple shear ε̇ = [ 0  ε̇xy; ε̇xy   0 ] =#
    Dxx = D_BC[1, 1]
    Dyy = -Dxx
    Dxy = D_BC[1, 2]
    Dkk = Dxx + Dyy

    ε̇ = @SVector([Dxx - Dkk / 3, Dyy - Dkk / 3, Dxy])

    # Normal vector of anisotropic direction
    n1 = -cos(θ)
    n2 = sin(θ)

    # compute isotropic and layered components for 𝐷
    Δ0 = 2 * n1^2 * n2^2
    Δ1 = n1 * n2^3 - n2 * n1^3
    Δ = @SMatrix([Δ0 -Δ0 2*Δ1; -Δ0 Δ0 -2*Δ1; Δ1 -Δ1 1-2*Δ0])
    A = @SMatrix([1 0 0; 0 1 0; 0 0 1])

    # compute 𝐷
    𝐷 = 2 * η * A - 2 * (η - η / δ) * Δ

    τ = 𝐷 * ε̇

    τ_II = sqrt(0.5 * (τ[1]^2 + τ[2]^2 + (-τ[1] - τ[2])^2) + τ[3]^2)
    return τ_II
end

@views function main(nc, nt, L, layering, BC_template, D_template, factorization, η1, η2, G1, G2, C1, C2, n1, n2; fabric_angle=nothing)
    #--------------------------------------------#   

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; compressible=false)
    materials.η0 .= [η1, η2]
    materials.n .= [n1, n2]
    materials.G .= [G1, G2]
    # materials.plasticity.C .= [C1, C2]
    preprocess!(materials)

    nmpc = (x = 1, y = 1)
    noise = false

    # Time steps
    Δt0 = 0.5

    # Newton solver
    iter_params = IterParams(niter=10, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases, nmpc, noise)
    τIIev = ones(nt)
    τxxev = ones(nt)
    τyyev = ones(nt)
    τxyev = ones(nt)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Only account for the subdomain
    imin_x = argmin(abs.(a.X.c_e.x .+ (0.3 * L.x)))
    imax_x = argmin(abs.(a.X.c_e.x .- (0.3 * L.x)))
    imin_y = argmin(abs.(a.X.c_e.y .+ (0.3 * L.y)))
    imax_y = argmin(abs.(a.X.c_e.y .- (0.3 * L.y)))
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

    if nmpc.x == 1
         # Set material geometry
        for i in inx_c, j in iny_c   # loop on centroids
            𝐱 = @SVector([a.X.c.x[i-1], a.X.c.y[j-1]])
            isin = inside(𝐱, layering)
            if isin
                a.phases.c[i, j] = 2
            end
        end

        for i in inx_v, j in iny_v  # loop on vertices
            𝐱 = @SVector([a.X.v.x[i-1], a.X.v.y[j-1]])
            isin = inside(𝐱, layering)
            if isin
                a.phases.v[i, j] = 2
            end
        end
        # Convert to phase ratios
        FillPhaseRatios!(a)
    else
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
    end

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
        τxxev[it] = mean(a.τ.xx[inner_x, inner_y])
        τyyev[it] = mean(a.τ.yy[inner_x, inner_y])
        τxyev[it] = mean(0.25 * (a.τ.xy[i, j] + a.τ.xy[i+1, j] + a.τ.xy[i, j+1] + a.τ.xy[i+1, j+1])
               for i in inner_x, j in inner_y)

        # if it == nt
            # cm.with_theme(cm.theme_latexfonts()) do
            #     fig = cm.Figure(size=(700, 600), px_per_unit=2)
            #     ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect(), xlabelsize=26, ylabelsize=26, titlesize=26)
            #     hm = cm.heatmap!(ax, a.X.c.x, a.X.c.y, a.τ.II[inx_c, iny_c], colormap=:bluesreds)
            #     # cm.poly!(ax, cm.Rect(a.X.c_e.x[imin_x], a.X.c_e.y[imin_y], a.X.c_e.x[imax_x] - a.X.c_e.x[imin_x], a.X.c_e.y[imax_y] - a.X.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
            #     st = 15
            #     # cm.arrows2d!(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
            #     cm.Colorbar(fig[1, 2], hm, label=cm.L"$\tau_{II} \ [-]$", labelsize=18)

            #     ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect())
            #     # hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.η.c[inx_c, iny_c], colormap=:roma)
            #     hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.ε̇.II[inx_c, iny_c], colormap=:bluesreds)
            #     # cm.poly!(ax2, cm.Rect(a.X.c_e.x[imin_x], a.X.c_e.y[imin_y], a.X.c_e.x[imax_x] - a.X.c_e.x[imin_x], a.X.c_e.y[imax_y] - a.X.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
            #     # cm.Colorbar(fig[1, 4], hm2, label="η")
            #     cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\dot\varepsilon_{II} \ [-]$", labelsize=18)

            #     ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect())
            #     hm3 = cm.heatmap!(ax3, a.X.c.x, a.X.c.y, a.V.x[inx_Vx, iny_Vx], colormap=:vik)
            #     cm.Colorbar(fig[2, 2], hm3, label=cm.L"$v_x \ [-]$", labelsize=18)

            #     ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect())
            #     hm4 = cm.heatmap!(ax4, a.X.c.x, a.X.c.y, a.V.y[inx_Vy, iny_Vy], colormap=:vik)
            #     cm.Colorbar(fig[2, 4], hm4, label=cm.L"$v_y \ [-]$", labelsize=18)

            #     # ax5_title = fabric_angle === nothing ? "Fabric inclination" : @sprintf("Fabric inclination = %.1f°", rad2deg(fabric_angle))
            #     ax5 = cm.Axis(fig[3, 1:4], xlabel="time step", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18) #, title=ax5_title)
            #     cm.xlims!(ax5, 0, nt)
            #     cm.lines!(ax5, 1:it, τIIev[1:it])
            #     display(fig)

        #         # Save 
        #         if save
        #             angle_deg = fabric_angle === nothing ? 0 : round(Int, rad2deg(fabric_angle))
        #             mkpath(domain_dir)
        #             figname = @sprintf("Layered_res%d_fabric%03ddeg.png", nc.x, angle_deg)
        #             cm.save(joinpath(domain_dir, figname), fig, px_per_unit=4)
        #         end
            # end
        # end
    end

    display(to)

    return τIIev, τxxev, τyyev, τxyev

end

let
    # Boundary condition templates
    BCs = [
        # :EW_periodic,
        # :all_Dirichlet,
        :free_slip,
    ]

    # Boundary deformation gradient matrix
    ε̇ref  = 1.0
    D_BCs = [
        @SMatrix([1 0; 0 -1]),
    ]
    nc = (x = 50, y = 50)
    nt = 1
    L = 1.

    # Discretise angle of layer 
    nθ = 17
    θ = LinRange(0, π, nθ)

    τpwl = [1.0] #, 4.0, 10.0
    npwl = [8.0] #10. 20. 50. 100.

    for it in eachindex(τpwl), in in eachindex(npwl)
        τref2 = τpwl[it]
        n1 = n2 = npwl[in]

        τ_lin  = (II = zeros(nθ, nt), xx =zeros(nθ,nt), yy = zeros(nθ,nt), xy = zeros(nθ,nt))
        τ_pwl = (II = zeros(nθ, nt), xx =zeros(nθ,nt), yy = zeros(nθ,nt), xy = zeros(nθ,nt))
        τ_ana  = zeros(nθ)
        τ_cart = zeros(nθ)

        #  Anisotropy parameters
        m  = 10
        τref1 = τref2 / m
        η2o = τref2 / (2 * ε̇ref)
        η1o = η2o / m
        C1 = τref1^(-n1) * ε̇ref
        C2 = τref2^(-n2) * ε̇ref
        η1 = 1/2 * C1^(-1/n1)
        η2 = 1/2 * C2^(-1/n2)

        α2 = 0.5
        α1 = 1 - α2

        ηn = α1 * η1 + α2 * η2
        δ  = (α1 + α2 * m) * (α1 + α2 / m)

        # elasticity 
        tmax = 1.0
        G2 = 1.0e6
        G1 = G2
        C2 = C1 = 1e6

        for iθ in eachindex(θ)
            layering = Layering(
                (0 * 0.25, 0 * 0.025),
                0.2,
                α2;
                θ=θ[iθ],
                perturb_amp=0. * 1.0,
                perturb_width=1.0
            )
            τ_lin.II[iθ, :], τ_lin.xx[iθ, :], τ_lin.yy[iθ, :], τ_lin.xy[iθ, :] = main(nc, nt, (x=L, y=L), layering, BCs[1], D_BCs[1], :lu, η1o, η2o, G1, G2, C1, C2, 1, 1; fabric_angle=θ[iθ])
            τ_pwl.II[iθ, :], τ_pwl.xx[iθ, :], τ_pwl.yy[iθ, :], τ_pwl.xy[iθ, :] = main(nc, nt, (x=L, y=L), layering, BCs[1], D_BCs[1], :lu, η1, η2, G1, G2, C1, C2, n1, n2; fabric_angle=θ[iθ])
            τ_ana[iθ]    = Analytical(θ[iθ], ηn, δ, D_BCs[1])

        end

        ε̇bg = sqrt( sum(1/2 .* D_BCs[1][:].^2))

        # Strongest end-member
        ηeff = α1*η1 + α2*η2
        @show τstrong    = 2*ηeff*ε̇bg

        # Weakest end-member
        ηeff = (α1/η1 + α2/η2)^(-1)
        @show τweak      = 2*ηeff*ε̇bg

        τ_cart .= τstrong * sqrt.(((δ^2 - 1) * cos.(2 .* θ).^2 .+ 1) / (δ^2))

        if save
            # phase_ratio = round(Int, α2*10)
            domain_dir = joinpath(figpath) # @sprintf("res%.1f", Int(nc.x)))
            mkpath(domain_dir)
            dataname = @sprintf("PowerLaw_multilayer_res%03d_%02d_pwl%03d.jl2d", nc.x, τref2, n1)
            jldsave(joinpath(domain_dir, dataname);
            τ_lin,
            τ_pwl,
            τ_cart,
            τ_ana,
            τstrong,
            τweak,
            η1,
            η2,
            G1,
            G2,
            n1,
            n2,
            α1,
            α2,
            m,
            δ,
            θ,
            η1o,
            η2o,
            τref1,
            τref2,
            ε̇ref,
            ηn)
        end

        a = b = 1

        for it in 1:nt
            cm.with_theme(cm.theme_latexfonts()) do
            fig   = cm.Figure(fontsize=15)
            ax    = cm.Axis(fig[1,1], xlabel= cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II}$ [-]", title="it $(it)")
            cm.lines!(ax, θ*180/π, τ_lin.II[:, it], label="Layering")
            cm.lines!(ax, θ*180/π, τ_pwl.II[:, it], label="Power Law")
            cm.lines!(ax, θ*180/π, τstrong*ones(size(θ)), color=:gray, linestyle=:dash, label="End-Member (Biot et al., 1965)")
            cm.lines!(ax, θ*180/π, τweak*ones(size(θ)), color=:gray, linestyle=:dash, label="End-Member (Biot et al., 1965)")
            cm.scatter!(ax, θ[1:b:end]*180/π, τ_cart[1:b:end], label="Expression", markersize=10)
            cm.scatter!(ax, θ[1:a:end]*180/π, τ_ana[1:a:end], label="Analytical", marker=:utriangle, markersize=10, color=cm.Cycled(2))
            cm.Legend(fig[2,1], ax, framevisible=false, orientation=:horizontal, unique=true, nbanks=3, cm.L"$\tau_{II}$    ($δ \approx$ %$(round(Int,δ)))")
            display(fig)
        end
    end
end
end