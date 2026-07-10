using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils, JLD2
import CairoMakie as cm
using DifferentiationInterface
using ForwardDiff: ForwardDiff
const save = false
const figpath = "/Users/filippozarabara/Documents/PHD/MEDIA/VEVP_Layered_Model/high_res_test/"
const backend = AutoForwardDiff()

function Analyticalviscous(θ, η, δ, D_BC)
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

function Analyticalstress(θ, τ, inner_x, inner_y)

    𝑛 = @SVector([-cos(θ), sin(θ), 0.])
    Σxx = mean(τ.xx[inner_x, inner_y])
    Σyy = mean(τ.yy[inner_x, inner_y])
    Σxy = mean(0.25 * (τ.xy[i, j] + τ.xy[i+1, j] + τ.xy[i, j+1] + τ.xy[i+1, j+1])
               for i in inner_x, j in inner_y)
    Σzz = -Σxx - Σyy

    Σ = @SMatrix [Σxx Σxy 0.0; Σxy Σyy 0.0; 0.0 0.0 Σzz]

    T = Σ * 𝑛
    σnn = dot(𝑛, T) # n·Σn
    Σₙ = sqrt((dot(T, T) - σnn^2))
    # Σd  = 0.5 * (tr(Σ) - 3*σnn)
    Σe2 = (0.5 * (Σxx^2 + Σyy^2 + Σzz^2) + Σxy^2)
    Σₗ = sqrt(max(Σe2 - Σₙ^2, 0))
    return Σₙ, Σₗ
end

function DissipationFunction(ω, α1, α2, C1, C2, Dl, Dn, Dm)
    return α1 * C1 * sqrt(Dl^2 + (1 + α2 * ω)^2 * Dn^2) + α2 * C2 * sqrt(Dl^2 + (1 - α1 * ω)^2 * Dn^2) + Dm
end

function Minimise(α1, α2, C1, C2, Dl, Dn, Dm; ω0=0., max_iter=100, tol=1e-10)

    # from Castañeda and deBotton 1992: minimize the (convex) dissipation function.
    # D is convex -> D' is monotonically increasing -> bracket the sign change and bisect.
    D(ω) = DissipationFunction(ω, α1, α2, C1, C2, Dl, Dn, Dm)
    ∂D∂ω(ω) = ForwardDiff.derivative(D, ω)

    # 1. expand outward from ω0 until D'(lo) ≤ 0 ≤ D'(hi)
    lo, hi = ω0 - 1.0, ω0 + 1.0
    it = 0
    while ∂D∂ω(lo) > 0 && (it += 1) < max_iter
        lo -= 2 * (hi - lo)
    end
    while ∂D∂ω(hi) < 0 && (it += 1) < max_iter
        hi += 2 * (hi - lo)
    end
    (∂D∂ω(lo) ≤ 0 ≤ ∂D∂ω(hi)) || error("could not find a minimum")

    # 2. bisect on the derivative
    for iter in 1:max_iter
        mid = (lo + hi) / 2
        if ∂D∂ω(mid) > 0
            hi = mid
        else
            lo = mid
        end
        (hi - lo) < tol && return mid
    end
    return (lo + hi) / 2
end

function AnalyticalDissipation(D_BC, θ, α1, α2, C1, C2)
    𝑛 = @SVector([-cos(θ), sin(θ)])
    Dd = 1 / 3 * tr(D_BC) - dot(𝑛, D_BC * 𝑛)
    Dn = 2 / sqrt(3) * sqrt(dot(D_BC * 𝑛, D_BC * 𝑛) - dot(𝑛, D_BC * 𝑛))
    De = 1 / 2 * D_BC[1, 1]^2 + 1 / 2 * D_BC[2, 2]^2 + D_BC[1, 2]^2
    Dl = sqrt((De - Dd^2 - Dn^2)^2 + Dd^2)
    Dm = tr(D_BC)

    ωₑ = Minimise(α1, α2, C1, C2, Dl, Dn, Dm)
    Dₑ = DissipationFunction(ωₑ, α1, α2, C1, C2, Dl, Dn, Dm)

    # Stress invariants from Σ = ∂D⋆/∂ε̇
    Σl = ForwardDiff.derivative(x -> DissipationFunction(ωₑ, α1, α2, C1, C2, x, Dn, Dm), Dl)
    Σn = ForwardDiff.derivative(x -> DissipationFunction(ωₑ, α1, α2, C1, C2, Dl, x, Dm), Dn)
    Σm = ForwardDiff.derivative(x -> DissipationFunction(ωₑ, α1, α2, C1, C2, Dl, Dn, x), Dm)

    # Second invariant of deviatoric stress
    Σₑ = sqrt(Σl^2 + Σn^2)
    return Σₑ
end

@views function main(nc, nt, L, layering, BC_template, D_template, factorization, α1, η1, η2, G1, G2, C1, C2; fabric_angle=nothing)
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
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases, nmpc, noise)
    τIIev = ones(nt)
    α2 = 1 - α1

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Only account for the subdomain
    imin_x = argmin(abs.(a.X.c_e.x .+ (0.6 * L.x)))
    imax_x = argmin(abs.(a.X.c_e.x .- (0.6 * L.x)))
    imin_y = argmin(abs.(a.X.c_e.y .+ (0.6 * L.y)))
    imax_y = argmin(abs.(a.X.c_e.y .- (0.6 * L.y)))
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

    # Saving
    angle_deg = fabric_angle === nothing ? 0 : round(Int, rad2deg(fabric_angle))
    domain_dir = joinpath(figpath, @sprintf("domainL%.1f", L.x))
    subdir = joinpath(domain_dir, @sprintf("fabric%03ddeg_evolution", angle_deg))
    save && mkpath(subdir)
    save_every_step = true

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

        # Save grid files
        save && mkpath(figpath)
        save_every_step = true
        angle_deg = fabric_angle === nothing ? 0 : round(Int, rad2deg(fabric_angle))
        if save && (it == nt || save_every_step)
            dataname = @sprintf("fields_res%d_fabric%03ddeg_step%03d.jld2", nc.x, angle_deg, it)
            jldsave(joinpath(subdir, dataname);
                it=it,
                t=it * Δ.t,
                angle=fabric_angle,
                x_c=a.X.c.x, y_c=a.X.c.y,
                τxx=a.τ.xx,
                τyy=a.τ.yy,
                τxy=a.τ.xy,
                τII=a.τ.II,
                ε̇II=a.ε̇.II,
                Pt=a.Pt,
                Vx=a.V.x,
                Vy=a.V.y,
                σ1x=σ1.x,
                σ1y=σ1.y,
                phase_ratio=a.phase_ratios.c,
                phases=a.m.phase
            )
        end

        if it == nt
            cm.with_theme(cm.theme_latexfonts()) do
                fig = cm.Figure(size=(700, 600), px_per_unit=2)
                ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect(), xlabelsize=26, ylabelsize=26, titlesize=26)
                hm = cm.heatmap!(ax, a.X.c.x, a.X.c.y, a.τ.II[inx_c, iny_c], colormap=cgrad(:roma, rev=true))
                # cm.poly!(ax, cm.Rect(a.X.c_e.x[imin_x], a.X.c_e.y[imin_y], a.X.c_e.x[imax_x] - a.X.c_e.x[imin_x], a.X.c_e.y[imax_y] - a.X.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
                st = 15
                # cm.arrows2d!(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
                cm.Colorbar(fig[1, 2], hm, label=cm.L"$\tau_{II} \ [-]$", labelsize=18)

                ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect())
                # hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.η.c[inx_c, iny_c], colormap=:roma)
                hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.ε̇.II[inx_c, iny_c], colormap=cgrad(:roma, rev=true))
                # cm.poly!(ax2, cm.Rect(a.X.c_e.x[imin_x], a.X.c_e.y[imin_y], a.X.c_e.x[imax_x] - a.X.c_e.x[imin_x], a.X.c_e.y[imax_y] - a.X.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
                # cm.Colorbar(fig[1, 4], hm2, label="η")
                cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\dot\varepsilon_{II} \ [-]$", labelsize=18)

                ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect())
                hm3 = cm.heatmap!(ax3, a.X.c.x, a.X.c.y, a.V.x[inx_Vx, iny_Vx], colormap=:vik)
                cm.Colorbar(fig[2, 2], hm3, label=cm.L"$v_x \ [-]$", labelsize=18)

                ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect())
                hm4 = cm.heatmap!(ax4, a.X.c.x, a.X.c.y, a.V.y[inx_Vy, iny_Vy], colormap=:vik)
                cm.Colorbar(fig[2, 4], hm4, label=cm.L"$v_y \ [-]$", labelsize=18)

                # ax5_title = fabric_angle === nothing ? "Fabric inclination" : @sprintf("Fabric inclination = %.1f°", rad2deg(fabric_angle))
                ax5 = cm.Axis(fig[3, 1:4], xlabel="time step", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18) #, title=ax5_title)
                cm.xlims!(ax5, 0, nt)
                cm.lines!(ax5, 1:it, τIIev[1:it])
                # display(fig)

                # Save 
                if save
                    angle_deg = fabric_angle === nothing ? 0 : round(Int, rad2deg(fabric_angle))
                    mkpath(domain_dir)
                    figname = @sprintf("LayeredVEVP_res%d_fabric%03ddeg.png", nc.x, angle_deg)
                    cm.save(joinpath(domain_dir, figname), fig, px_per_unit=4)
                end
            end
        end
    end

    τIIanS, τIIanE = Analyticalstress(fabric_angle, a.τ, inner_x, inner_y)
    # Dis = AnalyticalDissipation(D_BC, fabric_angle, α1, α2, C1, C2)

    τxx = mean(a.τ.xx[inner_x, inner_y])
    τyy = mean(a.τ.yy[inner_x, inner_y])
    τxy = mean(0.25 * (a.τ.xy[i, j] + a.τ.xy[i+1, j] + a.τ.xy[i, j+1] + a.τ.xy[i+1, j+1])
               for i in inner_x, j in inner_y)
    τIIsec = sqrt(0.5 * (τxx^2 + τyy^2 + (-τxx - τyy)^2) + τxy^2)

    display(to)

    return mean(a.τ.II[inner_x, inner_y]), τIIev, τIIanS, τIIanE, τIIsec

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

    nc = (x=50, y=50)
    nt = 50
    # L = [1.0, 2.0, 3., 4., 5.]
    L = 1.

    # Discretise angle of layer 
    # nθ = 1
    nθ = 31
    θ = LinRange(0, π / 2, nθ)
    # θ = LinRange(π / 8, π / 8, nθ)
    τ_cart = zeros(nθ)
    τ_cart_lay = zeros(length(L), nθ)
    τ_cart_anaS = zeros(length(L), nθ)
    τ_cart_anaE = zeros(length(L), nθ)
    τ_time = zeros(length(L), nθ, nt)
    τ_tensor = zeros(length(L), nθ)

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

    # Run them all
    for (iL, Lw) in enumerate(L)
        for iθ in 1:31

            layering = Layering(
                (0 * 0.25, 0.025),
                0.2,
                α2;
                θ=θ[iθ],
                perturb_amp=0. * 1.0,
                perturb_width=1.0
            )

            τ_cart_lay[iL, iθ], τ_time[iL, iθ, :], τ_cart_anaS[iL, iθ], τ_cart_anaE[iL, iθ], τ_tensor[iL, iθ] = main(nc, nt, (x=Lw, y=Lw), layering, BCs[1], D_BCs[1], :lu, α1, η1, η2, G1, G2, C1, C2; fabric_angle=θ[iθ])
        end
    end

    cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(size=(800, 650), px_per_unit=2)
        # colors = cm.cgrad(:roma, nL, categorical=true)   # color = domain width

        # iθ_mid = (nθ + 1) ÷ 2  # index of 45°
        # iθ_first = 1
        # iθ_last = nθ
        # linestyles = Dict(iθ_first => :solid, iθ_mid => :dash, iθ_last => :dot)
        # orientation_label = Dict(iθ_first => "Strong (hor)", iθ_mid => "Weak (45°)", iθ_last => "Strong (vert)")

        # ax = cm.Axis(fig[1, 1], title="m = $(m)", xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16)
        # for (iL, Lwidth) in enumerate(Lwidths)
        #     for iθ in (iθ_first, iθ_mid, iθ_last)
        #         label = iL == 1 ? orientation_label[iθ] : nothing  # only label once
        #         cm.lines!(ax, 1:nt, τ_time[iL, iθ, :], color=colors[iL], linestyle=linestyles[iθ], label=label)
        #     end
        # end
        # cm.Legend(fig[1, 2], ax, "orientation\n(color = domain width, see below)", labelsize=14, titlesize=13)

        ax2 = cm.Axis(fig[1, 1], title="m = $(m)", xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16)
        # for (iL, Lwidth) in enumerate(Lwidths)
        #     cm.scatterlines!(ax2, θ * 180 / π, τ_cart_lay[iL, :], color=:blue, label=@sprintf("domain = %.1f", Lwidth))
        #     cm.scatterlines!(ax2, θ * 180 / π, τ_cart_anaS[iL, :], color=:red, label=@sprintf("domain = %.1f", Lwidth))
        #     cm.scatterlines!(ax2, θ * 180 / π, τ_cart_anaE[iL, :], color=:green, label=@sprintf("domain = %.1f", Lwidth))

        # end

        for iL in 1:1
            cm.scatterlines!(ax2, θ * 180 / π, τ_cart_lay[iL, :], color=:blue, label=@sprintf("τII"))
            cm.scatterlines!(ax2, θ * 180 / π, τ_cart_anaS[iL, :], color=:red, label=@sprintf("Σₙ"))
            cm.scatterlines!(ax2, θ * 180 / π, τ_cart_anaE[iL, :], color=:green, label=@sprintf("Σₗ"))
            cm.scatterlines!(ax2, θ * 180 / π, τ_tensor[iL, :], color=:orange, label="τII from components")
            cm.scatterlines!(ax2, θ * 180 / π, sqrt.(τ_cart_anaE[iL, :] .^ 2 .+ τ_cart_anaS[iL, :] .^ 2), color=:purple, label="τII computed")
            cm.Legend(fig[1, 2], ax2, "domain width", labelsize=14, titlesize=13)
        end
        display(fig)

        if save
            mkpath(figpath)
            figname = @sprintf("LayeredVEVP_res%d_domainsp.png", nc.x)
            cm.save(joinpath(figpath, figname), fig, px_per_unit=4)
        end
    end

end