using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils
import CairoMakie as cm

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

@views function main(nc, nt, layering, BC_template, D_template, factorization, η1, η2, G1, G2, C1, C2)
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

        fig = cm.Figure()
        ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect())
        hm = cm.heatmap!(ax, a.X.c.x, a.X.c.y, a.τ.II[inx_c, iny_c], colormap=:bluesreds)
        cm.poly!(ax, cm.Rect(a.X.c_e.x[imin_x], a.X.c_e.y[imin_y], a.X.c_e.x[imax_x] - a.X.c_e.x[imin_x], a.X.c_e.y[imax_y] - a.X.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
        st = 15
        cm.arrows2d!(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
        cm.Colorbar(fig[1, 2], hm, label="τII")

        ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect())
        hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.η.c[inx_c, iny_c], colormap=:bluesreds)
        cm.Colorbar(fig[1, 4], hm2, label="η")

        ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect())
        hm3 = cm.heatmap!(ax3, a.X.c.x, a.X.c.y, a.V.x[inx_Vx, iny_Vx], colormap=:bluesreds)
        cm.Colorbar(fig[2, 2], hm3, label="Vx")

        ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect())
        hm4 = cm.heatmap!(ax4, a.X.c.x, a.X.c.y, a.V.y[inx_Vx, iny_Vx], colormap=:bluesreds)
        cm.Colorbar(fig[2, 4], hm4, label="Vy")

        ax5 = cm.Axis(fig[3, 1:4])
        cm.xlims!(ax5, 0, nt)
        # cm.ylims!(ax5, 0, 2.5)
        cm.lines!(ax5, 1:it, τIIev[1:it])
        display(fig)
        display(fig)

        ax6 = cm.Axis(fig[1, 4])
        cm.heatmap!
    end

    # display(to)

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

    nc = (x=200, y=200)
    nt = 40

    # Discretise angle of layer 
    nθ = 1
    θ = LinRange(-π / 2 + 5, -π / 2 + 5, nθ)
    τ_cart = zeros(nθ)
    τ_cart_lay = zeros(nθ)
    τ_cart_ana = zeros(nθ)
    τ_time = zeros(nθ, nt)

    #  Anisotropy parameters
    η2 = 1e2
    m = 1
    η1 = η2 / m

    α2 = 0.5
    α1 = 1 - α2

    ηn = α1 * η1 + α2 * η2
    δ = (α1 + α2 * m) * (α1 + α2 / m)

    # elasticity
    tmax = 1.0
    G2 = G1 = 1.0
    C2 = C1 = 10.
    C2 = 4.
    C1 = C2 / 4    # @abacaxi-seco HARDCODED factor 2, to remove

    # Run them all
    for iθ in eachindex(θ)

        layering = Layering(
            (0 * 0.25, 0.025),
            0.15,
            α2;
            θ=θ[iθ],
            perturb_amp=0 * 1.0,
            perturb_width=1.0
        )

        # @abacaxi-seco Note that I switched to LU factorisation as the Jacobian is already not symmetric with elasto-palsticity
        τ_cart_lay[iθ], τ_time[iθ, :] = main(nc, nt, layering, BCs[1], D_BCs[1], :lu, η1, η2, G1, G2, C1, C2)
        τ_cart_ana[iθ] = Analytical(θ[iθ], ηn, δ, D_BCs[1])

    end

    ε̇bg = sqrt(sum(1 / 2 .* D_BCs[1][:] .^ 2))

    # Strongest end-member
    ηeff = α1 * η1 + α2 * η2
    @show τstrong = 2 * ηeff * ε̇bg

    # Weakest end-member
    ηeff = (α1 / η1 + α2 / η2)^(-1)
    @show τweak = 2 * ηeff * ε̇bg

    τ_cart .= τstrong * sqrt.(((δ^2 - 1) * cos.(2 .* θ) .^ 2 .+ 1) / (δ^2))

    cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(fontsize=15)

        ax = cm.Axis(fig[0, 1], xlabel=cm.L"$$step]", ylabel=cm.L"$\tau_{II}$ [-]")
        for iθ in eachindex(θ)
            cm.lines!(ax, 1:nt, τ_time[iθ, :])
        end
        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II}$ [-]")
        cm.lines!(ax, θ * 180 / π, τ_cart_lay, label="Layering")
        # cm.lines!(ax, θ*180/π, τstrong*ones(size(θ)), color=:gray, linestyle=:dash, label="End-Member (Biot et al., 1965)")
        # cm.lines!(ax, θ*180/π, τweak*ones(size(θ)), color=:gray, linestyle=:dash, label="End-Member (Biot et al., 1965)")
        # cm.scatter!(ax, θ[1:6:end]*180/π, τ_cart[1:6:end], label="Expression", markersize=10)
        # cm.scatter!(ax, θ[1:8:end]*180/π, τ_cart_ana[1:8:end], label="Analytical", marker=:utriangle, markersize=10, color=cm.CGrid.c.yled(3))
        cm.Legend(fig[2, 1], ax, framevisible=false, orientation=:horizontal, unique=true, nbanks=3, cm.L"$\tau_{II}$    ($δ \approx$ %$(round(Int,δ)))")
        display(fig)
    end

end