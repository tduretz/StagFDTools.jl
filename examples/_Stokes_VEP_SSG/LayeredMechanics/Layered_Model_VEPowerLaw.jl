using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils, JLD2
import CairoMakie as cm
using DifferentiationInterface
using ForwardDiff: ForwardDiff
const save = true
const figpath = "/Users/filippozarabara/Documents/PHD/MEDIA/VEVP_Layered_Model/plastic_test_2/"
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
    Σxy = mean(av2D(τ.xy[inner_x, inner_y]))
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

function yieldSurface(θ, D_BC, C1, C2, α1, α2)
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
    Τ = sqrt(Dl^2*(α1*C1 + α2*C2)^2 + Dn^2 * C2^2)
    return Τ, Τxx, Τyy
end

@views function main(nc, nt, L, layering, BC_template, D_template, α1, η1, η2, G1, G2, C1, C2, n; fabric_angle=nothing)
    #--------------------------------------------#   

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 2
    # materials = initialize_materials(nphases; plasticity=VonMises, compressible=false)
    materials = initialize_materials(nphases; compressible=false)
    materials.η0 .= [η1, η2]
    materials.n .= [n, n]
    materials.G .= [G1, G2]
    # materials.plasticity.C .= [C1, C2]
    # materials.plasticity.ηvp .= [1e-2, 1e-2]

    preprocess!(materials)

    # MARKERS: markers-per-cell / position jitter (uncomment to use markers)
    nmpc = (x=4, y=4)
    noise = false

    # Time steps
    Δt0 = 0.1

    # Newton solver
    iter_params = IterParams(niter=20, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=(-L.x / 2), max=L.x / 2)
    y = (min=(-L.y / 2), max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases, nmpc, noise) # markers
    # a = Allocs(nc, config, x, y, Δ, nphases) # no markers
    Τev = (II=ones(nt), xx=ones(nt), yy=ones(nt), xy=ones(nt))
    α2 = 1 - α1

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Only account for the subdomain
    imin_x = argmin(abs.(a.X.c_e.x .+ (0.6 * L.x/2)))
    imax_x = argmin(abs.(a.X.c_e.x .- (0.6 * L.x/2)))
    imin_y = argmin(abs.(a.X.c_e.y .+ (0.6 * L.y/2)))
    imax_y = argmin(abs.(a.X.c_e.y .- (0.6 * L.y/2)))
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
    for I in CartesianIndices(a.m.phase)
        xm = a.m.Xm[I]
        ym = a.m.Ym[I]
        isin = inside(@SVector([xm, ym]), layering)
        a.m.phase[I] = isin ? 2 : 1
    end
    SetPhaseRatios!(a.phase_ratios, a.m, a.X.c_e.x, a.X.c_e.y, a.X.v_e.x, a.X.v_e.y, Δ, nphases)

    # NO MARKERS -----------------------------------------------------------
    # for i in inx_c, j in iny_c
    #     𝐱 = @SVector([a.X.c.x[i-1], a.X.c.y[j-1]])
    #     isin = inside(𝐱, layering)
    #     a.phases.c[i, j] = isin ? 2 : 1
    # end
    # for i in inx_v, j in iny_v
    #     𝐱 = @SVector([a.X.v.x[i-1], a.X.v.y[j-1]])
    #     isin = inside(𝐱, layering)
    #     a.phases.v[i, j] = isin ? 2 : 1
    # end
    # FillPhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    # Saving
    angle_deg = fabric_angle === nothing ? 0 : round(Int, rad2deg(fabric_angle))

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

        Τev.II[it] = mean(a.τ.II[inner_x, inner_y])
        Τev.xx[it] = mean(a.τ.xx[inner_x, inner_y])
        Τev.yy[it] = mean(a.τ.yy[inner_x, inner_y])
        Τev.xy[it] = mean(av2D(a.τ.xy[inner_x, inner_y]))

        # FIGURE: Fields at final timestep (adapted from plot_grid_fields in Layered_Post_Processing.jl,
        # using the live grid/field arrays already in scope here instead of a loaded `data` Dict)
        if it == nt
            cm.with_theme(cm.theme_latexfonts()) do
                fig = cm.Figure(size=(700, 600), px_per_unit=2)
                ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5], xlabelsize=26, ylabelsize=26, titlesize=20, title=@sprintf("θ = %d°", angle_deg))
                hm = cm.heatmap!(ax, a.X.c.x, a.X.c.y, a.τ.II[inx_c, iny_c], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # Inner-box overlay and principal-stress arrows (uncomment to use)
                # cm.poly!(ax, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]], a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]), strokecolor=:white, strokewidth=2, color=:white)
                # st = max(1, length(a.X.c_e.x) ÷ 25)
                # cm.arrows2d!(ax, a.X.c_e.x[1:st:end], a.X.c_e.y[1:st:end], σ1.x[1:st:end, 1:st:end], σ1.y[1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
                cm.Colorbar(fig[1, 2], hm, label=cm.L"$\tau_{II} \ [-]$", labelsize=18)

                # ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                # hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.ε̇.II[inx_c, iny_c], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                # #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                # #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                # #     strokecolor=:white, strokewidth=2, color=:transparent)
                # cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\dot\varepsilon_{II} \ [-]$", labelsize=18)

                # ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                # hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.η.c[inx_c, iny_c], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                # #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                # #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                # #     strokecolor=:white, strokewidth=2, color=:transparent)
                # cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\eta \ [-]$", labelsize=18)

                ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                hm2 = cm.heatmap!(ax2, a.X.v.x, a.X.v.y, a.τ.xy[inx_v, iny_v], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                #     strokecolor=:white, strokewidth=2, color=:transparent)
                cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\tau_{xy} \ [-]$", labelsize=18)

                ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                hm3 = cm.heatmap!(ax3, a.X.c.x, a.X.c.y, a.τ.yy[inx_c, iny_c], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                #     strokecolor=:white, strokewidth=2, color=:transparent)
                cm.Colorbar(fig[2, 2], hm3, label=cm.L"$\tau_{yy} \ [-]$", labelsize=18)

                ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                hm4 = cm.heatmap!(ax4, a.X.c.x, a.X.c.y, a.τ.xx[inx_c, iny_c], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                #     strokecolor=:white, strokewidth=2, color=:transparent)
                cm.Colorbar(fig[2, 4], hm4, label=cm.L"$\tau_{xx} \ [-]$", labelsize=18)

                # ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect())
                # hm3 = cm.heatmap!(ax3, a.X.vx.x, a.X.vx.y, a.V.x[inx_Vx, iny_Vx], colormap=:vik)
                # cm.Colorbar(fig[2, 2], hm3, label=cm.L"$v_x \ [-]$", labelsize=18)

                # ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect())
                # hm4 = cm.heatmap!(ax4, a.X.vy.x, a.X.vy.y, a.V.y[inx_Vy, iny_Vy], colormap=:vik)
                # cm.Colorbar(fig[2, 4], hm4, label=cm.L"$v_y \ [-]$", labelsize=18)

                # ax5 = cm.Axis(fig[3, 1:4], xlabel="time step", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18)
                # cm.xlims!(ax5, 0, nt)
                # cm.lines!(ax5, 1:it, Τev.II[1:it])

                # display(fig)
            end
        end
    end

    # Analytics
    # τIIanS, τIIanE = Analyticalstress(fabric_angle, a.τ, inner_x, inner_y)

    # Save mean (macroscopic) stress metrics at final timestep
    mΤxx = mean(a.τ.xx[inner_x, inner_y])
    mΤyy = mean(a.τ.yy[inner_x, inner_y])
    mΤxy = mean(av2D(a.τ.xy[inner_x, inner_y]))
    mΤII_macro = sqrt(0.5 * (mΤxx^2 + mΤyy^2 + (-mΤxx - mΤyy)^2) + mΤxy^2)
    mΤII = mean(a.τ.II[inner_x, inner_y])
    Macro = (Τxx=mΤxx, Τyy=mΤyy, Τxy=mΤxy, ΤII_macro=mΤII_macro, ΤII_mean=mΤII)

    display(to)

    return Τev, Macro
    # return mean(a.τ.II[inner_x, inner_y]), τIIev, τIIanS, τIIanE, τIIsec, τIIanDis, τxxev, τyyev, τxyev

end

let
    # Boundary condition templates
    BCs = [
        # :EW_periodic,
        # :all_Dirichlet,
        :free_slip,
    ]

    # Boundary deformation gradient matrix
    # D_BCs = [
    #     @SMatrix([1 0; 0 -1]),
    # ]
    D_BCs = @SMatrix([1 0; 0 -1])

    nc = (x=100, y=100)
    nt = 50
    L = 1.0

    # Discretise angle of layer 
    # nθ = 2
    nθ = 15
    θ = LinRange(0, π/2, nθ)
    # θ = LinRange(π / 8, π / 8, nθ)
    τ_cart = zeros(nθ)
    τ_II_mean = zeros(nθ)
    τ_II_macro = zeros(nθ)
    # τ_cart_anaS = zeros(length(L), nθ)
    # τ_cart_anaE = zeros(length(L), nθ)
    Τ_time = (II_mean=zeros(nθ, nt), II_macro=zeros(nθ, nt), xx=zeros(nθ, nt), yy=zeros(nθ, nt), xy=zeros(nθ, nt))
    Τ = (II_mean=zeros(nθ), II_macro=zeros(nθ), xx=zeros(nθ), yy=zeros(nθ), xy=zeros(nθ))
    ana = (Τ=zeros(nθ), Τxx=zeros(nθ), Τyy=zeros(nθ))

    #  Composite parameters
    m = 4
    # η2 = 1e10
    # η1 = η2
    η2 = 10
    η1 = η2/m

    G2 = 1e6
    G1 = G2 / m

    C2 = 10.
    C1 = C2 / 1 #m

    α2 = 0.5
    α1 = 1 - α2

    ηn = α1 * η1 + α2 * η2
    δ = (α1 + α2 * m) * (α1 + α2 / m)

    n_range = 1:20

    # elasticity
    tmax = 1.0

    # Single output file holding all power-law exponents, one subgroup per n
    # (e.g. "n5/τ_II", "n5/τ_xx", ...)
    jld2path = joinpath(@__DIR__, "Layered_pwl.jld2")
    jld2file = save ? jldopen(jld2path, "w") : nothing

    # Run them all
    try
        for n in n_range
            for iθ in 1:nθ

                layering = Layering(
                    (0 * 0.25, 0.025),
                    0.2,
                    α2;
                    θ=θ[iθ],
                    perturb_amp=0. * 1.0,
                    perturb_width=1.0
                )

                println("**Layering $(rad2deg(θ[iθ]))°, n = $n**")

                Τev, Macro = main(nc, nt, (x=L, y=L), layering, BCs[1], D_BCs, α1, η1, η2, G1, G2, C1, C2, n; fabric_angle=θ[iθ])

                # Evolutions of full fields
                Τ_time.II_mean[iθ, :] .= Τev.II
                Τ_time.xx[iθ, :] .= Τev.xx
                Τ_time.yy[iθ, :] .= Τev.yy
                Τ_time.xy[iθ, :] .= Τev.xy

                # Final timestep macroscopic invariants
                Τ.xx[iθ] = Macro.Τxx
                Τ.yy[iθ] = Macro.Τyy
                Τ.xy[iθ] = Macro.Τxy
                Τ.II_mean[iθ] = Macro.ΤII_mean
                Τ.II_macro[iθ] = Macro.ΤII_macro

                # ana.Τ[iL, iθ], ana.Τxx[iL, iθ], ana.Τyy[iL, iθ] = yieldSurface(θ[iθ], D_BCs, C2, C1, α2, α1)
            end

            if save
                group = "n$(n)"
                jld2file["$group/n"] = n
                jld2file["$group/θ"] = collect(θ)
                jld2file["$group/τ_II"] = Τ.II_macro
                jld2file["$group/τ_II_mean"] = Τ.II_mean
                jld2file["$group/τ_xx"] = Τ.xx
                jld2file["$group/τ_yy"] = Τ.yy
                jld2file["$group/τ_xy"] = Τ.xy
            end
        end

        if save
            jld2file["η2"] = η2
            jld2file["η1"] = η1
            jld2file["nθ"] = nθ
            jld2file["m"] = m
            jld2file["α2"] = α2
            jld2file["α1"] = α1
            jld2file["L"] = L
            jld2file["G2"] = G2
            jld2file["C2"] = C2
        end
    finally
        jld2file !== nothing && close(jld2file)
    end

    # # Anna's stuff
    # old = load(joinpath(@__DIR__, "Layered_pwl.jld2"))
    # θ_old = old["θ"]
    # τxx_old = old["τ_xx"]
    # τxy_old = old["τ_xy"]
    # τII_old = old["τ_II"]
    # τIIb_old = old["τ_IIb"]

    cm.with_theme(cm.theme_latexfonts()) do

        # FIGURE 1 ------------------------------------------------------------------
        iθ_first = 1
        iθ_mid = (nθ + 1) ÷ 2  # index of 45°
        iθ_last = nθ
        linestyles = Dict(iθ_first => :solid, iθ_mid => :dash, iθ_last => :dot)
        orientation_label = Dict(iθ_first => "Strong (hor)", iθ_mid => "Weak (45°)", iθ_last => "Strong (vert)")

        fig1 = cm.Figure(size=(800, 650), px_per_unit=2)
        ax1 = cm.Axis(fig1[1, 1], title="m = $(m)", xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16)
        for iθ in (iθ_first, iθ_mid, iθ_last)
            cm.lines!(ax1, 1:nt, Τ_time.II_mean[1, iθ, :], linestyle=linestyles[iθ], label=orientation_label[iθ])
        end
        cm.Legend(fig1[1, 2], ax1, "fabric orientation", labelsize=14, titlesize=13)
        display(fig1)

        # FIGURE 2 -----------------------------------------------------------------
        fig2 = cm.Figure(size=(800, 650), px_per_unit=2)
        ax2 = cm.Axis(fig2[1, 1], title="m = $(m)", xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16, aspect=2)
        for iL in 1:1
            # cm.scatterlines!(ax2, θ * 180 / π, ana.Τ[iL, :], color=:blue, label="τII (analytical)")
            # cm.scatterlines!(ax2, θ * 180 / π, ana.Τxx[iL, :], color=:red, label="Τxx (analytical)")
            # cm.scatterlines!(ax2, θ * 180 / π, ana.Τyy[iL, :], color=:green, label="Τyy (analytical)")
            cm.scatterlines!(ax2, θ * 180 / π, Τ.II_macro[iL, :], color=:orange, label="τII from components (simulated)")
            # cm.scatterlines!(ax2, θ * 180 / π, sqrt.(ana.Τxx[iL, :] .^ 2 .+ ana.Τyy[iL, :] .^ 2), color=:purple, label="τII computed")
            cm.scatterlines!(ax2, θ * 180 / π, Τ.II_mean[iL, :], color=:gray, label="τII mean (simulated)")
            cm.scatterlines!(ax2, θ * 180 / π, Τ.xx[iL, :], color=:blue, label="τxx mean (simulated)")
            cm.scatterlines!(ax2, θ * 180 / π, Τ.xy[iL, :], color=:green, label="τxy mean (simulated)")

            # cm.scatterlines!(ax2, θ_old * 180 / π, τIIb_old, color=:orange, linestyle=:dash, marker=:diamond, label="τII from components (old StagFD)")
            # cm.scatterlines!(ax2, θ_old * 180 / π, τII_old, color=:gray, linestyle=:dash, marker=:diamond, label="τII mean (old StagFD)")
            # cm.scatterlines!(ax2, θ_old * 180 / π, τxx_old, color=:blue, linestyle=:dash, marker=:diamond, label="τxx mean (old StagFD)")
            # cm.scatterlines!(ax2, θ_old * 180 / π, τxy_old, color=:green, linestyle=:dash, marker=:diamond, label="τxy mean (old StagFD)")

            cm.Legend(fig2[1, 2], ax2, "quantity", labelsize=14, titlesize=13)
        end
        # display(fig2)

        # # FIGURE 3 -------------------------------------------------------------
        # fig3 = cm.Figure(size=(800, 650), px_per_unit=2)
        # ax3 = cm.Axis(fig3[1, 1], title="predicted yield envelope in stress space", aspect=cm.DataAspect())
        # τS = vec(ana.Τxx)
        # τE = vec(ana.Τyy)
        # cm.scatter!(ax3, vcat(τS, -τS, τS, -τS), vcat(τE, τE, -τE, -τE))
        # display(fig3)

        # FIGURE 4 -------------------------------------------------------------
        fig4 = cm.Figure(size=(800, 650), px_per_unit=2)
        ax4 = cm.Axis(fig4[1, 1], title="stress-space trajectory (direct model) n=2", xlabel=cm.L"$\tau_{xx}' \ [-]$", ylabel=cm.L"$\tau_{xy}' \ [-]$", aspect=cm.DataAspect(), limits=(-15, 15, -6, 6))
        τ_yield = zeros(2, nθ)
        local sc
        for Iθ in eachindex(θ)
            𝐐_plot = @SMatrix([cos(θ[Iθ]) sin(θ[Iθ]);
                -sin(θ[Iθ]) cos(θ[Iθ])])
            τnn = zeros(nt)
            τnt = zeros(nt)
            for t in 1:nt
                τ_plot = @SMatrix([Τ_time.xx[Iθ, t] Τ_time.xy[Iθ, t]; Τ_time.xy[Iθ, t] Τ_time.yy[Iθ, t]])
                τ′_plot = 𝐐_plot * τ_plot * 𝐐_plot'
                τnn[t] = τ′_plot[1, 1]
                τnt[t] = τ′_plot[2, 1]
                if t == nt
                    τ_yield[1, Iθ] = τ′_plot[1, 1]
                    τ_yield[2, Iθ] = τ′_plot[2, 1]
                end
            end
            # sc = cm.scatter!(ax4, τnn, τnt, color=1:nt, colormap=:viridis, colorrange=(1, nt))
            # sc = cm.scatter!(ax4, -τnn, -τnt, color=1:nt, colormap=:viridis, colorrange=(1, nt))
            sc = cm.scatter!(ax4, τnn, τnt)
            sc = cm.scatter!(ax4, -τnn, -τnt)

        end
        scl = cm.scatterlines!(ax4, τ_yield[1, :], τ_yield[2, :], color=:red, label="yield surface at final timestep")
        scl = cm.scatterlines!(ax4, -τ_yield[1, :], -τ_yield[2, :], color=:red)
        cm.axislegend(ax4)
        # cm.Colorbar(fig4[1, 2], sc, label="time step")
        display(fig4)

        # # FIGURE 5 -------------------------------------------------------------
        # fig5 = cm.Figure(size=(800, 650), px_per_unit=2)
        # ax5 = cm.Axis(fig5[1, 1], title="simulated stress path vs analytical yield surface", xlabel=cm.L"$\Sigma_l \ [-]$", ylabel=cm.L"$\Sigma_n \ [-]$", aspect=cm.DataAspect())
        # local sc5
        # for Iθ in eachindex(θ)
        #     𝐐_plot = @SMatrix([cos(θ[Iθ]) sin(θ[Iθ]);
        #         -sin(θ[Iθ]) cos(θ[Iθ])])
        #     Σl = zeros(nt)
        #     Σn = zeros(nt)
        #     for t in 1:nt
        #         xx, yy, xy = Τ_time.xx[1, Iθ, t], Τ_time.yy[1, Iθ, t], Τ_time.xy[1, Iθ, t]
        #         τ_plot = @SMatrix([xx xy; xy yy])
        #         τ′_plot = 𝐐_plot * τ_plot * 𝐐_plot'
        #         τnt = τ′_plot[1, 2]
        #         τII_macro = sqrt(0.5 * (xx^2 + yy^2 + (-xx - yy)^2) + xy^2)
        #         Σn[t] = abs(τnt)
        #         Σl[t] = sqrt(max(τII_macro^2 - Σn[t]^2, 0))
        #     end
        #     sc5 = cm.scatter!(ax5, Σl, Σn, color=1:nt, colormap=:viridis, colorrange=(1, nt), markersize=6)
        # end
        # cm.Colorbar(fig5[1, 2], sc5, label="time step")
        # cm.lines!(ax5, ana.Τxx[1, :], ana.Τyy[1, :], color=:red, linewidth=2, label="analytical yield surface")
        # cm.Legend(fig5[1, 3], ax5)
        # display(fig5)

        # if save
        #     mkpath(figpath)
        #     figname = @sprintf("LayeredVEVP_res%d_domainsp.png", nc.x)
        #     cm.save(joinpath(figpath, figname), fig1, px_per_unit=4)
        # end
    end

end