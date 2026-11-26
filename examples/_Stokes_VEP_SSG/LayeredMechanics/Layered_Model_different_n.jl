using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils, JLD2
import CairoMakie as cm
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using JustPIC
using CellArraysIndexing: @index
const save = true
const figpath = "/Users/filippozarabara/Documents/PHD/MEDIA/I_manuscript_runs/power_law"
const ad_backend = AutoForwardDiff()
const backend = JustPIC.CPU

function set_phases_geometry!(phases, particles, layering)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                𝐱 = SVector(@index(particles.coords[1][ip, i, j]), @index(particles.coords[2][ip, i, j]))
                @index phases[ip, i, j] = inside(𝐱, layering) ? 2.0 : 1.0
            end
        end
    end
end

# Mean (macroscopic) stress metrics over the inner subdomain
function macro_stress(a, inner_x, inner_y)
    τxx       = mean(a.τ.xx[inner_x, inner_y])
    τyy       = mean(a.τ.yy[inner_x, inner_y])
    τxy       = mean(av2D(a.τ.xy[inner_x, inner_y]))
    τII_macro = sqrt(0.5 * (τxx^2 + τyy^2 + (-τxx - τyy)^2) + τxy^2)
    τII_mean  = mean(a.τ.II[inner_x, inner_y])
    return (Τxx=τxx, Τyy=τyy, Τxy=τxy, ΤII_macro=τII_macro, ΤII_mean=τII_mean)
end

@views function main(nc, nt, Δt, L, layering, BC_template, D_template, αs, η1, η2, G1, G2, n_w, n_s; fabric_angle=nothing)
    #--------------------------------------------#   

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 2
    # materials = initialize_materials(nphases; plasticity=VonMises, compressible=false)
    materials = initialize_materials(nphases; compressible=false, mixing=:reuss)
    materials.η0 .= [η1, η2]
    materials.n  .= [n_w, n_s]
    materials.G  .= [G1, G2]

    preprocess!(materials)

    # Newton solver
    iter_params = IterParams(niter=50, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt)
    x = (min=(-L.x / 2), max=L.x / 2)
    y = (min=(-L.y / 2), max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)
    # Nonlinear residuals per timestep (niter × nt)
    Err = (x=fill(NaN, iter_params.niter, nt), y=fill(NaN, iter_params.niter, nt), p=fill(NaN, iter_params.niter, nt), iter=zeros(Int, nt))

    # Macroscopic stress time series
    Τ_t = (t=collect(1:nt) .* Δt, xx=zeros(nt), yy=zeros(nt), xy=zeros(nt), II_macro=zeros(nt), II_mean=zeros(nt))

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
    nxcell = (4, 4)
    max_xcell = 25
    min_xcell = 10
    adv = Markers(backend, a, nxcell, max_xcell, min_xcell, nc, nphases; args=1)
    phases, = adv.particle_args
    set_phases_geometry!(phases, adv.particles, layering)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)
    Set_Grid_Phases!(a)

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
    # Set_PhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    # Saving
    angle_deg = fabric_angle === nothing ? 0 : round(Int, rad2deg(fabric_angle))

    for it = 1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
        Err.x[1:iter, it] .= err.x[1:iter]
        Err.y[1:iter, it] .= err.y[1:iter]
        Err.p[1:iter, it] .= err.p[1:iter]
        Err.iter[it] = iter

        # Macroscopic stress at this timestep
        M = macro_stress(a, inner_x, inner_y)
        Τ_t.xx[it]       = M.Τxx
        Τ_t.yy[it]       = M.Τyy
        Τ_t.xy[it]       = M.Τxy
        Τ_t.II_macro[it] = M.ΤII_macro
        Τ_t.II_mean[it]  = M.ΤII_mean

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

        # FIGURE: Fields at final timestep (adapted from plot_grid_fields in Layered_Post_Processing.jl,
        # using the live grid/field arrays already in scope here instead of a loaded `data` Dict)
        if it == nt
            cm.with_theme(cm.theme_latexfonts()) do
                fig = cm.Figure(size=(800, 800), px_per_unit=2)
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

                # ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                # hm2 = cm.heatmap!(ax2, a.X.v.x, a.X.v.y, a.τ.xy[inx_v, iny_v], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                #     strokecolor=:white, strokewidth=2, color=:transparent)
                # cm.Colorbar(fig[1, 4], hm2, label=cm.L"$\tau_{xy} \ [-]$", labelsize=18)

                # ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                # hm3 = cm.heatmap!(ax3, a.X.c.x, a.X.c.y, a.τ.yy[inx_c, iny_c], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                # #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                # #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                # #     strokecolor=:white, strokewidth=2, color=:transparent)
                # cm.Colorbar(fig[2, 2], hm3, label=cm.L"$\tau_{yy} \ [-]$", labelsize=18)

                # ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect(), xticks=[-0.5, 0.0, 0.5])
                # hm4 = cm.heatmap!(ax4, a.X.c.x, a.X.c.y, a.τ.xx[inx_c, iny_c], colormap=cm.cgrad(:vik)) #:roma, rev=true
                # # cm.poly!(ax2, cm.Rect(a.X.c_e.x[inner_x[1]], a.X.c_e.y[inner_y[1]],
                # #         a.X.c_e.x[inner_x[end]] - a.X.c_e.x[inner_x[1]],
                # #         a.X.c_e.y[inner_y[end]] - a.X.c_e.y[inner_y[1]]),
                # #     strokecolor=:white, strokewidth=2, color=:transparent)
                # cm.Colorbar(fig[2, 4], hm4, label=cm.L"$\tau_{xx} \ [-]$", labelsize=18)

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

    # Save mean (macroscopic) stress metrics at final timestep
    Macro = macro_stress(a, inner_x, inner_y)

    display(to)

    return Macro, Τ_t, Err
end

let
    # Boundary condition templates
    BCs = [
        # :EW_periodic,
        # :all_Dirichlet,
        :free_slip,
    ]

    # Boundary deformation gradient matrix
    D_BCs = @SMatrix([1 0; 0 -1])

    nc = (x=150, y=150)
    L = 3.0

    # Discretise angle of layer 
    # nθ = 2
    nθ = 9
    θ = LinRange(0, π/4, nθ)
    # θ = LinRange(π / 8, π / 8, nθ)
    τ_cart = zeros(nθ)
    τ_II_mean = zeros(nθ)
    τ_II_macro = zeros(nθ)
    Τ = (II_mean=zeros(nθ), II_macro=zeros(nθ), xx=zeros(nθ), yy=zeros(nθ), xy=zeros(nθ))

    #  Composite parameters
    m = 4.
    me = 4.
    # η2 = 1e10
    # η1 = η2
    η2 = 10
    η1 = η2/m

    G2 = 1.
    G1 = G2/me

    # Time stepping (Maxwell time η/G = 10 → Δt = tM/5, nt·Δt = 8 tM)
    nt = 40
    Δt = 2.0
    t  = collect(1:nt) .* Δt

    Τ_time = (II_mean=zeros(nθ, nt), II_macro=zeros(nθ, nt), xx=zeros(nθ, nt), yy=zeros(nθ, nt), xy=zeros(nθ, nt))

    α2 = 0.50 #strong
    α1 = 1 - α2

    ηn = α1 * η1 + α2 * η2
    δ = (α1 + α2 * m) * (α1 + α2 / m)

    n_ran = 1:20
    n1s = [1, 2, 3, 4, 4, 7]
    n2s = [1, 1, 1, 1, 2, 4]
    @assert length(n1s) == length(n2s)

    # Single output file 
    jld2path = joinpath(@__DIR__, "Layered_VEpwl_dn_nc150_astrong050_reuss.jld2")
    jld2file = save ? jldopen(jld2path, "w") : nothing

    try
        # Parameters shared by all n combinations (saved once)
        if save
            jld2file["nc"]  = nc.x
            jld2file["nt"]  = nt
            jld2file["t"]   = t
            jld2file["Δt"]  = Δt
            jld2file["η2"]  = η2
            jld2file["η1"]  = η1
            jld2file["G1"]  = G1
            jld2file["G2"]  = G2
            jld2file["D_BC"] = Matrix(D_BCs)
            jld2file["nθ"]  = nθ
            jld2file["m"]   = m
            jld2file["α2"]  = α2
            jld2file["α1"]  = α1
            jld2file["L"]   = L
            jld2file["n1s"] = n1s
            jld2file["n2s"] = n2s
        end

        # Run each (n1, n2) pair in order
        for (n1, n2) in zip(n1s, n2s)
            group = "n1_$(n1)n2_$(n2)"

            for iθ in 1:nθ

                layering = Layering(
                    (0 * 0.25, 0.025),
                    0.2,
                    α2;
                    θ=θ[iθ],
                    perturb_amp=0. * 1.0,
                    perturb_width=1.0
                )

                println("**Layering $(rad2deg(θ[iθ]))°, n₁ = $n1, n₂ = $(n2)x**")

                Macro, Τ_t, Err = main(nc, nt, Δt, (x=L, y=L), layering, BCs[1], D_BCs, α2, η1, η2, G1, G2, n1, n2; fabric_angle=θ[iθ])

                # Nonlinear residuals of each timestep (niter × nt)
                if save
                    run = "$(group)/err/θ$(iθ)"
                    jld2file["$run/θ"] = θ[iθ]
                    jld2file["$run/x"] = Err.x
                    jld2file["$run/y"] = Err.y
                    jld2file["$run/p"] = Err.p
                    jld2file["$run/iter"] = Err.iter
                end

                # Final timestep macroscopic invariants
                Τ.xx[iθ] = Macro.Τxx
                Τ.yy[iθ] = Macro.Τyy
                Τ.xy[iθ] = Macro.Τxy
                Τ.II_mean[iθ] = Macro.ΤII_mean
                Τ.II_macro[iθ] = Macro.ΤII_macro

                # Macroscopic stress time series
                Τ_time.xx[iθ, :]       .= Τ_t.xx
                Τ_time.yy[iθ, :]       .= Τ_t.yy
                Τ_time.xy[iθ, :]       .= Τ_t.xy
                Τ_time.II_macro[iθ, :] .= Τ_t.II_macro
                Τ_time.II_mean[iθ, :]  .= Τ_t.II_mean
            end

            if save
                jld2file["$group/θ"]         = collect(θ)
                jld2file["$group/τ_II"]      = Τ.II_macro
                jld2file["$group/τ_II_mean"] = Τ.II_mean
                jld2file["$group/τ_xx"]      = Τ.xx
                jld2file["$group/τ_yy"]      = Τ.yy
                jld2file["$group/τ_xy"]      = Τ.xy

                # Time series (nθ × nt)
                jld2file["$group/time/t"]         = t
                jld2file["$group/time/τ_II"]      = Τ_time.II_macro
                jld2file["$group/time/τ_II_mean"] = Τ_time.II_mean
                jld2file["$group/time/τ_xx"]      = Τ_time.xx
                jld2file["$group/time/τ_yy"]      = Τ_time.yy
                jld2file["$group/time/τ_xy"]      = Τ_time.xy
            end

            cm.with_theme(cm.theme_latexfonts()) do

                # # FIGURE 1 ------------------------------------------------------------------
                # iθ_first = 1
                # iθ_mid = (nθ + 1) ÷ 2  # index of 45°
                # iθ_last = nθ
                # linestyles = Dict(iθ_first => :solid, iθ_mid => :dash, iθ_last => :dot)
                # orientation_label = Dict(iθ_first => "Strong (hor)", iθ_mid => "Weak (45°)", iθ_last => "Strong (vert)")

                # fig1 = cm.Figure(size=(800, 650), px_per_unit=2)
                # ax1 = cm.Axis(fig1[1, 1], title="m = $(m)", xlabel=cm.L"$\mathrm{time step}$", ylabel=cm.L"$\tau_{II} \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16)
                # for iθ in (iθ_first, iθ_mid, iθ_last)
                #     cm.lines!(ax1, 1:nt, Τ_time.II_mean[iθ, :], linestyle=linestyles[iθ], label=orientation_label[iθ])
                # end
                # cm.Legend(fig1[1, 2], ax1, "fabric orientation", labelsize=14, titlesize=13)
                # display(fig1)

                # FIGURE 2 -----------------------------------------------------------------
                fig2 = cm.Figure(size=(800, 650), px_per_unit=2)
                ax2 = cm.Axis(fig2[1, 1], title="m = $(m)", xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau \ [-]$", xlabelsize=18, ylabelsize=18, titlesize=16, aspect=2)
                    # cm.scatterlines!(ax2, θ * 180 / π, ana.Τ[iL, :], color=:blue, label="τII (analytical)")
                    # cm.scatterlines!(ax2, θ * 180 / π, ana.Τxx[iL, :], color=:red, label="Τxx (analytical)")
                    # cm.scatterlines!(ax2, θ * 180 / π, ana.Τyy[iL, :], color=:green, label="Τyy (analytical)")
                    cm.scatterlines!(ax2, θ * 180 / π, Τ.II_macro, color=:orange, label="τII from components (simulated)")
                    # cm.scatterlines!(ax2, θ * 180 / π, sqrt.(ana.Τxx[iL, :] .^ 2 .+ ana.Τyy[iL, :] .^ 2), color=:purple, label="τII computed")
                    cm.scatterlines!(ax2, θ * 180 / π, Τ.II_mean, color=:gray, label="τII mean (simulated)")
                    cm.scatterlines!(ax2, θ * 180 / π, Τ.xx, color=:blue, label="τxx mean (simulated)")
                    cm.scatterlines!(ax2, θ * 180 / π, Τ.xy, color=:green, label="τxy mean (simulated)")

                    # cm.scatterlines!(ax2, θ_old * 180 / π, τIIb_old, color=:orange, linestyle=:dash, marker=:diamond, label="τII from components (old StagFD)")
                    # cm.scatterlines!(ax2, θ_old * 180 / π, τII_old, color=:gray, linestyle=:dash, marker=:diamond, label="τII mean (old StagFD)")
                    # cm.scatterlines!(ax2, θ_old * 180 / π, τxx_old, color=:blue, linestyle=:dash, marker=:diamond, label="τxx mean (old StagFD)")
                    # cm.scatterlines!(ax2, θ_old * 180 / π, τxy_old, color=:green, linestyle=:dash, marker=:diamond, label="τxy mean (old StagFD)")
                    cm.Legend(fig2[1, 2], ax2, "quantity", labelsize=14, titlesize=13)
                display(fig2)

                # # FIGURE 3 -------------------------------------------------------------
                # fig3 = cm.Figure(size=(800, 650), px_per_unit=2)
                # ax3 = cm.Axis(fig3[1, 1], title="predicted yield envelope in stress space", aspect=cm.DataAspect())
                # τS = vec(ana.Τxx)
                # τE = vec(ana.Τyy)
                # cm.scatter!(ax3, vcat(τS, -τS, τS, -τS), vcat(τE, τE, -τE, -τE))
                # display(fig3)

                # FIGURE 4 -------------------------------------------------------------
                fig4 = cm.Figure(size=(800, 650), px_per_unit=2)
                ax4 = cm.Axis(fig4[1, 1], title="stress space (direct model) n₁=$(n1), n₂=$(n2)", xlabel=cm.L"$\tau_{xx}' \ [-]$", ylabel=cm.L"$\tau_{xy}' \ [-]$", aspect=cm.DataAspect())
                τ_yield = zeros(2, nθ)
                for Iθ in eachindex(θ)
                    𝐐_plot = @SMatrix([cos(θ[Iθ]) sin(θ[Iθ]);
                        -sin(θ[Iθ]) cos(θ[Iθ])])
                    τ_plot = @SMatrix([Τ.xx[Iθ] Τ.xy[Iθ]; Τ.xy[Iθ] Τ.yy[Iθ]])
                    τ′_plot = 𝐐_plot * τ_plot * 𝐐_plot'
                    τ_yield[1, Iθ] = τ′_plot[1, 1]
                    τ_yield[2, Iθ] = τ′_plot[2, 1]
                end
                cm.scatterlines!(ax4, τ_yield[1, :], τ_yield[2, :], color=:red, label="yield surface")
                cm.scatterlines!(ax4, -τ_yield[1, :], -τ_yield[2, :], color=:red)
                cm.scatterlines!(ax4, -τ_yield[1, :], τ_yield[2, :], color=:red)
                cm.scatterlines!(ax4, τ_yield[1, :], -τ_yield[2, :], color=:red)
                cm.axislegend(ax4)
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
    finally
        jld2file !== nothing && close(jld2file)
    end

end
