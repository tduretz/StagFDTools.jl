using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs

@views function main(nc)
    #--------------------------------------------#

    # Resolution

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [ -1. 0.;
                          0  1 ])

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; plasticity=DruckerPrager,compressible=true)
    materials.g .= [0. , 0.]
    materials.ρ   .= [1.0 ,   1.0  ]
    materials.n   .= [1.0 ,   1.0  ]
    materials.η0  .= [1e2 ,   1e-1 ]
    materials.ξ0  .= [1e50,   1e50 ]
    materials.G   .= [1e1 ,   1e1  ]
    materials.plasticity.C   .= [150 ,   150  ]
    materials.plasticity.ϕ   .= [30. ,   30.  ]
    materials.plasticity.ηvp .= [0.5 ,   0.5  ]
    materials.β   .= [1e-2,   1e-2 ]
    materials.plasticity.ψ   .= [0.0 ,   0.0  ]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    =  40
    nmpc = (x=4, y=4)
    noise = false

    # Solver parameters
    Pic2Newt = 1.3    # more than 1.0 - always Newton
    iter_params = IterParams(niter=20, γ=1e5, ϵ_l=1e-11, ϵ_nl=1e-8, inexact=false, solver_type=:GCR, α=LinRange(0.05, 1.0, 6))

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x   = (min=-L.x/2, max=L.x/2)
    y   = (min=-L.y/2,   max=L.x/2  )

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)
    nVx = maximum(a.number.Vx)
    nVy = maximum(a.number.Vy)
    nPt = maximum(a.number.Pt)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1, 1] * a.X.vx_e.x .+ D_BC[1, 2] * a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1] * a.X.vy_e.x .+ D_BC[2, 2] * a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 0.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Set material geometry
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= 0.1^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= 0.1^2] .= 2
    # @views a.phases.v[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    # @views a.phases.v[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries
    # @views a.phases.c[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    # @views a.phases.c[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries
    FillPhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    # Note: this example switches between a full Newton solve and a cheaper
    # Picard solve depending on the convergence ratio, which is not something
    # `main_loop`/`Solve!` expose, so the Newton loop is kept explicit here.
    for it=1:nt #anim = @animate

        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)

        # Swap old values
        a.τ0.xx .= a.τ.xx
        a.τ0.yy .= a.τ.yy
        a.τ0.xy .= a.τ.xy
        a.Pt0   .= a.Pt

        @printf("Time step %04d (nthreads = %03d)\n", it, Threads.nthreads())
        iter, ϵ0, ϵ = 0, 0.0, 0.0
        niter = 10

        compute_grid_fields!(a.G, a.β, a.ρ, a.ξ, materials, a.phase_ratios, nc, nphases)

        # @time
        while iter<niter

            iter +=1
            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check
            @timeit to "Residual" begin
                TangentOperator!(a.𝐷, a.𝐷_ctl, a.τ, a.τ0, a.ε̇, a.λ̇, a.η, a.G, a.V, a.Pt, a.Pt0, a.ΔPt, a.type, BC, materials, a.phase_ratios, Δ)
                @show extrema(a.λ̇.c[inx_c, iny_c])
                @show extrema(a.λ̇.v[inx_v, iny_v])
                ResidualContinuity2D!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ, materials, a.number, a.type, BC, nc, Δ)
                ResidualMomentum2D_x!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, materials, a.number, a.type, BC, nc, Δ)
                ResidualMomentum2D_y!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ, materials, a.number, a.type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(a.R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(a.R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(a.R.p[inx_c,iny_c])/sqrt(nPt)
            ϵ =  max(err.x[iter], err.y[iter])
            (iter == 1) && (ϵ0 = ϵ)
            ϵ < iter_params.ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(a.r, a.R, a.number, a.type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                # Jacobian
                AssembleContinuity2D!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.β, a.ξ, materials, a.number, a.pattern, a.type, BC, nc, Δ)
                AssembleMomentum2D_x!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, materials, a.number, a.pattern, a.type, BC, nc, Δ)
                AssembleMomentum2D_y!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, a.ρ, materials, a.number, a.pattern, a.type, BC, nc, Δ)
                # Preconditioner
                AssembleContinuity2D!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ, materials, a.number, a.pattern, a.type, BC, nc, Δ)
                AssembleMomentum2D_x!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, materials, a.number, a.pattern, a.type, BC, nc, Δ)
                AssembleMomentum2D_y!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ, materials, a.number, a.pattern, a.type, BC, nc, Δ)
            end
            #--------------------------------------------#
            # Stokes operator as block matrices
            a.𝐊  .= [a.M.Vx.Vx a.M.Vx.Vy; a.M.Vy.Vx a.M.Vy.Vy]
            a.𝐐  .= [a.M.Vx.Pt; a.M.Vy.Pt]
            a.𝐐ᵀ .= [a.M.Pt.Vx a.M.Pt.Vy]
            a.𝐏  .= a.M.Pt.Pt
            # Picard preconditioner
            a.𝐊_PC  .= [a.M_PC.Vx.Vx a.M_PC.Vx.Vy; a.M_PC.Vy.Vx a.M_PC.Vy.Vy]
            a.𝐐_PC  .= [a.M_PC.Vx.Pt; a.M_PC.Vy.Pt]
            a.𝐐ᵀ_PC .= [a.M_PC.Pt.Vx a.M_PC.Pt.Vy]
            a.𝐏_PC  .= a.M_PC.Pt.Pt
            #--------------------------------------------#

            # Inexact Newton-Raphson
            ϵ_l = iter_params.inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : iter_params.ϵ_l
            Newton = (ϵ/ϵ0 < Pic2Newt) ? true : false
            @printf("Abs. res. = %02e --- Rel. res = %02e  --- ϵ_l = %1.2e --- Newton = %01d\n", ϵ, ϵ/ϵ0, ϵ_l, Newton)

            # Direct-iterative solver
            @timeit to "Linear solve" begin
                if Newton
                    mechanical_solver!( a.dx, a.M,    a.r, a.𝐊,    a.𝐐,    a.𝐐ᵀ,    a.𝐏,    a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC; solver=iter_params.solver_type, ηb=iter_params.γ, ϵ_l=ϵ_l, niter_l=10, restart=20)
                else
                    mechanical_solver!( a.dx, a.M_PC, a.r, a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC, a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC; solver=iter_params.solver_type, ηb=iter_params.γ, ϵ_l=ϵ_l, niter_l=10, restart=20)
                end
            end

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, iter_params.α, a.dx, a.R, a.V, a.Pt, a.ε̇, a.τ, a.Vi, a.Pti, a.ΔPt, a.Pt0, a.τ0, a.λ̇, a.η, a.G, a.β, a.ξ, a.ρ, a.𝐷, a.𝐷_ctl, a.number, a.type, BC, materials, a.phase_ratios, nc, Δ)
            UpdateSolution!(a.V, a.Pt, iter_params.α[imin] * a.dx, a.number, a.type, nc)

        end

        # Update pressure
        a.Pt .+= a.ΔPt.c

        #--------------------------------------------#
        fig = Figure(size=(900,700), fontsize=14)

        ax1 = Axis(fig[1,1], xlabel="Iterations @ step $(it)", ylabel="log₁₀ error", title="Convergence")
        scatter!(ax1, 1:iter, log10.(err.x[1:iter]), markersize=6, label="Vx")
        scatter!(ax1, 1:iter, log10.(err.y[1:iter]), markersize=6, label="Vy")
        axislegend(ax1, position=:rt)

        ax2 = Axis(fig[1,2], title="Vx", aspect=DataAspect())
        heatmap!(ax2, a.X.v.x, a.X.c.y, a.V.x[inx_Vx,iny_Vx]')
        xlims!(ax2, extrema(a.X.v.x))

        ax3 = Axis(fig[2,1], title="ε̇II", aspect=DataAspect())
        hm3 = heatmap!(ax3, a.X.c.x, a.X.c.y, log10.(a.ε̇.II[inx_c,iny_c])'; colormap=:coolwarm, colorrange=(-0.4,0.4))
        xlims!(ax3, extrema(a.X.c.x))
        Colorbar(fig[2,1, Right()], hm3, width=12)

        ax4 = Axis(fig[2,2], title="τxx", aspect=DataAspect())
        hm4 = heatmap!(ax4, a.X.c.x, a.X.c.y, a.τ.xx[inx_c,iny_c]'; colormap=:turbo)
        xlims!(ax4, extrema(a.X.c.x))
        Colorbar(fig[2,2, Right()], hm4, width=12)

        display(fig)

        @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))

    end
    # gif(anim, "./results/ShearBanding.gif", fps = 5)

    display(to)

end

let
    main((x = 100, y = 100))
end
