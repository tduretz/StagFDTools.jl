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
    materials_properties      = initialize_materials( 2, compressible = true, plasticity = :DruckerPrager )
    materials_properties.ρ   .= [1.0 ,   1.0  ]
    materials_properties.n   .= [1.0 ,   1.0  ]
    materials_properties.η0  .= [1e0 ,   1e0  ]
    materials_properties.ξ0  .= [1e50,   1e50]
    materials_properties.G   .= [1e0 ,   0.5  ]
    materials_properties.C   .= [1.6/cosd(30) ,   1.6/cosd(30)]
    materials_properties.ϕ   .= [30. ,   30.  ]
    materials_properties.ηvp .= [8e-3,   8e-3  ]
    materials_properties.β   .= [1/4,    1/4 ]
    materials_properties.ψ   .= [0.0 ,   0.0  ]
    materials                 = preprocess_materials( materials_properties )

    # Time steps
    Δt0   = 0.175
    nt    = 15

    # Newton solver
    niter    = 10     # max. number of non-linear iters
    γ        = 1e5    # penalty viscosity
    ϵ_l      = 1e-11  # linear solver tolerance
    ϵ_nl     = 1e-9   # non-linear solver tolerance
    inexact  = false  # inexact Newton
    Pic2Newt = 1e10   # more than 1.0 - always Newton
    solver   = :GCR   # :GCR or :PH
    α        = LinRange(0.05, 1.0, 6)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    set_boundaries_template!(type, config, nc)

    #--------------------------------------------#
    # Equation numbering
    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    #--------------------------------------------#
    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    M_PC = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊    = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐊_PC = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐    = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐_PC = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ   = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐐ᵀ_PC= ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏    = ExtendableSparseMatrix(nPt, nPt)
    𝐏_PC = ExtendableSparseMatrix(nPt, nPt)
    dx   = zeros(nVx + nVy + nPt)
    r    = zeros(nVx + nVy + nPt)

    #--------------------------------------------#
    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    ξ       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    ΔPt     = (c=zeros(size_c...), Vx = zeros(size_x...), Vy = zeros(size_y...))

    Dc      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    # Mesh coordinates
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    @views V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    @views V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    @views Pt[inx_c, iny_c ]  .= 0.0                 
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
        BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
        BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
        BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)
    end

    # Set material geometry 
    @views phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    @views phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2
    # @views phases.v[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    # @views phases.v[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries
    # @views phases.c[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    # @views phases.c[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries

    #--------------------------------------------#

    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    probes = (τII = zeros(nt), t = zeros(nt))
    to = TimerOutput()

    #--------------------------------------------#

    for it=1:nt #anim = @animate 

        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)
        
        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        @printf("Time step %04d (nthreads = %03d)\n", it, Threads.nthreads())
        iter, ϵ0, ϵ = 0, 0.0, 0.0

        @time while iter<niter

            iter +=1
            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                @info "Tangent"
                @time TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, ξ, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
                @info "Residual"
                @time ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                @time ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                @time ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(R.p[inx_c,iny_c])/sqrt(nPt)
            ϵ =  max(err.x[iter], err.y[iter])
            (iter == 1) && (ϵ0 = ϵ)

            #--------------------------------------------#
            # Set global residual vector
            @info "Right-hand side"
            @time SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                # Jacobian
                @info "Jacobian Assembly"
                @time AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                @time AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                @time AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                # Preconditioner
                @info "Picard Assembly" 
                @time AssembleContinuity2D!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
                @time AssembleMomentum2D_x!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
                @time AssembleMomentum2D_y!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt
            # Picard preconditioner
            𝐊_PC  .= [M_PC.Vx.Vx M_PC.Vx.Vy; M_PC.Vy.Vx M_PC.Vy.Vy]
            𝐐_PC  .= [M_PC.Vx.Pt; M_PC.Vy.Pt]
            𝐐ᵀ_PC .= [M_PC.Pt.Vx M_PC.Pt.Vy]
            𝐏_PC  .= M_PC.Pt.Pt
            #--------------------------------------------#
     
            # Inexact Newton-Raphson
            ϵ_l = inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : ϵ_l
            Newton = (ϵ/ϵ0 < Pic2Newt) ? true : false 
            @printf("Abs. res. = %02e --- Rel. res = %02e  --- ϵ_l = %1.2e --- Newton = %01d\n", ϵ, ϵ/ϵ0, ϵ_l, Newton)
            ϵ < ϵ_nl ? break : nothing

            # Direct-iterative solver
            @timeit to "Linear solve" begin
                if Newton
                    mechanical_solver!( dx, M,    r, 𝐊,    𝐐,    𝐐ᵀ,    𝐏,    𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC; solver=solver, ηb=γ, ϵ_l=ϵ_l, niter_l=10, restart=20) 
                else
                    mechanical_solver!( dx, M_PC, r, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC; solver=solver, ηb=γ, ϵ_l=ϵ_l, niter_l=10, restart=20) 
                end
            end

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" begin
                @info "Line search"
                @time imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, ξ, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            end
            @info "Solution update"
            @time UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)

        end

        # Update pressure
        Pt .+= ΔPt.c

        # Probe
        probes.τII[it] = mean(τ.II) 
        probes.t[it]   = it*Δ.t 

        #--------------------------------------------#

    end
    # gif(anim, "./results/ShearBanding.gif", fps = 5)

    display(to)
    
end

let
    main((x = 41, y = 41))
end