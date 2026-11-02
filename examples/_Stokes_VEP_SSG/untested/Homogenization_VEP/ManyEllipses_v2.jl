using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
using Statistics
using DifferentiationInterface
using Random
Random.seed!(1234)
using TimerOutputs

function laplacian(A, dx, dy)
    nx, ny = size(A)
    L = fill(NaN, nx, ny)

    for i in 2:nx-1, j in 2:ny-1
        if !isnan(A[i,j]) &&
           !isnan(A[i+1,j]) && !isnan(A[i-1,j]) &&
           !isnan(A[i,j+1]) && !isnan(A[i,j-1])

            L[i,j] =
                (A[i+1,j] - 2A[i,j] + A[i-1,j]) / dx^2 +
                (A[i,j+1] - 2A[i,j] + A[i,j-1]) / dy^2
        end
    end

    return L
end

# @views function PorousMediumEllipses!(phase, inx, iny, X, Y)

#     for i=1:100
#         # Ellipse 1
#         x0, y0 = rand() - 0.5, rand() - 0.5
#         α  = rand() * 90
#         ar = rand() * 10
#         r  = rand() * 0.02
#         𝑋 = cosd(α)*X .- sind(α).*Y'
#         𝑌 = sind(α)*X .+ cosd(α).*Y'
#         phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2
#     end
# end

@views function PorousMediumEllipses!(phase, inx, iny, X, Y)

    for i=1:1000
        # Ellipse 1
        x0, y0 = rand() - 0.5, rand() - 0.5
        α  = rand() * 90
        ar = rand() * 4
        r  = rand() * 0.02
        𝑋 = cosd(α)*X .- sind(α).*Y'
        𝑌 = sind(α)*X .+ cosd(α).*Y'
        phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2
    end
end

@views function main(nc)
    #--------------------------------------------#

    # Scales
    sc = (σ = 1e0, L = 1e-0, t=1e0)

    # Boundary loading type
    config = :free_slip
    ε̇bg    = -1e0*sc.t
    # D_BC   = @SMatrix( [ -ε̇bg 0.;
    #                       0.  ε̇bg ])
    D_BC   = @SMatrix( [  0.0 ε̇bg*2.;
                          0.  0.0 ])

    # Material parameters
    materials = ( 
        compressible = false,
        plasticity   = :none,
        g    = [1.0    1.0  ],
        ρ    = [0.0    0.0  ], 
        n    = [1.0    1.0  ],
        η0   = [1e0    1.00001e0 ]./(sc.σ * sc.t), 
        G    = [3e50   1e50  ]./(sc.σ),
        C    = [50e6   1e60 ]./(sc.σ),
        ϕ    = [30.    0.   ],
        ηvp  = [1e17   0.   ]./(sc.σ * sc.t),
        β    = [1e-11  4e-10].*(sc.σ),
        ψ    = [0.0    0.0  ],
        B    = [0.0    0.0  ],
        cosϕ = [0.0    0.0  ],
        sinϕ = [0.0    0.0  ],
        sinψ = [0.0    0.0  ],
    )
    # For power law
    materials.B   .= (2*materials.η0).^(-materials.n)

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)

    # Time steps
    Δt0   = 1e8/sc.t
    nt    = nc.t

    # Newton solver
    niter = 20
    ϵ_nl  = 1e-8
    α     = LinRange(0.05, 1.0, 10)

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
    𝐊  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏  = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r  = zeros(nVx + nVy + nPt)

    M_PC = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊_PC  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐_PC  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ_PC = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏_PC  = ExtendableSparseMatrix(nPt, nPt)

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
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
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
    @views V.x .= D_BC[1,1]*X.vx_e.x .+ D_BC[1,2]*X.vx_e.y' 
    @views V.y .= D_BC[2,1]*X.vy_e.x .+ D_BC[2,2]*X.vy_e.y'
    @views Pt[inx_c, iny_c ]  .= 1e7/sc.σ                 
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

    PorousMediumEllipses!(phases.c, inx_c, iny_c, xc, yc)
    PorousMediumEllipses!(phases.v, inx_v, iny_v, xv, yv)

    # PorousMediumCircles!(phases.c, inx_c, iny_c, xc, yc)
    # PorousMediumCircles!(phases.v, inx_v, iny_v, xv, yv)

    # # Set material geometry 
    # @views phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    # @views phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2

    #--------------------------------------------#

    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    
    ϕ = sum(phases.c.==2)/ *(size(phases.c)...)
    @show ϕ
    
    probes = ( 
        Pt =  zeros(nt),
        Pf =  zeros(nt),
        Ps =  zeros(nt),
        τt =  zeros(nt),
        τf =  zeros(nt),
        τs =  zeros(nt),
    )
    to   = TimerOutput()

    #--------------------------------------------#

    anim = @animate for it=1:nt

        @printf("Step %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)
        
        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        for iter=1:niter

            @info "Newton iteration $(iter)"

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
   TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, ξ, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
                @show extrema(λ̇.c)
                @show extrema(λ̇.v)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(R.p[inx_c,iny_c])/sqrt(nPt)
            
            @info err.x

            max(err.x[iter], err.y[iter]) < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            end

            @timeit to "Assembly" begin
                AssembleContinuity2D!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt

            # Stokes operator as block matrices
            𝐊_PC  .= [M_PC.Vx.Vx M_PC.Vx.Vy; M_PC.Vy.Vx M_PC.Vy.Vy]
            𝐐_PC  .= [M_PC.Vx.Pt; M_PC.Vy.Pt]
            𝐐ᵀ_PC .= [M_PC.Pt.Vx M_PC.Pt.Vy]
            𝐏_PC  .= M_PC.Pt.Pt
            
            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = @views -r[1:size(𝐊,1)]
            fp   = @views -r[size(𝐊,1)+1:end]
            @timeit to "Solver" u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e5, niter_l=10, ϵ_l=1e-9, 𝐊_PC=𝐊_PC)
            @views dx[1:size(𝐊,1)]     .= u
            @views dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, ξ, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)

            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, ξ, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)

        end

        # Update pressure
        Pt .+= ΔPt.c

        #--------------------------------------------#

        τxyc = av2D(τ.xy)
        τII  = sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + (-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        ε̇xyc = av2D(ε̇.xy)
        ε̇II  = sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + (-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )
        
        fluid = phases.c .== 2
        solid = phases.c .== 1
        if sum(fluid) == 0
            τ_fluid = 0.
            P_fluid = 0.
        else
            τ_fluid = sum(τII[fluid[inx_c,iny_c]])/sum(fluid)
            P_fluid = sum(Pt[fluid])/sum(fluid)
        end
        P_solid = sum(Pt[solid])/sum(solid)
        P_total = ϕ*P_fluid + (1-ϕ)*P_solid
        τ_solid = sum(τII[solid[inx_c,iny_c]])/sum(solid)
        τ_total = ϕ*τ_fluid + (1-ϕ)*τ_solid

        τ_dry  = materials.C[1]*cosd(materials.ϕ[1]) +            P_total *sind(materials.ϕ[1])
        τ_terz = materials.C[1]*cosd(materials.ϕ[1]) + (P_total-  P_fluid)*sind(materials.ϕ[1])
        τ_shi  = materials.C[1]*cosd(materials.ϕ[1]) + (P_total-ϕ*P_fluid)*sind(materials.ϕ[1])

        @show (ϕ)
        @show (P_total)
        @show (P_total -  P_fluid)

        @show sum(fluid), P_fluid, τ_fluid
        @show sum(solid), P_solid, τ_solid
        probes.Pf[it] = P_fluid
        probes.Ps[it] = P_solid
        probes.Pt[it] = P_total
        probes.τf[it] = τ_fluid
        probes.τs[it] = τ_solid
        probes.τt[it] = τ_total

        #############################

        Wz    = zeros(size_v...)
        dVxdy = zeros(size_v...)
        dVydx = zeros(size_v...)
        Wz_m = 0.0
        Wz_f = 0.0
        Wz_s = 0.0
        iwz_m, iwz_f, iwz_s = 0, 0, 0

        for I in CartesianIndices(ε̇.xy)
            i, j = I[1], I[2]
            if i>2 && j>2 && i<nc.x+0  && j<nc.y+0
                iwz_m +=1
                wz    = 1/2 * ((V.x[i+0,j+1] -  V.x[i+0,j+0])/Δ.y - (V.y[i+1,j+0] - V.y[i+0,j+0])/Δ.x )
                dVxdy[I] = (V.x[i+0,j+1] -  V.x[i+0,j+0])/Δ.y
                dVydx[I] = (V.y[i+1,j+0] -  V.y[i+0,j+0])/Δ.x
                Wz[I] = wz
                Wz_m += wz
                if phases.v[I] == 1
                    iwz_f +=1
                    Wz_f  += wz
                elseif phases.v[I] == 2
                    iwz_s += 1
                    Wz_s  += wz
                end
            end
        end

        Wz_m /= iwz_m
        Wz_s /= iwz_s
        Wz_f /= iwz_f

        @show Wz_m
        @show Wz_f
        @show Wz_s

        @show (Wz_f-Wz_m)/Wz_m*100
        @show (Wz_s-Wz_m)/Wz_m*100

        ###############################

        # Block analysis
        k = [2, 4, 8, 16, 32, 64, 128]

        R_block       = Vector{Matrix{Float64}}(undef, length(k))

        tau_xx = zeros(size_v...)
        tau_yy = zeros(size_v...)
        tau_xx[inx_v,iny_v] .= av2D(τ.xx)
        tau_yy[inx_v,iny_v] .= av2D(τ.yy)
        tau_xy = τ.xy
        tau_yx = τ.xy


        for ik = 1:length(k)
            kx = k[ik]   # block size in x
            ky = k[ik]   # block size in y
            nbx = div(nc.x, kx)
            nby = div(nc.y, ky)

             @info "nbx = $nbx, nby=$nby"

            # =========================
            # 0. Grid / geometry
            # =========================
            dx = L.x / nc.x
            dy = L.y / nc.y

            Lx_block = L.x / nbx
            Ly_block = L.y / nby
            A_block  = Lx_block * Ly_block

            # =========================
            # 1. Allocate fields
            # =========================
            Tz_block     = fill(NaN, nbx, nby)
            D12_block    = fill(NaN, nbx, nby)
            σ12_block    = fill(NaN, nbx, nby)
            φ_block      = fill(NaN, nbx, nby)
            R_block[ik]  = fill(NaN, nbx, nby)

            # =========================
            # 2. Block loop
            # =========================
            for bx in 1:nbx, by in 1:nby

                i_start = (bx-1)*kx + 1
                i_end   = bx*kx
                j_start = (by-1)*ky + 1
                j_end   = by*ky

                ntot = 0
                ns   = 0

                D_sum = 0.0
                σ_sum = 0.0
                T_sum = 0.0

                # --- block center (physical) ---
                x_c = ((i_start + i_end)/2 - 0.5) * dx
                y_c = ((j_start + j_end)/2 - 0.5) * dy

                for i in i_start:i_end, j in j_start:j_end

                    # --- physical coordinates ---
                    x = (i - 0.5) * dx
                    y = (j - 0.5) * dy

                    # --- gradients ---
                    dxy = dVxdy[i,j]
                    dyx = dVydx[i,j]

                    # --- kinematics ---
                    D12 = 0.5 * (dxy + dyx)

                    # --- stresses ---
                    τxy = tau_xy[i,j]
                    τyx = tau_yx[i,j]  # usually equal

                    # --- lever arm ---
                    rx = x - x_c
                    ry = y - y_c

                    # --- torque contribution ---
                    T_sum += rx * τxy - ry * τyx

                    # --- accumulate ---
                    D_sum += D12
                    σ_sum += τxy
                    ntot += 1

                    if phases.v[i,j] == 2
                        ns += 1
                    end
                end

                if ntot > 0
                    D12_block[bx,by] = D_sum / ntot
                    σ12_block[bx,by] = σ_sum / ntot
                    φ_block[bx,by]   = ns / ntot

                    # --- normalize torque (intensive) ---
                    Tz_block[bx,by] = T_sum / A_block
                    R_block[ik][bx,by] = Tz_block[bx,by]
                end
            end

            # =========================
            # 1. Compute Laplacian
            # =========================
            L_D12 = laplacian(D12_block, Lx_block, Ly_block)

            # =========================
            # 2. Build dataset
            # =========================
            mask = .!isnan.(L_D12) .&
                .!isnan.(D12_block) .&
                .!isnan.(σ12_block) .&
                (φ_block .> 0.2) .&
                (φ_block .< 0.8)

            X1 = D12_block[mask]
            X2 = L_D12[mask]
            Y  = σ12_block[mask]

            println("Number of valid blocks: ", length(X1))

            # =========================
            # 3. Statistics
            # =========================

            println("\n--- Statistics ---")

            println("mean(|D|) ≈ ", mean(abs.(X1)))
            println("mean(|∇²D|) ≈ ", mean(abs.(X2)))

            println("\nCorrelations:")
            println("cor(D, σ)    = ", cor(X1, Y))
            println("cor(∇²D, σ)  = ", cor(X2, Y))
            println("cor(D, ∇²D)  = ", cor(X1, X2))

            # =========================
            # 4. Nonlocal regression
            # =========================

            # normalize (important!)
            s1 = std(X1)
            s2 = std(X2)

            X1n = X1 ./ s1
            X2n = X2 ./ s2

            A = hcat(2 .* X1n, X2n)

            coeffs = A \ Y

            μ_fit = coeffs[1] / s1
            β_fit = coeffs[2] / s2

            println("\n--- Nonlocal fit ---")
            println("μ = ", μ_fit)
            println("β = ", β_fit)
            println("β/μ = ", β_fit / μ_fit)

            # =========================
            # 5. Model comparison
            # =========================

            μ_classical = (2 .* X1) \ Y

            Y_classical = 2 .* μ_classical .* X1
            Y_nonlocal  = 2 .* μ_fit .* X1 .+ β_fit .* X2

            rmse_classical = sqrt(mean((Y - Y_classical).^2))
            rmse_nonlocal  = sqrt(mean((Y - Y_nonlocal).^2))

            println("\n--- Model comparison ---")
            println("RMSE classical = ", rmse_classical)
            println("RMSE nonlocal  = ", rmse_nonlocal)
            println("Improvement    = ", rmse_classical / rmse_nonlocal)

            # =========================
            # 6. Internal length scale
            # =========================

            if μ_fit != 0
                ℓ = sqrt(abs(β_fit / μ_fit))
                println("\nEstimated internal length ℓ = ", ℓ)
            end

            valid = .!isnan.(R_block[1])
            R_mean = mean(R_block[1][valid])
            R_rms  = sqrt(mean(R_block[1][valid].^2))

            ω_mean = mean(abs.(Wz))
            χ = R_rms / ω_mean
            @show χ
        end
 
        ###############################

        # p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
        # p2 = heatmap(xc, yc,  (Pt[inx_c,iny_c].-mean(Pt[inx_c,iny_c]))'.*sc.σ, aspect_ratio=1, xlim=extrema(xc), title="Pt")
        p3  = heatmap(xc, yc,  (phases.c[inx_c,iny_c])', aspect_ratio=1, xlim=extrema(xc), title="phases")
        # p2 = heatmap(R_block', aspect_ratio=1, title="R_block")
        p2 = histogram(R_block[1][.!isnan.(R_block[1])])
        # p3 = heatmap(xc, yc,  log10.(ε̇II)', aspect_ratio=1, xlim=extrema(xc), title="ε̇II", c=:coolwarm)
        # p3 = plot(xlabel="time", ylabel="stress")
        # p3 = plot!([1:it].*Δ.t,  probes.τf[1:it].*sc.σ, label="τ fluid")
        # p3 = plot!([1:it].*Δ.t,  probes.τs[1:it].*sc.σ, label="τ solid")
        # p3 = plot!([1:it].*Δ.t,  probes.τt[1:it].*sc.σ, label="τ total")
        # p3 = plot!([1:it].*Δ.t,  probes.Pf[1:it].*sc.σ, label="P fluid")
        # p3 = plot!([1:it].*Δ.t,  probes.Ps[1:it].*sc.σ, label="P solid")
        # p3 = plot!([1:it].*Δ.t,  probes.Pt[1:it].*sc.σ, label="P total")
        # p3 = plot!([1:it].*Δ.t,  τ_dry *ones(it).*sc.σ, label="τ dry",  linewidth=2)
        # p3 = plot!([1:it].*Δ.t,  τ_terz*ones(it).*sc.σ, label="τ Terz", linewidth=2)
        # p3 = plot!([1:it].*Δ.t,  τ_shi *ones(it).*sc.σ, label="τ Shi",  linewidth=2, legend=:outertopright)

        p4 = heatmap(xc, yc,  τII'.*sc.σ, aspect_ratio=1, xlim=extrema(xc), title="τII", c=:turbo)
        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright)
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))

        @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))

    end
    # gif(anim, "./results/ShearBanding.gif", fps = 5)

    display(to)
    
end

let
    main((x = 600, y = 600, t=1))
end