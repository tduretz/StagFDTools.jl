using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics:mean
using DifferentiationInterface
using TimerOutputs, CairoMakie

# comment rand() in initial pressure
# make large VP
# made :free_slip

function softening(Ep, Cmax, Cmin, E0, w)
    return Cmin + (Cmax-Cmin)*(1 - tanh((Ep-E0)/w))
end

@views function main(nc, nt, θgouge)
    #--------------------------------------------#

    # Scaling
    sc  = (σ=1e9, L=1e-3, t=1e6)

    # Parameters
    width     = (12e-3 + 38e-3)/sc.L
    height    = 100e-3/sc.L
    thickness = 2*2e-3/sc.L
    θgouge    = (θgouge) /180*π
    Δt0       = 1e2/sc.t
    ε̇yy       = 1.31e-6*sc.t
    Pbg       = 1e7/sc.σ

    𝐱_probe_salt  = (0.023/sc.L,   height/2) # apparatus response (plate and salt)
    𝐱_probe_plate = (0.000/sc.L, 0.002/sc.L) # apparatus response (plate and salt)
    # 𝐱_probe_MC1   = (0.003/sc.L, height/2) # sandstone
    𝐱_probe_MC1   = (0.006/sc.L, height/1.5) # sandstone
    𝐱_probe_MC2   = (0.001/sc.L, height/2) # gouge

    # Boundary loading type
    config = :EW_stress
    # config = :free_slip
    D_BC   = @SMatrix( [  ε̇yy  0.;
                          0  -ε̇yy ])
    σ_BC   = @SMatrix( [ -Pbg  0.;
                          0  -Pbg ])
                          
    # Material parameters
    nphases                    = 4
    materials                  = initialize_materials( nphases, compressible = true, Dzz=-0.0*ε̇yy, plasticity = DruckerPrager )
    materials.n               .= [  1.0,    1.0,     1.0,    1.0]             # Power law exponent
    materials.η0              .= [ 1e48,   1e28,    1e10,   1e48]./sc.σ./sc.t # Reference viscosity 
    materials.G               .= [ 1e10,    1e9,    1e60,   1e10]./sc.σ       # Shear modulus
    materials.β               .= [1e-11,  1e-10,   1e-12,  1e-12].*sc.σ       # Compressibility
    materials.plasticity.C    .= [ 45e6, 0.67e6,   15e60,  15e60]./sc.σ       # Cohesion
    materials.plasticity.ϕ    .= [  31.,   30.5,     35.,    35.]             # Friction angle
    materials.plasticity.ψ    .= [  0.0,    0.0,     0.0,    0.0]             # Dilation angle
    materials.plasticity.ηvp  .= [ 1e10,    1e9,    1e10,   1e10].*5/sc.σ./sc.t # Viscoplastic regularisation
    materials.plasticity.soft .= [  0.0,    0.0,     0.0,    0.0]             # softening ?
    #                            rock     gouge     salt   plates
    preprocess!(materials)

    @show  π*180/π/4 +  materials.plasticity.ϕ[2]/2 
    @show  π*180/π/4 + (materials.plasticity.ϕ[2] + materials.plasticity.ψ[2])/4 
    # error()

    # Geometry
    L     = (x=width, y=height)
    gouge = (
        Rectangle((0.0/sc.L, L.y/2), thickness, 20.0e-2/sc.L; θ = θgouge),
    )
    salt = (
        Rectangle((-L.x/2, L.y/2), 6e-3/sc.L, L.y; θ = 0),
        Rectangle(( L.x/2, L.y/2), 6e-3/sc.L, L.y; θ = 0),
    )
    plate = (
        Rectangle((0.0/sc.L,    2e-3/sc.L/2), 1.1e-1/sc.L, 7e-3/sc.L; θ = 0),
        Rectangle((0.0/sc.L,L.y-2e-3/sc.L/2), 1.1e-1/sc.L, 7.0e-3/sc.L; θ = 0),
    )
    # seed = (
    #     Ellipse((0.0, 0.0), 0.1/sc.L, 0.1/sc.L; θ = 0),
    # )

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
    dx = zeros(nVx + nVy + nPt)
    r  = zeros(nVx + nVy + nPt)

    #--------------------------------------------#
    # Discretisation
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    ξ       = (c=ones(size_c...), v=ones(size_v...))
    G       = (c=zeros(size_c...), v=zeros(size_v...))
    β       = (c=zeros(size_c...), v=zeros(size_v...))
    ρ       = (c=zeros(size_c...), v=zeros(size_v...))    
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    εp      = (c  = zeros(size_c...), v  = zeros(size_v...) )
    sp      = (c  =  ones(size_c...), v  =  ones(size_v...) )
    C       = (c  = zeros(size_c...), v  = zeros(size_v...) )
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
    x   = (min=-L.x/2, max=L.x/2)
    # y   = (min=-L.y/2, max=L.y/2)
    y   = (min=0.0, max=L.y)
    X   = GenerateGrid(x, y, Δ, nc)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    @views V.x .= D_BC[1,1]*X.vx_e.x .+ D_BC[1,2]*X.vx_e.y' 
    @views V.y .= D_BC[2,1]*X.vy_e.x .+ D_BC[2,2]*X.vy_e.y'
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]  .+ (type.Vx[     1, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]  .+ (type.Vx[   end, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]  .+ (type.Vy[inx_Vy,     1 ] .== :normal_stress) .* σ_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]  .+ (type.Vy[inx_Vy,   end ] .== :normal_stress) .* σ_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[1]   .+ D_BC[2,2]*X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[end] .+ D_BC[2,2]*X.v.y)
    end
    
    # Set material geometry 
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([X.c_e.x[i], X.c_e.y[j]])

        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                phases.c[i, j] = 2
            end
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                phases.c[i, j] = 3
            end
        end
        for igeom in eachindex(plate) # Plate: phase 4
            if inside(𝐱, plate[igeom])
                phases.c[i, j] = 4
            end
        end
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([X.v_e.x[i], X.v_e.y[j]])

        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                phases.v[i, j] = 2
            end  
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                phases.v[i, j] = 3
            end  
        end
        for igeom in eachindex(plate) # Plate: phase 4
            if inside(𝐱, plate[igeom])
                phases.v[i, j] = 4
            end  
        end
    end

    # Markers 
    nmpc = (x=4, y=4)
    noise = false

    # Initialise marker field
    m = InitialiseMarkerField(nc, nmpc, L, Δ, x, y, noise)
    phase_ratios, phase_weights = InitialisePhaseRatios(nphases, ε̇)

    # Set material geometry: circle
    for k in eachindex(m.phase) 
        m.phase[k] = 1
        𝐱 = @SVector([m.Xm[k], m.Ym[k]])
        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                 m.phase[k] = 2
            end  
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                 m.phase[k] = 3
            end  
        end
        for igeom in eachindex(plate) # Plate: phase 4
            if inside(𝐱, plate[igeom])
                 m.phase[k] = 4
            end  
        end

    end

    # Set phase ratios
    SetPhaseRatios!(phase_ratios, phase_weights, m, X.c_e.x, X.c_e.y, X.v_e.x, X.v_e.y, Δ, nphases)

    # check 
    for I in CartesianIndices(phase_ratios.c)
        s = sum(phase_ratios.c[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.c at $I: sum = $s, values = $(phase_ratios.c[I])"
        end
    end

    ΔCc = materials.plasticity.C[1]/5 .* rand(size(C.c)...)
    ΔCv = materials.plasticity.C[1]/5 .* rand(size(C.v)...)

    Pt  .= Pbg #*rand(size(Pt)...)
    Pt0 .= Pt
    Pti .= Pt

    #--------------------------------------------#

    rvec   = zeros(length(α))
    err    = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    probes = (τII = zeros(nt), τIIW = zeros(nt), τIIE = zeros(nt), τIIS = zeros(nt), τIIN = zeros(nt), true_fric = zeros(nt), app_fric = zeros(nt), app_fric_lab = zeros(nt), t = zeros(nt), εxx=zeros(nt), εyy=zeros(nt), σyyN=zeros(nt), σyyS=zeros(nt), σxxW=zeros(nt), σxxE=zeros(nt), PW=zeros(nt), PE=zeros(nt), minθ=zeros(nt), maxθ=zeros(nt), meanθ=zeros(nt))
    probe_salt  = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    probe_plate = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    probe_MC1   = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    probe_MC2   = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    to     = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        @printf("Step %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)
        
        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        compute_grid_fields!(G, β, ρ, ξ, C, materials, phase_ratios, nc, nphases)

        # G.c .*= sp.c 
        # G.v .*= sp.v 

        # C.c[phases.c.==1] .+= ΔCc[phases.c.==1]
        # C.v[phases.v.==1] .+= ΔCv[phases.v.==1]

        @printf("Time step %04d (nthreads = %03d)\n", it, Threads.nthreads())
        iter, ϵ0, ϵ = 0, 0.0, 0.

        # Time integration
        while iter<niter

            iter +=1
            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, sp, C, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
                @show extrema(λ̇.c[inx_c,iny_c])
                @show extrema(λ̇.v[inx_v,iny_v])
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(R.p[inx_c,iny_c])/sqrt(nPt)
            ϵ =  max(err.x[iter], err.y[iter])
            (iter == 1) && (ϵ0 = ϵ)

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                # Jacobian
                AssembleContinuity2D!(   M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, ξ, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(   M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(   M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, ρ, materials, number, pattern, type, BC, nc, Δ)
                # Preconditioner
                AssembleContinuity2D!(M_PC, V, Pt, Pt0, ΔPt, τ0,     𝐷, β, ξ, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M_PC, V, Pt, Pt0, ΔPt, τ0,     𝐷, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M_PC, V, Pt, Pt0, ΔPt, τ0,     𝐷, G, ρ, materials, number, pattern, type, BC, nc, Δ)
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
                    mechanical_solver!( dx, M,    r, 𝐊,    𝐐,    𝐐ᵀ,    𝐏,    𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC; solver=solver, ηb=γ, ϵ_l=ϵ_l, niter_l=10, restart=20, maxit=100) 
                else
                    mechanical_solver!( dx, M_PC, r, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC; solver=solver, ηb=γ, ϵ_l=ϵ_l, niter_l=10, restart=20, maxit=100) 
                end
            end

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, sp, C, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
        end

        # Update pressure
        Pt   .+= ΔPt.c
        εp.c .+= Δ.t .* λ̇.c
        εp.v .+= Δ.t .* λ̇.v

        # Softening
        sp.c .= softening.(εp.c, 1.0, 0.2, 0.0, 0.1)
        sp.v .= softening.(εp.v, 1.0, 0.2, 0.0, 0.1)
        @show extrema(sp.c )
        @show extrema(sp.v )

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = av2D(τ.xy)

        isrock      = zeros(Bool, size_c)
        isgouge     = zeros(Bool, size_c)
        [isrock[I]  = phase_ratios.c[I][1]≈1.0 for I in eachindex(phase_ratios.c)]
        [isgouge[I] = phase_ratios.c[I][2]≈1.0 for I in eachindex(phase_ratios.c)]

        τII_rock  = τ.II[isrock]
        P_rock    =   Pt[isrock]
        λ̇_rock    =  λ̇.c[isrock]

        τII_gouge = τ.II[isgouge]
        P_gouge   =   Pt[isgouge]
        λ̇_gouge   =  λ̇.c[isgouge]

        # Compute apparent and true friction locally
        app_fric       = zeros(size(Pt))
        true_fric      = zeros(size(Pt))
        Rot            = @SMatrix[cos(π/2 - θgouge) sin(π/2 - θgouge) 0; -sin(π/2 - θgouge) cos(π/2 - θgouge) 0; 0 0 1.0]
        app_fric_sum   = 0.0
        true_fric_sum  = 0.0
        app_sum        = 0
        true_sum       = 0
        ind_mid_x      = Int64(floor((nc.x+2)/2))
        ind_mid_y      = Int64(floor((nc.y+2)/2))

        for i in inx_c, j in iny_c
            if phases.c[i,j] == 2 # λ̇.c[i,j] > 1e-10
                app_sum     += 1
                σ  = @SMatrix[-Pt[i,j]+τ.xx[i,j] τxyc[i,j] 0.; τxyc[i,j] -Pt[i,j]+τ.yy[i,j] 0.; 0. 0. -Pt[i,j]+(-τ.xx[i,j]-τ.yy[i,j])]
                # Compute apparent friction
                σ′             = Rot * σ * Rot'
                app_fric[i,j]  = σ′[1,2] / σ′[2,2]
                app_fric_sum  += app_fric[i,j]
                # Compute true friction
                if λ̇.c[i,j] > abs(ε̇yy) &&  i==ind_mid_x && j==ind_mid_y
                    true_sum      += 1
                    ph             = phases.c[i,j]
                    cxcosϕ         = materials.plasticity.C[ph] * materials.plasticity.cosϕ[ph]
                    ηvp            = materials.plasticity.ηvp[ph]
                    true_fric[i,j] = tand(asind( 1/Pt[i,j] * (τ.II[i,j] - cxcosϕ - ηvp*λ̇.c[i,j])  ))
                    true_fric_sum += true_fric[i,j]
                end
            end
        end

        # Store probes data
        𝐈_probe_salt  = (argmin(abs.(X.c_e.x .- 𝐱_probe_salt[1] ))), argmin(abs.(X.c_e.y .- 𝐱_probe_salt[2] )) # apparatus response (plate and salt)
        𝐈_probe_plate = (argmin(abs.(X.c_e.x .- 𝐱_probe_plate[1]))), argmin(abs.(X.c_e.y .- 𝐱_probe_plate[2])) # apparatus response (plate and salt)
        𝐈_probe_MC1   = (argmin(abs.(X.c_e.x .- 𝐱_probe_MC1[1] ))),  argmin(abs.(X.c_e.y .- 𝐱_probe_MC1[2]  )) 
        𝐈_probe_MC2   = (argmin(abs.(X.c_e.x .- 𝐱_probe_MC2[1]))),   argmin(abs.(X.c_e.y .- 𝐱_probe_MC2[2]  )) 
    
        σyy = τ.yy .- Pt
        σxx = τ.xx .- Pt
        σzz = τ.yy .- Pt
        ind_mid_x = Int64(floor(nc.x/2))
        ind_mid_y = Int64(floor(nc.y/2))
        probes.t[it]    = it*Δ.t
        probes.τII[it]  = mean(τ.II)
        probes.τIIW[it] = τ.II[2,     ind_mid_y]  
        probes.τIIE[it] = τ.II[end-1, ind_mid_y] 
        probes.τIIS[it] = τ.II[ind_mid_x,     2]  
        probes.τIIN[it] = τ.II[ind_mid_x, end-1]  
        probes.PW[it]   = Pt[2,     ind_mid_y] 
        probes.PE[it]   = Pt[end-1, ind_mid_y] 
        probes.σxxW[it] = τ.xx[2,     ind_mid_y] - Pt[2,     ind_mid_y] 
        probes.σxxE[it] = τ.xx[end-1, ind_mid_y] - Pt[end-1, ind_mid_y] 
        probes.σyyS[it] = mean(τ.yy[inx_c,     2] .- Pt[inx_c,     2]) 
        probes.σyyN[it] = mean(τ.yy[inx_c, end-1] .- Pt[inx_c, end-1]) 
        
        probe_salt.σxx[it]  = σxx[𝐈_probe_salt...]
        probe_salt.σyy[it]  = σyy[𝐈_probe_salt...]
        probe_salt.σzz[it]  = σzz[𝐈_probe_salt...]
        probe_salt.σxy[it]  = τ.xy[𝐈_probe_salt...]
        probe_salt.P[it]    = Pt[𝐈_probe_salt...]
        probe_salt.τII[it]  = τ.II[𝐈_probe_salt...]

        probe_plate.σxx[it]  = σxx[𝐈_probe_plate...]
        probe_plate.σyy[it]  = σyy[𝐈_probe_plate...]
        probe_plate.σzz[it]  = σzz[𝐈_probe_plate...]
        probe_plate.σxy[it]  = τ.xy[𝐈_probe_plate...]
        probe_plate.P[it]    = Pt[𝐈_probe_plate...]
        probe_plate.τII[it]  = τ.II[𝐈_probe_plate...]

        probe_MC1.σxx[it]  = σxx[𝐈_probe_MC1...]
        probe_MC1.σyy[it]  = σyy[𝐈_probe_MC1...]
        probe_MC1.σzz[it]  = σzz[𝐈_probe_MC1...]
        probe_MC1.σxy[it]  = τ.xy[𝐈_probe_MC1...]
        probe_MC1.P[it]    = Pt[𝐈_probe_MC1...]
        probe_MC1.τII[it]  = τ.II[𝐈_probe_MC1...]

        probe_MC2.σxx[it]  = σxx[𝐈_probe_MC2...]
        probe_MC2.σyy[it]  = σyy[𝐈_probe_MC2...]
        probe_MC2.σzz[it]  = σzz[𝐈_probe_MC2...]
        probe_MC2.σxy[it]  = τ.xy[𝐈_probe_MC2...]
        probe_MC2.P[it]    = Pt[𝐈_probe_MC2...]
        probe_MC2.τII[it]  = τ.II[𝐈_probe_MC2...]

        # σd = probes.σyyS[it] - probes.σxxW[it]
        # σm = -probes.PW[it]
        σd = 2 *  1/2*(mean(τII_gouge) )
        σm = 1 *  1/2*(mean(  P_gouge) )
        τr = σd/2 * sin(2*(π/2 - θgouge))
        τn = σm + σd/2 * cos(2*(π/2 - θgouge))
        # probes.app_fric[it] =  τr / τn
        probes.app_fric[it]    = app_fric_sum / app_sum 
        probes.true_fric[it]   = true_fric_sum / true_sum 
        
        # σ  = @SMatrix[probes.σxxW[it] 0.0 0.; 0.0 probes.σyyS[it] 0.; 0. 0. 1/2*(probes.σxxW[it] + probes.σyyS[it])]
        σ  = @SMatrix[probe_salt.σxx[it] 0.0 0.; 
                      0. probe_plate.σyy[it] 0.; 
                      0. 0. -probe_salt.P[it]]

        σ′             = Rot * σ * Rot'
        probes.app_fric_lab[it] = σ′[1,2] / σ′[2,2]

        # Stress angles
        i1 = argmin(abs.(X.c_e.x .- (-0.01/sc.L)))
        i2 = argmin(abs.(X.c_e.x .- ( 0.01/sc.L)))
        j1 = argmin(abs.(X.c_e.y .- ( 0.025/sc.L)))
        j2 = argmin(abs.(X.c_e.y .- ( 0.075/sc.L)))
        θawayBC = τ.θ[i1:i2, j1:j2]
        probes.minθ[it]  = minimum(θawayBC[phases.c[i1:i2, j1:j2] .== 2])
        probes.meanθ[it] =    mean(θawayBC[phases.c[i1:i2, j1:j2] .== 2])
        probes.maxθ[it]  = maximum(θawayBC[phases.c[i1:i2, j1:j2] .== 2])

        # Visualise
        function figure()
            ftsz = 25
            fig = Figure(size=(1000, 1000)) 
            empty!(fig)

            # Split heatmap of the apparatus
            # ax  = Axis(fig[1:3,1], aspect=DataAspect(), title="Apparent / True friction", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            ax  = Axis(fig[1:3,1], aspect=DataAspect(), title="Plastic strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            eps   = 1e-1
            # field = (τ.xy)[inx_c,iny_c]  .* sc.σ / 1e6
            # field = app_fric[inx_c,iny_c]
            # field = (τ.yy .- Pt)[inx_c,iny_c]  .* sc.σ / 1e6
            field = log10.((λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            crange= (minimum(field)-eps, maximum(field)+eps)
            
            # field = C.c*sc.σ./1e6
            # crange = (5, 7)
            
            field1 = app_fric  .+ eps
            field2 = true_fric .+ eps
            # field = phases.c[inx_c,iny_c]
            hm = heatmap!(ax, X.c.x.*sc.L, X.c.y.*sc.L, field, colormap=:bluesreds, colorrange=crange)
            # hm1 = heatmap!(ax, X.c.x[1:ind_mid_x].*sc.L, X.c.y.*sc.L, field1[1:ind_mid_x,:], colormap=:bluesreds, colorrange=(minimum(field1)-eps, maximum(field1)+eps))
            # hm2 = heatmap!(ax, X.c.x[ind_mid_x:end].*sc.L, X.c.y.*sc.L, field2[ind_mid_x:end,:], colormap=:bluesreds, colorrange=(minimum(field2)-eps, maximum(field2)+eps))
            contour!(ax, X.c.x.*sc.L, X.c.y.*sc.L,  phases.c[inx_c,iny_c], color=:white)
            # Colorbar(fig[4, 1], hm1, label = L"$\phi^\text{app}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            # Colorbar(fig[4, 2], hm2, label = L"$\phi^\text{true}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            Vxc = (0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1]))[2:end-1,2:end-1].*sc.L/sc.t
            Vyc = (0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end]))[2:end-1,2:end-1].*sc.L/sc.t
            step = 10
            # arrows2d!(ax, X.c.x[1:step:end].*sc.L, X.c.y[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=50000.4, color=:white)
            arrows2d!(ax, X.c.x[1:step:end].*sc.L, X.c.y[1:step:end].*sc.L, cos.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], sin.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], lengthscale=0.001, color=:white, tiplength = 0)
            xlims!(ax, minimum(X.v.x).*sc.L, maximum(X.v.x).*sc.L)
            lines!(ax, X.c.x[ind_mid_x].*sc.L *  ones(size(X.c.y)), X.c.y.*sc.L, color=:white, linewidth=4)
            # Probes
            scatter!(ax, X.c.x[𝐈_probe_salt[1]].*sc.L,  X.c.y[𝐈_probe_salt[2]].*sc.L,  marker=:star5, markersize=20)
            scatter!(ax, X.c.x[𝐈_probe_plate[1]].*sc.L, X.c.y[𝐈_probe_plate[2]].*sc.L, marker=:star5, markersize=20)
            # Subdomain
            scatter!(ax, X.c.x[i1].*sc.L,  X.c.y[j1].*sc.L,  marker=:cross, color=:gray, markersize=20)
            scatter!(ax, X.c.x[i2].*sc.L,  X.c.y[j1].*sc.L,  marker=:cross, color=:gray, markersize=20)
            scatter!(ax, X.c.x[i1].*sc.L,  X.c.y[j2].*sc.L,  marker=:cross, color=:gray, markersize=20)
            scatter!(ax, X.c.x[i2].*sc.L,  X.c.y[j2].*sc.L,  marker=:cross, color=:gray, markersize=20)
            # Mohr-circles
            scatter!(ax, X.c.x[𝐈_probe_MC1[1]].*sc.L, X.c.y[𝐈_probe_MC1[2]].*sc.L, marker=:xcross, color=:white, markersize=20)
            scatter!(ax, X.c.x[𝐈_probe_MC2[1]].*sc.L, X.c.y[𝐈_probe_MC2[2]].*sc.L, marker=:xcross, color=:white, markersize=20)

            # # Zoom on the gouge
            # ax  = Axis(fig[2,2], aspect=DataAspect(), title="Plastic Strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            # eps   = 1e-1
            # # field = (τ.xy)[inx_c,iny_c]  .* sc.σ / 1e6
            # # field = app_fric[inx_c,iny_c]
            # # field = (τ.yy .- Pt)[inx_c,iny_c]  .* sc.σ / 1e6
            # field = log10.((λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            # # field = phases.c[inx_c,iny_c]
            # hm = heatmap!(ax, X.c.x.*sc.L, X.c.y.*sc.L, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, X.c.x.*sc.L, X.c.y.*sc.L,  phases.c[inx_c,iny_c], color=:white)
            # # arrows2d!(ax, X.c.x[1:step:end].*sc.L, X.c.y[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=50000.4, color=:white)
            # arrows2d!(ax, X.c.x[1:step:end].*sc.L, X.c.y[1:step:end].*sc.L, cos.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], sin.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], lengthscale=0.001, color=:white, tiplength = 0)
            # xlims!(ax, -0.02, 0.02)
            # ylims!(ax, 0.04, 0.06)

            ax  = Axis(fig[2,2], aspect=DataAspect(), title="Mohr circles", xlabel="σn", ylabel="τ", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            θ   = LinRange(0, π, 100)

            MC1 = (x = (probe_MC1.τII[it].*cos.(θ).+probe_MC1.P[it]), y = (probe_MC1.τII[it].*sin.(θ))) 
            MC2 = (x = (probe_MC2.τII[it].*cos.(θ).+probe_MC2.P[it]), y = (probe_MC2.τII[it].*sin.(θ))) 
            
            τII_MC3 = -(probe_plate.σyy[it] - probe_salt.σxx[it])/2
            P_MC3   = -(probe_plate.σyy[it] + probe_salt.σxx[it])/2
            MC3     = (x = (τII_MC3.*cos.(θ).+P_MC3), y = (τII_MC3.*sin.(θ))) 
            MC3_μ   = (x = (τII_MC3.*(cos.(2*(π/2-θgouge))).+P_MC3), y = (τII_MC3.*sin.(2*(π/2-θgouge)))) 

            lines!(ax, MC1.x*sc.σ./1e6, MC1.y*sc.σ./1e6, label="probe 1")
            lines!(ax, MC2.x*sc.σ./1e6, MC2.y*sc.σ./1e6, label="probe 2")
            lines!(ax, MC3.x*sc.σ./1e6, MC3.y*sc.σ./1e6, label="apparatus")
            σmin = minimum([minimum(MC1.x), minimum(MC2.x), minimum(MC3.x)])
            σmax = maximum([maximum(MC1.x), maximum(MC2.x), maximum(MC3.x)])
            scatter!(ax, MC3_μ.x*sc.σ./1e6, MC3_μ.y*sc.σ./1e6)
            probes.app_fric_lab[it] = MC3_μ.y / MC3_μ.x
            σn_ax = LinRange(σmin-0.1*(σmin), σmax+0.1*σmin, 100)
            # τ1    = materials.plasticity.cosϕ[1]/(1+materials.plasticity.sinϕ[1]) * (materials.plasticity.C[1] * materials.plasticity.cosϕ[1] .+ σn_ax.*materials.plasticity.sinϕ[1])
            # τ2    = materials.plasticity.cosϕ[2]/(1+materials.plasticity.sinϕ[2]) * (materials.plasticity.C[2] * materials.plasticity.cosϕ[2] .+ σn_ax.*materials.plasticity.sinϕ[2])
            τ1    = materials.plasticity.C[1] .+ σn_ax.*materials.plasticity.sinϕ[1]./materials.plasticity.cosϕ[1]
            τ2    = materials.plasticity.C[2] .+ σn_ax.*materials.plasticity.sinϕ[2]./materials.plasticity.cosϕ[2]
            lines!(ax, σn_ax*sc.σ/1e6, τ1*sc.σ/1e6, color=:black)
            lines!(ax, σn_ax*sc.σ/1e6, τ2*sc.σ/1e6, color=:black)

            # ax  = Axis(fig[1,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, 1:niter, log10.(err.x[1:niter]./err.x[1]) )
            # scatter!(ax, 1:niter, log10.(err.y[1:niter]./err.y[1]) )
            # scatter!(ax, 1:niter, log10.(err.p[1:niter]./err.p[1]) )
            # ylims!(ax, -15, 1)
            ax  = Axis(fig[3,2], title=L"$$Stress space", xlabel=L"$P$", ylabel=L"$\tau_{II}$", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            P_ax       = LinRange(minimum(P_rock), maximum(P_rock), 100)
            τ_ax_rock  = materials.plasticity.C[1] * materials.plasticity.cosϕ[1] .+ P_ax.*materials.plasticity.sinϕ[1]
            τ_ax_gouge = materials.plasticity.C[2] * materials.plasticity.cosϕ[2] .+ P_ax.*materials.plasticity.sinϕ[2]
            lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:black)
            lines!(ax, P_ax*sc.σ/1e6, τ_ax_rock*sc.σ/1e6, color=:black)
            scatter!(ax,  P_rock*sc.σ/1e6, ( τII_rock .- λ̇_rock.* materials.plasticity.ηvp[2])*sc.σ/1e6, color=:blue )
            scatter!(ax, P_gouge*sc.σ/1e6, (τII_gouge .- λ̇_gouge.*materials.plasticity.ηvp[2])*sc.σ/1e6, color= :red )

            # τ_ax_gouge = materials.C[2]*materials.cosϕ[2] .+ P_ax.*materials.sinϕ[2]
            # lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:red)
            # scatter!(ax, P_gouge*sc.σ/1e6, τII_gouge*sc.σ/1e6, color=:red )

            ax  = Axis(fig[0,1], xlabel="Axial shortening", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.PW[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.PE[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σxxW[1:it]*sc.σ./1e6, marker=:star5 ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σxxE[1:it]*sc.σ./1e6, marker=:star5 ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σyyN[1:it]*sc.σ./1e6, marker=:circle )
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σyyS[1:it]*sc.σ./1e6, marker=:circle )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, -probe_salt.σxx[1:it]*sc.σ./1e6, marker=:circle )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, -probe_plate.σyy[1:it]*sc.σ./1e6, marker=:circle )

            # ax  = Axis(fig[0,1], xlabel="Axial shortening", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.τII[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20
 
            # ax  = Axis(fig[0,2], xlabel="time [hrs]", ylabel="τII [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τII[1:it]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIW[1:it]*sc.σ./1e6, marker=:star5, markersize=20 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIE[1:it]*sc.σ./1e6, marker=:star5, markersize=20 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIS[1:it]*sc.σ./1e6, marker=:diamond, markersize=20 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIN[1:it]*sc.σ./1e6, marker=:diamond, markersize=20 )

            ax  = Axis(fig[0,2], xlabel="Axial shortening", ylabel="-τ/σ", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            lines!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, ones(it)*tand(materials.plasticity.ϕ[2]), linestyle=:dash, color=:gray )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.app_fric[1:it] )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.app_fric_lab[1:it] )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.true_fric[1:it], marker=:star5) #, markersize=20  
            tanα = cosd(materials.plasticity.ψ[2])*sind(materials.plasticity.ϕ[2]) / (1 - sind(materials.plasticity.ψ[2])*sind(materials.plasticity.ϕ[2]))
            lines!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, ones(it)*tanα, linestyle=:dash, color=:gray )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_salt.σxy[1:it]./probe_salt.σyy[1:it], marker=:circle )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_plate.σxy[1:it]./probe_plate.σyy[1:it], marker=:star5 )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_MC1.σxy[1:it]./probe_MC1.σyy[1:it], marker=:diamond )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_MC2.σxy[1:it]./probe_MC2.σyy[1:it], marker=:xcross )


            ax  = Axis(fig[1,2], xlabel="time [hrs]", ylabel="σ1 angle gouge", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.minθ[1:it]*180/π )
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.meanθ[1:it]*180/π )
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.maxθ[1:it]*180/π )
            
            display(fig)

        end
        with_theme(figure, theme_latexfonts())
    end
    display(to)
end

let
    # main((x = 150, y = 250), 1000, 30) # matches friction
    main((x = 100, y = 200), 100, 30)
end