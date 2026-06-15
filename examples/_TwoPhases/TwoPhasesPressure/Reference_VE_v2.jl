using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
@views function main(nc, Ωl, Ωη, viscoelastic)

    homo   = false

    # Linear solver
    solver      = :GCR
    GCR_restart = 25
    GCR_maxit   = 2000

    # Non-linear solver
    niter       = 4

    if viscoelastic
        nt           = 120*1
        make_elastic = 1.0
    else
        nt           = 1
        make_elastic = 1e30
    end

    # Adimensionnal numbers
    Ωr     = 0.1             # Ratio inclusion radius / L
    Ωηi    = 1e-1            # Ratio (inclusion viscosity) / (matrix viscosity)
    Ωp     = 1.              # Ratio (ε̇bg * ηs) / P0
    # Independent
    ηsi    = 1.              # Shear viscosity
    L      = 1.              # Box size
    Pi     = 1.              # Initial ambiant pressure
    Φi     = 1e-2            # Reference
    n_CK   = 3.0
    # Dependent
    @show Ωl, Ωr, L
    δ      = Ωl * Ωr * L     # δ = δ/r * r/L where L = 1
    ηbi    = Ωη * ηsi        # Bulk viscosity
    k_ηΦ   = δ^2 / (ηbi + 4/3 * ηsi) # Permeability / fluid viscosity
    r      = Ωr * L          # Inclusion radius
    ηs_inc = Ωηi * ηsi# * 5
          # Inclusion shear viscosity
    ε̇bg      = Ωp * Pi / ηsi #* 5 # Background strain rate
    # Time integration
    Δt0    = 2.5e-4 #1 / ε̇ / nc.x / 2 / 40  

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇bg 0; 0 -ε̇bg] )

    τxx_ini = 0.0
    τyy_ini = 0.0

    # Material parameters
    nphases = 2
    materials = initialize_materials_TwoPhases(nphases,
        oneway       = false,
        compressible = true,
        linearizeΦ   = false, 
        single_phase = false,
        conservative = false,
        plasticity   = DruckerPrager,
    )
    materials.η0             .= [ηsi,          ηs_inc      ] 
    materials.n_CK           .= [n_CK,         n_CK        ] 
    materials.ξ0             .= [ηbi,          ηbi         ]
    materials.k_ηf0          .= [k_ηΦ/Φi^n_CK, k_ηΦ/Φi^n_CK]
    materials.G              .= [1e0,   1e0] * 2000  * make_elastic
    materials.Ks             .= [1e0,   1e0] * 1.1e4 * make_elastic
    materials.Kf             .= [1e0,   1e0] * 1e4   * make_elastic
    materials.KΦ             .= [1e0,   1e0] * 9e3   * make_elastic
    materials.plasticity.C   .= [1e50,  1e50]
    materials.plasticity.ϕ   .= [30. ,  30. ]
    materials.plasticity.ηvp .= [8e-3,  8e-3]
    materials.plasticity.ψ   .= [0.0 ,  0.0 ]
    preprocess!(materials)

    Φ0 =    Φi  
    # Φ0 = (materials.KΦ[1] .* Δt0 .* (Pf_ini - Pt_ini)) ./ (materials.KΦ[1] .* materials.ξ0[1])
    @show Φ0
    # error()
    Φ_ini   = Φ0

    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx]  .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
    type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy]  .= :in       
    type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
    type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Dirichlet 
    type.Pf[end,:]           .= :Dirichlet 
    type.Pf[:,1]             .= :Dirichlet
    type.Pf[:,end]           .= :Dirichlet
    
    # Equation Fields
    number = Fields(
        fill(0, (nc.x+3, nc.y+4)),
        fill(0, (nc.x+4, nc.y+3)),
        fill(0, (nc.x+2, nc.y+2)),
        fill(0, (nc.x+2, nc.y+2)),
    )
    Numbering!(number, type, nc)

    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1;  1 1 1]),        @SMatrix([1 1 1;  1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1]),        @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                       @SMatrix([1]),                   @SMatrix([1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                       @SMatrix([1]),                   @SMatrix([1 1 1; 1 1 1; 1 1 1])),
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    nPf   = maximum(number.Pf)
    M     = Fields(
        Fields(spzeros(nVx, nVx), spzeros(nVx, nVy), spzeros(nVx, nPt), spzeros(nVx, nPt)),
        Fields(spzeros(nVy, nVx), spzeros(nVy, nVy), spzeros(nVy, nPt), spzeros(nVy, nPt)),
        Fields(spzeros(nPt, nVx), spzeros(nPt, nVy), spzeros(nPt, nPt), spzeros(nPt, nPf)),
        Fields(spzeros(nPf, nVx), spzeros(nPf, nVy), spzeros(nPf, nPt), spzeros(nPf, nPf)),
    )
    M_PC  = Fields(
        Fields(spzeros(nVx, nVx), spzeros(nVx, nVy), spzeros(nVx, nPt), spzeros(nVx, nPt)),
        Fields(spzeros(nVy, nVx), spzeros(nVy, nVy), spzeros(nVy, nPt), spzeros(nVy, nPt)),
        Fields(spzeros(nPt, nVx), spzeros(nPt, nVy), spzeros(nPt, nPt), spzeros(nPt, nPf)),
        Fields(spzeros(nPf, nVx), spzeros(nPf, nVy), spzeros(nPf, nPt), spzeros(nPf, nPf)),
    )
    # Global arrays
    dx   = zeros(nVx + nVy + nPt + nPf)
    r    = zeros(nVx + nVy + nPt + nPf)
    solver_cache = 0 

    #--------------------------------------------#
    # Intialise field
    L   = (x=L, y=L)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...), Φ=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    Vi  = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ   = (c=Φ_ini.*ones(size_c...), v=Φ_ini.*ones(size_v...) )
    Φ0  = (c=Φ_ini.*ones(size_c...), v=Φ_ini.*ones(size_v...) )
    εp  = zeros(size_c...)
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = τxx_ini.*ones(size_c...), yy = τyy_ini.*ones(size_c...), xy = zeros(size_v...) )
    τ       = (xx = τxx_ini.*ones(size_c...), yy = τyy_ini.*ones(size_c...), xy = zeros(size_v...), II = zeros(size_c...), f = zeros(size_c...) )
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)
    
    ξ0      = (c  =  ones(size_c...), v  =  ones(size_v...) )
    m       = (c=zeros(size_c...),)
    k_ηf0   = (c=zeros(size_c...),)
    n_CK    = (c=zeros(size_c...),)
    G       = (c=zeros(size_c...), v=zeros(size_v...))
    ρsi     = (c=zeros(size_c...),)
    ρfi     = (c=zeros(size_c...),)
    Ks      = (c=zeros(size_c...), v=zeros(size_v...))
    KΦ      = (c=zeros(size_c...), v=zeros(size_v...))
    Kf      = (c=zeros(size_c...), v=zeros(size_v...))
    
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t = 0.0*ones(size_c...), f = 0.0.*ones(size_c...))
    P0      = (t = zeros(size_c...), f = zeros(size_c...))
    ΔP      = (t = zeros(size_c...), f = zeros(size_c...))
    Pi      = (t = 0.0*ones(size_c...), f = 0.0.*ones(size_c...))
    ρ       = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))
    ρ0      = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))
   
    # Generate grid coordinates 
    x = (min=-L.x/2, max=L.x/2)
    y = (min=-L.y/2, max=L.y/2)
    X = GenerateGrid(x, y, Δ, nc)

    # Find nodes for monitoring
    ix     = argmin(abs.(X.c.x .- 0.15))
    iy     = argmin(abs.(X.c.y .- 0.15))
    ix_mid = argmin(abs.(X.c.x .- 0.00))
    iy_mid = argmin(abs.(X.c.y .- 0.00))

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'

    if !homo
        # for I in CartesianIndices(Φ.c)
        #     i, j = I[1], I[2]
        #     if i>1 && i<size(Φ.c,1) && j>1 && j<size(Φ.c,2)
        #         if (X.c.x[i-1]^2 + X.c.y[j-1]^2) < rad^2
        #             Φ.c[i,j] = 1.1*Φ_ini
        #         end
        #     end 
        # end

        # Set material geometry 
        rad = Ωr
        @views phases.c[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y').^2) .<= rad^2] .= 2
        @views phases.v[inx_v, iny_v][(X.v.x.^2 .+ (X.v.y').^2) .<= rad^2] .= 2
    end

    phase_ratios = InitialisePhaseRatios(phases, nphases)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[1]   .+ D_BC[2,2]*X.v.y)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[end] .+ D_BC[2,2]*X.v.y)
    
    #--------------------------------------------#

    # Newton solver
    niter  = 25
    ϵ_nl   = 1e-8
    α      = LinRange(0.05, 1.0, 5)
    rvec   = zeros(length(α))

    probes = (
        maxPt = zeros(nt),
        maxPf = zeros(nt),
        maxτ  = zeros(nt),
        Pti = zeros(nt),
        Pfi = zeros(nt),
        Pei = zeros(nt),
        ΔPt = zeros(nt),
        ΔPf = zeros(nt),
        ΔPe = zeros(nt),
        normτ   = zeros(nt),
        normPe  = zeros(nt),
        normPt  = zeros(nt),
        normPf  = zeros(nt),
        meanτ   = zeros(nt),
        meanPe  = zeros(nt),
        meanPt  = zeros(nt),
        meanPf  = zeros(nt),
        t   = zeros(nt),
    )

    err  = (x = zeros(niter), y = zeros(niter), pt = zeros(niter), pf = zeros(niter))
    
    for it=1:nt

        @printf("\nStep %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.pt, 0e0)
        fill!(err.pf, 0e0)

        # Swap old values 
        P0.t  .= P.t
        P0.f  .= P.f
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Φ0.c  .= Φ.c 
        ρ0.s  .= ρ.s
        ρ0.f  .= ρ.f

        # Compute bulk and shear moduli
        compute_grid_fields_two_phases!(G, Ks, KΦ, Kf, ξ0, m, ρfi, ρsi, k_ηf0, n_CK, materials, phase_ratios, nc, nphases)

        old  = τ0, P0, Φ0, ρ0
        rheo = G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK

        for iter=1:niter

            @printf("     Step %04d --- Iteration %04d\n", it, iter)

            λ̇.c   .= 0.0
            λ̇.v   .= 0.0

            #--------------------------------------------#
            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, rheo, Δ)
            ResidualMomentum2D_x!(     R, V, P, ΔP, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(     R, V, P, ΔP, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(     R, V, P, ΔP, old,    rheo, materials, number, type, BC, nc, Δ) 
            ResidualFluidContinuity2D!(R, V, P, ΔP, old,    rheo, materials, number, type, BC, nc, Δ) 
            println("min/max λ̇.c  - ",  extrema(λ̇.c[inx_c,iny_c]))
            println("min/max λ̇.v  - ",  extrema(λ̇.v[3:end-2,3:end-2]))
            println("min/max ΔP.t - ",  extrema(ΔP.t[inx_c,iny_c]))
            println("min/max ΔP.f - ",  extrema(ΔP.f[inx_c,iny_c]))

            @info "Residuals"
            @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

            err.x[iter]  = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter]  = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.pt[iter] = @views norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            err.pf[iter] = @views norm(R.pf[inx_c,iny_c])/sqrt(nPt)
            if max(err.x[iter], err.y[iter], err.pt[iter], err.pf[iter]) < ϵ_nl 
                println("Converged")
                break 
            end

            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
            
            # Assemble global Jacobian
            @info "Assemble Jacobian, ndof  = $(nVx + nVy + nPt + nPf)"
            M_PC_threads = reset_parallel_storage(number)
            @time AssembleMomentum2D_x!(     M_PC_threads, V, P, ΔP, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
            @time AssembleMomentum2D_y!(     M_PC_threads, V, P, ΔP, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
            @time AssembleContinuity2D!(     M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ)
            @time AssembleFluidContinuity2D!(M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ)
            reduce_sparse_matrix!(M, M_PC_threads)
            
            # Assemble preconditionner
            @info "Assemble PC, ndof  = $(nVx + nVy + nPt + nPf)"
            M_PC_threads = reset_parallel_storage(number)
            @time AssembleMomentum2D_x!(     M_PC_threads, V, P, ΔP, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
            @time AssembleMomentum2D_y!(     M_PC_threads, V, P, ΔP, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
            @time AssembleContinuity2D!(     M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
            @time AssembleFluidContinuity2D!(M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
            reduce_sparse_matrix!(M_PC, M_PC_threads)

            #--------------------------------------------#
            @info "Solver"
            # Prepare work space (symbolic factorization)
            if iter==1 && it==1 && solver == :GCR
                solver_cache = KSP_GCR_TwoPhases_setup( M_PC; restart=GCR_restart, maxit=GCR_maxit)
            end

            # Sparse-direct-iterative solver
            two_phases_mechanical_solver!(dx, M, r, M_PC;
                solver=solver, solver_cache=solver_cache,
                ηb=1e5, ϵ_l=1e-9, niter_l=10, restart=20, noisy=true )

            #--------------------------------------------#
            imin = LineSearch!(rvec, α, dx, R, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, old, rheo, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, P, α[imin]*dx, number, type, nc)
        end
    
        #--------------------------------------------#

        # Include plasticity corrections
        P.t .= P.t .+ ΔP.t
        P.f .= P.f .+ ΔP.f
        εp  .+= ε̇.II*Δ.t
        
        k_ηΦ_x = materials.k_ηf0[1] .* ((Φ.c[2:end,:] .+ Φ.c[1:end-1,:]) / 2).^ materials.n_CK[1]
        k_ηΦ_y = materials.k_ηf0[1] .* ((Φ.c[:,2:end] .+ Φ.c[:,1:end-1]) / 2).^ materials.n_CK[1]

        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
        Vs   = (x=Vxsc, y=Vysc )
        Vs_mag   = sqrt.( Vxsc.^2 .+ Vysc.^2)
        Vxf  = -k_ηΦ_x .* diff(P.f, dims=1)/Δ.x
        Vyf  = -k_ηΦ_y .* diff(P.f, dims=2)/Δ.y
        Vxfc = 0.5*(Vxf[1:end-1,2:end-1] .+ Vxf[2:end,2:end-1])
        Vyfc = 0.5*(Vyf[2:end-1,1:end-1] .+ Vyf[2:end-1,2:end])
        Vf   = (x=Vxfc, y=Vyfc )
        Vf_mag   = sqrt.( Vxfc.^2 .+ Vyfc.^2)

        dΦdt = (Φ.c .- Φ0.c) / Δ.t

        #--------------------------------------------#
        probes.Pti[it]   = mean(P.t[phases.c.==2])
        probes.Pfi[it]   = mean(P.f[phases.c.==2])
        probes.Pei[it]   = mean(P.t[phases.c.==2] .- P.f[phases.c.==2])
        probes.ΔPt[it]   = maximum(P.t) - minimum(P.t)
        probes.ΔPf[it]   = maximum(P.f) - minimum(P.f)
        probes.ΔPe[it]   = maximum(P.t .- P.f) - minimum(P.t .- P.f) 
        probes.normτ[it]  = norm(τ.II[inx_c,iny_c])
        probes.normPe[it] = norm(P.t[inx_c,iny_c] .- P.f[inx_c,iny_c])
        probes.normPt[it] = norm(P.t[inx_c,iny_c])
        probes.normPf[it] = norm(P.f[inx_c,iny_c])
        probes.meanτ[it]  = mean(τ.II[inx_c,iny_c])
        probes.meanPe[it] = mean(P.t[inx_c,iny_c] .- P.f[inx_c,iny_c])
        probes.meanPt[it] = mean(P.t[inx_c,iny_c])
        probes.meanPf[it] = mean(P.f[inx_c,iny_c])
        probes.t[it]     = it*Δ.t
        # probes.maxPt[it] = maximum(P.t.-mean(P.t[inx_c,iny_c]) )
        # probes.maxPf[it] = maximum(P.f.-mean(P.f[inx_c,iny_c]) )
        probes.maxPt[it] = (P.t .- 0*mean(P.t[inx_c,iny_c]))[ix, iy_mid]
        probes.maxPf[it] = (P.f .- 0*mean(P.f[inx_c,iny_c]))[ix, iy_mid]
        probes.maxτ[it]  = τ.II[ix,iy]

        @show mean(P.t[phases.c.==2])
        @show mean(P.f[phases.c.==2])

        #-------------------------------------------# 

        # Visualise
        function figure()

            xc = X.c.x
            yc = X.c.y
            cmap = :jet1
            st  = 15
            ind = st:st:size(xc,1)-st

            fig = Figure(fontsize = 14, size = (675, 600) ) 

            ax1 = Axis(fig[3,1],  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$V^\text{s}$"
            hmVs = heatmap!(ax1, xc, yc, Vs_mag, colormap=cmap, colorrange=(0,0.75)) 
            arrows2d!(ax1, xc[ind], yc[ind], Vs.x[ind,ind], Vs.y[ind,ind], lengthscale = 1e-1, color = :white)

            ax2 = Axis(fig[3,2], xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$V^\text{f} \times 1000$"
            hmVf = heatmap!(ax2, xc, yc, Vf_mag*1000, colormap=cmap, colorrange=(0,0.2)) 
            arrows2d!(ax2, xc[ind], yc[ind], Vf.x[ind,ind], Vf.y[ind,ind], lengthscale = 500, color = :white)
            # arrowsize = V.arrow, lengthscale = V.scale)

            ax2 = Axis(fig[3,3], xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$V^\text{f} \times 1000$"
            hmτ = heatmap!(ax2, xc, yc, τ.II[inx_c,iny_c], colormap=cmap, colorrange=(0,3)) 
            # arrows2d!(ax2, xc[ind], yc[ind], σ1.x[ind,ind], σ1.y[ind,ind], lengthscale = 7e-2, color = :white, tipwidth = 0)

            ax1 = Axis(fig[2,1],  xlabel=L"$x$ [-]",  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$P^\text{t}$"
            hm1=heatmap!(ax1, xc, yc, P.t[inx_c,iny_c].-mean(P.t[inx_c,iny_c]), colormap=cmap, colorrange=(-3,3))
            # hm1=heatmap!(ax1, xc, yc, Vs.x, colormap=cmap) 

            ax2 = Axis(fig[2,2],  xlabel=L"$x$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) # , title=L"$P^\text{f}$"
            hm2=heatmap!(ax2, xc, yc, P.f[inx_c,iny_c].-mean(P.f[inx_c,iny_c]), colormap=cmap, colorrange=(-3,3))
            
            ax3 = Axis(fig[2,3],  xlabel=L"$x$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) # , title=L"$\dot{\phi}$"
            hm3=heatmap!(ax3, xc, yc, dΦdt[inx_c,iny_c]*100, colormap=cmap, colorrange=(-10.e-1, 10.e-1)) 

            # contour!( ax3, xc, yc, Pe[inx_c,iny_c], levels=[0.1], color=:white)
            
            Colorbar(fig[4,   1], hmVs, label = L"D) $|V^\text{s}|$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = false )
            Colorbar(fig[4,   2], hmVf, label = L"E) $|Q^\text{f}| \times 1000$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = false )
            Colorbar(fig[4,   3], hmτ,  label = L"F) $\tau_{II}$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = false )

            Colorbar(fig[1, 1], hm1, label = L"A) $P^\text{t}$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = true )
            Colorbar(fig[1, 2], hm2, label = L"B) $P^\text{f}$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = true )
            Colorbar(fig[1, 3], hm3, label = L"C) $\dot{\phi} \times 100$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = true )

            display(fig)

            # save("./figures/benchmark_v2.png", f, px_per_unit=4)

            # save("./examples/_TwoPhases/TwoPhasesPressure/PoroviscousReference.jld2", "Ωl", Ωl, "Ωη", Ωη,"x", (c=xc, v=xv), "y", (c=yc, v=yv), "P", P, "dΦdt", dΦdt, "Φ", Φ, "τ", τ, "Vs", (x=Vxsc, y=Vysc), "Vf", (x=Vxfc, y=Vyfc))

            fig = Figure(fontsize = 14, size = (600, 600) )  
            ax = Axis(fig[1,1], xlabelsize=20, ylabelsize=20, title=L"$\text{max} P^t, P^f, \tau_\text{II}$", xlabel = L"$t$ [-]", ylabel = L"$P, \tau$ [-]")
            lines!(ax,  probes.t[1:it], probes.maxPt[1:it], label=L"$$P^t")
            lines!(ax,  probes.t[1:it], probes.maxPf[1:it], label=L"$$P^f")
            lines!(ax,  probes.t[1:it], probes.maxτ[1:it],  label=L"$$\tau_\text{II}")

            if viscoelastic
                # Values at specific locations
                ΔPt_viscous = 1.53
                ΔPf_viscous = 1.44
                τ_viscous   = 2.37
                lines!(ax,  probes.t[1:it], ΔPt_viscous * ones(it)[1:it], label=L"$P^t$ -- V")
                lines!(ax,  probes.t[1:it], ΔPf_viscous * ones(it)[1:it], label=L"$P^f$ -- V")
                lines!(ax,  probes.t[1:it], τ_viscous  * ones(it)[1:it], label=L"$\tau_\text{II}$ -- V")
            end

            axislegend(framevisible = false, position=:rb, nbanks = 2)
            display(fig) 
        end
        with_theme(figure, theme_latexfonts())

        #-------------------------------------------# 

        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_omega_l$(Ωl)_step$(@sprintf("%04d", it)).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )

    end

    #--------------------------------------------#

    @show ix, iy
    @show (P.t .- mean(P.t[inx_c,iny_c]))[ix, iy_mid]
    @show (P.f .- mean(P.f[inx_c,iny_c]))[ix, iy_mid]
    @show τ.II[ix, iy]
    @show Δt0

    # if viscoelastic 
        save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_conservtative.jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_Kphi$(materials.KΦ[1]).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_Kphi$(materials.KΦ[1])_etaphi$(materials.ξ0[1]).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_G$(materials.G[1]).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_Kf$(materials.Kf[1]).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_etaphi$(materials.ξ0[1]).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_etas$(materials.η0[1]).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_ebg$(ε̇bg).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )
        # save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_syst_omega_l$(Ωl)_etasinc$(ηs_inc).jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ )

    # else
    #     save("./examples/_TwoPhases/TwoPhasesPressure/ReferenceModel.jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "X", X, "P", P, "phases", phases, "τ", τ)
    # end

    return 
end

function Run()

    nc = (x=200, y=200) # paper figure
    nc = (x=50, y=50) 

    # Mode 0   
    Ωη = 10^(2)
    Ωl = 0.15
    # Ωl = .045
    # Ωl = 2.0   
    # Ωl = 1.5
    # Ωl = 1.0

    # Ωl = 1.5e-1 # with kphi*3 
    # Ωl = 1.0e-0 # with kphi*3, kphi_3, G*3 
    # Ωl = 1.5e-0 # with kphi*3 
    # Ωl = .55 # with kphi*3 
   
    # main(nc, Ωl, Ωη, false);
    main(nc, Ωl, Ωη, true);

    # nc = (x=50, y=50)
    # main(nc, Ωl, Ωη, true);
end

Run()
