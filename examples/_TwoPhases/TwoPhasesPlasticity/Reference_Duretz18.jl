using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs, MAT
import Statistics:mean

@views function main_Duretz18(D_BC, nc, nt, n_nt; homo=false, niter=20, Φini=5e-2, ηvp=0.0, r_fact=1.0, ε̇_fact=1.0, visualization=true, n_CK=0.0)

    # Load data
    filepath = joinpath(@__DIR__, "DataM2Di_EP_test01.mat")
    data = matread(filepath)
    @show keys(data)

   sc = (σ =1e7, L = 1e3, t = 1e10)

    # Load data
    filepath = joinpath(@__DIR__, "DataM2Di_EP_test01.mat")
    data = matread(filepath)
    @show keys(data)

    homo   = false

    # Time steps
    nt     = 30
    Δt0    = 1e10/sc.t 

    # Linear solver
    solver       = :GCR
    GCR_restart  = 25
    GCR_maxit    = 100
    ϵ_l          = 1e-11
    Pic2Newt     = 1.8   # more than 1.0 - always Newton
    solver_ready = false

    # Newton solver
    ϵ_nl  = 1e-9
    α     = LinRange(0.05, 1.0, 5)

    rad     = 1e2/sc.L 
    Pt_ini  = 0*1e8/sc.σ
    Pf_ini  = 0*1e6/sc.σ
    ε̇bg     = -5e-15.*sc.t
    τ_ini   = 0*(sind(35)*(Pt_ini-Pf_ini) + 0*1e7/sc.σ*cosd(35))  

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇bg 0; 0 -ε̇bg] )

    τxx_ini = τ_ini*D_BC[1,1]/abs(ε̇bg)
    τyy_ini = τ_ini*D_BC[2,2]/abs(ε̇bg)

    # Material parameters
    nphases = 2
    materials = initialize_materials_TwoPhases(nphases,
        oneway       = false,
        compressible = true,
        linearizeΦ   = false, 
        single_phase = true,
        conservative = false,
        
        # plasticity   = Tensile,
        plasticity   = DruckerPrager,
        # plasticity   = DruckerHyperbolic,
        # plasticity   = DruckerPragerCap,
        # plasticity   = Golchin2021,
    )

    materials.n     .= [  1.0,    1.0 ]
    materials.m     .= [  0.0,    0.0 ]
    materials.n_CK  .= [  0.0,    0.0 ]
    materials.η0    .= [ 1e32,   1e32 ]/sc.σ/sc.t 
    materials.ξ0    .= [ 2e32,   2e32 ]/sc.σ/sc.t
    materials.G     .= [1e10,   0.25e10]./sc.σ 
    materials.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials.Ks    .= [ 2e10,   2e10 ]./sc.σ
    materials.KΦ    .= [  5e9,    5e9 ]./sc.σ
    materials.Kf    .= [  2e9,    2e9 ]./sc.σ 
    materials.k_ηf0 .= [1e-15,  1e-15 ]./(sc.L^2/sc.σ/sc.t)
    materials.plasticity.ϕ   .= [ 30.,     30. ] * 1
    materials.plasticity.ψ   .= [ 10.,     10. ] * 1
    materials.plasticity.C   .= [ 3e7,     3e7 ]./sc.σ
    materials.plasticity.ηvp .= [ 0.0,     0.0 ]./sc.σ/sc.t 
    # materials.plasticity.Pt  .= [-1.e6,    -1e6 ]./sc.σ 
    # materials.plasticity.Pt  .= [-1.e6,    -1e6 ]./sc.σ 
    # materials.plasticity.Pc  .= [ 1e8,     1e8 ]./sc.σ
    # materials.plasticity.a   .= [ 0.8,     0.8 ]
    # materials.plasticity.b   .= [ 0.0,     0.0 ]
    # materials.plasticity.c   .= [ 0.8,     0.8 ]

    preprocess!(materials)

    Φ0      = Φini
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
    type.Vx[inx_Vx,2]       .= :Neumann_tangent
    type.Vx[inx_Vx,end-1]   .= :Neumann_tangent
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy]  .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann_tangent
    type.Vy[end-1,iny_Vy]   .= :Neumann_tangent
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    type.Pt[1,:]             .= :Neumann 
    type.Pt[end,:]           .= :Neumann 
    type.Pt[:,1]             .= :Neumann
    type.Pt[:,end]           .= :Neumann
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Neumann 
    type.Pf[end,:]           .= :Neumann 
    type.Pf[:,1]             .= :Neumann
    type.Pf[:,end]           .= :Neumann
    
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
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                   @SMatrix([1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                   @SMatrix([1 1 1; 1 1 1; 1 1 1])),
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
    L       = (x=4e3/sc.L, y=2e3/sc.L)
    Δ       = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R       = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...), Φ=zeros(size_c...))
    V       = (x=zeros(size_x...), y=zeros(size_y...))
    Vi      = (x=zeros(size_x...), y=zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ       = (c=Φ_ini.*ones(size_c...), v=Φ_ini.*ones(size_v...) )
    Φ0      = (c=Φ_ini.*ones(size_c...), v=Φ_ini.*ones(size_v...) )
    εp      = zeros(size_c...)
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = τxx_ini.*ones(size_c...), yy = τyy_ini.*ones(size_c...), xy = zeros(size_v...) )
    τ       = (xx = τxx_ini.*ones(size_c...), yy = τyy_ini.*ones(size_c...), xy = zeros(size_v...), II = zeros(size_c...), f = zeros(size_c...) )
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )

    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t = Pt_ini.*ones(size_c...), f = Pf_ini.*ones(size_c...))
    Pi      = (t = Pt_ini.*ones(size_c...), f = Pf_ini.*ones(size_c...))
    P0      = (t = zeros(size_c...), f = zeros(size_c...))
    ΔP      = (t = zeros(size_c...), f = zeros(size_c...))
    ρ       = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))
    ρ0      = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))
    div_qD  = (c  = zeros(size_c...), v  = zeros(size_v...) )
    div_Vs  = (c  = zeros(size_c...), v  = zeros(size_v...) )

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

    # Generate grid coordinates 
    x = (min=-L.x/2, max=L.x/2)
    y = (min=-L.y/2, max=L.y/2)
    X = GenerateGrid(x, y, Δ, nc)

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

    rvec   = zeros(length(α))
    probes = (
        Pe  = zeros(nt),
        Pt  = zeros(nt),
        Pf  = zeros(nt),
        τ   = zeros(nt),
        Φ   = zeros(nt),
        λ̇   = zeros(nt),
        t   = zeros(nt),
        τII = zeros(nt),
        str = zeros(nt),
    )

    err  = (x = zeros(niter), y = zeros(niter), pt = zeros(niter), pf = zeros(niter))
    
    to   = TimerOutput()
    solver_ready = false

    for it=1:nt

        @printf("\nStep %04d\n", it)
        fill!( err.x, 0e0)
        fill!( err.y, 0e0)
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
        iter, ϵ0, ϵ = 0, 0.0, 0.0

        # Newton-Raphson iterations
        for iter=1:niter

            @printf("     Step %04d --- Iteration %04d\n", it, iter)

            λ̇.c   .= 0.0
            λ̇.v   .= 0.0

            # Residual check
            @timeit to "Tangent operator" begin
                TangentOperator!( 𝐷, 𝐷_ctl, τ, ε̇, λ̇, η, V, P, ΔP, Φ, ρ, old, div_Vs, div_qD, type, BC, materials, phases, rheo, Δ)
            end
            @timeit to "Residual" begin
                ResidualMomentum2D_x!(     R, V, P, ΔP, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(     R, V, P, ΔP, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
                ResidualContinuity2D!(     R, V, P, ΔP, old,    rheo, materials, number, type, BC, nc, Δ) 
                ResidualFluidContinuity2D!(R, V, P, ΔP, old,    rheo, materials, number, type, BC, nc, Δ) 
            end
            @info "Residuals"
            @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            @show norm(R.pt[inx_c,iny_c]) /sqrt(nPt)
            @show norm(R.pf[inx_c,iny_c]) /sqrt(nPf)
            err.x[iter]  = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter]  = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.pt[iter] = @views norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            err.pf[iter] = @views norm(R.pf[inx_c,iny_c])/sqrt(nPt)
            println("min/max F    - ",  extrema(τ.f[inx_c,iny_c]))
            println("min/max λ̇.c  - ",  extrema(λ̇.c[inx_c,iny_c]))
            println("min/max λ̇.v  - ",  extrema(λ̇.v[3:end-2,3:end-2]))
            println("min/max ΔP.t - ",  extrema(ΔP.t[inx_c,iny_c]))
            println("min/max ΔP.f - ",  extrema(ΔP.f[inx_c,iny_c]))
            ϵ = max(err.x[iter], err.y[iter], err.pt[iter], err.pf[iter])
            (iter == 1) && (ϵ0 = ϵ)
            if ϵ < ϵ_nl || ϵ/ϵ0 < ϵ_nl
                println("Converged")
                break 
            end

            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            @timeit to "Assembly" begin
                # Assemble global Jacobian
                @info "Assemble Jacobian, ndof  = $(nVx + nVy + nPt + nPf)"
                M_PC_threads = reset_parallel_storage(number)
                AssembleMomentum2D_x!(     M_PC_threads, V, P, ΔP, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(     M_PC_threads, V, P, ΔP, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleContinuity2D!(     M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleFluidContinuity2D!(M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ)
                @timeit to "Reduction" begin
                    reduce_sparse_matrix!(M, M_PC_threads)
                end
                # Assemble preconditionner
                @info "Assemble PC, ndof  = $(nVx + nVy + nPt + nPf)"
                M_PC_threads = reset_parallel_storage(number)
                AssembleMomentum2D_x!(     M_PC_threads, V, P, ΔP, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(     M_PC_threads, V, P, ΔP, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleContinuity2D!(     M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
                AssembleFluidContinuity2D!(M_PC_threads, V, P, ΔP, old,        rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
                @timeit to "Reduction" begin
                    reduce_sparse_matrix!(M_PC, M_PC_threads)
                end
            end

            Newton = (ϵ/ϵ0 < Pic2Newt) ? true : false 

            @info "Solver - Newton = $(Newton)"
            # Prepare work space (symbolic factorization)
            if !solver_ready && solver == :GCR
                solver_cache = KSP_GCR_TwoPhases_setup( M_PC; restart=GCR_restart, maxit=GCR_maxit)
                solver_ready = true
            end

            # Sparse-direct-iterative solver
            @timeit to "Linear solve" begin
                Newton && two_phases_mechanical_solver!(dx, M, r, M_PC;
                    solver=solver, solver_cache=solver_cache,
                    ηb=1e5, ϵ_l=ϵ_l, niter_l=10, restart=10, noisy=false )
                !Newton && two_phases_mechanical_solver!(dx, M_PC, r, M_PC;
                    solver=solver, solver_cache=solver_cache,
                    ηb=1e5, ϵ_l=ϵ_l, niter_l=10, restart=10, noisy=false )
            end

            #--------------------------------------------#
            @timeit to "Line search" begin
                # imin = LineSearch!(rvec, α, dx, R, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, old, rheo, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
                # UpdateSolution!(V, P, α[imin]*dx, number, type, nc)
                αmax   = Newton ? 1.0 : 3.0
                α_best, R_trial, success = BackTrackingLineSearch!(R, dx, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, ρ, div_Vs, div_qD, old, rheo, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ, α0=αmax )
                UpdateSolution!(V, P, α_best*dx, number, type, nc)
            end

        end

        # if norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx) > ϵ_nl || norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy) > ϵ_nl
        #     error("Global convergence failed !")
        # end 

        #--------------------------------------------#

        # Include plasticity corrections
        P.t  .= P.t .+ ΔP.t
        P.f  .= P.f .+ ΔP.f
        εp  .+= ε̇.II*Δ.t
        
    #     # τxyc = av2D(τ.xy)
    #     # ε̇xyc = av2D(ε̇.xy)

    #     # # # Post process 
    #     # # @time for i in eachindex(Φ.c)
    #     # #     KΦ     = materials.KΦ[phases.c[i]]
    #     # #     ηΦ     = materials.ξ0[phases.c[i]] 
    #     # #     sinψ   = materials.sinψ[phases.c[i]] 
    #     # #     dPtdt  = (P.t[i] - P0.t[i]) / Δ.t
    #     # #     dPfdt  = (P.f[i] - P0.f[i]) / Δ.t
    #     # #     dΦdt   = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (P.f[i] - P.t[i]) + λ̇.c[i]*sinψ
    #     # #     Φ.c[i] = Φ0.c[i] + dΦdt*Δ.t
    #     # # end

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
        probes.Pe[it]   = mean(P.t[inx_c,iny_c] .- P.f[inx_c,iny_c])*sc.σ
        probes.Pt[it]   = mean(P.t[inx_c,iny_c])*sc.σ
        probes.Pf[it]   = mean(P.f[inx_c,iny_c])*sc.σ
        probes.τ[it]    = mean(τ.II[inx_c,iny_c])*sc.σ
        probes.Φ[it]    = mean(Φ.c[inx_c,iny_c])
        probes.λ̇[it]    = mean(λ̇.c[inx_c,iny_c])/sc.t
        probes.t[it]    = it*Δ.t*sc.t
        probes.str[it]  = abs(ε̇bg)*it*Δ.t

        #-------------------------------------------# 

        @show extrema(P.t[inx_c,iny_c]*sc.σ), mean(P.t[inx_c,iny_c]*sc.σ)

        # fname = @sprintf("PoroVEP_%03d.jld2",  it)
        # save("./examples/_TwoPhases/TwoPhasesPlasticity/results/$(fname)", "X", X, "sc", sc, "probes", probes,
        # "λ̇", λ̇, "P", P, "τ", τ, "ε̇", ε̇, "V", V, "η", η, "Φ", Φ, "εp", εp, "niter", niter, "err", err ) 
      
        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 600) )    
            step = 10
            ftsz = 15
            eps  = 1e-10

            ax   = Axis(fig[1,1], aspect=DataAspect(), title=L"$$Strain", xlabel=L"x", ylabel=L"y")
            # field = log10.((λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            field = log10.(εp[inx_c,iny_c])
            hm = heatmap!(ax, X.c.x, X.c.y, field, colormap=:jet, colorrange=(-3, -2.3))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 1], hm, label = L"$\lambda$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # arrows2d!(ax, X.c.x[1:step:end], X.c.y[1:step:end], Vxsc[1:step:end,1:step:end], Vysc[1:step:end,1:step:end], lengthscale=10000.4, color=:white)

            ax    = Axis(fig[3,1], aspect=DataAspect(), title=L"$$Porosity", xlabel=L"x", ylabel=L"y")
            field = Φ.c[inx_c,iny_c]
            hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 1], hm, label = L"$\dot\lambda$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax    = Axis(fig[1,2], aspect=DataAspect(), title=L"$P^t$ [MPa]", xlabel=L"x", ylabel=L"y")
            field = (P.t)[inx_c,iny_c].*sc.σ./1e6 
            hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:jet, colorrange=(-6, 4))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 2], hm, label = L"$P^t$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # arrows2d!(ax, X.c.x[1:step:end], X.c.y[1:step:end], Vxsc[1:step:end,1:step:end], Vysc[1:step:end,1:step:end], lengthscale=10000.4, color=:white)

            #######################
            # ax    = Axis(fig[3,2], aspect=DataAspect(), title=L"$P^e - \tau$", xlabel=L"P^e", ylabel=L"\tau")
                 
            # (materials.single_phase) ? α1 = 0.0 : α1 = 1.0 
            # Pe    = (P.t .- α1*P.f)[inx_c,iny_c].*sc.σ

            # τII       = (τ.II)[inx_c,iny_c].*sc.σ
            # P_ax      = LinRange(-5e6, 5e6, 100)
            # τ_ax_rock = materials.C[1]*sc.σ*materials.cosϕ[1] .+ P_ax.*materials.sinϕ[1]
            # lines!(ax, P_ax/1e6, τ_ax_rock/1e6, color=:black)
            # scatter!(ax, Pe[:]/1e6, τII[:]/1e6, color=:black )
            # F_post = @. τ.II - materials.C[1]*materials.cosϕ[1] - (P.t .- α1*P.f)*materials.sinϕ[1]
            # maxF   =  maximum( F_post[inx_c,iny_c] )
            # @info maxF, maxF .*sc.σ /1e6
            # @show maximum(τ.f[inx_c,iny_c]),  maximum(τ.f[inx_c,iny_c]) .*sc.σ /1e6
            #######################

            # # Previous stress states
            # τxyc0 = av2D(τ0.xy)
            # τII0  = sqrt.( 0.5.*(τ0.xx[inx_c,iny_c].^2 + τ0.yy[inx_c,iny_c].^2 + (-τ0.xx[inx_c,iny_c]-τ0.yy[inx_c,iny_c]).^2) .+ τxyc0[inx_c,iny_c].^2 )
            # Pe    = (P0.t .- α1*P0.f)[inx_c,iny_c].*sc.σ
            # τII   = τII0.*sc.σ
            # scatter!(ax, Pe[:]/1e6, τII[:]/1e6, color=:gray )

            # ax    = Axis(fig[1,3], aspect=DataAspect(), title=L"$\tau_\text{II}$ [MPa]", xlabel=L"x", ylabel=L"y")
            # field = (τ.II)[inx_c,iny_c].*sc.σ./1e6
            # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[2, 3], hm, label = L"$\tau_\text{II}$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax  = Axis(fig[3,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error")
            scatter!(ax, 1:niter, log10.(err.x[1:niter]./err.x[1]) )
            scatter!(ax, 1:niter, log10.(err.y[1:niter]./err.x[1]) )
            scatter!(ax, 1:niter, log10.(err.pt[1:niter]./err.pt[1]) )
            scatter!(ax, 1:niter, log10.(err.pf[1:niter]./err.pf[1]) )
            ylims!(ax, -10, 1.1)

            ax  = Axis(fig[1,3], xlabel="Strain", ylabel="Mean pressure")
            lines!(  ax, data["strvec"][1:end], data["Pvec"][1:end] )
            scatter!(ax, probes.str[1:2:nt], probes.Pt[1:2:nt] )

            ax  = Axis(fig[3,3], xlabel="Strain", ylabel="Mean stress invariant")
            lines!(  ax, data["strvec"][1:end], data["Tiivec"][1:end] )
            scatter!(ax, probes.str[1:2:nt], probes.τ[1:2:nt] )

            # field = P.f.*sc.σ
            # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[4, 2], hm, label = L"$P^f$", height=20, width = 200, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            
            display(fig) 
        end
        
        visualization && with_theme(figure, theme_latexfonts())
        
        #-------------------------------------------# 

    end

    #--------------------------------------------#

    display(to)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DP_PS_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DPC_PS_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DPHyp_PS_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DP_tens_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DPC_tens_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DPHyp_tens_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DP_comp_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DPC_comp_v2.jld2", "probes", probes)
    # homo && save("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DPHyp_comp_v2.jld2", "probes", probes)
    return 
end

function Run()

    n_nx = 1
    n_nt = 1
    nc   = (x=n_nx*150, y=n_nx*100)
    nt   = Int64(30*n_nt)
    D_BC = @SMatrix([-1 0; 0 1] )
    main_Duretz18(D_BC, nc, nt, n_nt; ηvp=0*1e19, homo=false, n_CK=0.0, r_fact=1.0, ε̇_fact=2.5, Φini=1e-3, niter=30, visualization=true); #1e20
    
end

@time Run()