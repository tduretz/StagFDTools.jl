#---------------------------------------------------------------------------------------
# Crustal-scale shear banding with a visco-elasto-viscoplastic (V-E-VP) rheology, computed
# with the two-phase code in its single-phase limit.
# Duretz, de Borst, Yamato & Le Pourhiet (2020), GRL, https://doi.org/10.1029/2019GL086027
# Single-phase reference: examples/_Stokes_VEP_SSG/ShearBanding_Duretz2020/Example_VEVP_Duretz2020.jl
#
# Single-phase limit: single_phase = true (Pe = Pt, porosity frozen), ρs = ρf (body force does not
# depend on porosity), Ks → ∞ (incompressible: Pt pinned in the top centroid on the symmetry axis).
# The fluid pressure is still solved for but decoupled from the mechanics.
# The setup is perfectly symmetric and symmetry is never enforced.
#---------------------------------------------------------------------------------------
using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs
import Statistics:mean

# Copy a saved state (nested NamedTuples of arrays) into the preallocated fields
function restore_state!(dst, src)
    for k in keys(src)
        if src[k] isa NamedTuple
            restore_state!(dst[k], src[k])
        else
            n = min(length(dst[k]), length(src[k])) # probes: nt may differ between runs
            copyto!(dst[k], 1, src[k], 1, n)
        end
    end
end

# Relative mirror asymmetry x -> -x (odd: field changes sign under the mirror, e.g. Vx or τxy)
asymmetry(A; odd=false) = maximum(abs.(A .- (odd ? -1 : 1) .* reverse(A, dims=1))) / maximum(abs, A)

# Default physics: single-phase limit. Any subset can be overridden through the physics keyword to
# switch two-phase ingredients on one at a time (e.g. physics=(single_phase=false, ψ=5.0))
const single_phase_limit = (single_phase=true, compressible=false, ξ0=1e50, Ks=1e50, KΦ0=1e50, m=0.0, ρf=2700.0, ψ=0.0)

# restart: step to resume from (0: start from scratch); a checkpoint is written every nsave steps
# tag: label added to the checkpoint and figure file names
@views function main(nc; mode=:compression, nt=100, Δt=4e11, ηvp=1e21, niter=15, ϵ_nl=5e-10, restart=0,
    physics=(;), tag="", checkpoint_dir="./examples/_TwoPhases/TwoPhasesPlasticity/results", save_dir=nothing)

    phys = merge(single_phase_limit, physics)
    @assert isodd(nc.x) "nc.x must be odd: the pressure is pinned in the top centroid lying on the symmetry axis"
    save_dir === nothing || mkpath(save_dir)

    sc = (σ=1e7, t=1e10, L=1e3)
    nsave = 10

    # Linear solver
    solver      = :LU
    GCR_restart = 25
    GCR_maxit   = 100
    ϵ_l         = 1e-11
    Pic2Newt    = 1.8   # more than 1.0 - always Newton

    # Time steps
    Δt0     = Δt/sc.t

    # Boundary loading: pure shear, Vx = -ε̇ x, Vy = ε̇ y in compression (opposite in extension)
    ε̇bg  = (mode == :compression ? 1.0 : -1.0) * 1e-15 * sc.t
    D_BC = @SMatrix([ -ε̇bg 0.0;
                       0.0  ε̇bg ])

    # Materials: 1 = Westerly granite (Hansen & Carter, 1983), 2 = weak seed
    # Creep law ε̇II = A exp(-Q/(RT)) (τII/2)^n  <=>  η = η0 exp(Q/(nR) (1/T - 1/Tref)) ε̇II^(1/n-1)
    nphases = 2
    A    = [3.1623e-26, 1e-20]  # [Pa^-n s^-1]
    n    = [3.3, 1.0]
    Q    = [186.5e3, 0.0]       # [J/mol]
    Tref = 773.15               # [K]
    η0   = @. A^(-1/n) * exp(Q / (n * 8.314462618 * Tref))
    ϕ    = atand(0.6)
    C    = [5e7, 1e4]           # [Pa], yield τII = C + P sinϕ as in the M2Di code (cohesion term without cosϕ)

    materials = initialize_materials_TwoPhases(nphases,
        oneway       = false,
        compressible = phys.compressible,
        linearizeΦ   = false,
        single_phase = phys.single_phase,
        conservative = false,
        plasticity   = DruckerPrager,
        g            = [0., -10.0] / (sc.L/sc.t^2),
    )

    materials.n     .= n
    materials.m     .= phys.m
    materials.n_CK  .= [  0.0,    0.0 ]
    materials.η0    .= @. η0 / (sc.σ * sc.t^(1/n))  # creep prefactor at Tref
    materials.Ea    .= Q
    materials.Tref  .= Tref
    materials.ξ0    .= phys.ξ0/sc.σ/sc.t
    materials.G     .= [ 1e10,   1e10 ]./sc.σ
    materials.ρs    .= [ 2700,   2700 ]/(sc.σ*sc.t^2/sc.L^2)
    materials.ρf    .= phys.ρf/(sc.σ*sc.t^2/sc.L^2)   # = ρs: ρt does not depend on porosity
    materials.Ks    .= phys.Ks./sc.σ                  # → ∞: incompressible
    materials.KΦ0   .= phys.KΦ0./sc.σ
    materials.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials.k_ηf0 .= [1e-15,  1e-15 ]./(sc.L^2/sc.σ/sc.t)
    materials.Φ0    .= [ 1e-2,   1e-2 ]
    materials.plasticity.ϕ   .= [ ϕ,       0.0 ]
    materials.plasticity.ψ   .= [ phys.ψ,  0.0 ]
    materials.plasticity.C   .= C ./ cosd(ϕ) ./ sc.σ
    materials.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t
    preprocess!(materials)

    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Define node types and set BC flags: free slip on all sides,
    # pressure fixed in the top centroid on the symmetry axis if incompressible (P defined up to a constant)
    type =  set_BC_template(:free_slip, nc)
    ipin, jpin = (nc.x + 1) ÷ 2 + 1, nc.y + 1
    phys.compressible || (type.Pt[ipin, jpin] = :constant)

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
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                         @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),          @SMatrix([1 1 1;  1 1 1]),        @SMatrix([1 1 1;  1 1 1])),
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),          @SMatrix([1 1 1; 1 1 1; 1 1 1]),                         @SMatrix([1 1; 1 1; 1 1]),        @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                               @SMatrix([0 0; 1 1; 0 0]),                               @SMatrix([1]),                    @SMatrix([1])),
        Fields(@SMatrix([1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1]),  @SMatrix([1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1]), @SMatrix([1 1 1; 1 1 1; 1 1 1]),  @SMatrix([1 1 1; 1 1 1; 1 1 1])),
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    nPf   = maximum(number.Pf)
    M     = Fields(
        Fields(spzeros(nVx, nVx), spzeros(nVx, nVy), spzeros(nVx, nPt), spzeros(nVx, nPf)),
        Fields(spzeros(nVy, nVx), spzeros(nVy, nVy), spzeros(nVy, nPt), spzeros(nVy, nPf)),
        Fields(spzeros(nPt, nVx), spzeros(nPt, nVy), spzeros(nPt, nPt), spzeros(nPt, nPf)),
        Fields(spzeros(nPf, nVx), spzeros(nPf, nVy), spzeros(nPf, nPt), spzeros(nPf, nPf)),
    )
    M_PC  = Fields(
        Fields(spzeros(nVx, nVx), spzeros(nVx, nVy), spzeros(nVx, nPt), spzeros(nVx, nPf)),
        Fields(spzeros(nVy, nVx), spzeros(nVy, nVy), spzeros(nVy, nPt), spzeros(nVy, nPf)),
        Fields(spzeros(nPt, nVx), spzeros(nPt, nVy), spzeros(nPt, nPt), spzeros(nPt, nPf)),
        Fields(spzeros(nPf, nVx), spzeros(nPf, nVy), spzeros(nPf, nPt), spzeros(nPf, nPf)),
    )
    # Global arrays
    dx   = zeros(nVx + nVy + nPt + nPf)
    r    = zeros(nVx + nVy + nPt + nPf)
    solver_cache = 0

    #--------------------------------------------#
    # Intialise field
    L       = (x=100e3/sc.L, y=30e3/sc.L)
    rad     = 2e3/sc.L
    Δ       = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R       = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...), Φ=zeros(size_c...))
    V       = (x=zeros(size_x...), y=zeros(size_y...))
    Vi      = (x=zeros(size_x...), y=zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ       = (c=materials.Φ0[1].*ones(size_c...), v=materials.Φ0[1].*ones(size_v...) )
    Φ0      = (c=materials.Φ0[1].*ones(size_c...), v=materials.Φ0[1].*ones(size_v...) )
    εp      = zeros(size_c...)
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), f = zeros(size_c...) )
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    T       = (c  = zeros(size_c...), v  = zeros(size_v...) )   # temperature [K]

    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t = zeros(size_c...), f = zeros(size_c...))
    Pi      = (t = zeros(size_c...), f = zeros(size_c...))
    P0      = (t = zeros(size_c...), f = zeros(size_c...))
    ΔP      = (t = zeros(size_c...), f = zeros(size_c...))
    ρ       = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = materials.ρs[1]*ones(size_c...))
    ρ0      = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = materials.ρs[1]*ones(size_c...))
    ΔΦ      = zeros(size_c...)
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
    y = (min=-L.y, max=0.0)
    X = GenerateGrid(x, y, Δ, nc)

    # Geotherm: 20 °C at the surface, 15 °C/km (ghost nodes included)
    T.c .= [293.15 - 15e-3 * yc * sc.L for _ in X.c_e.x, yc in X.c_e.y]
    T.v .= [293.15 - 15e-3 * yv * sc.L for _ in X.v_e.x, yv in X.v_e.y]

    # Initial configuration: pure shear velocity
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y'
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'

    # Weak seed in the middle of the domain
    @views phases.c[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y' .+ L.y/2).^2) .< rad^2] .= 2
    @views phases.v[inx_v, iny_v][(X.v.x.^2 .+ (X.v.y' .+ L.y/2).^2) .< rad^2] .= 2
    phase_ratios = InitialisePhaseRatios(phases, nphases)

    # Initial pressure fields: lithostatic / hydrostatic (0 in the top centroid)
    ρt_ini = (1 - materials.Φ0[1]) * materials.ρs[1] + materials.Φ0[1] * materials.ρf[1]
    P.t .= [-ρt_ini         * materials.g[2] * (-yc - Δ.y/2) for _ in X.c_e.x, yc in X.c_e.y]
    P.f .= [-materials.ρf[1] * materials.g[2] * (-yc - Δ.y/2) for _ in X.c_e.x, yc in X.c_e.y]

    # Boundary condition values (free slip: zero tangential stress, no Darcy flux)
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))

    #--------------------------------------------#

    probes = (
        iters = zeros(Int, nt),
        asym  = zeros(nt),
    )

    err  = (x = zeros(niter), y = zeros(niter), pt = zeros(niter), pf = zeros(niter))
    to   = TimerOutput()

    solver_ready  = false
    pattern_ready = false
    M_fixed = M_PC_fixed = nothing

    # Checkpointing: all fields carried from one step to the next (the rest is recomputed every iteration)
    state = (; V, P, τ, Φ, ρ, εp, ε̇, λ̇, η, ΔP, ΔΦ, div_Vs, div_qD, probes)
    checkpoint_file(it) = joinpath(checkpoint_dir, @sprintf("checkpoint_Duretz2020%s_%s_%04d.jld2", tag, mode, it))
    if restart > 0
        restore_state!(state, load(checkpoint_file(restart), "state"))
        @info "Restarting after step $(restart) from $(checkpoint_file(restart))"
    end

    for it=restart+1:nt

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
        compute_grid_fields_two_phases!(G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK, materials, phase_ratios, nc, nphases)

        old  = τ0, P0, Φ0, ρ0
        rheo = G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK
        iter, ϵ0, ϵ = 0, 0.0, 0.0

        # Newton-Raphson iterations
        for outer iter=1:niter

            λ̇.c   .= 0.0
            λ̇.v   .= 0.0

            # Residual check
            @timeit to "Tangent operator" begin
                TangentOperator!( 𝐷, 𝐷_ctl, τ, ε̇, λ̇, η, V, P, ΔP, Φ, ΔΦ, ρ, old, div_Vs, div_qD, type, BC, materials, phases, rheo, T, Δ)
            end
            @timeit to "Residual" begin
                ResidualMomentum2D_x!(     R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(     R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
                ResidualContinuity2D!(     R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
                ResidualFluidContinuity2D!(R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
            end
            err.x[iter]  = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter]  = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.pt[iter] = @views norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            err.pf[iter] = @views norm(R.pf[inx_c,iny_c])/sqrt(nPf)
            ϵ = max(err.x[iter], err.y[iter], err.pt[iter], err.pf[iter])
            (iter == 1) && (ϵ0 = ϵ)
            @printf("    iter %02d: abs. res. = %1.3e --- rel. res. = %1.3e (x %1.1e, y %1.1e, pt %1.1e, pf %1.1e)\n",
                iter, ϵ, ϵ/ϵ0, err.x[iter], err.y[iter], err.pt[iter], err.pf[iter])
            (ϵ/ϵ0 < ϵ_nl || ϵ < ϵ_nl) && break

            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            @timeit to "Assembly" begin
                # Record the sparsity pattern once (first assembly), then write straight into the CSC nonzeros
                if !pattern_ready
                    @timeit to "Pattern" begin
                        K_pat = pattern_storage()
                        AssembleMomentum2D_x!(     K_pat, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                        AssembleMomentum2D_y!(     K_pat, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                        AssembleContinuity2D!(     K_pat, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                        AssembleFluidContinuity2D!(K_pat, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                        set_sparsity_pattern!(M, K_pat)
                        K_pat = pattern_storage()
                        AssembleMomentum2D_x!(     K_pat, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
                        AssembleMomentum2D_y!(     K_pat, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
                        AssembleContinuity2D!(     K_pat, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
                        AssembleFluidContinuity2D!(K_pat, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
                        set_sparsity_pattern!(M_PC, K_pat)
                        M_fixed, M_PC_fixed = fixed_pattern(M), fixed_pattern(M_PC)
                        pattern_ready = true
                    end
                end
                # Assemble global Jacobian
                zero_values!(M)
                AssembleMomentum2D_x!(     M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(     M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleContinuity2D!(     M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleFluidContinuity2D!(M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
                # Assemble preconditionner
                zero_values!(M_PC)
                AssembleMomentum2D_x!(     M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(     M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
                AssembleContinuity2D!(     M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
                AssembleFluidContinuity2D!(M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
            end

            Newton = (ϵ/ϵ0 < Pic2Newt) ? true : false

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
                αmax   = Newton ? 1.0 : 3.0
                α_best, R_trial, success = BackTrackingLineSearch!(R, dx, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, ΔΦ, ρ, div_Vs, div_qD, old, rheo, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, T, nc, Δ, α0=αmax )
                UpdateSolution!(V, P, α_best*dx, number, type, nc)
            end

        end
        (ϵ/ϵ0 < ϵ_nl || ϵ < ϵ_nl) || error(@sprintf("Step %d: Newton did not converge in %d iterations (rel. res. = %1.3e)", it, niter, ϵ/ϵ0))

        #--------------------------------------------#

        # Include plasticity corrections
        P.t  .= P.t .+ ΔP.t
        P.f  .= P.f .+ ΔP.f
        εp  .+= ε̇.II*Δ.t

        probes.iters[it] = iter
        probes.asym[it]  = max(asymmetry(V.x[inx_Vx, iny_Vx]; odd=true), asymmetry(P.t[inx_c, iny_c]), asymmetry(ε̇.II[inx_c, iny_c]))
        @printf("Step %03d (%s): %02d iterations, max λ̇ = %1.2e s⁻¹, mirror asymmetry (Vx, P, ε̇II) = %1.2e\n",
            it, mode, iter, maximum(λ̇.c) / sc.t, probes.asym[it])

        if mod(it, nsave)==0
            jldsave(checkpoint_file(it); it, state)
        end

        #--------------------------------------------#
        # Visualise
        xc, yc   = X.c.x * sc.L / 1e3, X.c.y * sc.L / 1e3
        ε̇II      = ε̇.II[inx_c, iny_c] / sc.t
        j10      = argmin(abs.(yc .+ 10))                         # profile at y = -10 km
        rock     = phases.c[inx_c, iny_c] .== 1
        depth    = repeat(yc', nc.x)[rock]
        Plith    = @. 2700 * 10 * (-yc * 1e3 - Δ.y * sc.L / 2)    # lithostatic pressure (top centroid pinned to 0)
        sinϕ     = sind(ϕ)
        τ_DP     = mode == :compression ? @.((C[1] + Plith * sinϕ) / (1 - sinϕ)) : @.((C[1] + Plith * sinϕ) / (1 + sinϕ))
        P_DP     = mode == :compression ? Plith .+ τ_DP : Plith .- τ_DP

        fig = Figure(size=(1200, 900), fontsize=16)
        ax  = Axis(fig[1, 1:3], aspect=DataAspect(), title="Two-phase code, $(isempty(tag) ? "single-phase limit" : tag): $(mode), $(nc.x)×$(nc.y) cells, t = $(@sprintf("%1.2f", it * Δt / (1e6 * 365.25 * 24 * 3600))) My", xlabel="x [km]", ylabel="y [km]")
        hm  = heatmap!(ax, xc, yc, log10.(ε̇II), colormap=:roma, colorrange=(-16, -14))
        hlines!(ax, yc[j10], color=:white, linestyle=:dash)
        Colorbar(fig[1, 4], hm, label=L"\log_{10} \dot{\varepsilon}_{II}")

        ax  = Axis(fig[2, 1:3], title="Profile at y = $(@sprintf("%1.1f", yc[j10])) km (dashed: mirrored)", xlabel="x [km]", ylabel=L"\log_{10} \dot{\varepsilon}_{II}")
        lines!(ax, xc, log10.(ε̇II[:, j10]))
        lines!(ax, xc, log10.(reverse(ε̇II[:, j10])), linestyle=:dash)

        g   = fig[3, 1:4] = GridLayout()
        ax  = Axis(g[1, 1], title="Newton iterations", xlabel="Time step", ylabel="Iterations")
        scatter!(ax, 1:it, probes.iters[1:it])
        ax  = Axis(g[1, 2], title=L"\tau_{II}", xlabel=L"\tau_{II}\ \mathrm{[MPa]}", ylabel="y [km]")
        scatter!(ax, τ.II[inx_c, iny_c][rock] * sc.σ / 1e6, depth, markersize=2, color=:black)
        lines!(ax, τ_DP / 1e6, yc, color=:green)
        ax  = Axis(g[1, 3], title=L"P", xlabel=L"P\ \mathrm{[MPa]}", ylabel="y [km]")
        scatter!(ax, P.t[inx_c, iny_c][rock] * sc.σ / 1e6, depth, markersize=2, color=:black)
        lines!(ax, P_DP / 1e6, yc, color=:green)
        lines!(ax, Plith / 1e6, yc, color=:gray, linestyle=:dash)
        display(fig)
        save_dir === nothing || save(joinpath(save_dir, @sprintf("Duretz2020_TwoPhases%s_%s_%dx%d_step%04d.png", tag, mode, nc.x, nc.y, it)), fig)
    end

    # Animation of the png sequence (ffmpeg with a generated palette for clean colours)
    if save_dir !== nothing
        frames = joinpath(save_dir, "Duretz2020_TwoPhases$(tag)_$(mode)_$(nc.x)x$(nc.y)_step%04d.png")
        gif    = joinpath(save_dir, "Duretz2020_TwoPhases$(tag)_$(mode)_$(nc.x)x$(nc.y).gif")
        run(`ffmpeg -y -loglevel error -framerate 10 -i $frames -vf "scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" $gif`)
    end

    display(to)
    return (; probes.iters, probes.asym, ε̇II=ε̇.II[inx_c, iny_c] / sc.t, xc=X.c.x * sc.L / 1e3, yc=X.c.y * sc.L / 1e3)
end

let
    nc = (x=201, y=61) # Figure 2a resolution (nc.x odd)
    main(nc; mode=:compression)
    main(nc; mode=:extension)
end
