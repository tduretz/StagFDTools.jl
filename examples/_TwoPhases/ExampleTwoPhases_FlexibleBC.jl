#---------------------------------------------------------------------------------------
# Two-phase equivalent of examples/_Stokes_VEP_SSG/ExampleADStokes_FlexibleBC.jl:
# one visco-elastic step with a weak inclusion for each boundary-condition template of set_BC_template.
# The *_stress templates prescribe the normal stress σ_BC on S/N or W/E; the boundary strain rates and
# normal stresses reached are reported to check that the boundary conditions are imposed.
#---------------------------------------------------------------------------------------
using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean

@views function main(BC_template, D_template, σ_template=@SMatrix(zeros(2, 2)); niter=10, ϵ_nl=1e-10)

    # Resolution
    nc = (x = 20, y = 20)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template
    σ_BC   = σ_template

    # Linear solver
    solver   = :LU
    Pic2Newt = 1.8   # more than 1.0 - always Newton

    # Material parameters: 1 = matrix, 2 = weak inclusion (no plasticity: yield stress never reached)
    nphases = 2
    materials = initialize_materials_TwoPhases(nphases,
        oneway       = false,
        compressible = true,
        linearizeΦ   = false,
        single_phase = false,
        conservative = false,
        plasticity   = DruckerPrager,
        g            = [0., 0.],
    )
    materials.n     .= [  1.0,   1.0 ]
    materials.m     .= [  0.0,   0.0 ]
    materials.n_CK  .= [  0.0,   0.0 ]
    materials.η0    .= [  1e2,  1e-1 ]
    materials.ξ0    .= [  1e2,  1e-1 ]
    materials.G     .= [  1e1,   1e1 ]
    materials.ρs    .= [  1.0,   1.0 ]
    materials.ρf    .= [  1.0,   1.0 ]
    materials.Ks    .= [  1e3,   1e3 ]
    materials.KΦ0   .= [  1e3,   1e3 ]
    materials.Kf    .= [  1e3,   1e3 ]
    materials.k_ηf0 .= [  1.0,   1.0 ]
    materials.Φ0    .= [ 5e-2,  5e-2 ]
    materials.plasticity.ϕ   .= [ 30.0, 30.0 ]
    materials.plasticity.ψ   .= [  0.0,  0.0 ]
    materials.plasticity.C   .= [ 1e10, 1e10 ]
    materials.plasticity.ηvp .= [  0.0,  0.0 ]
    preprocess!(materials)

    # Time step
    Δt0 = 0.5

    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Define node types and set BC flags
    type = set_BC_template(config, nc)

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
    dx   = zeros(nVx + nVy + nPt + nPf)
    r    = zeros(nVx + nVy + nPt + nPf)

    #--------------------------------------------#
    # Intialise field
    L       = (x=1.0, y=1.0)
    Δ       = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R       = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...), Φ=zeros(size_c...))
    V       = (x=zeros(size_x...), y=zeros(size_y...))
    Vi      = (x=zeros(size_x...), y=zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ       = (c=materials.Φ0[1].*ones(size_c...), v=materials.Φ0[1].*ones(size_v...) )
    Φ0      = (c=materials.Φ0[1].*ones(size_c...), v=materials.Φ0[1].*ones(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), f = zeros(size_c...) )
    𝐷       = (c = [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)], v = [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)])
    𝐷_ctl   = (c = [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)], v = [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)])
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    T       = (c  = ones(size_c...),  v  = ones(size_v...) )   # temperature [K]: unused (no activation energy)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )
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
    y = (min=-L.y/2, max=L.y/2)
    X = GenerateGrid(x, y, Δ, nc)

    # Initial velocity field
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y'
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'
    SetPeriodicGhosts!(V, P, type, nc) # does nothing if not periodic

    # Set material geometry
    @views phases.c[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y').^2) .<= 0.1^2] .= 2
    @views phases.v[inx_v, iny_v][(X.v.x.^2 .+ (X.v.y').^2) .<= 0.1^2] .= 2
    phase_ratios = InitialisePhaseRatios(phases, nphases)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1] .+ (type.Vx[     1, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1] .+ (type.Vx[   end, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[1]  ) .+ (type.Vx[inx_Vx,     2] .== :tangential_stress) .* σ_BC[1,2]
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[end]) .+ (type.Vx[inx_Vx, end-1] .== :tangential_stress) .* σ_BC[1,2]
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2] .+ (type.Vy[inx_Vy,     1 ] .== :normal_stress) .* σ_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2] .+ (type.Vy[inx_Vy,   end ] .== :normal_stress) .* σ_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[1]   .+ D_BC[2,2]*X.v.y) .+ (type.Vy[    2, iny_Vy] .== :tangential_stress) .* σ_BC[2,1]
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[end] .+ D_BC[2,2]*X.v.y) .+ (type.Vy[end-1, iny_Vy] .== :tangential_stress) .* σ_BC[2,1]

    #--------------------------------------------#

    err = (x = zeros(niter), y = zeros(niter), pt = zeros(niter), pf = zeros(niter))
    compute_grid_fields_two_phases!(G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK, materials, phase_ratios, nc, nphases)
    old  = τ0, P0, Φ0, ρ0
    rheo = G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK
    pattern_ready = false
    M_fixed = M_PC_fixed = nothing
    iter, ϵ0, ϵ = 0, 0.0, 0.0

    # Newton-Raphson iterations (one time step)
    for outer iter=1:niter
        λ̇.c .= 0.0
        λ̇.v .= 0.0
        TangentOperator!( 𝐷, 𝐷_ctl, τ, ε̇, λ̇, η, V, P, ΔP, Φ, ΔΦ, ρ, old, div_Vs, div_qD, type, BC, materials, phases, rheo, T, Δ)
        ResidualMomentum2D_x!(     R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(     R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
        ResidualContinuity2D!(     R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
        ResidualFluidContinuity2D!(R, V, P, ΔP, ΔΦ, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)
        err.x[iter]  = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
        err.y[iter]  = norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
        err.pt[iter] = norm(R.pt[inx_c,iny_c])/sqrt(nPt)
        err.pf[iter] = norm(R.pf[inx_c,iny_c])/sqrt(nPf)
        ϵ = max(err.x[iter], err.y[iter], err.pt[iter], err.pf[iter])
        (iter == 1) && (ϵ0 = ϵ)
        @printf("    iter %02d: abs. res. = %1.3e --- rel. res. = %1.3e\n", iter, ϵ, ϵ/ϵ0)
        (ϵ/ϵ0 < ϵ_nl || ϵ < ϵ_nl) && break

        SetRHS!(r, R, number, type, nc)

        # Record the sparsity pattern once, then write straight into the CSC nonzeros
        if !pattern_ready
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
        zero_values!(M)
        AssembleMomentum2D_x!(     M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
        AssembleMomentum2D_y!(     M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
        AssembleContinuity2D!(     M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
        AssembleFluidContinuity2D!(M_fixed, V, P, ΔP, ΔΦ, old, 𝐷_ctl, rheo, materials, number, pattern, type, BC, nc, Δ)
        zero_values!(M_PC)
        AssembleMomentum2D_x!(     M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
        AssembleMomentum2D_y!(     M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ)
        AssembleContinuity2D!(     M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)
        AssembleFluidContinuity2D!(M_PC_fixed, V, P, ΔP, ΔΦ, old, 𝐷,     rheo, materials, number, pattern, type, BC, nc, Δ; PC=true)

        Newton = (ϵ/ϵ0 < Pic2Newt)
        two_phases_mechanical_solver!(dx, Newton ? M : M_PC, r, M_PC; solver=solver, solver_cache=0, ηb=1e5, ϵ_l=1e-11, niter_l=10, restart=10, noisy=false)

        α_best, _, _ = BackTrackingLineSearch!(R, dx, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, ΔΦ, ρ, div_Vs, div_qD, old, rheo, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, T, nc, Δ, α0=Newton ? 1.0 : 3.0)
        UpdateSolution!(V, P, α_best*dx, number, type, nc)
    end
    (ϵ/ϵ0 < ϵ_nl || ϵ < ϵ_nl) || error(@sprintf("Newton did not converge in %d iterations (rel. res. = %1.3e)", niter, ϵ/ϵ0))

    # Boundary strain rates of the solution (W/E: ∂Vx/∂x, S/N: ∂Vy/∂y), to compare with D_BC
    Vx, Vy = V.x[inx_Vx,iny_Vx], V.y[inx_Vy,iny_Vy]
    ε̇_bnd = (
        W = mean(Vx[2,:]   .- Vx[1,:])     / Δ.x,
        E = mean(Vx[end,:] .- Vx[end-1,:]) / Δ.x,
        S = mean(Vy[:,2]   .- Vy[:,1])     / Δ.y,
        N = mean(Vy[:,end] .- Vy[:,end-1]) / Δ.y,
    )

    # Normal stresses (σ = τ - Pt) in the centroids next to each boundary, to compare with σ_BC
    σxx = τ.xx[inx_c,iny_c] .- P.t[inx_c,iny_c]
    σyy = τ.yy[inx_c,iny_c] .- P.t[inx_c,iny_c]
    σ_bnd = (W = mean(σxx[1,:]), E = mean(σxx[end,:]), S = mean(σyy[:,1]), N = mean(σyy[:,end]))
    # Normal stresses reached on the boundaries (linear extrapolation of the first two centroids)
    σ_face = (
        W = mean(1.5 .* σxx[1,:]   .- 0.5 .* σxx[2,:]),
        E = mean(1.5 .* σxx[end,:] .- 0.5 .* σxx[end-1,:]),
        S = mean(1.5 .* σyy[:,1]   .- 0.5 .* σyy[:,2]),
        N = mean(1.5 .* σyy[:,end] .- 0.5 .* σyy[:,end-1]),
    )

    # Shear stresses reached on the boundaries (linear extrapolation of the first two vertex rows/columns inside)
    τxy = τ.xy[inx_v, iny_v]
    τ_face = (
        W = mean(2 .* τxy[2,:]   .- τxy[3,:]),
        E = mean(2 .* τxy[end-1,:] .- τxy[end-2,:]),
        S = mean(2 .* τxy[:,2]   .- τxy[:,3]),
        N = mean(2 .* τxy[:,end-1] .- τxy[:,end-2]),
    )

    return (; X, Vx, Vy, Pt=P.t[inx_c,iny_c], Pf=P.f[inx_c,iny_c], err, iter, ε̇_bnd, σ_bnd, σ_face, τ_face)
end

let
    # Boundary condition templates
    BCs = [
        :pure_shear,
        :free_slip,
        :NS_stress,
        :EW_stress,
        :simple_shear,
        :EW_periodic_shear_stress,
        :NS_periodic_shear_stress,
    ]

    # Boundary velocity gradient matrix
    D_BCs = [
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [0 1; 0  0] ),
        @SMatrix( [0 1; 0  0] ), # initial guess and S velocity (V = 0 would give a NaN tangent: ε̇II = √(…) at 0)
        @SMatrix( [0 0; 1  0] ), # initial guess and W velocity
    ]

    # Boundary stresses: normal (diagonal, *_stress templates) and shear (off-diagonal, *_shear_stress templates)
    σ_BCs = [
        @SMatrix( [  0.0   0.0;   0.0   0.0] ),
        @SMatrix( [  0.0   0.0;   0.0   0.0] ),
        @SMatrix( [-50.0   0.0;   0.0 -50.0] ),
        @SMatrix( [-50.0   0.0;   0.0 -50.0] ),
        @SMatrix( [  0.0   0.0;   0.0   0.0] ),
        @SMatrix( [  0.0   1.0;   1.0   0.0] ),
        @SMatrix( [  0.0   1.0;   1.0   0.0] ),
    ]

    # Run them all and collect the results on a single figure (one row per BC template)
    fig = Figure(size=(1600, 330*length(BCs)), fontsize=14)
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        r = main(BCs[iBC], D_BCs[iBC], σ_BCs[iBC])
        D, σ = D_BCs[iBC], σ_BCs[iBC]
        e, s, b, t = r.ε̇_bnd, r.σ_bnd, r.σ_face, r.τ_face
        @printf("%-14s boundary σxx W, E = %8.3f, %8.3f   σyy S, N = %8.3f, %8.3f   (σ_BC = %g, %g)   boundary τxy S, N = %8.3f, %8.3f   W, E = %8.3f, %8.3f   (τ_BC = %g)\n", BCs[iBC], b.W, b.E, b.S, b.N, σ[1,1], σ[2,2], t.S, t.N, t.W, t.E, σ[1,2])
        Label(fig[iBC, 0], "$(BCs[iBC])\nD = $(D[1,:]) $(D[2,:])\nσ_BC = $(σ[1,1]), $(σ[2,2]), τ_BC = $(σ[1,2])\n\n∂Vx/∂x W, E: $(round(e.W, digits=3)), $(round(e.E, digits=3))\n∂Vy/∂y S, N: $(round(e.S, digits=3)), $(round(e.N, digits=3))\nσxx W, E: $(round(s.W, digits=2)), $(round(s.E, digits=2))\nσyy S, N: $(round(s.S, digits=2)), $(round(s.N, digits=2))\nat boundary:\nσxx W, E: $(round(b.W, digits=2)), $(round(b.E, digits=2))\nσyy S, N: $(round(b.S, digits=2)), $(round(b.N, digits=2))\nτxy S, N: $(round(t.S, digits=3)), $(round(t.N, digits=3))\nτxy W, E: $(round(t.W, digits=3)), $(round(t.E, digits=3))", tellheight=false)
        ax = Axis(fig[iBC, 1], aspect=DataAspect(), title="Vx")
        hm = heatmap!(ax, r.X.v.x, r.X.c.y, r.Vx); Colorbar(fig[iBC, 2], hm)
        ax = Axis(fig[iBC, 3], aspect=DataAspect(), title="Vy")
        hm = heatmap!(ax, r.X.c.x, r.X.v.y, r.Vy); Colorbar(fig[iBC, 4], hm)
        ax = Axis(fig[iBC, 5], aspect=DataAspect(), title="Pt")
        hm = heatmap!(ax, r.X.c.x, r.X.c.y, r.Pt); Colorbar(fig[iBC, 6], hm)
        ax = Axis(fig[iBC, 7], aspect=DataAspect(), title="Pf")
        hm = heatmap!(ax, r.X.c.x, r.X.c.y, r.Pf); Colorbar(fig[iBC, 8], hm)
        ax = Axis(fig[iBC, 9], title="Convergence", xlabel="Iterations", ylabel="log₁₀ error")
        scatter!(ax, 1:r.iter, log10.(r.err.x[1:r.iter]),  label="Vx")
        scatter!(ax, 1:r.iter, log10.(r.err.y[1:r.iter]),  label="Vy")
        scatter!(ax, 1:r.iter, log10.(r.err.pt[1:r.iter]), label="Pt")
        scatter!(ax, 1:r.iter, log10.(r.err.pf[1:r.iter]), label="Pf")
        iBC == 1 && axislegend(ax)
    end
    display(fig)
end
