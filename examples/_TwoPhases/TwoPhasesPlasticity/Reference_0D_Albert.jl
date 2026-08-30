using StagFDTools, StagFDTools.TwoPhases
using StaticArrays, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs
import Statistics:mean
import FiniteDiff, ForwardDiff
using MuladdMacro

 bulk_viscosity(ϕ::T, η0, m) where T = iszero(m) ? T(η0) : η0*abs(ϕ)^m

 function PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, λ̇, sinψ, Δt)  
    ηΦ      = bulk_viscosity(Φ, ξ0, m)
    dPtdt   = @muladd (Pt - Pt0) / Δt
    dPfdt   = @muladd (Pf - Pf0) / Δt
    dΦdt    = @muladd ((dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sinψ)
    return dΦdt, ηΦ
end

 function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, λ̇, sinψ, Δt) 
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, λ̇, sinψ, Δt)[1] 
    r    = @muladd Φ - (Φ0  + dΦdt * Δt)  
    return r 
end

 function StressVector_P!(ε̇::SVector{N, T}, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) where {N,T}
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
    return τ
end

function residual_two_phase_P2(x, ηve, Δt, ε̇II_eff, τII_trial, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )
     
    τII, ΔPt, ΔPf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]
    α1 = single_phase ? 0.0 : 1.0 
    
    Ptc = (Pt_trial+ΔPt) 
    Pfc = (Pf_trial+ΔPf)

    fp = F(τII, Ptc, Pfc, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)   
    # fΦ = PorosityResidual(Φ, Φ0, Ptc, Pfc, Pt0, Pf0, KΦ, ηΦ, m, λ̇, sinψ, Δt) 

    dPfdt   = (Pfc - Pf0) / Δt
    dPsdt   = 1/Δt* ((Φ - 1) .* (-Pf0 .* Φ0 + Pt0) + (Φ0 - 1) .* (Pfc .* Φ - Ptc)) ./ ((Φ - 1) .* (Φ0 - 1))
    dPtdt   = (Ptc - Pt0) / Δt
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pfc - Ptc)/ηΦ + λ̇*sinψ

    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/Ks * ( dPsdt ) 

    return @SVector [ 
        ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        dlnρsdt   - dΦdt/(1-Φ) +   divVs,
        Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD,
        fp,     
        Φ - (Φ0  + dΦdt * Δt), 
    ]
end

function LocalRheology_P2(ε̇::SVector{N, D}, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) where {N, D}

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt = ε̇[4]
    Pf = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    m    = materials.m[phases]
    η0   = materials.η0[phases]
    G    = materials.G[phases]
    C    = materials.plasticity.C[phases]
    ηΦ0  = materials.ξ0[phases]
    KΦ   = materials.KΦ[phases]
    Ks   = materials.Ks[phases]
    Kf   = materials.Kf[phases]

    ηvp  = materials.plasticity.ηvp[phases]
    sinψ = materials.plasticity.sinψ[phases]    
    sinϕ = materials.plasticity.sinϕ[phases] 
    cosϕ = materials.plasticity.cosϕ[phases]  
    
    α1 = materials.single_phase ? zero(D) : one(D)

    # Initial guess
    η         = η0 * ε̇II_eff^(1 / n - 1 )
    ηve       = inv(1/η + 1/(G*Δ.t))
    τII       = 2*ηve*ε̇II_eff
    ηvep      = ηve
    
    # Trial porosity: numerics (nested AD)
    Φ = Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, 0.0, sinψ, Δ.t)[1]

    # Check yield
    λ̇  = zero(D)

    #############################

    f  = F(τII, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    x = @SVector [τII, 0.0, 0.0, λ̇, Φ]
    plastic_correction = false

    nr   = D(1.0)
    nr0  = D(1.0)
    tol  = D(1e-10)

    # Return mapping
    if f > D(-1e-13)
        for iter=1:10
            R, J = fd_value_and_jacobian(residual_two_phase_P2, x, ηve, Δ.t, ε̇II_eff, τII, Pt, Pf, divVs, divqD, Φ,       Pt0, Pf0, Φ0, ηΦ0, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, materials.single_phase)
            x   -= J \ R
            nr   = norm(R)
            if iter==1 
                nr0 = nr
            end
            nr/nr0 < tol && break
        end
    end
    τII, dPt, dPf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]
   
    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    return  @SVector [ηvep, λ̇, dPt, dPf] #ηvep, λ̇, Pt, Pf#, τII, Φ, f, dlnρsdt, dlnρfdt 
end


function StressVector_P2!(ε̇::SVector{N, T}, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) where {N,T}
    Pt_trial, Pf_trial = ε̇[4], ε̇[5] 
    x = LocalRheology_P2(ε̇, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ)
    τ  = @SVector([2 * x[1] * ε̇[1],
                   2 * x[1] * ε̇[2],
                   2 * x[1] * ε̇[3],
                   Pt_trial+x[3],
                   Pf_trial+x[4],])
    return τ
end

@views function main(nc, nt, n_nt; 
    homo=false, niter=100, Φini=5e-2, ηvp=0.0)

    sc = (σ=1e7, t=1e10, L=1e3)
    ky = 1e3*365*24*3600

    # Time steps
    Δt     = 1e10/sc.t / n_nt 

    rad     = 1e3/sc.L 
    Pt_ini  = 1e6/sc.σ
    Pf_ini  = 1e6/sc.σ
    ε̇       = 2e-15.*sc.t
    τ_ini   = 0*(sind(35)*(Pt_ini-Pf_ini) + 0*1e7/sc.σ*cosd(35))  

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )

    τxx_ini = τ_ini*D_BC[1,1]/ε̇
    τyy_ini = τ_ini*D_BC[2,2]/ε̇

    ε̇kk, divqD = 0.0, 0.0

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

    materials.n     .= [  1.0,    1.0 ]
    materials.m     .= [  0.0,    0.0 ]
    materials.n_CK  .= [  0.0,    0.0 ]
    materials.η0    .= [ 1e22,   1e19 ]/sc.σ/sc.t 
    materials.ξ0    .= [ 2e22,   2e22 ]/sc.σ/sc.t
    materials.G     .= [ 3e10,   3e10 ]./sc.σ 
    materials.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials.Ks    .= [ 1e11,   1e11 ]./sc.σ
    materials.KΦ    .= [ 1e10,   1e10 ]./sc.σ
    materials.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials.k_ηf0 .= [1e-15,  1e-15 ]./(sc.L^2/sc.σ/sc.t)
    materials.plasticity.ϕ   .= [ 35.,     35. ]
    materials.plasticity.ψ   .= [ 10.,     10. ] .* 1
    materials.plasticity.C   .= [ 1e7,     1e7 ]./sc.σ
    materials.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t 
    preprocess!(materials)

    Φ0      = Φini
    # Φ0 = (materials.KΦ[1] .* Δt0 .* (Pf_ini - Pt_ini)) ./ (materials.KΦ[1] .* materials.ξ0[1])
    @show Φ0
    # error()
    Φ_ini   = Φ0

    probes = (
        Pe  = zeros(nt),
        Pt  = zeros(nt),
        Pf  = zeros(nt),
        τ1  = zeros(nt),
        τ   = zeros(nt),
        Φ   = zeros(nt),
        λ̇   = zeros(nt),
        t   = zeros(nt),
        τII = zeros(nt),
    )

    τII, λ̇ = 0.0, 0.0
    Pt, Pf = Pt_ini, Pf_ini
    τxx, τyy, τxy =  0.0, 0.0, 0.0
    ΔPt, ΔPf = 0.0, 0.0
    Φ = Φ0
    ρs, ρf = materials.ρs[1], materials.ρf[1]
    Δ = (t=Δt,)

    for it=1:nt

        @printf("\nStep %04d\n", it)

        # Swap old values 
        Pt0  = Pt
        Pf0  = Pf
        τxx0 = τxx
        τyy0 = τyy
        τxy0 = τxy
        Φ0   = Φ 

        # Trial deviatoric stress
        ε̇xx_eff = ε̇ + τxx0/(2*materials.G[1]*Δt)
        ε̇yy_eff =-ε̇ + τyy0/(2*materials.G[1]*Δt)
 
        # NEW STYLE 
        ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )

        x    = LocalRheology_P2(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
        
        τ_vec, jac2 = fd_value_and_jacobian(StressVector_P2!, ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
   
        display(jac2)

        # function Stress(x)
        #     StressVector_P2!(
        #         x, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ
        #     )
        # end
        # jac_FD1 = ForwardDiff.jacobian(Stress, ε̇vec)
        # jac_FD2 = FiniteDiff.finite_difference_jacobian(Stress, ε̇vec)
        # display(jac_FD1)
        # display(jac_FD2)

        #--------------------------------------------#

        # Include plasticity corrections
        η   = x[1]
        Pt  = τ_vec[4]
        Pf  = τ_vec[5]
        τxx = 2 * η * ε̇vec[1]
        τyy = 2 * η * ε̇vec[2]
        τxy = 2 * η * ε̇vec[3]

        #-------------------------------------------# 
    end
    #--------------------------------------------#
    return 
end

function Run()
    # Homogeneous test
    n_nx = 1
    n_nt = 1
    nc   = (x=n_nx*50, y=n_nx*25)
    nt   = 9 #8#40*n_nt
    main(nc, nt, n_nt, homo=true, niter=2)
end

Run()