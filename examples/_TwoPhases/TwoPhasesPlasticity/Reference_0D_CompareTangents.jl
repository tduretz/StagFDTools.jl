using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs
import Statistics:mean
import FiniteDiff, ForwardDiff
using MuladdMacro

################## COMMON ##################
@inline mynorm(x) = sum(xi^2 for xi in x)
@inline bulk_viscosity(ϕ::T, η0, m) where T = iszero(m) ? T(η0) : η0*abs(ϕ)^m

@inline function PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, λ̇, sinψ, Δt)  
    ηΦ      = bulk_viscosity(Φ, ξ0, m)
    dPtdt   = @muladd (Pt - Pt0) / Δt
    dPfdt   = @muladd (Pf - Pf0) / Δt
    # @show λ̇*sinψ, λ̇, sinψ
    dΦdt    = @muladd ((dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sinψ)
    return dΦdt, ηΦ
end

@inline function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, λ̇, sinψ, Δt) 
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, λ̇, sinψ, Δt)[1] 
    r    = @muladd Φ - (Φ0  + dΦdt * Δt)  
    return r 
end

################## OLD ROUTINES (CORRECT) ##################
function ΔP_Trial(x, Pt, Pf, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, sinψ, Δt )

    Pt, Pf = x[1], x[2]

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dlnρfdt = dPfdt / Kf

    Φ, dΦdt = Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ, m, λ̇, sinψ, Δt)  
    dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    dlnρsdt = 1/Ks * ( dPsdt ) 

    return @SVector [ 
        dlnρsdt   - dΦdt/(1-Φ),
        (Φ*dlnρfdt + dΦdt     ),
    ]
end

function ΔP(Pt_trial, Pf_trial, divVs, divqD, λ̇::Tλ, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, sinψ, Δt) where Tλ

    x   = @SVector[zero(Tλ), zero(Tλ)]  # typed to match λ̇ so J\R doesn't change x's type
    r0  = one(Tλ)
    tol = 1e-13

    for iter=1:10
        R, J = ad_value_and_jacobian(ΔP_Trial, x, Pt_trial, Pf_trial, 0 * divVs, 0 * divqD, λ̇, 0 * Pt0, 0 * Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, sinψ, Δt)
        x  = x .- J \ R
        nr = mynorm(R)
        if iter==1 && nr>1e-17
            r0 = nr
        end
        r = nr/r0
        if r<tol
            break
        end
    end
    return x[1], x[2]
end

function residual_two_phase_P(x, ηve, Δt, ε̇II_eff, τII_trial, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )
     
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], (x[4])
    α1 = single_phase ? 0.0 : 1.0 

    # Pressure corrections: closed form
    # ΔPt_1 = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
    # ΔPf   = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    
    # Pressure corrections: numerics (nested AD)
    ΔPt_1, ΔPf = ΔP(Pt_trial, Pf_trial, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ, m,  KΦ, Ks, Kf, sinψ, Δt)

    # Check yield
    f = if single_phase
            τII - C*cosϕ - Pt*sinϕ 
        else
            F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)
        end

    ΔPt = if single_phase
        Ks .* sinψ .* Δt .* λ̇
        else
            ΔPt_1
        end

    return @SVector [ 
        # ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        τII - (τII_trial - ηve*λ̇),
        Pt - (Pt_trial + ΔPt),
        Pf - (Pf_trial + ΔPf),
        f, 
    ]
end

function LocalRheology_P(ε̇::SVector{N, D}, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) where {N, D}

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
    ηΦ0   = materials.ξ0[phases]
    KΦ   = materials.KΦ[phases]
    Ks   = materials.Ks[phases]
    Kf   = materials.Kf[phases]

    ηvp  = materials.plasticity.ηvp[phases]
    sinψ = materials.plasticity.sinψ[phases]    
    sinϕ = materials.plasticity.sinϕ[phases] 
    cosϕ = materials.plasticity.cosϕ[phases]  

    # ηvep, λ̇, Pt, Pf, τII, Φ, f  = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    
    α1 = materials.single_phase ? zero(D) : one(D)

    # Initial guess
    η         = η0 * ε̇II_eff^(1 / n - 1 )
    ηve       = inv(1/η + 1/(G*Δ.t))
    τII       = 2*ηve*ε̇II_eff
    ηvep      = ηve

    Φ = if materials.single_phase
        zero(D)
    else
        # Trial porosity: closed form
        # Φ = (KΦ * Δ.t * (Pf - Pt) + KΦ * Φ0 * ηΦ0 + ηΦ0 * (Pf - Pf0 - Pt + Pt0)) / (KΦ * ηΦ0)
    
        # Trial porosity: numerics (nested AD)
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, 0.0, sinψ, Δ.t)[1]
    end

    # Check yield
    λ̇  = zero(D)

    #############################

    f  = F(τII, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    x = @SVector [τII, Pt, Pf, λ̇]
    plastic_correction = false

    nr   = D(1.0)
    nr0  = D(1.0)
    tol  = D(1e-10)

    # Return mapping
    if f > D(-1e-13)
        plastic_correction = true
        # This is the proper return mapping with plasticity
        for iter=1:10
            R, J = fd_value_and_jacobian(residual_two_phase_P, x, ηve, Δ.t, ε̇II_eff, τII,       Pt,       Pf,       divVs, divqD, Φ,       Pt0, Pf0, Φ0, ηΦ0, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, materials.single_phase)
            x   -= J \ R
            nr   = mynorm(R)
            if iter==1 
                nr0 = nr
            end
            nr/nr0 < tol && break
        end
    end

    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]

    Φ = if materials.single_phase
        zero(D)
    else
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δ.t)[1]
    end

    dΦdt = if materials.single_phase
        zero(D)
    else
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δ.t)[2]
    end

    # Φ, dΦdt = if materials.linearizeΦ ||  materials.single_phase
    #     TΦ      = promote_type_to_dual(Pt_loc, Pf_loc)
    #     Φ       = TΦ(Φ0)
    #     dΦdt    = TΦ(zeros(3,3))
    #     Φ, dΦdt 
    # else
    #     Φ, dΦdt = compute_Φ_and_dΦdt(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δ.t)
    #     Φ, dΦdt 
    # end


    # EOS
    dPtdt   = (Pt - Pt0) / Δ.t
    dPfdt   = (Pf - Pf0) / Δ.t
    dlnρfdt = dPfdt / Kf
    # dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1/Ks * ( dPsdt ) 

    #############################

    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    # Optional: check f
    if plastic_correction
        f    = F(τII, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)
    end

    return ηvep, λ̇, Pt, Pf, τII, Φ, f, dlnρsdt, dlnρfdt 
end

@inline function StressVector_P!(ε̇::SVector{N, T}, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) where {N,T}
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
    return τ
end

################## NEW ROUTINES (CORRECT) ##################
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

    α1   = materials.single_phase ? zero(D) : one(D)

    # Initial guess
    η         = η0 * ε̇II_eff^(1 / n - 1 )
    ηve       = inv(1/η + 1/(G*Δ.t))
    τII       = 2*ηve*ε̇II_eff
    ηvep      = ηve

    Φ = if materials.single_phase
        zero(D)
    else
        # Trial porosity: numerics (nested AD)
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, 0.0, sinψ, Δ.t)[1]
    end

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

        plastic_correction = true
        # This is the proper return mapping with plasticity
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
    
    #############################

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

################## MAIN ##################
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

        # OLD STYLE 

        # # Trial pressures - not needed with LocalRheology_P2 !!!
        # let
            div = @SVector[ε̇kk, divqD]
            x   = Pressures(div, Pt0, Pf0, Φ0, materials.KΦ[1],  materials.Ks[1],  materials.Kf[1], materials.ξ0[1], Δ.t)
            Pt, Pf, Φ = x[1], x[2], x[3]

            # Correction
            ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )
            η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
            τ_vec1, jac1 = fd_value_and_jacobian(StressVector_P!, ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
            # @show τ_vec1, Pt1, Pf1
            # display(jac1)
        # end

        # # NEW STYLE 
        # ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )
        # x    = LocalRheology_P2(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
        # τ_vec, jac2 = fd_value_and_jacobian(StressVector_P2!, ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
        # @show  τ_vec, x[3], x[4]

        # function Stress(x)
        #     StressVector_P2!(
        #         x, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ
        #     )
        # end
        # jac_FD1 = ForwardDiff.jacobian(Stress, ε̇vec)
        # jac_FD2 = FiniteDiff.finite_difference_jacobian(Stress, ε̇vec)
        # display(jac2)

        #--------------------------------------------#

        # Include plasticity corrections
        # η   = x[1]
        # Pt  = τ_vec[4]
        # Pf  = τ_vec[5]
        τxx = 2 * η * ε̇vec[1]
        τyy = 2 * η * ε̇vec[2]
        τxy = 2 * η * ε̇vec[3]

        @show τxx, τyy, τII, Pt, Pf
     
        #--------------------------------------------#
        probes.Pe[it]   = (Pt .- Pf)*sc.σ
        probes.Pt[it]   = Pt*sc.σ
        probes.Pf[it]   = Pf*sc.σ
        probes.τ1[it]   = τII*sc.σ
        probes.τ[it]    = τII*sc.σ
        probes.Φ[it]    = Φ
        probes.λ̇[it]    = λ̇/sc.t
        probes.t[it]    = it*Δt*sc.t

        #-------------------------------------------# 

        @info τ_ini*sc.σ
        @show τxx_ini*sc.σ, τyy_ini*sc.σ

        # fname = @sprintf("PoroVEP_%03d.jld2",  it)
        # save("./examples/_TwoPhases/TwoPhasesPlasticity/results/$(fname)", "X", X, "sc", sc, "probes", probes,
        # "λ̇", λ̇, "P", P, "τ", τ, "ε̇", ε̇, "V", V, "η", η, "Φ", Φ, "εp", εp, "niter", niter, "err", err ) 
      
        data = load("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_remix2.jld2")
        probes2D = data["probes"]

        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 600) )    
            step = 10
            ftsz = 15
            eps  = 1e-10
            # ax    = Axis(fig[1,1], title=L"$\dot{\lambda}$ [1/s]", xlabel=L"x", ylabel=L"y")
            ax    = Axis(fig[1,1], title=L"$\tau_\text{II}$", xlabel=L"$t$ [ky]", ylabel=L"$\tau_\text{II}$ [MPa]")
            lines!(ax, probes.t[1:it]/ky, probes.τ[1:it]/1e6)
            scatter!(ax, probes.t[1:it]/ky, probes.τ1[1:it]/1e6)
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.τ[1:it]/1e6)
            ax    = Axis(fig[2,1], title=L"$P$", xlabel=L"$t$ [ky]", ylabel=L"$P$ [MPa]")
            lines!(ax, probes.t[1:it]/ky, probes.Pt[1:it]/1e6)
            lines!(ax, probes.t[1:it]/ky, probes.Pf[1:it]/1e6)
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.Pt[1:it]/1e6)
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.Pf[1:it]/1e6)
            ax    = Axis(fig[1,2], title=L"$\dot{\lambda}$", xlabel=L"$t$ [ky]", ylabel=L"$\dot{\lambda}$ [1/s]")
            lines!(ax, probes.t[1:it]/ky, probes.λ̇[1:it])
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.λ̇[1:it])
            ax    = Axis(fig[2,2], title=L"$\Phi$", xlabel=L"$t$ [ky]", ylabel=L"$\Phi$ [-]")
            lines!(ax, probes.t[1:it]/ky, probes.Φ[1:it])
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.Φ[1:it])


            ax    = Axis(fig[3:4,1:2], title=L"$$Yield", xlabel=L"$P^\text{eff}$ [MPa]", ylabel=L"$\tau$ [MPa]")
            Peff = (probes.Pt[1:it].-probes.Pf[1:it])
            pmin = Peff[it] - 1.1*Peff[it] 
            pmax = Peff[it] + 0.1*Peff[it]  + 0.01
            p    = LinRange(pmin, pmax, 5)
            C    = materials.plasticity.C[1] * sc.σ 
            cosϕ = materials.plasticity.cosϕ[1] 
            sinϕ = materials.plasticity.sinϕ[1]
            τ    = C*cosϕ .+ p*sinϕ
            lines!(ax, p./1e6, τ./1e6)
            scatter!(ax, Peff/1e6, probes2D.τ[1:it]./1e6)

            display(fig) 
        end
        with_theme(figure, theme_latexfonts())
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
    nt   = 8 #8#40*n_nt
    main(nc, nt, n_nt, homo=true, niter=2)
end

Run()