using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs
import Statistics:mean
import FiniteDiff, ForwardDiff
using MuladdMacro

@inline mynorm(x) = sum(xi^2 for xi in x)

# bulk_viscosity(ϕ, η0, m) = η0*abs(ϕ)^m
@inline bulk_viscosity(ϕ::T, η0, m) where T = iszero(m) ? T(η0) : η0*abs(ϕ)^m

# Trial VE
@inline function PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)  
    ηΦ      = bulk_viscosity(Φ, ξ0, m)
    dPtdt   = @muladd (Pt - Pt0) / Δt
    dPfdt   = @muladd (Pf - Pf0) / Δt
    dΦdt    = @muladd ((dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ)
    return dΦdt, ηΦ
end

# Corrected VEP
@inline function PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)  
    ηΦ      = bulk_viscosity(Φ, ξ0, m)
    dPtdt   = @muladd (Pt - Pt0) / Δt
    dPfdt   = @muladd (Pf - Pf0) / Δt
    P_eff   = Pt - Pf
    ∂Q∂p    = ForwardDiff.derivative( P_eff -> Q(pl, τII, P_eff, 0.0, λ̇, ph), P_eff)
    # ∂Q∂p    = ForwardDiff.derivative( P_eff -> (τII, P_eff, 0.0, pl.C, pl.cosϕ, pl.sinψ, λ̇, pl.ηvp), P_eff)
    dΦdt    = @muladd ((dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ - λ̇*∂Q∂p)
    return dΦdt, ηΦ
end

# Trial VE
@inline function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, Δt) 
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)[1] 
    r    = @muladd Φ - (Φ0  + dΦdt * Δt)  
    return r 
end

# Corrected VEP
@inline function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt) 
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1] 
    r    = @muladd Φ - (Φ0  + dΦdt * Δt)  
    return r 
end

# Trial VE
@inline function Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt) 

    dΦdt, ηΦ = PorosityRate(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)
    Φ        = Φ0  + dΦdt * Δt
    if iszero(m)
        return Φ, dΦdt, ηΦ
    end

    r0       = one(Φ)  # typed to match Φ so r0 doesn't change type after first iter
    for iter=1:10
        r, dresdΦ = ad_value_and_derivative(PorosityResidual, Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)
        if iter==1 r0 = abs(r) + 1e-10 end
        # @show iter, abs(r), abs(r)/r0
        if min(abs(r), abs(r)/r0 ) < 1e-10 break end
        Φ    -=  r / dresdΦ
    end
    dΦdt, ηΦ = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)
    return Φ, dΦdt, ηΦ 
end

# Corrected VEP
@inline function Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt) 

    dΦdt, ηΦ = PorosityRate(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)
    Φ        = Φ0  + dΦdt * Δt
    if iszero(m)
        return Φ, dΦdt, ηΦ
    end

    r0       = one(Φ)  # typed to match Φ so r0 doesn't change type after first iter
    for iter=1:10
        r, dresdΦ = ad_value_and_derivative(PorosityResidual, Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)
        if iter==1 r0 = abs(r) + 1e-10 end
        # @show iter, abs(r), abs(r)/r0
        if min(abs(r), abs(r)/r0 ) < 1e-10 break end
        Φ    -=  r / dresdΦ
    end
    dΦdt, ηΦ = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)
    return Φ, dΦdt, ηΦ 
end

function divergence(x, Δt, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m)
     
    Pt, Pf = x[1], x[2]

    poro = Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)
    Φ, dΦdt = poro[1], poro[2]

    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dPsdt   = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = dPsdt / Ks 
    dlnρfdt = dPfdt / Kf

    # dlnρsdt   - dΦdt/(1-Φ) + divVs,
    # Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD, 
    # -Φ*dlnρsdt   + Φ*dΦdt/(1-Φ) - Φ*divVs, # (* -Φ)
    # -Φ*dlnρsdt   + Φ*dΦdt/(1-Φ) - Φ*divVs, # (* -Φ)
    # Φ*dlnρfdt -Φ*dlnρsdt + dΦdt +  Φ*dΦdt/(1-Φ) + divqD

    return @SVector [ 
        -(dlnρsdt   - dΦdt/(1-Φ)),            # = divVs
        -(Φ*(dlnρfdt - dlnρsdt) + dΦdt/(1-Φ)) # = divqD
    ]
end

function ΔP_residual(x, Φ, Pt, Pf, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇, Δt )

    Pt, Pf = x[1], x[2]

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dlnρfdt = dPfdt / Kf
    # dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    # Φ, dΦdt = Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ, m, λ̇, sinψ, Δt)  
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1]  
    # dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    # dPsdt = (dPtdt - Φ*dPfdt) /(1-Φ)
    dlnρsdt = 1/Ks * ( dPsdt ) 

    # Ps     = (Pt - phi*Pf)/(1-phi) 
    # dPsdt = (dPtdt - phi*dPfdt) /(1-phi)
    # # dPsdt = ((Pt - phi*Pf)/(1-phi) - (Pt0 - phi0*Pf0)/(1-phi0))/dt
    # # dPsdt = dphidt*(Pt - Pf*phi)/(1-phi)**2 + (dPtdt - phi*dPfdt - 0*Pf*dphidt) / (1 - phi)
    # dlnrhosdt = elastic * 1/K_s * ( dPsdt ) 

    return @SVector [ 
        dlnρsdt   - dΦdt/(1-Φ),
        Φ*dlnρfdt + dΦdt      , 
    ]
end

function ΔP(Pt_trial, Pf_trial, divVs, divqD, Φ, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇::Tλ, Δt) where Tλ

    x   = @SVector[zero(Tλ), zero(Tλ)]  # typed to match λ̇ so J\R doesn't change x's type
    r0  = one(Tλ)
    tol = 1e-13

    for iter=1:10
        R, J = ad_value_and_jacobian(ΔP_residual, x, Φ, Pt_trial, Pf_trial, 0 * divVs, 0 * divqD, 0 * Pt0, 0 * Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇, Δt)
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


function residual_two_phase_P(x, ηve, Δt, ε̇II_eff, τII_trial, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, single_phase )
     
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    α1 = single_phase ? 0.0 : 1.0 

    Pe = if single_phase
        Pt
    else
         Pt .- Pf
    end

    ∂Q∂τ  = ForwardDiff.derivative( τII -> Q(pl, τII, Pe, 0.0,  λ̇, ph), τII )

    # Pressure corrections: closed form
    # ΔPt_1 = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
    # ΔPf   = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    
    # Pressure corrections: numerics (nested AD)
    ΔPt_1, ΔPf = ΔP(Pt_trial, Pf_trial, divVs, divqD, Φ, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇, Δt)

    # Check yield
    fy =  F(pl, τII, Pe, 0.0, λ̇, ph)

    ΔPt = if single_phase
        Ks .* sinψ .* Δt .* λ̇
        else
            ΔPt_1
        end
    
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1]  
    fΦ   =  @muladd Φ - (Φ0  + dΦdt * Δt)  

    return @SVector [ 
        # ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        τII - (τII_trial - ηve*λ̇*∂Q∂τ),
        Pt - (Pt_trial + ΔPt),
        Pf - (Pf_trial + ΔPf),
        fy, 
        fΦ,
    ]
end


function bt_line_search(Δx, J, x, r, args; α=1.0, ρ=0.5, c=1.0e-4, α_min=1.0e-8)
    # Borrowed from RheologicalCalculator
    perturbed_x = @. x + α * Δx
    perturbed_r = residual_two_phase_P(x, args... )

    J_times_Δx = -J * Δx
    while sqrt(sum(perturbed_r .^ 2)) > sqrt(sum((r + (c * α * (J_times_Δx))) .^ 2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_x = @. x + α * Δx
        perturbed_r = residual_two_phase_P(x, args... )
    end
    return α
end

function LocalRheology_P(ε̇::SVector{N, D}, divVs, divqD, Pt0, Pf0, Φ0, materials, ph, Δ) where {N, D}

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt = ε̇[4]
    Pf = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[ph]
    m    = materials.m[ph]
    η0   = materials.η0[ph]
    G    = materials.G[ph]
    ξ0   = materials.ξ0[ph]
    KΦ   = materials.KΦ[ph]
    Ks   = materials.Ks[ph]
    Kf   = materials.Kf[ph]

    pl   = materials.plasticity

    # C    = materials.plasticity.C[ph]
    # ηvp  = materials.plasticity.ηvp[ph]
    # sinψ = materials.plasticity.sinψ[ph]    
    # sinϕ = materials.plasticity.sinϕ[ph] 
    # cosϕ = materials.plasticity.cosϕ[ph] 
    
    # pl = (C   =C   ,
    #       ηvp =ηvp ,
    #       sinψ=sinψ,
    #       sinϕ=sinϕ,
    #       cosϕ=cosϕ,)

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
        # Φ = (KΦ * Δ.t * (Pf - Pt) + KΦ * Φ0 * ξ0 + ξ0 * (Pf - Pf0 - Pt + Pt0)) / (KΦ * ξ0)
    
        # Trial porosity: numerics (nested AD)
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δ.t)[1]
    end

    # Check yield
    λ̇  = zero(D)

    #############################

    Peff =  Pt - Pf
    f    = F(pl, τII, Peff, Φ, λ̇, ph)

    x = @SVector [τII, Pt, Pf, λ̇, Φ]
    x0 =copy(x)
    plastic_correction = false

    nr   = D(1.0)
    nr0  = D(1.0)
    tol  = D(1e-10)

    # Return mapping
    if f > D(-1e-13)
        plastic_correction = true
        # This is the proper return mapping with plasticity
        args = (ηve, Δ.t, ε̇II_eff, τII,       Pt,       Pf,       divVs, divqD, Φ,       Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, materials.single_phase)
        for iter=1:20
            r, J = fd_value_and_jacobian(residual_two_phase_P, x, args...)
            Δx   = -J \ r
            α    = bt_line_search(Δx, J, x, r, args, α=1.0, ρ=0.5, c=1.0e-4, α_min=1.0e-8)
            x   += α*Δx
            nr   = mynorm(r)
            if iter==1 
                nr0 = nr
            end
            if iter==20
                error("Local iteration failed: nr=$(nr) nr0=$(nr0) f = $(f) x0 = $(x0), α = $(α) ")
            end
            nr/nr0 < tol && break
        end
    end

    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    Φ = if materials.single_phase
        zero(D)
    else
        Φ
    end

    dΦdt = if materials.single_phase
        zero(D)
    else
        PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δ.t)[1]  
    end

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
# function residual_two_phase_P_v2(x, ηve, Δt, ε̇II_eff, τII_trial, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, single_phase )
     
#     τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

#     Pe    = Pt .- Pf
#     dPtdt = (Pt - Pt0) / Δt
#     dPfdt = (Pf - Pf0) / Δt
#     dΦdt = StagFDTools.TwoPhases.PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1]  
#     dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    
#     ∂Q∂τ  = ForwardDiff.derivative( τII -> Q(pl, τII, Pe, 0.0,  λ̇, ph), τII )

#     # Check yield
#     fy =  F(pl, τII, Pe, 0.0, λ̇, ph)
    
#     # dΦdt
#     fΦ   =  @muladd Φ - (Φ0  + dΦdt * Δt)  

#     dlnρsdt = dPsdt / Ks 
#     dlnρfdt = dPfdt / Kf

#     return @SVector [ 
#         # ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
#         τII - (τII_trial - ηve*λ̇*∂Q∂τ),
#         dlnρsdt   - dΦdt/(1-Φ) + divVs,
#         # Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD, 
#         Φ*dlnρfdt -Φ*dlnρsdt + dΦdt +  Φ*dΦdt/(1-Φ)  + divqD,
#         fy, 
#         fΦ,
#     ]
# end

function residual_two_phase_P_v2(x, ηve, Δt, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, single_phase )
     
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    Pe    = Pt .- Pf
    dPtdt = (Pt - Pt0) / Δt
    dPfdt = (Pf - Pf0) / Δt
    dΦdt = StagFDTools.TwoPhases.PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1]  
    dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    
    ∂Q∂τ  = ForwardDiff.derivative( τII -> Q(pl, τII, Pe, 0.0,  λ̇, ph), τII )

    # Check yield
    fy =  F(pl, τII, Pe, 0.0, λ̇, ph)
    
    # dΦdt
    fΦ   =  @muladd Φ - (Φ0  + dΦdt * Δt)  

    dlnρsdt = dPsdt / Ks 
    dlnρfdt = dPfdt / Kf


    return @SVector [ 
        # ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        # τII - (τII_trial - ηve*λ̇*∂Q∂τ),
        τII - 2*ηve*(ε̇II_eff - λ̇*∂Q∂τ/2),
        dlnρsdt   - dΦdt/(1-Φ) + divVs,
        Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD, 
        fy, 
        fΦ,
    ]
end



function LocalRheology_P2(ε̇::SVector{N, D}, Pt0, Pf0, Φ0, materials, ph, Δ) where {N, D}

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt = Pt0
    Pf = Pf0
    divVs, divqD = ε̇[4], ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[ph]
    m    = materials.m[ph]
    η0   = materials.η0[ph]
    G    = materials.G[ph]
    ξ0   = materials.ξ0[ph]
    KΦ   = materials.KΦ[ph]
    Ks   = materials.Ks[ph]
    Kf   = materials.Kf[ph]

    pl   = materials.plasticity

    α1 = materials.single_phase ? zero(D) : one(D)

    # Initial guess
    η         = η0 * ε̇II_eff^(1 / n - 1 )
    ηve       = inv(1/η + 1/(G*Δ.t))
    τII       = 2*ηve*ε̇II_eff
    ηvep      = ηve

    # div = @SVector[divVs, divqD]
    # x   = Pressures(div, Pt0, Pf0, Φ0, materials.KΦ[1],  materials.Ks[1],  materials.Kf[1], materials.ξ0[1], Δ.t)
    # Pt, Pf, Φ = x[1], x[2], x[3]

    Φ = if materials.single_phase
        zero(D)
    else
        # Trial porosity: closed form
        # Φ = (KΦ * Δ.t * (Pf - Pt) + KΦ * Φ0 * ξ0 + ξ0 * (Pf - Pf0 - Pt + Pt0)) / (KΦ * ξ0)
    
        # Trial porosity: numerics (nested AD)
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δ.t)[1]
    end

    # Check yield
    λ̇  = zero(D)

    #############################

    Peff =  Pt - Pf
    f    = F(pl, τII, Peff, Φ, λ̇, ph)

    x = @SVector [τII, Pt, Pf, λ̇, Φ]
    x0 = copy(x)
    plastic_correction = false

    nr   = D(1.0)
    nr0  = D(1.0)
    tol  = D(1e-10)

    # Return mapping
    if f > D(-1e-13)
        plastic_correction = true
        # This is the proper return mapping with plasticity
        args = (ηve, Δ.t, ε̇II_eff,     divVs, divqD,       Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, materials.single_phase)
        for iter=1:20
            r, J = fd_value_and_jacobian(residual_two_phase_P_v2, x, args...)
            Δx   = -J \ r
            # α    = bt_line_search(Δx, J, x, r, args, α=1.0, ρ=0.5, c=1.0e-4, α_min=1.0e-8)
            x   += 1*Δx
            nr   = mynorm(r)
            if iter==1 
                nr0 = nr
            end
            if iter==20
                error("Local iteration failed: nr=$(nr) nr0=$(nr0) f = $(f) x0 = $(x0), α = $(α) ")
            end
            nr/nr0 < tol && break
        end
    end

    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    Φ = if materials.single_phase
        zero(D)
    else
        Φ
    end

    dΦdt = if materials.single_phase
        zero(D)
    else
        PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δ.t)[1]  
    end

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

    return ηvep, λ̇, Pt, Pf, τII, Φ, f, dlnρsdt, dlnρfdt 
end

@inline function StressVector_P2!(ε̇::SVector{N, T}, Pt0, Pf0, Φ0, materials, phases, Δ) where {N,T}
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P2(ε̇, Pt0, Pf0, Φ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
    return τ
end


function residual_two_phase_P_v3(x, ηve, Δt, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, single_phase )
     
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    ϵ = -1e-13

    Pe    = Pt .- Pf
    dPtdt = (Pt - Pt0) / Δt
    dPfdt = (Pf - Pf0) / Δt
    dΦdt = StagFDTools.TwoPhases.PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1]  
    dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    
    ∂Q∂τ  = ForwardDiff.derivative( τII -> Q(pl, τII, Pe, 0.0,  λ̇, ph), τII )

    # Plasticity residual
    fy =  F(pl, τII, Pe, 0.0, λ̇, ph)
    
    # Porosity residual
    fΦ   =  @muladd Φ - (Φ0  + dΦdt * Δt)  

    # Equations of state
    dlnρsdt = dPsdt / Ks 
    dlnρfdt = dPfdt / Kf

    return @SVector [ 
        # ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        τII - 2*ηve*(ε̇II_eff - λ̇*∂Q∂τ/2),
        dlnρsdt   - dΦdt/(1-Φ) + divVs,
        # Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD,
        Φ*(dlnρfdt - dlnρsdt) + dΦdt/(1-Φ)  + divqD, 
        (fy>=0)*fy + (fy<0)*λ̇, 
        fΦ,
    ]
end


function LocalRheology_P3(ε̇::SVector{N, D}, Pt0, Pf0, Φ0, materials, ph, Δ) where {N, D}

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt = Pt0
    Pf = Pf0
    divVs, divqD = ε̇[4], ε̇[5]

    # Parameters
    n    = materials.n[ph]
    m    = materials.m[ph]
    η0   = materials.η0[ph]
    G    = materials.G[ph]
    ξ0   = materials.ξ0[ph]
    KΦ   = materials.KΦ[ph]
    Ks   = materials.Ks[ph]
    Kf   = materials.Kf[ph]

    pl   = materials.plasticity

    # Initial guess
    η         = η0 * ε̇II_eff^(1 / n - 1 )
    ηve       = inv(1/η + 1/(G*Δ.t))
    τII       = 2*ηve*ε̇II_eff
    ηvep      = ηve

    x = @SVector [τII, Pt, Pf, 0.0, Φ0]

    nr   = D(1.0)
    nr0  = D(1.0)
    tol  = D(1e-10)

    # Return mapping
    args = (ηve, Δ.t, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, materials.single_phase )
    for iter=1:20
        r, J = fd_value_and_jacobian(residual_two_phase_P_v3, x, args...)
        Δx   = -J \ r
        # α    = bt_line_search(Δx, J, x, r, args, α=1.0, ρ=0.5, c=1.0e-4, α_min=1.0e-8)
        x   += 1*Δx
        nr   = mynorm(r)
        if iter==1 
            nr0 = nr
        end

        ((nr/nr0  < tol) || (nr < tol)) && break
    end

    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    #############################

    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    return ηvep, λ̇, Pt, Pf, τII, Φ
end

@inline function StressVector_P3!(ε̇::SVector{N, T}, Pt0, Pf0, Φ0, materials, phases, Δ) where {N,T}
    η, λ̇, Pt, Pf, τII, Φ = LocalRheology_P3(ε̇, Pt0, Pf0, Φ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
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

        # P_trial = @SVector( [Pt; Pf] )
        # residual_two_phase_trial_P(x, Δt, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m )


        # OLD STYLE 
        @info "Old style"
        # # Trial pressures - not needed with LocalRheology_P2 !!!
        # let
            div = @SVector[ε̇kk, divqD]
            x   = Pressures(div, Pt0, Pf0, Φ0, materials.KΦ[1],  materials.Ks[1],  materials.Kf[1], materials.ξ0[1], Δ.t)
            Pt, Pf, Φ = x[1], x[2], x[3]

            # Correction
            ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )
            η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
            τ_vec1, jac1 = fd_value_and_jacobian(StressVector_P!, ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)
            display(jac1)
        # end

        # NEW STYLE
        @info "New style - pt 1"
        P =  @SVector[Pt, Pf]
        J_pp = ForwardDiff.jacobian(P -> divergence(P, Δt, Pt0, Pf0, Φ0, materials.KΦ[1],  materials.Ks[1],  materials.Kf[1], materials.ξ0[1], materials.m[1]), P)
        Mpp = Matrix{Float64}(I, 5, 5)
        Mpp[4:5,4:5] .= J_pp

        @info "New style - pt 2"
        ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; ε̇kk; divqD] )
        η, λ̇, Pt, Pf, τII, Φ    = LocalRheology_P3(ε̇vec, Pt0, Pf0, Φ0, materials, 1, Δ)
        τ_vec, jac2 = fd_value_and_jacobian(StressVector_P3!, ε̇vec, Pt0, Pf0, Φ0, materials, 1, Δ)
        display(jac2*Mpp)

        # function Stress(x)
        #     StressVector_P2!(
        #         x, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ
        #     )
        # end
        # jac_FD1 = ForwardDiff.jacobian(Stress, ε̇vec)
        # jac_FD2 = FiniteDiff.finite_difference_jacobian(Stress, ε̇vec)
        # display(jac2*Mpp)
        # display(jac_FD1)
        # display(jac_FD2)

        #--------------------------------------------#

        # Include plasticity corrections
        # η   = x[1]
        # Pt  = τ_vec[4]
        # Pf  = τ_vec[5]
        τxx = 2 * η * ε̇vec[1]
        τyy = 2 * η * ε̇vec[2]
        τxy = 2 * η * ε̇vec[3]
     
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