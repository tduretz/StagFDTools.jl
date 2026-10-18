using StaticArrays, Printf, LinearAlgebra, ForwardDiff, CairoMakie

abstract type AbstractRheology end

"""
    AbstractPlasticity <: AbstractRheology

Supertype for plastic yield or flow-rule elements.
"""
abstract type AbstractPlasticity <: AbstractRheology end 

struct DruckerPrager{T} <: AbstractPlasticity
    C::T
    ϕ::T        # in degrees for now
    ψ::T        # in degrees for now
    η_vp::T     # regularisation viscosity

    # computational parameters (precomputed, to speed up later calculations)
    sinϕ::T     # Friction angle
    cosϕ::T     # Friction angle
    sinΨ::T     # Dilation angle
    cosΨ::T     # Dilation angle
end

function compute_F(r::DruckerPrager, τII, P, λ̇)
    C, cosΦ, sinΦ, ηvp = r.C, r.cosΦ, r.sinΦ, r.η_vp 
    return F = τII - sinΦ * P  - C*cosΦ - λ̇*ηvp
end

function compute_Q(r::DruckerPrager, τ, P)     
    return τ - r.sinΨ * (P )  
end

function DruckerPrager(; C=10e6, ϕ=30.0, ψ=0.0, η_vp=1e0) 
    sinϕ = sind(ϕ) # Friction angle
    cosϕ = cosd(ϕ) # Friction angle
    sinΨ = sind(ψ) # Dilation angle
    cosΨ = cosd(ψ) # Dilation angle
    return DruckerPrager(C, ϕ, ψ, η_vp, sinϕ, cosϕ, sinΨ, cosΨ)
end

struct DruckerPragerCap{T} <: AbstractPlasticity
    C::T
    ϕ::T        # in degrees for now
    ψ::T        # in degrees for now
    η_vp::T     # regularisation viscosity
    Pt::T       # Tensile strength

    # computational parameters (precomputed, to speed up later calculations)
    sinϕ::T     # Friction angle
    cosϕ::T     # Friction angle
    sinΨ::T     # Dilation angle
    cosΨ::T     # Dilation angle

    k ::T
    kq::T
    c ::T 
    a ::T
    b ::T
    py::T 
    Ry::T 
    pd::T 
    τd::T 
    pq::T 
end

function ismode2_yield(v::DruckerPragerCap{_T}, τII::_T1, P::_T2)  where {_T,_T1,_T2}
    py, τd, pd = v.py, v.τd, v.pd
    return τII*(py - pd) >= τd*(py - P)
end

function ismode2_flowpotential(v::DruckerPragerCap{_T}, τII::_T1, P::_T2)  where {_T,_T1,_T2}
    pq, τd, pd = v.pq, v.τd, v.pd
    return τII*(pq - pd) >= τd*(pq - P)
end


function compute_F(r::DruckerPragerCap, τII, P, λ̇)
    k, c, py, a, Ry, ηvp  = r.k, r.c, r.py, r.a, r.Ry, r.η_vp 

    if ismode2_yield(r, τII, P)
        # Mode 2
        F = τII - k * (P)  - c # with fluid pressure (set to zero by default)
    else
        # Mode 1
        Rf   = sqrt(τII^2 + (P - py)^2)
        
        F    = a*(Rf - Ry)  
    end

    # Note that viscoplastic regularisation is taken into account in the residual function
    return F - λ̇*ηvp #*(F>-1e-8) 
end

function compute_Q(r::DruckerPragerCap, τ, P) 

    # These parameters are required to compute the constant in the plastic flow
    # potential. Note that this constant does not matter apart when plotting,
    # as we only need derivates of Q in general 
    Rf      = r.pq - r.Pt
    sd      = r.c + r.k*r.pd
    normvRf = sqrt((r.pd - r.pq)^2 + sd^2)/Rf
    pdf     = (r.pd - r.pq)/normvRf + r.pq
    sdf     = sd/normvRf

    if ismode2_flowpotential(r, τ, P) 
        cons =  sdf - r.kq*pdf 
        Q    =  τ - r.kq * (P )  - cons
    else 
        cons =  Rf 
        Rq   =  sqrt(τ^2 + (P - r.pq)^2)
        Q    =  r.b*(Rq - cons)  
    end
    return Q
end 

function DruckerPragerCap(; C=10e6, ϕ=30.0, ψ=0.0, η_vp=1e0, Pt=-1e5) 
    sinϕ = sind(ϕ) # Friction angle
    cosϕ = cosd(ϕ) # Friction angle
    sinΨ = sind(ψ) # Dilation angle
    cosΨ = cosd(ψ) # Dilation angle
    k    = sinϕ
    kq   = sinΨ
    c    = C*cosϕ
    a    = sqrt(1.0 + k^2)
    b    = sqrt(1.0 + kq^2)
    py   = (Pt + c/a)/(1-k/a)
    Ry   = py - Pt
    pd   = py - Ry*k/a
    τd   = k*pd + c
    pq   = pd + kq*τd
    return DruckerPragerCap(C, ϕ, ψ, η_vp, Pt, sinϕ, cosϕ, sinΨ, cosΨ, k, kq, c, a, b, py, Ry, pd, τd, pq)
end

function residual(x, ε̇II_eff, τII_trial, rheo, ηve, K, Pt_trial, Δt)
    τII, Pt, λ̇ = x[1], x[2], x[3]

    ∂Q∂p = ForwardDiff.derivative( x -> compute_Q(rheo,   x, Pt),  Pt )
    ∂Q∂τ = ForwardDiff.derivative( x -> compute_Q(rheo, τII,  x), τII )
    f    = compute_F(rheo, τII, Pt, λ̇)
    ΔPt  = -K*Δt*∂Q∂p*λ̇

    # r_τ = ε̇II_eff - τII/(2*ηve) - λ̇*∂Q∂τ/2
    r_τ = τII - (τII_trial - ηve*λ̇*∂Q∂τ)
    r_P = Pt - (Pt_trial + ΔPt)
    r_λ = (f>=0)*f + (f<0)*λ̇
    return @SVector [r_τ, r_P, r_λ]
end

function main(; nt=10)

    sc = (σ=1e7, t=1e10, L=1e3)
    ky = 1e3*365*24*3600

    # Time steps
    Δt     = 1e10/sc.t 

    Pt      = 1e6/sc.σ
    ε̇       = 2e-15.*sc.t

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -0.9*ε̇] )
    divV = tr(D_BC)
    ε̇kk  = 0.0
 
    rheo = DruckerPragerCap(C=10e6/sc.σ, ϕ=30.0, ψ=0.0, η_vp=0e20/sc.σ/sc.t, Pt=-1e5/sc.σ)
    G    = 3e10/sc.σ
    K    = 8e10/sc.σ
    η    = 1e24/sc.σ/sc.t
    ηve  = inv(1/η + 1/G)  

    P    = 1e6/sc.σ
    τ    = [0.0, 0.0, 0.0]

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
    Pt     = 0.0
    τxx, τyy, τxy =  0.0, 0.0, 0.0
    ΔPt = 0.0
    Δ   = (t=Δt,)

    for it=1:nt

        @printf("\nStep %04d\n", it)

        τxx0 = τxx
        τyy0 = τyy
        τxy0 = τxy
        P0   = P

        # Trial deviatoric stress
        ε̇xx_eff =( ε̇ -1/3*divV) + τxx0/(2*G*Δt)
        ε̇yy_eff =(-ε̇ -1/3*divV) + τyy0/(2*G*Δt)
        ε̇xy_eff =-0*ε̇ + τxy0/(2*G*Δt)

        τxx = 2*ηve*ε̇xx_eff
        τyy = 2*ηve*ε̇yy_eff
        τxy = 2*ηve*ε̇xy_eff
        P   = P0 - K*Δt*divV
        τII = sqrt(0.5*(τxx^2 + τyy^2 + τxy^2 + τxy^2))
        ε̇II_eff = sqrt(0.5*(ε̇xx_eff^2 + ε̇yy_eff^2 + ε̇xy_eff^2 + ε̇xy_eff^2)) 

        x = @SVector [τII, P, 0.0]
        
        r = residual(x, ε̇II_eff, τII, rheo, ηve, K, P, Δt)
        if norm(r)>1e-13
            for iter=1:50
                r = residual(x, ε̇II_eff, τII, rheo, ηve, K, P, Δt)
                J = ForwardDiff.jacobian(x -> residual(x, ε̇II_eff, τII, rheo, ηve, K, P, Δt), x )
                x -= J\r 
                @show r
            end
        end

        probes.τ[it]  = x[1]
        probes.Pt[it] = x[2]

        τ_ax = LinRange( 0.0/sc.σ, 4e7/sc.σ, 100)
        P_ax = LinRange(-5e6/sc.σ, 4e7/sc.σ, 100)
        F    = zeros(length(τ_ax), length(P_ax))
        for j in eachindex(P_ax), i in eachindex(τ_ax)
            F[i,j] = compute_F(rheo, τ_ax[i], P_ax[j], 0.0)
        end

        fig = Figure()
        ax = Axis(fig[1,1], aspect=DataAspect())
        contour!(ax, P_ax*sc.σ./1e6, τ_ax*sc.σ./1e6, F'; levels= [0.0])
        scatter!(ax, probes.Pt*sc.σ./1e6, probes.τ*sc.σ./1e6,)

        display(fig)

        @show probes.Pt*sc.σ
        @show probes.τ*sc.σ

    end

    

end

main()