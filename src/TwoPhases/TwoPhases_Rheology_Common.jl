using StaticArrays, LinearAlgebra, ForwardDiff

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + 1/2*(-x[1]-x[2])^2 + x[3]^2) 

function StrainRateTrial(τII, Pt, Pf, ηve, ηϕ, Kϕ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δt)
    ε̇II_trial = τII/2/ηve
    return ε̇II_trial
end

# function F(p::DruckerPrager{Vector{Float64}}, τII, p_eff, ϕ, λ̇, ph) 
#     c, cosϕ, sinϕ, ηvp = p.C[ph], p.cosϕ[ph], p.sinϕ[ph], p.ηvp[ph] 
#     return τII - p_eff*sinϕ - c*cosϕ - λ̇*ηvp
# end

# function Q(p::DruckerPrager{Vector{Float64}}, τII, p_eff, ϕ, λ̇, ph) 
#     c, cosψ, sinψ, ηvp = p.C[ph], p.cosψ[ph], p.sinψ[ph], p.ηvp[ph] 
#     return τII - p_eff*sinψ - c*cosψ - λ̇*ηvp
# end


function F(p::DruckerPrager{Vector{Float64}}, τII, P, ϕ, λ̇, ph)
    C, cosϕ, sinϕ, ηvp = p.C[ph], p.cosϕ[ph], p.sinϕ[ph], p.ηvp[ph] 
    return τII - sinϕ * P - C*cosϕ - λ̇*ηvp
end

function Q(p::DruckerPrager{Vector{Float64}}, τ, P, ϕ, λ̇, ph)   
    C, cosψ, sinψ, ηvp = p.C[ph], p.cosψ[ph], p.sinψ[ph], p.ηvp[ph]   
    return τ - sinψ * P - C*cosψ  
end

function F(p::Tensile{Vector{Float64}}, τII, P, ϕ, λ̇, ph)
    return τII - P - λ̇*p.ηvp[ph] + p.Pt[ph]
end

function Q(p::Tensile{Vector{Float64}}, τ, P, ϕ, λ̇, ph)     
    return τ - P + p.Pt[ph]
end

function ismode2_yield(v::DruckerPragerCap{Vector{Float64}}, τII::_T1, P::_T2, ph)  where {_T1,_T2}
    py, τd, pd = v.py[ph], v.τd[ph], v.pd[ph]
    return τII*(py - pd) >= τd*(py - P)
end

function ismode2_flowpotential(v::DruckerPragerCap{Vector{Float64}}, τII::_T1, P::_T2, ph)  where {_T1,_T2}
    pq, τd, pd = v.pq[ph], v.τd[ph], v.pd[ph]
    return τII*(pq - pd) >= τd*(pq - P)
end

function F(r::DruckerPragerCap{Vector{Float64}}, τII, P, ϕ, λ̇, ph)
    k, c, py, a, Ry, ηvp  = r.k[ph], r.c[ph], r.py[ph], r.a[ph], r.Ry[ph], r.ηvp[ph] 

    if ismode2_yield(r, τII, P, ph)
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

function Q(r::DruckerPragerCap{Vector{Float64}}, τ, P, ϕ, λ̇, ph) 
    # These parameters are required to compute the constant in the plastic flow
    # potential. Note that this constant does not matter apart when plotting,
    # as we only need derivates of Q in general 
    Rf      = r.pq[ph] - r.Pt[ph]
    sd      = r.c[ph] + r.k[ph]*r.pd[ph]
    normvRf = sqrt((r.pd[ph] - r.pq[ph])^2 + sd^2)/Rf
    pdf     = (r.pd[ph] - r.pq[ph])/normvRf + r.pq[ph]
    sdf     = sd/normvRf

    if ismode2_flowpotential(r, τ, P, ph) 
        cons =  sdf - r.kq[ph]*pdf 
        Q    =  τ - r.kq[ph] * (P )  - cons
    else 
        cons =  Rf 
        Rq   =  sqrt(τ^2 + (P - r.pq[ph])^2)
        Q    =  r.b[ph]*(Rq - cons)  
    end
    return Q
end 


@inline Af(p, pc, pt, γ) = (pc - pt) / (2 * π) * (2 * atan(γ * (pc + pt - 2p) / (2 * pc)) + π)
@inline Bf(p, pc, pt, M, C, α) = M * C * exp(α * (p - C) / (pc - pt))
@inline Cf(pc, pt, γ) = (pc - pt) / π * atan(γ / 2) + (pc + pt) / 2

yield_Golchin(τ, P, A, B, C, β, λ̇, ηvp) = B * (P - λ̇ * ηvp - C)^2 / A + A * (τ - λ̇ * ηvp - β * (P - λ̇ * ηvp))^2 / B - A * B

function F(r::Golchin2021{Vector{Float64}}, τ, P, ϕ, λ̇, ph)
    M, N, Pt, Pc, α, β, γ, ηvp  = r.M[ph], r.N[ph], r.Pt[ph], r.Pc[ph], r.a[ph], r.b[ph], r.c[ph], r.ηvp[ph] 
    C = Cf(Pc, Pt, γ)
    B = Bf(P, Pc, Pt, M, C, α)
    A = Af(P, Pc, Pt, γ)
    F = yield_Golchin(τ, P, A, B, C, β, λ̇, 0 * ηvp)
    return F - λ̇ * ηvp
end

function Q(r::Golchin2021{Vector{Float64}}, τ, P, ϕ, λ̇, ph)
    M, N, Pt, Pc, α, β, γ, ηvp  = r.M[ph], r.N[ph], r.Pt[ph], r.Pc[ph], r.a[ph], r.b[ph], r.c[ph], r.ηvp[ph] 
    C = Cf(Pc, Pt, γ)
    B = Bf(P, Pc, Pt, N, C, α)
    A = Af(P, Pc, Pt, γ)
    Q = yield_Golchin(τ, P, A, B, C, β, λ̇, 0 * ηvp)
    return Q
end

yield_Hyperbolic(τ, P, C, cosΨ, sinΨ, σT) = sqrt(τ^2 + (C * cosΨ - σT * sinΨ)^2) - (P * sinΨ + C * cosΨ)

function F(r::DruckerHyperbolic{Vector{Float64}}, τ, P, ϕ, λ̇, ph)
    C, cosϕ, sinϕ, σT, ηvp = r.C[ph], r.cosϕ[ph], r.sinϕ[ph], -r.Pt[ph], r.ηvp[ph]
    F = yield_Hyperbolic(τ, P, C, cosϕ, sinϕ, σT)
    return (F - λ̇ * ηvp) 
end

function Q(r::DruckerHyperbolic{Vector{Float64}}, τ, P, ϕ, λ̇, ph)
    C, cosψ, sinψ, σT = r.C[ph], r.cosψ[ph], r.sinψ[ph], -r.Pt[ph]
    Q = yield_Hyperbolic(τ, P, C, cosψ, sinψ, σT)
    return Q
end
