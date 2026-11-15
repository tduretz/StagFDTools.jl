using ForwardDiff

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + 1/2*(-x[1]-x[2])^2 + x[3]^2) 

function line(p, K, dt, η_ve, ψ, p1, t1)
    p2 = p1 + K * dt * sind(ψ)  # introduce sinϕ ?
    t2 = t1 - η_ve
    a = (t2 - t1) / (p2 - p1)
    b = t2 - a * p2
    return a * p + b
end

# Yield and Potential Functions ----------------------------------
yield_DruckerPrager(τ, P, C, cosΨ, sinΨ) = τ - C * cosΨ - P * sinΨ

function Yield(x, p, model::DruckerPrager1)
    (; C, cosϕ, sinϕ, cosψ, sinψ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    F = yield_DruckerPrager(τ, P, p.C, p.cosϕ, p.sinϕ)
    return (F - λ̇ * ηvp) * (F > ϵ) + (F < ϵ) * λ̇ * ηvp
end

function Potential(x, p, model::DruckerPrager1)
    (; C, cosϕ, sinϕ, cosψ, sinψ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    Q = yield_DruckerPrager(τ, P, C, cosψ, sinψ)
    return Q
end

yield_Hyperbolic(τ, P, C, cosΨ, sinΨ, σT) = sqrt(τ^2 + (C * cosΨ - σT * sinΨ)^2) - (P * sinΨ + C * cosΨ)

function Yield(x, p, model::DruckerHyperbolic)
    (; C, cosϕ, sinϕ, cosψ, sinψ, σT, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    F = yield_Hyperbolic(τ, P, C, cosϕ, sinϕ, σT)
    return (F - λ̇ * ηvp) * (F >= ϵ) + (F < ϵ) * λ̇ * ηvp
end

function Potential(x, p, model::DruckerHyperbolic)
    (; C, cosϕ, sinϕ, cosψ, sinψ, σT, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    Q = yield_Hyperbolic(τ, P, C, cosψ, sinψ, σT)
    return Q
end

@inline Af(p, pc, pt, γ) = (pc - pt) / (2 * π) * (2 * atan(γ * (pc + pt - 2p) / (2 * pc)) + π)
@inline Bf(p, pc, pt, M, C, α) = M * C * exp(α * (p - C) / (pc - pt))
@inline Cf(pc, pt, γ) = (pc - pt) / π * atan(γ / 2) + (pc + pt) / 2

yield_Golchin(τ, P, A, B, C, β, λ̇, ηvp) = B * (P - λ̇ * ηvp - C)^2 / A + A * (τ - λ̇ * ηvp - β * (P - λ̇ * ηvp))^2 / B - A * B

function Yield(x, p, model::Golchin2021)
    (; M, N, Pt, Pc, α, β, γ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    C = Cf(Pc, Pt, γ)
    B = Bf(P, Pc, Pt, M, C, α)
    A = Af(P, Pc, Pt, γ)
    F = yield_Golchin(τ, P, A, B, C, β, λ̇, 0 * ηvp)
    return (F - λ̇ * ηvp) * (F >= ϵ) + (F < ϵ) * λ̇ * ηvp
    # return (F)*(F>=ϵ) + (F<ϵ)*λ̇*ηvp
end

function Potential(x, p, model::Golchin2021)
    (; M, N, Pt, Pc, α, β, γ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    C = Cf(Pc, Pt, γ)
    B = Bf(P, Pc, Pt, N, C, α)
    A = Af(P, Pc, Pt, γ)
    Q = yield_Golchin(τ, P, A, B, C, β, λ̇, 0 * ηvp)
    return Q
end

# Residual -------------------------------------------
function ResidualDeviator(x, τ_trial, ε̇_eff, ηve, p, model)
    τ, P, λ̇ = x[1], x[2], x[3]
    ∂Q∂σ = ad_gradient(Potential, x, p, model)
    # return ε̇_eff -  τ/2/ηve  - λ̇/2*∂Q∂σ[1][1]
    return τ - τ_trial + ηve * λ̇ * ∂Q∂σ[1]
end

function ResidualVolume(x, P_trial, Dkk, P0, K, Δt, p, model)
    τ, P, λ̇ = x[1], x[2], x[3]
    ∂Q∂σ = ad_gradient(Potential, x, p, model)
    return P - P_trial + K * Δt * λ̇ * ∂Q∂σ[2]
end

function RheologyResidual(x, trial, plastic, model)
    τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt = trial
    return @SVector([
        ResidualDeviator(x, τ_trial, ε̇_eff, ηve, plastic, model),
        ResidualVolume(x, P_trial, Dkk, P0, K, Δt, plastic, model),
        Yield(x, plastic, model),
    ])
end

function bt_line_search(Δx, J, x, r, trial, plastic, model; α=1.0, ρ=0.5, c=1.0e-4, α_min=1.0e-8)
    # Borrowed from RheologicalCalculator
    perturbed_x = @. x + α * Δx
    perturbed_r = RheologyResidual(x, trial, plastic, model)

    J_times_Δx = -J * Δx
    while sqrt(sum(perturbed_r .^ 2)) > sqrt(sum((r + (c * α * (J_times_Δx))) .^ 2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_x = @. x + α * Δx
        perturbed_r = RheologyResidual(x, trial..., plastic, model)
    end
    return α
end

# Return mapping functions ------------------------------------
function NonLinearReturnMapping(τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, plastic, model)

    tol = 1e-5
    λ̇ = zero(τII)
    K = 1 / β
    τ_trial = τII
    P_trial = P
    itermax = 100

    T = typeof(τII)
    x = SVector{3,T}(τII, P, λ̇)
    αvec = @SVector([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    Fvec = MVector{length(αvec),T}(zeros(T, length(αvec)))

    trial = (τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt)

    R = RheologyResidual(x, trial, plastic, model)
    nR = abs(R[3])#norm(R)
    iter, nR0 = 0, nR
    R0 = copy(R)

    while nR > tol && (nR / nR0) > tol && iter < itermax

        iter += 1
        x0 = copy(x)
        R, J = ad_value_and_jacobian(RheologyResidual, x, trial, plastic, model)
        δx = -J \ R
        nR = abs(R[3])

        # x .= x0 .+  1*δx

        # α = bt_line_search(δx, J.derivs[1], x0, J.val, trial, plastic, model)
        # x .= x0 .+  α*δx

        for ils in eachindex(αvec)
            x = @. x0 + αvec[ils] * δx
            R = RheologyResidual(x, trial, plastic, model)
            Fvec[ils] = norm(ForwardDiff.value.(R))
        end
        # ibest = argmin(Fvec)
        # x .= x0 .+  αvec[ibest]*δx

        # @show iter, nR,  αvec[ibest], x

        # if isnan(norm(δx))
        #     @show R0
        #     @show J.val
        #     @show J.derivs[1]
        #     @show δx
        #     @show iter, nR,  αvec[ibest]
        #     error()
        # end
    end

    if iter == itermax && (nR > tol && (nR / nR0) > tol)
        R = RheologyResidual(x, trial, plastic, model)
        @warn "Failed return mapping after $iter iterations"
        @show τII, P, ε̇_eff, ηve, β
        @show nR, nR0, nR / nR0, tol
        @show trial
        @show plastic
        @show R0
        @show R
        @show x
        # Relax tolerance and retry once
        tol_relax = tol * 100
        if nR > tol_relax || (nR / nR0) > tol_relax
            error("Failed return mapping")
        end
    end

    if x[1] < 0
        @show R, x
        error()
    end

    return x[1], x[2], x[3]
end

function Kiss2023ReturnMapping(τ, P, η_ve, comp, β, Δt, C, φ, ψ, ηvp, σ_T, δσ_T, pc1, τc1, pc2, τc2)
    K = 1 / β
    λ̇ = zero(τ)
    Pc = P
    τc = τ
    l1 = line(P, K, Δt, η_ve, 90.0, pc1, τc1)
    l2 = line(P, K, Δt, η_ve, 90.0, pc2, τc2)
    l3 = line(P, K, Δt, η_ve, ψ, pc2, τc2)
    if max(τ - P * sind(φ) - C * cosd(φ), τ - P - σ_T, -P - (σ_T - δσ_T)) > 0.0
        if τ <= τc1
            # pressure limiter 
            dqdp = -1.0
            f = -P - (σ_T - δσ_T)
            λ̇ = f / (K * Δt)
            τc = τ
            Pc = P - K * Δt * λ̇ * dqdp
            f = -Pc - (σ_T - δσ_T)
            domain_pl = 1.0
        elseif τc1 < τ <= l1
            # corner 1 
            τc = τ - η_ve * (τ - τc1) / (η_ve + ηvp)
            Pc = P - K * Δt * (P - pc1) / (K * Δt + ηvp)
            domain_pl = 2.0
        elseif l1 < τ <= l2            # mode-1
            # tension
            dqdp = -1.0
            dqdτ = 1.0
            f = τ - P - σ_T
            λ̇ = f / (K * Δt + η_ve + ηvp)
            τc = τ - η_ve * λ̇ * dqdτ
            Pc = P - K * Δt * λ̇ * dqdp
            domain_pl = 3.0
        elseif l2 < τ <= l3 # 2nd corner
            # corner 2
            τc = τ - η_ve * (τ - τc2) / (η_ve + ηvp)
            Pc = P - K * Δt * (P - pc2) / (K * Δt + ηvp)
            domain_pl = 4.0
        elseif l3 < τ
            # Drucker-Prager
            dqdp = -sind(ψ)
            dqdτ = 1.0
            f = τ - P * sind(φ) - C * cosd(φ)
            λ̇ = f / (K * Δt * sind(φ) * sind(ψ) + η_ve + ηvp)
            τc = τ - η_ve * λ̇ * dqdτ
            Pc = P - K * Δt * λ̇ * dqdp
            domain_pl = 5.0
        end
    end
    return τc, Pc, λ̇
end

function AnalyticalReturnMapping(τII, P, ηve, comp, β, Δt, C, cosϕ, sinϕ, sinψ, ηvp)
    λ̇ = zero(τII)
    F = τII - C * cosϕ - P * sinϕ - λ̇ * ηvp
    if F > 1e-10
        λ̇ = F / (ηve + ηvp + comp * Δt / β * sinϕ * sinψ)
        τII -= λ̇ * ηve
        P += comp * λ̇ * sinψ * Δt / β
        F = τII - C * cosϕ - P * sinϕ - λ̇ * ηvp
        (F > 1e-10) && error("Failed return mapping")
    end
    return τII, P, λ̇
end

function TensileReturnMapping(τII, P, ηve, comp, β, Δt, σT, ηvp)
    λ̇ = zero(τII)
    F = τII - σT - P - λ̇ * ηvp
    if F > 1e-10
        λ̇ = F / (ηve + ηvp + comp * Δt / β)
        τII -= λ̇ * ηve
        P += comp * λ̇ * Δt / β
        F = τII - σT - P - λ̇ * ηvp
        (F > 1e-10) && error("Failed return mapping")
        (τII < 0.0) && error("Plasticity without condom")
    end
    return τII, P, λ̇
end

# Drucker-Prager with tension cap: same yield function and flow potential as DruckerPragerCap in the two-phase
# routines (mode 2: Drucker-Prager line, mode 1: circle tangent to it and passing through the tensile strength Pt)
function yield_Cap(τ, P, p)
    (; k, c, py, Ry, pd, τd, a, Pt) = p
    if τ * (py - pd) >= τd * (py - P)
        return τ - k * P - c # mode 2
    else
        # mode 1: a(Rf - Ry) written as a(Rf² - Ry²)/(Rf + Ry), no cancellation when C (and py, Ry) is huge
        Rf = sqrt(τ^2 + (P - py)^2)
        return a * (τ^2 + (P - Pt) * (P + Pt - 2py)) / (Rf + Ry)
    end
end

# Derivatives of the flow potential (∂Q/∂τ, ∂Q/∂P)
function ∂Q_Cap(τ, P, p)
    (; kq, b, pq, pd, τd) = p
    if τ * (pq - pd) >= τd * (pq - P)
        return one(τ), -kq # mode 2
    else
        Rq = sqrt(τ^2 + (P - pq)^2) # mode 1
        return b * τ / Rq, b * (P - pq) / Rq
    end
end

function CapResidual(x, τ_trial, P_trial, ηve, KΔt, p)
    τ, P, λ̇ = x[1], x[2], x[3]
    ∂Q∂τ, ∂Q∂P = ∂Q_Cap(τ, P, p)
    return SVector(τ - τ_trial + ηve * λ̇ * ∂Q∂τ,
                   P - P_trial + KΔt * λ̇ * ∂Q∂P,
                   yield_Cap(τ, P, p) - λ̇ * p.ηvp)
end

# Local Newton on (τII, P, λ̇), converged to machine precision so that the global Newton is not polluted.
# Backtracking line search on |R|: full Newton steps can cycle between the cap and the Drucker-Prager branches
# (e.g. trial stress far in tension)
function CapReturnMapping(τII, P, ηve, comp, β, Δt, p; α_min=1e-8)
    # Common number type of the unknowns and of the parameters (implicit hardening: p depends on dual numbers), so that the
    # local jacobian nests its dual numbers outside those of p
    Tp = promote_type(typeof(τII), typeof(P), typeof(ηve), typeof(p.k), typeof(p.c), typeof(p.py), typeof(p.pq))
    τII, P, ηve = Tp(τII), Tp(P), Tp(ηve)
    yield_Cap(τII, P, p) <= 0 && return τII, P, zero(τII)
    KΔt = comp * Δt / β
    x   = SVector(τII, P, zero(τII))
    tol = 1e-13 * (abs(τII) + abs(P)) + 1e-30
    for _ in 1:100
        R, J = fd_value_and_jacobian(CapResidual, x, τII, P, ηve, KΔt, p)
        maximum(abs, R) < tol && return x[1], x[2], x[3]
        δx  = -J \ R
        nR  = sum(abs2, R)
        α   = one(nR)
        while α > α_min
            sum(abs2, CapResidual(x + α * δx, τII, P, ηve, KΔt, p)) < (1 - 1e-4 * α) * nR && break # sufficient decrease
            α /= 2
        end
        x = x + α * δx
    end
    R = CapResidual(x, τII, P, ηve, KΔt, p)
    maximum(abs, R) < 1e3 * tol || error("Failed cap return mapping: τII = $τII, P = $P, R = $R")
    return x[1], x[2], x[3]
end

# Return mapping --------------------------------------------
return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, ::NoPlasticity, phases) = τII, P, 0.0

# εp: accumulated plastic strain (∫λ̇dt) of the cell, only used by plasticity models with strain hardening
return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::AbstractPlasticity, phases, εp) =
    return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl, phases)

# Parameters of the cap model of phase ph. With strain hardening (Δϕ or ΔC ≠ 0), the friction angle is ϕ + Δϕ tanh(εp/εh)
# and the cohesion C + ΔC tanh(εp/εh), and the quantities derived from them are recomputed (same as preprocess!);
# εp: start-of-step value (lagged); see HardeningReturnMapping for the implicit (end-of-step) update
@inline function cap_parameters(pl::DruckerPragerCap, ph, εp)
    # No hardening: εp = 0, hardening fields not set (DruckerPragerCap built without initialize) or Δϕ = ΔC = 0
    if εp <= 0 || isempty(pl.Δϕ) || (iszero(pl.Δϕ[ph]) && iszero(pl.ΔC[ph]))
        return (k=pl.k[ph], kq=pl.kq[ph], c=pl.c[ph], a=pl.a[ph], b=pl.b[ph], py=pl.py[ph], Ry=pl.Ry[ph],
                pd=pl.pd[ph], τd=pl.τd[ph], pq=pl.pq[ph], Pt=pl.Pt[ph], ηvp=pl.ηvp[ph])
    end
    return hardened_parameters(pl, ph, εp)
end

# Cap parameters for the plastic strain εp (strain hardening/softening active)
@inline function hardened_parameters(pl::DruckerPragerCap, ph, εp)
    h  = tanh(max(εp, zero(εp)) / pl.εh[ph])
    ϕ  = pl.ϕ[ph] + pl.Δϕ[ph] * h
    k  = sind(ϕ)
    c  = (pl.C[ph] + pl.ΔC[ph] * h) * cosd(ϕ)
    a  = sqrt(1.0 + k^2)
    Pt = pl.Pt[ph]
    py = (Pt + c / a) / (1 - k / a)
    Ry = py - Pt
    pd = py - Ry * k / a
    τd = k * pd + c
    pq = pd + pl.kq[ph] * τd
    return (k=k, kq=pl.kq[ph], c=c, a=a, b=pl.b[ph], py=py, Ry=Ry, pd=pd, τd=τd, pq=pq, Pt=Pt, ηvp=pl.ηvp[ph])
end

return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerPragerCap, phases) =
    return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl, phases, 0.0)

# Strain hardening/softening: implicit (end-of-step εp, default) or lagged (start-of-step εp)
const implicit_hardening = Ref(true)
@inline hardening_on(pl::DruckerPragerCap, ph) = !isempty(pl.Δϕ) && !(iszero(pl.Δϕ[ph]) && iszero(pl.ΔC[ph]))

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerPragerCap, phases, εp)
    if implicit_hardening[] && hardening_on(pl, phases)
        return HardeningReturnMapping(τII, P, ηvep, comp, β, Δt, pl, phases, εp)
    end
    return CapReturnMapping(τII, P, ηvep, comp, β, Δt, cap_parameters(pl, phases, εp))
end

# Implicit strain hardening/softening: the cap parameters are those of the end-of-step plastic strain εp, solution of
#     g(εp) = εp - εp0 - Δt ε̇II_pl(εp) = 0,   ε̇II_pl(εp) = (τ_trial - τ(εp)) / (2ηve)
# where τ(εp) is the return mapping (CapReturnMapping, unchanged) with the parameters of εp, and ε̇II_pl the plastic part of
# the deviatoric strain rate invariant (as phase_vep). Since 0 ≤ ε̇II_pl ≤ τ_trial/(2ηve), the root is bracketed by
# [εp0, εp0 + Δt τ_trial/(2ηve)]: safeguarded Newton (bisection when the Newton step leaves the bracket, e.g. strong
# softening). A last Newton step makes the derivatives (dual numbers) those of the implicit function: the AD tangent of the
# global Newton is consistent.
_hardening_g(ε, εp0, τII, P, ηve, comp, β, Δt, pl, ph) =
    ε - εp0 - Δt * (τII - CapReturnMapping(τII, P, ηve, comp, β, Δt, hardened_parameters(pl, ph, ε))[1]) / (2ηve)

function HardeningReturnMapping(τII, P, ηve, comp, β, Δt, pl, ph, εp0; tol=1e-12, niter=50)
    p0 = cap_parameters(pl, ph, εp0)
    yield_Cap(τII, P, p0) <= 0 && return τII, P, zero(τII) # elastic trial with the start-of-step parameters
    lo, hi = εp0, εp0 + Δt * τII / (2ηve)
    ε = εp0 + Δt * (τII - CapReturnMapping(τII, P, ηve, comp, β, Δt, p0)[1]) / (2ηve) # lagged estimate
    scale = Δt * τII / (2ηve) + 1e-30
    for _ in 1:niter
        g, dg = fd_value_and_derivative(_hardening_g, ε, εp0, τII, P, ηve, comp, β, Δt, pl, ph)
        abs(g) < tol * scale && break
        g > 0 ? (hi = ε) : (lo = ε)
        εn = ε - g / dg
        ε = (dg > 0 && lo < εn < hi) ? εn : (lo + hi) / 2
    end
    g, dg = fd_value_and_derivative(_hardening_g, ε, εp0, τII, P, ηve, comp, β, Δt, pl, ph)
    dg > 0 && (ε -= g / dg) # implicit-function derivatives
    return CapReturnMapping(τII, P, ηve, comp, β, Δt, hardened_parameters(pl, ph, ε))
end

# Ω: rate-and-state variable of the cell, only used by plasticity models with rate-and-state friction
return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::AbstractPlasticity, phases, εp, Ω) =
    return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl, phases, εp)

rate_state_on(pl::DruckerPragerCap, ph) = !isempty(pl.aRS) && pl.aRS[ph] > 0

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerPragerCap, phases, εp, Ω)
    p = cap_parameters(pl, phases, εp)
    # Rate-and-state on the Drucker-Prager branch (mode 2); the tension cap (mode 1) keeps the static parameters
    if rate_state_on(pl, phases) && τII * (p.py - p.pd) >= p.τd * (p.py - P)
        rs = (a=pl.aRS[phases], b=pl.bRS[phases], V0=pl.V0RS[phases], D=pl.DRS[phases], η=pl.ηRS[phases])
        return RateStateReturnMapping(τII, P, ηvep, comp, β, Δt, p, rs, Ω)
    end
    return return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl, phases, εp) # (implicit) hardening
end

# Rate-and-state friction coefficient: μ = a asinh(A λ̇), A = D/(2V0) exp((μ0 + bΩ)/a), V = Dλ̇.
# Steady state (Ω = ln(V0/V)): μ ≈ μ0 + (a - b) ln(V/V0)
@inline rate_state_A(μ0, rs, Ω) = rs.D / (2 * rs.V0) * exp((μ0 + rs.b * Ω) / rs.a)

# Weight of the static regularisation ηvp: 1 for P ≤ 0, smoothly → 0 at P = c (smoothstep)
@inline function rate_state_wvp(P, c)
    x = clamp(1 - P / c, zero(P), one(P))
    return x^2 * (3 - 2x)
end

# Return mapping with the rate-and-state friction coefficient in place of k = sinϕ (Drucker-Prager branch, flow potential slope kq):
#   τ = τII - ηve λ̇,  P = P_trial + KΔt kq λ̇,  f(λ̇) = τ - c - F(P, λ̇) - η λ̇ = 0,
#   F = P a asinh(A λ̇) for P ≥ 0, k P for P < 0 (static Drucker-Prager in tension: continuous at P = 0),
#   η = ηRS + w(P_trial) ηvp: radiation damping in compression, static regularisation for P_trial ≤ 0 (w = 1, smooth up to P_trial = c),
#   so that the mapping is continuous with the static cap at the mode 1/2 boundary.
# There is no yield check: λ̇ > 0 as soon as f(0) > 0 (hypoplastic, the creep rate is negligible well below μ0 P + c).
# f is decreasing in λ̇ and bracketed by f(0) > 0, f(f(0)/ηve) < 0: safeguarded Newton on s = ln λ̇,
# then plain Newton steps so that the derivatives (dual numbers of the global tangent) are converged as well.
function RateStateReturnMapping(τII, P, ηve, comp, β, Δt, p, rs, Ω)
    (; c, k, kq) = p
    KΔt = comp * Δt / β
    A   = rate_state_A(k, rs, Ω)
    η   = rs.η + rate_state_wvp(P, c) * p.ηvp
    Pλ(λ̇) = P + KΔt * kq * λ̇
    F(λ̇)  = (Pl = Pλ(λ̇); Pl >= 0 ? Pl * rs.a * asinh(A * λ̇) : k * Pl)
    dF(λ̇) = (Pl = Pλ(λ̇); Pl >= 0 ? KΔt * kq * rs.a * asinh(A * λ̇) + Pl * rs.a * A / sqrt(1 + (A * λ̇)^2) : k * KΔt * kq)
    f(λ̇)  = (τII - ηve * λ̇) - c - F(λ̇) - η * λ̇
    df(λ̇) = -ηve - dF(λ̇) - η
    f0 = f(zero(τII))
    f0 <= 0 && return τII, P, zero(τII)
    g(s)  = f(exp(s))
    dg(s) = df(exp(s)) * exp(s)
    s_hi = log(f0 / ηve)                      # f < 0
    # μ ≈ aAλ̇ ≪ 1: f ≈ f0 - (ηve + aA|P| + η + kKΔt kq) λ̇ ≥ f0/2 > 0 (f0 can be tiny)
    s_lo = min(s_hi - 1, log(1e-3 / A), log(f0 / (2 * (ηve + rs.a * A * abs(P) + η + k * KΔt * kq))))
    for _ in 1:20
        g(s_lo) > 0 && break
        s_lo -= 5
    end
    # Initial guess: λ̇ at the trial stress, sinh(f0/(aP))/A ≈ exp(f0/(aP))/(2A)
    s   = P > 0 ? clamp(f0 / (rs.a * P) - log(2A), s_lo, s_hi) : (s_lo + s_hi) / 2
    tol = 1e-13 * (abs(τII) + abs(P) + abs(c))
    for _ in 1:200
        gs = g(s)
        abs(gs) < tol && break
        gs > 0 ? (s_lo = s) : (s_hi = s)
        s_new = s - gs / dg(s)
        s = (s_lo < s_new < s_hi) ? s_new : (s_lo + s_hi) / 2
        s_hi - s_lo < 1e-15 * max(1, abs(s)) && break
    end
    for _ in 1:2 # derivative propagation
        s -= g(s) / dg(s)
    end
    abs(g(s)) < 1e3 * tol || error("Failed rate-and-state return mapping: τII = $τII, P = $P, Ω = $Ω, f = $(g(s))")
    λ̇ = exp(s)
    return τII - ηve * λ̇, Pλ(λ̇), λ̇
end

"""
    Ω = update_rate_state(Ω_old, V, Δt, L, V0)

Aging-law update of the rate-and-state variable Ω = ln(V0θ/L) over Δt at constant slip rate V (exact integration of
dθ/dt = 1 - Vθ/L, as `update_Ω` in RheologyCalculator.jl).
"""
@inline function update_rate_state(Ω_old, V, Δt, L, V0)
    if V * Δt / L <= 1e-6
        Θ = exp(Ω_old) * (1 - V * Δt / L) + V0 * Δt / L
    else
        Θ = V0 / V + (exp(Ω_old) - V0 / V) * exp(-V * Δt / L)
    end
    return log(max(Θ, 1e-300))
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::VonMises, phases)
    return AnalyticalReturnMapping(τII, P, ηvep, comp, β, Δt,
        pl.C[phases], pl.cosϕ[phases], 0.0, 0.0, pl.ηvp[phases])
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerPrager, phases)
    return AnalyticalReturnMapping(τII, P, ηvep, comp, β, Δt,
        pl.C[phases], pl.cosϕ[phases], pl.sinϕ[phases], pl.sinψ[phases], pl.ηvp[phases])
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerPrager1, phases)
    p = (C=pl.C[phases], cosϕ=pl.cosϕ[phases], sinϕ=pl.sinϕ[phases], sinψ=pl.sinψ[phases], cosψ=pl.cosψ[phases], ηvp=pl.ηvp[phases])
    return NonLinearReturnMapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, p, DruckerPrager1())
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerHyperbolic, phases)
    p = (C=pl.C[phases], cosϕ=pl.cosϕ[phases], sinϕ=pl.sinϕ[phases], sinψ=pl.sinψ[phases], cosψ=pl.cosψ[phases], σT=pl.σT[phases], ηvp=pl.ηvp[phases])
    return NonLinearReturnMapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, p, DruckerHyperbolic())
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::Golchin2021, phases)
    Pt = -pl.σT[phases]
    p = (M=pl.M[phases], N=pl.N[phases], Pt, Pc=pl.Pc[phases], α=pl.a[phases], β=pl.b[phases], γ=pl.c[phases], ηvp=pl.ηvp[phases])
    return NonLinearReturnMapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, p, Golchin2021())
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::Kiss2023, phases)
    return Kiss2023ReturnMapping(τII, P, ηvep, comp, β, Δt,
        pl.C[phases], pl.ϕ[phases], pl.ψ[phases], pl.ηvp[phases],
        pl.σT[phases], pl.δσT[phases], pl.P1[phases], pl.τ1[phases], pl.P2[phases], pl.τ2[phases])
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::Tensile, phases)
    return TensileReturnMapping(τII, P, ηvep, comp, β, Δt, pl.σT[phases], pl.ηvp[phases])
end

# Strain rate trial ------------------------------------------
function StrainRateTrial(τII, G, Δt, B, n)
    ε̇II_vis = B * τII^n
    ε̇II_trial = ε̇II_vis + τII / (2 * G * Δt)
    return ε̇II_trial
end

# Phase mixing in interface cells (materials.mixing): :default, :voigt (Rheology_Voigt.jl), :reuss (Rheology_Reuss.jl),
# :isowork (Rheology_IsoWork.jl, per-phase stresses as Voigt)
@inline _mixing(materials) = hasproperty(materials, :mixing) ? materials.mixing : :default
# Schemes with per-phase stresses (τ0 per phase on the grid and on the markers, Rheology_Voigt.jl)
@inline _per_phase_stress(materials) = _mixing(materials) === :voigt || _mixing(materials) === :isowork
# Per-phase old pressure (materials.pressure = :phase, with per-phase stresses only)
@inline _phase_pressure(materials) = hasproperty(materials, :pressure) && materials.pressure === :phase

# Plastic strain (strain hardening) in interface cells (materials.plastic_strain):
#   :mixed  one εp per grid point, εp += Δt Σᵢφᵢε̇IIplᵢ, used by all phases
#   :phase  one εpᵢ per phase,     εpᵢ += Δt ε̇IIplᵢ,     phase i hardens with its own εpᵢ
# εp = ∫ε̇II_pl dt: invariant of the plastic part of the deviatoric strain rate (phase_vep), carried by the markers
# (plastic_strain_ToParticles!, plastic_strain_ToGrid!)
@inline _plastic_strain(materials) = hasproperty(materials, :plastic_strain) ? materials.plastic_strain : :mixed

# Plastic strain seen by the phases at a grid point: the local rheology gets either a number (shared by all phases) or
# a PhaseStrain, and phase p reads _εp(εp, p). One concrete type for both modes (no type instability in the kernels).
struct PhaseStrain{V}
    mix::Float64            # :mixed: value of the grid point
    ph::V                   # :phase: per-phase fields (Vector of arrays)
    I::CartesianIndex{2}
    phase::Bool
end
@inline _εp(εp::Real, p) = εp
@inline _εp(εp::PhaseStrain, p) = εp.phase ? εp.ph[p][εp.I] : εp.mix

# Per-phase plastic strain rates ε̇IIplᵢ at a grid point (default mixing: each phase sees the effective strain rate built with
# the grid G), written to ε̇plph[p][I] (0 where phase p is absent). ε̇: effective strain rate (ϵ̇xx, ϵ̇yy, ϵ̇xy, P).
function phase_plastic_rates_default!(ε̇plph, I, ε̇, Dkk, P0, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0)
    ε̇II = sqrt((ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1] - ε̇[2])^2) / 2 + ε̇[3]^2) + 1e-17
    for p in eachindex(materials.n)
        ε̇plph[p][I] = phase_ratios[p] < eps() ? 0.0 : phase_vep(ε̇II, ε̇[4], Dkk, P0, Ggrid, materials, p, Δ, Tk, _εp(εp, p), Ω)[4]
    end
    return nothing
end

# Dispatch on materials.mixing (shared-τ0 schemes; :voigt and :isowork: see TangentOperator_Voigt!)
function phase_plastic_rates!(ε̇plph, I, ε̇, Dkk, P0, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0)
    if _mixing(materials) === :reuss
        phase_plastic_rates_partition!(ε̇plph, I, ReussRule(), ε̇, Dkk, P0, materials, phase_ratios, Δ, Tk, εp, Ω)
    else
        phase_plastic_rates_default!(ε̇plph, I, ε̇, Dkk, P0, Ggrid, materials, phase_ratios, Δ, Tk, εp, Ω)
    end
    return nothing
end

# Phase average ----------------------------------------------
function PhaseAverage_summand(a, phase_ratio, averaging)
    if averaging === :harmonic && a != 0.0
        # Hⱼ = w′ᵢ * aᵢ⁻¹
        a_j = phase_ratio / a
    elseif averaging === :geometric && a > 0.0
        # Gⱼ = w′ᵢ * ln(aᵢ)
        a_j = phase_ratio * log(a)
    else # arithmetic
        # Aⱼ =w′ᵢ * aᵢ
        a_j = phase_ratio * a
    end
    return a_j
end

function PhaseAverage(a_average, averaging)
    if averaging === :harmonic && a_average != 0.0
        # H = (Σⁿᵢ₌₁ w′ᵢ * aᵢ⁻¹)⁻¹ = (Σⁿᵢ₌₁ Hⱼ)⁻¹
        a_avg = 1 / a_average
    elseif averaging === :geometric
        # G = exp(Σⁿᵢ₌₁ w′ᵢ * ln(aᵢ)) = exp(Σⁿᵢ₌₁ Gⱼ)
        a_avg = exp(a_average)
    else # arithmetic
        # A = Σⁿᵢ₌₁ w′ᵢ * aᵢ = Σⁿᵢ₌₁ Aⱼ
        a_avg = a_average
    end
    return a_avg
end

# Tk: temperature [K] for temperature-dependent creep (nothing: η0 used as is)
# εp: accumulated plastic strain (∫λ̇dt) for strain hardening (see cap_parameters)
# Ω: rate-and-state variable (see RateStateReturnMapping)
function LocalRheology(ε̇, Dkk, P0, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0; phase_avg=:arithmetic)

    nphases = length(materials.n)
    eps0 = 1e-17

    # Effective strain rate & pressure
    ε̇II = sqrt((ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1] - ε̇[2])^2) / 2 + ε̇[3]^2) + eps0
    P = ε̇[4]

    η_average = zero(ε̇II)
    λ̇_average = zero(ε̇II)
    P_average = zero(ε̇II)
    τ_average = zero(ε̇II)

    for phases = 1:nphases

        phase_ratios[phases] < eps() && continue

        # P = P_trial
        # Parameters
        ϵ = 1e-10 # tolerance
        n = materials.n[phases]
        η0 = materials.η0[phases]
        B = materials.B[phases]
        G = Ggrid
        β = materials.β[phases]
        comp = materials.compressible
        if Tk !== nothing
            f  = arrhenius_factor(materials.Ea[phases], n, Tk, materials.Tref[phases])
            η0 = η0 * f
            B  = B * f^(-n)
        end

        # Initial guess
        η = (η0 .* ε̇II .^ (1 ./ n .- 1.0))[1]
        ηvep = inv(1 / η + 1 / (G * Δ.t))
        τII = 2 * ηvep * ε̇II
        # P = P0 - comp * Δ.t / β * P_trial

        # Visco-elastic powerlaw
        for it = 1:20
            r = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
            # @show abs(r)
            (abs(r) < ϵ) && break
            _, ∂ε̇II∂τII = fd_value_and_derivative(StrainRateTrial, τII, G, Δ.t, B, n)
            ∂τII∂ε̇II = inv(∂ε̇II∂τII)
            τII += ∂τII∂ε̇II * r
        end
        isnan(τII) && error()

        # ηvep for analytical solution
        ηvep = τII / 2 / ε̇II

        # Viscoplastic return mapping (from the trial pressure P of the cell: Pp is the corrected pressure of this phase only;
        # overwriting P would make the next phases start from it and compound the dilatancy correction in mixed cells)
        τII, Pp, λ̇ = return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δ.t, comp, materials.plasticity, phases, _εp(εp, phases), Ω)

        # Effective viscosity
        ηvep = τII / (2 * ε̇II)

        # Phase averaging
        η_average += PhaseAverage_summand(ηvep, phase_ratios[phases], phase_avg)
        P_average += PhaseAverage_summand(Pp, phase_ratios[phases], phase_avg)
        λ̇_average += PhaseAverage_summand(λ̇, phase_ratios[phases], phase_avg)
        τ_average += PhaseAverage_summand(τII, phase_ratios[phases], phase_avg)
    end

    η_average = PhaseAverage(η_average, phase_avg)
    P_average = PhaseAverage(P_average, phase_avg)
    λ̇_average = PhaseAverage(λ̇_average, phase_avg)
    τ_average = PhaseAverage(τ_average, phase_avg)

    return η_average, λ̇_average, P_average, τ_average
end

function StressVector!(ε̇::SVector{N,T}, ε̇kk, P0, G, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0) where {N,T}
    η, λ̇, P, τII = _mixing(materials) === :reuss ?
        LocalRheology_Reuss(ε̇, ε̇kk, P0, G, materials, phase_ratios, Δ, Tk, εp, Ω) :
        LocalRheology(ε̇, ε̇kk, P0, G, materials, phase_ratios, Δ, Tk, εp, Ω)
    τ = @SVector([2 * η * ε̇[1],
        2 * η * ε̇[2],
        2 * η * ε̇[3],
        P])
    return τ, η, λ̇, τII
end

