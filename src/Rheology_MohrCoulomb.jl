# Mohr-Coulomb elasto-viscoplasticity (plane strain, out-of-plane stress τzz = -τxx - τyy carried through the deviator).
#
# The yield surface is not a circle in the deviatoric plane, so the stress is not coaxial with the effective strain rate
# (unlike Drucker-Prager, where the return is a scaling of τII): tensorial return mapping on x = (τxx, τyy, τxy, P, λ̇),
#   τ = τ_trial - 2ηve λ̇ dev(∂Q/∂σ),   P = P_trial + KΔt λ̇ tr(∂Q/∂σ),   F(σ) - ηvp λ̇ = 0,
# σ = τ - P I (tension positive), Q = F with ψ in place of ϕ (and no cohesion). For Q = Drucker-Prager this reduces to the
# scalar mappings of Rheology.jl (ε̇II_pl = λ̇/2, ε̇kk_pl = λ̇ sinψ).
#
# Yield functions (plasticity.θt):
#   θt = 0: exact Mohr-Coulomb from the principal stresses (de Borst, 1990), F = (σ1 - σ3)/2 + (σ1 + σ3)/2 sinϕ - C cosϕ,
#           σ1 ≥ σ2 ≥ σ3 (the in-plane principal stresses and σzz), corners at the triaxial states (σ2 = σ1 or σ3)
#   θt > 0: Abbo & Sloan (1995), F = k(θ) τII - P sinϕ - C cosϕ, k = cosθ - sinθ sinϕ/√3 for |θ| ≤ θt, k = A - B sin3θ above
#           (smooth corners), θ ∈ [-30°, 30°] the Lode angle

# Voigt σ = (σxx, σyy, σzz, σxy). q = k(θ)τII is the deviatoric part of the Mohr-Coulomb function, F = q - P sinϕ - C cosϕ.
# Values and analytic gradients ∂q/∂σ (Voigt: the xy entry is ∂q/∂σxy = 2(∂q/∂σ)xy); the gradients are deviatoric.

# θt = 0: principal stresses, q = (σ1 - σ3)/2 + ((σ1 + σ3)/2 + P) sinϕ
@inline function mc_q_principal(σ, s)
    T    = promote_type(eltype(σ), typeof(s))
    d    = (σ[1] - σ[2]) / 2
    R    = sqrt(d^2 + σ[4]^2 + T(1e-40))
    σc   = (σ[1] + σ[2]) / 2
    σa, σb = σc + R, σc - R
    o, z, h = one(T), zero(T), T(0.5)
    ga   = SVector(h + d / (2R), h - d / (2R), z, σ[4] / R)       # ∂σa/∂σ
    gb   = SVector(h - d / (2R), h + d / (2R), z, -σ[4] / R)      # ∂σb/∂σ
    gz   = SVector(z, z, o, z)
    σmax, gmax = σa >= σ[3] ? (σa, ga) : (σ[3], gz)
    σmin, gmin = σb <= σ[3] ? (σb, gb) : (σ[3], gz)
    P    = -(σ[1] + σ[2] + σ[3]) / 3
    gP   = SVector(-o / 3, -o / 3, -o / 3, z)
    q    = (σmax - σmin) / 2 + ((σmax + σmin) / 2 + P) * s
    g    = (gmax - gmin) / 2 + ((gmax + gmin) / 2 + gP) * s
    return q, g
end

# θt > 0: Abbo & Sloan (1995), q = k(θ)τII, sin3θ = -3√3/2 J3/τII³
@inline function mc_q_as95(σ, s, θt)
    T   = promote_type(eltype(σ), typeof(s), typeof(θt))
    P   = -(σ[1] + σ[2] + σ[3]) / 3
    sxx, syy, szz, sxy = σ[1] + P, σ[2] + P, σ[3] + P, σ[4]
    τII = sqrt((sxx^2 + syy^2 + szz^2) / 2 + sxy^2 + T(1e-40))
    J3  = szz * (sxx * syy - sxy^2)
    gτ  = SVector(sxx, syy, szz, 2sxy) / (2τII)                       # ∂τII/∂σ
    dJ  = SVector(szz * syy, szz * sxx, sxx * syy - sxy^2, -2szz * sxy) # ∂J3/∂s
    mJ  = (dJ[1] + dJ[2] + dJ[3]) / 3
    gJ  = SVector(dJ[1] - mJ, dJ[2] - mJ, dJ[3] - mJ, dJ[4])            # ∂J3/∂σ (deviatoric projection)
    c3  = -3 * sqrt(T(3)) / 2
    x   = c3 * J3 / τII^3
    # Lode angle: deviatoric principal stresses s₂ = 2/√3 τII sinθ, θ = 30° at σ2 = σ1 (triaxial extension), -30° at σ2 = σ3
    s3θ, gs3 = abs(x) < 1 ? (x, c3 * (gJ / τII^3 - 3J3 * gτ / τII^4)) : (sign(x), zero(gτ))
    θ   = asin(s3θ) / 3
    if abs(θ) > θt
        sθ = sign(θ)
        A  = cos(θt) / 3 * (3 + tan(θt) * tan(3θt) + sθ * (tan(3θt) - 3 * tan(θt)) * s / sqrt(T(3)))
        B  = (sθ * sin(θt) + s * cos(θt) / sqrt(T(3))) / (3 * cos(3θt))
        k  = A - B * s3θ
        gk = -B * gs3
    else
        k  = cos(θ) - sin(θ) * s / sqrt(T(3))
        gk = (-sin(θ) - cos(θ) * s / sqrt(T(3))) / (3 * cos(3θ)) * gs3   # dk/dθ ∂θ/∂σ, cos3θ > 0 for |θ| ≤ θt < 30°
    end
    return k * τII, k * gτ + τII * gk
end

@inline mc_q(σ, s, θt) = θt > 0 ? mc_q_as95(σ, s, θt) : mc_q_principal(σ, s)
@inline mc_P(σ) = -(σ[1] + σ[2] + σ[3]) / 3

# Yield function without cap: F = q - P sinϕ - C cosϕ (exact Mohr-Coulomb for θt = 0)
@inline mc_yield(σ, sinϕ, ccosϕ, θt) = mc_q(σ, sinϕ, θt)[1] - mc_P(σ) * sinϕ - ccosϕ

# Tension cap (plasticity.Pt finite): the cap of DruckerPragerCap is used in the (q, P) plane: yield_Cap(q, P), ∂Q_Cap(q, P)
# (circle tangent to the Mohr-Coulomb line, through P = Pt at q = 0). The shear branch is exactly Mohr-Coulomb; the circle
# meets q = 0 at right angle, so the return of tensile states has no apex singularity.
# Flow direction (both cases): ∂Q/∂σ = dq ∂qψ/∂σ + dP ∂P/∂σ, qψ = kψ(θ)τII; without cap dq = 1, dP = -sinψ (Mohr-Coulomb
# potential), with cap (dq, dP) = ∂Q_Cap(qϕ, P). For k ≡ 1 this is DruckerPragerCap.
@inline function mc_cap(sinϕ, ccosϕ, sinψ, Pt)
    k, c, kq = sinϕ, ccosϕ, sinψ
    a  = sqrt(1 + k^2)
    b  = sqrt(1 + kq^2)
    py = (Pt + c / a) / (1 - k / a)
    Ry = py - Pt
    pd = py - Ry * k / a
    τd = k * pd + c
    pq = pd + kq * τd
    return (k=k, kq=kq, c=c, a=a, b=b, py=py, Ry=Ry, pd=pd, τd=τd, pq=pq, Pt=Pt)
end

@inline function mc_yield_full(σ, p)
    p.hascap || return mc_yield(σ, p.sinϕ, p.ccosϕ, p.θt)
    return yield_Cap(mc_q(σ, p.sinϕ, p.θt)[1], mc_P(σ), p.cap)
end

# Plastic part of the return: residual rows for (τxx, τyy, τxy, P, F) given the unknowns and the parameters p
@inline function mc_rows(τxx, τyy, τxy, P, λ̇, τt, Pt, ηve, KΔt, p)
    T  = typeof(τxx)
    σ  = SVector(τxx - P, τyy - P, -τxx - τyy - P, τxy)
    _, gq = mc_q(σ, p.sinψ, p.θt)                                       # ∂qψ/∂σ (deviatoric)
    dq, dP = if p.hascap
        a, b = ∂Q_Cap(mc_q(σ, p.sinϕ, p.θt)[1], P, p.cap)
        T(a), T(b)
    else
        one(T), -T(p.sinψ)
    end
    return SVector(τxx - τt[1] + 2ηve * λ̇ * dq * gq[1],
                   τyy - τt[2] + 2ηve * λ̇ * dq * gq[2],
                   τxy - τt[3] + 2ηve * λ̇ * dq * gq[4] / 2,
                   P - Pt + KΔt * λ̇ * dP,                                # tr(∂Q/∂σ) = -dP (∂qψ/∂σ is deviatoric)
                   mc_yield_full(σ, p) - p.ηvp * λ̇)
end

MCResidual(x, τt, Pt, ηve, KΔt, p) = mc_rows(x[1], x[2], x[3], x[4], x[5], τt, Pt, ηve, KΔt, p)

# Implicit hardening: εp is the 6th unknown, parameters p(εp), εp = εp0 + Δt ‖τ_trial - τ‖_II/(2ηve)
function MCResidualH(x, τt, Pt, ηve, KΔt, εp0, Δt, pl, ph)
    p  = mc_parameters(pl, ph, x[6])
    r  = mc_rows(x[1], x[2], x[3], x[4], x[5], τt, Pt, ηve, KΔt, p)
    dτ = SVector(τt[1] - x[1], τt[2] - x[2], τt[3] - x[3])
    rε = x[6] - εp0 - Δt * sqrt((dτ[1]^2 + dτ[2]^2 + (dτ[1] + dτ[2])^2) / 2 + dτ[3]^2 + 1e-40) / (2ηve)
    return SVector(r[1], r[2], r[3], r[4], r[5], rε)
end

# Newton with backtracking on |R|², converged to machine precision (dual numbers of the global tangent propagate through it).
# Specialised on F and on the argument types (no dynamic dispatch, no allocation).
@inline function mc_newton(F::FF, x, tol, args::Vararg{Any,N}; α_min=1e-8) where {FF,N}
    for _ in 1:100
        R, J = fd_value_and_jacobian(F, x, args...)
        maximum(abs, R) < tol && return x, true
        δx = -J \ R
        nR = sum(abs2, R)
        α  = one(nR)
        while α > α_min
            sum(abs2, F(x + α * δx, args...)) < (1 - 1e-4 * α) * nR && break
            α /= 2
        end
        x = x + α * δx
    end
    return x, maximum(abs, F(x, args...)) < 1e3 * tol
end

# Common number type of the unknowns and the parameters (new names, not the arguments: every variable keeps one type)
@inline mc_promote(τt_in, P_in, ηve_in, extra) = promote_type(eltype(τt_in), typeof(P_in), typeof(ηve_in), extra)

# Return mapping with fixed parameters p
function MCReturnMapping(τt_in, P_in, ηve_in, comp, β, Δt, p)
    Tp  = mc_promote(τt_in, P_in, ηve_in, promote_type(typeof(p.sinϕ), typeof(p.ccosϕ)))
    τt, P, ηve = SVector{3,Tp}(τt_in), Tp(P_in), Tp(ηve_in)
    σt  = SVector(τt[1] - P, τt[2] - P, -τt[1] - τt[2] - P, τt[3])
    mc_yield_full(σt, p) <= 0 && return τt, P, zero(Tp)
    KΔt = comp * Δt / β
    tol = 1e-13 * (maximum(abs, τt) + abs(P) + abs(p.ccosϕ)) + 1e-30
    x, ok = mc_newton(MCResidual, SVector(τt[1], τt[2], τt[3], P, zero(Tp)), tol, τt, P, ηve, KΔt, p)
    ok || mc_failed(τt, P, MCResidual(x, τt, P, ηve, KΔt, p))
    return SVector(x[1], x[2], x[3]), x[4], x[5]
end

# Return mapping with implicit strain hardening: predictor with the start-of-step parameters (lagged), then the coupled
# Newton on (τ, P, λ̇, εp)
function MCHardeningReturnMapping(τt_in, P_in, ηve_in, comp, β, Δt, pl, ph, εp0)
    p0  = mc_parameters(pl, ph, εp0)
    Tp  = mc_promote(τt_in, P_in, ηve_in, typeof(εp0))
    τt, P, ηve = SVector{3,Tp}(τt_in), Tp(P_in), Tp(ηve_in)
    τ0, P0, λ̇0 = MCReturnMapping(τt, P, ηve, comp, β, Δt, p0)
    iszero(λ̇0) && return τ0, P0, λ̇0                                    # elastic with the start-of-step parameters
    KΔt = comp * Δt / β
    dτ  = τt - τ0
    εp  = εp0 + Δt * mc_strain_rate_II(dτ) / (2ηve)                       # lagged estimate
    x0  = SVector(τ0[1], τ0[2], τ0[3], P0, λ̇0, Tp(εp))
    tol = 1e-13 * (maximum(abs, τt) + abs(P) + abs(p0.ccosϕ)) + 1e-30
    x, ok = mc_newton(MCResidualH, x0, tol, τt, P, ηve, KΔt, εp0, Δt, pl, ph)
    ok || mc_failed(τt, P, MCResidualH(x, τt, P, ηve, KΔt, εp0, Δt, pl, ph))
    return SVector(x[1], x[2], x[3]), x[4], x[5]
end

# Error path kept out of the kernels (string interpolation of dual numbers)
@noinline mc_failed(τt, P, R) = error("Failed Mohr-Coulomb return mapping: τ_trial = $(value_all.(τt)), P = $(value_all(P)), R = $(value_all.(R))")
@inline value_all(x::Real) = x
@inline value_all(x::ForwardDiff.Dual) = value_all(ForwardDiff.value(x))

@inline mc_strain_rate_II(ε̇) = sqrt((ε̇[1]^2 + ε̇[2]^2 + (ε̇[1] + ε̇[2])^2) / 2 + ε̇[3]^2)

# Parameters of phase ph for the plastic strain εp: ϕ + Δϕ tanh(εp/εh), C + ΔC tanh(εp/εh) (as DruckerPragerCap)
@inline mc_hardening_on(pl::MohrCoulomb, ph) = !isempty(pl.Δϕ) && !(iszero(pl.Δϕ[ph]) && iszero(pl.ΔC[ph]))
@inline function mc_parameters(pl::MohrCoulomb, ph, εp)
    θt = deg2rad(pl.θt[ph])
    # one formula for all cases (type stable: εp is a dual number in the implicit hardening); h = 0 without hardening
    h  = mc_hardening_on(pl, ph) ? tanh(max(εp, zero(εp)) / pl.εh[ph]) : zero(εp)
    ϕ  = pl.ϕ[ph] + pl.Δϕ[ph] * h
    sinϕ, ccosϕ = sind(ϕ), (pl.C[ph] + pl.ΔC[ph] * h) * cosd(ϕ)
    Pt     = isempty(pl.Pt) ? -Inf : pl.Pt[ph]
    hascap = isfinite(Pt)
    cap    = mc_cap(sinϕ, ccosϕ, pl.sinψ[ph], hascap ? Pt : -one(Pt))   # dummy geometry without cap (not used)
    return (sinϕ=sinϕ, ccosϕ=ccosϕ, sinψ=pl.sinψ[ph], ηvp=pl.ηvp[ph], θt=θt, hascap=hascap, cap=cap)
end

# Visco-elastic (power-law) effective viscosity of phase ph for the effective strain rate invariant ϵ̇II, as LocalRheology
function mc_ηve(ϵ̇II, G, materials, ph, Δ, Tk)
    n, η0, B = materials.n[ph], materials.η0[ph], materials.B[ph]
    if Tk !== nothing
        f  = arrhenius_factor(materials.Ea[ph], n, Tk, materials.Tref[ph])
        η0 = η0 * f
        B  = B * f^(-n)
    end
    η   = η0 * ϵ̇II^(1 / n - 1)
    τII = 2 * inv(1 / η + 1 / (G * Δ.t)) * ϵ̇II
    for _ in 1:20
        r = ϵ̇II - StrainRateTrial(τII, G, Δ.t, B, n)
        abs(r) < 1e-10 && break
        _, dε̇ = fd_value_and_derivative(StrainRateTrial, τII, G, Δ.t, B, n)
        τII += r / dε̇
    end
    return τII / (2ϵ̇II)
end

# One phase: effective strain rate ϵ̇ = (ϵ̇xx, ϵ̇yy, ϵ̇xy) (with the old stress of the phase), trial pressure Ptr, shear modulus G,
# start-of-step plastic strain εp0. Returns τ = (τxx, τyy, τxy), P, λ̇ and the plastic deviatoric strain rate (τ_trial - τ)/(2ηve)
function mc_phase(ϵ̇, Ptr, G, materials, ph, Δ, Tk=nothing, εp0=0.0)
    pl  = materials.plasticity
    ϵ̇II = mc_strain_rate_II(ϵ̇) + 1e-17
    ηve = mc_ηve(ϵ̇II, G, materials, ph, Δ, Tk)
    τt  = 2ηve * SVector(ϵ̇[1], ϵ̇[2], ϵ̇[3])
    β, comp = materials.β[ph], materials.compressible
    τ, P, λ̇ = implicit_hardening[] && mc_hardening_on(pl, ph) ?
        MCHardeningReturnMapping(τt, Ptr, ηve, comp, β, Δ.t, pl, ph, εp0) :
        MCReturnMapping(τt, Ptr, ηve, comp, β, Δ.t, mc_parameters(pl, ph, εp0))
    return τ, P, λ̇, (τt - τ) / (2ηve)
end

"""
    τ, P, λ̇, ε̇IIpl, η = phase_vep_tensor(ϵ̇, Ptr, Dkk, P0, G, materials, p, Δ, Tk, εp, Ω)

Tensorial visco-elasto-plastic update of phase p for the effective strain rate ϵ̇ = (ϵ̇xx, ϵ̇yy, ϵ̇xy): Mohr-Coulomb (stress
not coaxial with ϵ̇), or phase_vep (invariant models: τ = 2η ϵ̇). η = τII/(2ϵ̇II), ε̇IIpl: plastic strain rate invariant.
Used by the per-phase stress schemes (Voigt, iso-work, Reuss, markers).
"""
@inline phase_vep_tensor(ϵ̇, Ptr, Dkk, P0, G, materials, p, Δ, Tk=nothing, εp=0.0, Ω=0.0) =
    _phase_vep_tensor(materials.plasticity, ϵ̇, Ptr, Dkk, P0, G, materials, p, Δ, Tk, εp, Ω)

@inline function _phase_vep_tensor(::MohrCoulomb, ϵ̇, Ptr, Dkk, P0, G, materials, p, Δ, Tk, εp, Ω)
    ϵ̇II = mc_strain_rate_II(ϵ̇) + 1e-17
    τ, Pp, λ̇, ε̇pl = mc_phase(ϵ̇, Ptr, G, materials, p, Δ, Tk, εp)
    return τ, Pp, λ̇, mc_strain_rate_II(ε̇pl), mc_strain_rate_II(τ) / (2ϵ̇II)
end

@inline function _phase_vep_tensor(::AbstractPlasticity, ϵ̇, Ptr, Dkk, P0, G, materials, p, Δ, Tk, εp, Ω)
    ϵ̇II = mc_strain_rate_II(ϵ̇) + 1e-17
    τII, Pp, λ̇, ε̇pl = phase_vep(ϵ̇II, Ptr, Dkk, P0, G, materials, p, Δ, Tk, εp, Ω)
    η = τII / (2ϵ̇II)
    return 2η * SVector(ϵ̇[1], ϵ̇[2], ϵ̇[3]), Pp, λ̇, ε̇pl, η
end

# ------------------------------------------------------------------------------------------------------------------
# Default mixing (shared τ0, grid G): same interface as StressVector! (τ = (τxx, τyy, τxy, P), η = τII/(2ε̇II), λ̇, τII);
# phases averaged arithmetically on the stress tensor (equivalent to the η averaging of LocalRheology for coaxial stresses)
function StressVector_MC(ε̇, ε̇kk, P0, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0)
    ε̇II  = mc_strain_rate_II(ε̇) + 1e-17
    τ_av = zero(SVector{3,typeof(ε̇II)})
    P_av, λ̇_av = zero(ε̇II), zero(ε̇II)
    for ph in eachindex(materials.n)
        phase_ratios[ph] < eps() && continue
        τ, P, λ̇, _ = mc_phase(ε̇, ε̇[4], Ggrid, materials, ph, Δ, Tk, _εp(εp, ph))
        τ_av += phase_ratios[ph] * τ
        P_av += phase_ratios[ph] * P
        λ̇_av += phase_ratios[ph] * λ̇
    end
    τII = mc_strain_rate_II(τ_av)
    return SVector(τ_av[1], τ_av[2], τ_av[3], P_av), τII / (2ε̇II), λ̇_av, τII
end

function phase_plastic_rates_MC!(ε̇plph, I, ε̇, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0)
    for p in eachindex(materials.n)
        ε̇plph[p][I] = phase_ratios[p] < eps() ? 0.0 : mc_strain_rate_II(mc_phase(ε̇, ε̇[4], Ggrid, materials, p, Δ, Tk, _εp(εp, p))[4])
    end
    return nothing
end

# ------------------------------------------------------------------------------------------------------------------
# Iso-stress (Reuss) mixing, tensorial (materials.mixing = :reuss): the phase stresses are not coaxial with the strain rate,
# so the scalar partition of Rheology_Reuss.jl does not apply. Find the per-phase effective strain rates eᵢ (3 components)
#     τᵢ(eᵢ) = τ*  for all i,     Σᵢ φᵢ eᵢ = ϵ̇   (effective strain rate built with the harmonic grid G)
# Newton, Kᵢ = ∂τᵢ/∂eᵢ:  τ* = (Σφᵢ Kᵢ⁻¹)⁻¹ (ϵ̇ - Σφᵢeᵢ + Σφᵢ Kᵢ⁻¹τᵢ),   eᵢ += Kᵢ⁻¹(τ* - τᵢ)
_mc_phase_τ(e, Ptr, G, materials, p, Δ, Tk, εp) = mc_phase(e, Ptr, G, materials, p, Δ, Tk, εp)[1]

function partition_strain_rate_MC(ε̇, materials, phase_ratios, Δ, Tk, εp; tol=1e-11, niter=50)
    nphases = length(materials.n)
    T   = eltype(ε̇)
    ϵ̇   = SVector{3,T}(ε̇[1], ε̇[2], ε̇[3])
    ϵ̇II = mc_strain_rate_II(ϵ̇) + 1e-17
    # initial guess: linear visco-elastic partition, eᵢ ∝ 1/ηveᵢ
    w   = SVector{reuss_max_phases,T}(ntuple(_ -> zero(T), reuss_max_phases))
    Σφw = zero(T)
    for p = 1:nphases
        phase_ratios[p] > eps() || continue
        n, η0 = materials.n[p], materials.η0[p]
        Tk === nothing || (η0 *= arrhenius_factor(materials.Ea[p], n, Tk, materials.Tref[p]))
        wp  = inv(η0 * ϵ̇II^(1 / n - 1)) + inv(materials.G[p] * Δ.t)
        w   = setindex(w, wp, p)
        Σφw += phase_ratios[p] * wp
    end
    # let: w and Σφw are reassigned above, a closure capturing them directly would box them (allocation, type instability)
    e = let w = w, s = ϵ̇ / Σφw
        SVector{reuss_max_phases,SVector{3,T}}(ntuple(p -> s * w[p], reuss_max_phases))
    end
    for _ = 1:niter
        A, b, Σφe = zero(SMatrix{3,3,T,9}), zero(SVector{3,T}), zero(SVector{3,T})
        τs = SVector{reuss_max_phases,SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), reuss_max_phases))
        Ks = SVector{reuss_max_phases,SMatrix{3,3,T,9}}(ntuple(_ -> zero(SMatrix{3,3,T,9}), reuss_max_phases))
        for p = 1:nphases
            φ = phase_ratios[p]
            φ > eps() || continue
            τp, K = fd_value_and_jacobian(_mc_phase_τ, e[p], ε̇[4], materials.G[p], materials, p, Δ, Tk, _εp(εp, p))
            K   = K + 1e-12 * (abs(K[1, 1]) + abs(K[2, 2]) + abs(K[3, 3])) * I   # perfectly plastic phase: K singular
            Ki  = inv(K)
            τs  = setindex(τs, τp, p)
            Ks  = setindex(Ks, Ki, p)
            A  += φ * Ki
            b  += φ * (Ki * τp)
            Σφe += φ * e[p]
        end
        τstar = A \ (ϵ̇ - Σφe + b)
        err = zero(T)
        for p = 1:nphases
            phase_ratios[p] > eps() || continue
            err = max(err, maximum(abs, τs[p] - τstar))
            e = setindex(e, e[p] + Ks[p] * (τstar - τs[p]), p)
        end
        (err < tol * (maximum(abs, τstar) + 1e-30) && maximum(abs, Σφe - ϵ̇) < tol * ϵ̇II) && break
    end
    return e
end

function StressVector_MC_Reuss(ε̇, ε̇kk, P0, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0)
    nphases = length(materials.n)
    nphases > reuss_max_phases && error("StressVector_MC_Reuss: at most $reuss_max_phases phases")
    nact, pact = _active_phases(phase_ratios, nphases)
    nact == 1 && return StressVector_MC(ε̇, ε̇kk, P0, Ggrid, materials, phase_ratios, Δ, Tk, εp)
    e    = partition_strain_rate_MC(ε̇, materials, phase_ratios, Δ, Tk, εp)
    T    = eltype(ε̇)
    τ_av = zero(SVector{3,T})
    P_av, λ̇_av = zero(T), zero(T)
    for p = 1:nphases
        φ = phase_ratios[p]
        φ > eps() || continue
        τ, P, λ̇, _ = mc_phase(e[p], ε̇[4], materials.G[p], materials, p, Δ, Tk, _εp(εp, p))
        τ_av += φ * τ
        P_av += φ * P
        λ̇_av += φ * λ̇
    end
    τII = mc_strain_rate_II(τ_av)
    return SVector(τ_av[1], τ_av[2], τ_av[3], P_av), τII / (2 * (mc_strain_rate_II(ε̇) + 1e-17)), λ̇_av, τII
end

function phase_plastic_rates_MC_Reuss!(ε̇plph, I, ε̇, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0)
    nphases = length(materials.n)
    nact, _ = _active_phases(phase_ratios, nphases)
    nact == 1 && return phase_plastic_rates_MC!(ε̇plph, I, ε̇, Ggrid, materials, phase_ratios, Δ, Tk, εp)
    e = partition_strain_rate_MC(ε̇, materials, phase_ratios, Δ, Tk, εp)
    for p = 1:nphases
        ε̇plph[p][I] = phase_ratios[p] < eps() ? 0.0 :
            mc_strain_rate_II(mc_phase(e[p], ε̇[4], materials.G[p], materials, p, Δ, Tk, _εp(εp, p))[4])
    end
    return nothing
end
