# Iso-stress (Reuss) mixing of the phases in a cell (materials.mixing = :reuss)
#
# All phases carry the same stress τ (and share the old stress τ0); the strain rate is partitioned:
#     ε̇ = Σᵢ φᵢ ε̇ᵢ
# With the effective visco-elastic strain rate of phase i, ϵ̇ᵢ = ε̇ᵢ + τ0/(2GᵢΔt), all ϵ̇ᵢ are parallel to τ and
#     Σᵢ φᵢ ϵ̇ᵢ = ε̇ + τ0 Σᵢ φᵢ/(2GᵢΔt) = ε̇ + τ0/(2GΔt)   with G the harmonic average of the Gᵢ,
# i.e. the effective strain rate the solver builds with the (default, harmonic) grid G. The problem therefore reduces
# to a scalar one on the invariants: find the per-phase ϵ̇IIᵢ such that
#     τIIᵢ(ϵ̇IIᵢ) = τ*  for all i,     Σᵢ φᵢ ϵ̇IIᵢ = ϵ̇II
# where τIIᵢ(·) is the full visco-elasto-(visco)plastic update of phase i with its own Gᵢ.
# Newton on the ϵ̇IIᵢ: with kᵢ = dτIIᵢ/dϵ̇IIᵢ, the linearised conditions give τ* in closed form
#     τ* = (ϵ̇II - Σφᵢϵ̇IIᵢ + Σφᵢτᵢ/kᵢ) / Σ(φᵢ/kᵢ),   ϵ̇IIᵢ += (τ* - τᵢ)/kᵢ
# A phase at yield (kᵢ → 0) then sets τ* and takes the remaining strain rate.
# The iterations are plain Newton steps, so the outer AD (tangent operator) differentiates through them.
#
# Requires the grid G to be the harmonic average of the phase G (default of compute_grid_fields!).
# λ̇ and P are returned as volume averages Σφᵢ(⋅)ᵢ. Plastic strain: shared or per phase (materials.plastic_strain), Ω is shared.

const reuss_max_phases = 8

# Visco-elasto-plastic stress update of a single phase p, for the effective strain rate invariant ε̇II and shear modulus G.
# Same as the body of the phase loop of LocalRheology (Rheology.jl), but with G as an argument.
function phase_vep(ε̇II, P, Dkk, P0, G, materials, p, Δ, Tk=nothing, εp=0.0, Ω=0.0)
    ϵ = 1e-10 # tolerance
    n = materials.n[p]
    η0 = materials.η0[p]
    B = materials.B[p]
    β = materials.β[p]
    comp = materials.compressible
    if Tk !== nothing
        f  = arrhenius_factor(materials.Ea[p], n, Tk, materials.Tref[p])
        η0 = η0 * f
        B  = B * f^(-n)
    end

    # Initial guess
    η = η0 * ε̇II^(1 / n - 1.0)
    ηvep = inv(1 / η + 1 / (G * Δ.t))
    τII = 2 * ηvep * ε̇II

    # Visco-elastic powerlaw
    for it = 1:20
        r = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
        (abs(r) < ϵ) && break
        _, ∂ε̇II∂τII = fd_value_and_derivative(StrainRateTrial, τII, G, Δ.t, B, n)
        τII += inv(∂ε̇II∂τII) * r
    end
    isnan(τII) && error("phase_vep: NaN stress (phase $p)")
    ηvep = τII / 2 / ε̇II

    # Viscoplastic return mapping (common number type: the Reuss iteration differentiates w.r.t. ε̇II only, P would
    # otherwise stay a Float64 and the mixed-type call of the return mapping is not type stable)
    Tp = promote_type(typeof(τII), typeof(P))
    τve = τII
    τII, P, λ̇ = return_mapping(Tp(τII), Tp(P), Tp(ε̇II), Dkk, P0, Tp(ηvep), β, Δ.t, comp, materials.plasticity, p, εp, Ω)
    # Plastic part of the deviatoric strain rate invariant: the return mapping reduces the visco-elastic stress τve by
    # 2ηve ε̇II_pl (any flow rule: λ̇ ∂Q/∂τ / 2 for DruckerPragerCap, λ̇/2 on the Drucker-Prager branch)
    ε̇pl = max(ε̇II * (1 - τII / τve), zero(τII))
    return τII, P, λ̇, ε̇pl
end

_phase_τII(ε̇II, P, Dkk, P0, G, materials, p, Δ, Tk, εp, Ω) = phase_vep(ε̇II, P, Dkk, P0, G, materials, p, Δ, Tk, εp, Ω)[1]

# ------------------------------------------------------------------------------------------------------------------
# Partition of the effective strain rate invariant between the phases of a cell (iso-stress)
#
# Find the per-phase ϵ̇IIᵢ such that hᵢ(ϵ̇IIᵢ) = H for all phases i and Σᵢ φᵢ ϵ̇IIᵢ = ϵ̇II, where hᵢ is the quantity the
# mixing rule makes uniform (τIIᵢ: iso-stress, Reuss; the rule is a type, see _partition_h).
# Newton on the ϵ̇IIᵢ, with h′ᵢ = dhᵢ/dϵ̇IIᵢ; the linearised conditions give H in closed form:
#     H = (ϵ̇II - Σφᵢϵ̇IIᵢ + Σφᵢhᵢ/h′ᵢ) / Σ(φᵢ/h′ᵢ),   ϵ̇IIᵢ += (H - hᵢ)/h′ᵢ
abstract type PartitionRule end
struct ReussRule <: PartitionRule end
# (h, h′) from τIIᵢ, kᵢ = dτIIᵢ/dϵ̇IIᵢ and ϵ̇IIᵢ
@inline _partition_h(::ReussRule, τ, k, e) = (τ, k)
# Initial guess: exact partition for linear visco-elastic phases (ηveᵢ = (1/η0ᵢ + 1/(GᵢΔt))⁻¹), ϵ̇IIᵢ ∝ w(ηveᵢ)
@inline _partition_w(::ReussRule, ηve) = inv(ηve)

# εp: number (shared) or PhaseStrain (per phase, see Rheology.jl)
function partition_strain_rate(rule::PartitionRule, ε̇II, P, Dkk, P0, materials, phase_ratios, Δ, Tk, εp, Ω; tol=1e-11, niter=50)
    nphases = length(materials.n)
    T = typeof(ε̇II)
    # immutable SVectors (functional updates with setindex): no heap allocation
    # Initial guess: linear visco-elastic partition (power law: η0 evaluated at ε̇II)
    e  = SVector{reuss_max_phases,T}(ntuple(_ -> ε̇II, reuss_max_phases))
    Σφw = zero(T)
    for p = 1:nphases
        phase_ratios[p] > eps() || continue
        n  = materials.n[p]
        η0 = materials.η0[p]
        Tk === nothing || (η0 *= arrhenius_factor(materials.Ea[p], n, Tk, materials.Tref[p]))
        ηve = inv(inv(η0 * ε̇II^(1 / n - 1)) + inv(materials.G[p] * Δ.t))
        w = _partition_w(rule, ηve)
        e = setindex(e, w, p)
        Σφw += phase_ratios[p] * w
    end
    for p = 1:nphases
        phase_ratios[p] > eps() && (e = setindex(e, e[p] * ε̇II / Σφw, p))
    end
    h  = SVector{reuss_max_phases,T}(ntuple(_ -> zero(T), reuss_max_phases))
    dh = SVector{reuss_max_phases,T}(ntuple(_ -> zero(T), reuss_max_phases))
    for it = 1:niter
        dhmax = zero(T)
        for p = 1:nphases
            phase_ratios[p] > eps() || continue
            τp, kp = fd_value_and_derivative(_phase_τII, e[p], P, Dkk, P0, materials.G[p], materials, p, Δ, Tk, _εp(εp, p), Ω)
            hp, dhp = _partition_h(rule, τp, kp, e[p])
            h  = setindex(h, hp, p)
            dh = setindex(dh, dhp, p)
            dhmax = max(dhmax, dhp)
        end
        dhmin = 1e-12 * dhmax # perfectly plastic phase (Reuss): k = 0
        num, den, Σφe = zero(T), zero(T), zero(T)
        for p = 1:nphases
            φ = phase_ratios[p]
            φ > eps() || continue
            dh = setindex(dh, max(dh[p], dhmin), p)
            num += φ * h[p] / dh[p]
            den += φ / dh[p]
            Σφe += φ * e[p]
        end
        H = (ε̇II - Σφe + num) / den
        err = zero(T)
        for p = 1:nphases
            phase_ratios[p] > eps() || continue
            err = max(err, abs(h[p] - H))
            e = setindex(e, max(e[p] + (H - h[p]) / dh[p], 0.1 * e[p]), p) # keep the strain rates positive
        end
        (err < tol * abs(H) && abs(Σφe - ε̇II) < tol * ε̇II) && break
    end
    return e
end

# Local rheology with a partition rule: τII = Σφᵢ τIIᵢ(ϵ̇IIᵢ) (= the common τ for Reuss), parallel to the effective strain rate
function LocalRheology_partition(rule::PartitionRule, ε̇, Dkk, P0, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0; tol=1e-11, niter=50)

    nphases = length(materials.n)
    nphases > reuss_max_phases && error("LocalRheology_partition: at most $reuss_max_phases phases")
    eps0 = 1e-17

    # Effective strain rate invariant (built with the harmonic grid G) & pressure
    ε̇II = sqrt((ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1] - ε̇[2])^2) / 2 + ε̇[3]^2) + eps0
    P = ε̇[4]
    T = typeof(ε̇II)

    # Single phase cell: nothing to mix
    nact, pact = _active_phases(phase_ratios, nphases)
    if nact == 1
        τII, Pp, λ̇ = phase_vep(ε̇II, P, Dkk, P0, materials.G[pact], materials, pact, Δ, Tk, _εp(εp, pact), Ω)
        return τII / (2 * ε̇II), λ̇, Pp, τII
    end

    e = partition_strain_rate(rule, ε̇II, P, Dkk, P0, materials, phase_ratios, Δ, Tk, εp, Ω; tol, niter)

    # Final evaluation at the converged partition
    τII, P_average, λ̇_average = zero(T), zero(T), zero(T)
    for p = 1:nphases
        φ = phase_ratios[p]
        φ > eps() || continue
        τp, Pp, λ̇p = phase_vep(e[p], P, Dkk, P0, materials.G[p], materials, p, Δ, Tk, _εp(εp, p), Ω)
        τII += φ * τp
        P_average += φ * Pp
        λ̇_average += φ * λ̇p
    end
    return τII / (2 * ε̇II), λ̇_average, P_average, τII
end

function _active_phases(phase_ratios, nphases)
    nact, pact = 0, 1
    for p = 1:nphases
        if phase_ratios[p] > eps()
            nact += 1
            pact = p
        end
    end
    return nact, pact
end

LocalRheology_Reuss(ε̇, Dkk, P0, Ggrid, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0; tol=1e-11, niter=50) =
    LocalRheology_partition(ReussRule(), ε̇, Dkk, P0, Ggrid, materials, phase_ratios, Δ, Tk, εp, Ω; tol, niter)

# Per-phase plastic strain rates ε̇IIplᵢ at the converged partition, written to ε̇plph[p][I] (0 where phase p is absent)
function phase_plastic_rates_partition!(ε̇plph, I, rule::PartitionRule, ε̇, Dkk, P0, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0)
    nphases = length(materials.n)
    ε̇II = sqrt((ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1] - ε̇[2])^2) / 2 + ε̇[3]^2) + 1e-17
    P = ε̇[4]
    nact, _ = _active_phases(phase_ratios, nphases)
    e = nact == 1 ? SVector{reuss_max_phases}(ntuple(_ -> ε̇II, reuss_max_phases)) :
                    partition_strain_rate(rule, ε̇II, P, Dkk, P0, materials, phase_ratios, Δ, Tk, εp, Ω)
    for p = 1:nphases
        ε̇plph[p][I] = phase_ratios[p] > eps() ? phase_vep(e[p], P, Dkk, P0, materials.G[p], materials, p, Δ, Tk, _εp(εp, p), Ω)[4] : 0.0
    end
    return nothing
end
