# Iso-work rate mixing of the phases in a cell (materials.mixing = :isowork)
#
# Built on the per-phase stress framework of Voigt (Rheology_Voigt.jl): each phase i keeps its own old stress τ0ᵢ
# (grid and markers) and modulus Gᵢ. Phase i sees the strain rate aᵢ ε̇ (ε̇: deviatoric strain rate of the cell), with
#     Σᵢ φᵢ aᵢ = 1                                         (the strain rates add up)
#     Wᵢ = (τᵢ(aᵢ) - τᵢ(0)) : (aᵢ ε̇) = W  for all i        (same work rate)
#     τ  = Σᵢ φᵢ τᵢ(aᵢ)                                    (as Voigt, which is aᵢ = 1)
# τᵢ(a) is the full visco-elasto-plastic update of phase i (voigt_phase) and τᵢ(0) the stress phase i would have
# without deformation (its relaxed old stress). Wᵢ is the work rate of the stress change produced by the strain rate:
#  - viscous phases (τᵢ(0) = 0): the mechanical power τᵢ:ε̇ᵢ. Linear viscous: aᵢ ∝ ηᵢ^(-1/2),
#    η = Σφᵢηᵢ^(1/2) / Σφᵢηᵢ^(-1/2), between the Reuss and Voigt bounds;
#  - elastic phases: aᵢ ∝ Gᵢ^(-1/2), independent of the loading direction (loading followed by the reverse strain
#    brings every phase back to zero stress). The mechanical power τᵢ:ε̇ᵢ itself would include the pre-stress
#    (τ0ᵢ:ε̇ᵢ), which changes sign at unloading: the partition is then not defined (and with a shared τ0 the stresses
#    blow up, first version of this scheme).
# With a monotonic stress update, Wᵢ ≥ 0 and increasing in aᵢ: the partition is unique.
#
# Newton on the aᵢ (same closed form as partition_strain_rate, Rheology_Reuss.jl):
#     W = (1 - Σφᵢaᵢ + ΣφᵢWᵢ/W′ᵢ) / Σ(φᵢ/W′ᵢ),   aᵢ += (W - Wᵢ)/W′ᵢ
# The iterations are plain Newton steps, so the outer AD (tangent operator) differentiates through them.

# τ:ε̇ in plane strain (deviatoric, τzz = -τxx - τyy, ε̇zz = -ε̇xx - ε̇yy)
@inline _power(τ, ε̇) = τ[1] * ε̇[1] + τ[2] * ε̇[2] + (τ[1] + τ[2]) * (ε̇[1] + ε̇[2]) + 2 * τ[3] * ε̇[3]

# Work rate of phase p for the multiplier a (stress change produced by the strain rate a ε̇)
function _isowork_power(a, ε̇, ε̇kk, P0, T0, I, materials, p, Δ, Tk, εp, Ω, βbar=1.0)
    τa = voigt_phase(ε̇, ε̇kk, P0, T0, I, materials, p, Δ, Tk, εp, Ω, 1e-17, a, βbar)[1]
    τr = voigt_phase(ε̇, ε̇kk, P0, T0, I, materials, p, Δ, Tk, εp, Ω, 1e-17, zero(a), βbar)[1]
    return a * _power(τa - τr, ε̇)
end

# Strain rate multipliers aᵢ of the phases: 1 for Voigt, iso-work partition for :isowork
function phase_multipliers(ε̇, ε̇kk, P0, T0, I, materials, phase_ratios, Δ, Tk, εp, Ω; tol=1e-11, niter=50)
    T = promote_type(eltype(ε̇), typeof(P0))
    a = SVector{reuss_max_phases,T}(ntuple(_ -> one(T), reuss_max_phases))
    _mixing(materials) === :isowork || return a
    nphases = length(materials.n)
    nphases > reuss_max_phases && error("phase_multipliers: at most $reuss_max_phases phases")

    # Single phase cell: nothing to partition
    nact = 0
    for p = 1:nphases
        phase_ratios[p] > eps() && (nact += 1)
    end
    nact == 1 && return a

    # Initial guess: linear visco-elastic partition aᵢ ∝ ηveᵢ^(-1/2)
    ε̇II = sqrt((ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1] - ε̇[2])^2) / 2 + ε̇[3]^2) + 1e-17
    Σφw = zero(T)
    for p = 1:nphases
        phase_ratios[p] > eps() || continue
        n  = materials.n[p]
        η0 = materials.η0[p]
        Tk === nothing || (η0 *= arrhenius_factor(materials.Ea[p], n, Tk, materials.Tref[p]))
        w  = inv(sqrt(inv(inv(η0 * ε̇II^(1 / n - 1)) + inv(materials.G[p] * Δ.t))))
        a  = setindex(a, w, p)
        Σφw += phase_ratios[p] * w
    end
    for p = 1:nphases
        phase_ratios[p] > eps() && (a = setindex(a, a[p] / Σφw, p))
    end

    βbar = grid_β(materials, phase_ratios)
    W  = SVector{reuss_max_phases,T}(ntuple(_ -> zero(T), reuss_max_phases))
    dW = W
    for it = 1:niter
        for p = 1:nphases
            phase_ratios[p] > eps() || continue
            Wp, dWp = fd_value_and_derivative(_isowork_power, a[p], ε̇, ε̇kk, P0, T0, I, materials, p, Δ, Tk, εp, Ω, βbar)
            dWp > 0 || return SVector{reuss_max_phases,T}(ntuple(_ -> one(T), reuss_max_phases)) # safeguard (non-monotonic update): Voigt
            W  = setindex(W, Wp, p)
            dW = setindex(dW, dWp, p)
        end
        num, den, Σφa = zero(T), zero(T), zero(T)
        for p = 1:nphases
            φ = phase_ratios[p]
            φ > eps() || continue
            num += φ * W[p] / dW[p]
            den += φ / dW[p]
            Σφa += φ * a[p]
        end
        Ws = (1 - Σφa + num) / den
        err = zero(T)
        for p = 1:nphases
            phase_ratios[p] > eps() || continue
            err = max(err, abs(W[p] - Ws))
            a = setindex(a, max(a[p] + (Ws - W[p]) / dW[p], 0.1 * a[p]), p) # keep the multipliers positive
        end
        (err < tol * abs(Ws) && abs(Σφa - 1) < tol) && break
    end
    return a
end
