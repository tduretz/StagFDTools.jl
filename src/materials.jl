abstract type AbstractPlasticity end

# Temperature dependence of the viscosity: η0(T) = η0 exp(Ea/(nR) (1/T - 1/Tref)), T and Tref in K (unscaled)
const R_gas = 8.314462618
@inline arrhenius_factor(Ea, n, Tk, Tref) = iszero(Ea) ? one(Tk) : exp(Ea / (n * R_gas) * (inv(Tk) - inv(Tref)))

Base.@kwdef struct IterParams
    niter::Int64        = 10
    γ::Float64          = 1e5
    ϵ_l::Float64        = 1e-11
    ϵ_nl::Float64       = 1e-8
    ϵ_nl_last::Float64  = 1e-5                  # last resort: after niter iterations, accept the step (with a warning) if abs. res. < ϵ_nl_last
    niter_KSP::Int64    = 1000                  # max. number of KSP iterations per linear solve (solver_type = :GCR)
    inexact::Bool       = false
    solver_type::Symbol = :PH
    line_search::Symbol = :backtracking         # :backtracking (first α that reduces the residual) or :grid (best of α)
    α::Vector{Float64}  = collect(LinRange(0.05, 1.0, 6)) # only used by line_search = :grid
    Pic2Newt::Float64   = Inf                   # Picard until ϵ/ϵ0 < Pic2Newt, then Newton (> 1: Newton only)
    α_Picard::Float64   = 3.0                   # first trial step of the backtracking for Picard iterations
    PC_refresh::Symbol  = :step                 # solver_type = :GCR: Picard preconditioner assembled and factorised once per time step (:step) or at each Newton iteration (:iteration); always at Picard iterations
    pc_type::Symbol     = :picard               # solver_type = :GCR, Newton steps: Cholesky factor of the Schur complement built from the Picard Kuu (:picard) or from the symmetrised Newton Kuu, ½(Kuu + Kuuᵀ) (:sym_newton, refreshed at each Newton iteration; better for plasticity, e.g. ~2× fewer GCR iterations on Duretz et al. 2018)
    Δt_red_fac::Float64 = 2.0                   # adaptive time stepping: a failed step is restarted with Δt/Δt_red_fac...
    ntry::Int64         = 4                     # ...at most ntry times
    Δt_inc_fac::Float64 = 1.25                  # after a successful step, Δt is multiplied by Δt_inc_fac
    nt_max::Int64       = typemax(Int64)        # the run stops after nt_max steps...
    time_max::Float64   = Inf                   # ...or at time_max (same units as Δ.t), whichever comes first
end
Base.@kwdef struct VonMises{T} <: AbstractPlasticity
    C::T    = Float64[]
    cosϕ::T = Float64[]
    ηvp::T  = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end
Base.@kwdef struct DruckerPrager{T} <: AbstractPlasticity
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

# Mohr-Coulomb (σ2-independent), tensorial return mapping (Rheology_MohrCoulomb.jl). Non-associated: flow potential = yield
# function with ψ in place of ϕ. θt [°]: 0 → exact Mohr-Coulomb written with the principal stresses (de Borst, 1990; corners
# at the triaxial states), > 0 → Abbo & Sloan (1995) rounding of the corners for |θ| > θt (Lode angle, e.g. 25-29°)
Base.@kwdef struct MohrCoulomb{T} <: AbstractPlasticity
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    θt::T   = Float64[]
    Pt::T   = Float64[]  # Tension cap (as DruckerPragerCap, in the (k(θ)τII, P) plane), must be > -C/tanϕ; -Inf: no cap
    Δϕ::T   = Float64[]  # Strain hardening: ϕ(εp) = ϕ + Δϕ tanh(εp/εh) [°] (0: no hardening)
    ΔC::T   = Float64[]  # Strain hardening: C(εp) = C + ΔC tanh(εp/εh)
    εh::T   = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct DruckerPragerCap{T} <: AbstractPlasticity
    C::T    = Float64[]
    Pt::T   = Float64[]  # Tensile strength
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
    k::T    = Float64[]
    kq::T   = Float64[]
    c::T    = Float64[]
    a::T    = Float64[]
    b::T    = Float64[]
    py::T   = Float64[]
    Ry::T   = Float64[]
    pd::T   = Float64[]
    τd::T   = Float64[]
    pq::T   = Float64[]
    Δϕ::T   = Float64[]  # Strain hardening: ϕ(εp) = ϕ + Δϕ tanh(εp/εh) [°], εp = ∫λ̇dt (0: no hardening)
    ΔC::T   = Float64[]  # Strain hardening: C(εp) = C + ΔC tanh(εp/εh) (0: no hardening)
    εh::T   = Float64[]  # Plastic strain scale of the hardening
    # Rate-and-state friction (Herrendörfer et al., 2018; as RateStateFriction in RheologyCalculator.jl), off where aRS = 0:
    # the Drucker-Prager slope k = sinϕ is replaced by μ(V, Ω) = aRS asinh(V/(2V0RS) exp((sinϕ + bRS Ω)/aRS)),
    # V = DRS λ̇ (slip rate of a band of width DRS), Ω = ln(V0RS θ/LRS) (lagged, start of the time step)
    aRS::T  = Float64[]  # direct effect
    bRS::T  = Float64[]  # evolution effect
    LRS::T  = Float64[]  # characteristic slip distance
    V0RS::T = Float64[]  # reference slip rate (μ = sinϕ at V = V0RS in steady state)
    DRS::T  = Float64[]  # width of the slipping band (usually the cell size)
    ηRS::T  = Float64[]  # viscoplastic regularisation of the rate-and-state branch (e.g. radiation damping DRS G/(2cs)); ηvp is kept for the cap
end

Base.@kwdef struct DruckerPrager1{T} <: AbstractPlasticity
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct DruckerHyperbolic{T} <: AbstractPlasticity
    Pt::T   = Float64[]
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct DruckerAniso{T} <: AbstractPlasticity
    δ::T    = Float64[]
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct Tensile{T} <: AbstractPlasticity
    Pt::T   = Float64[]
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct Golchin2021{T} <: AbstractPlasticity
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
    M::T    = Float64[]
    N::T    = Float64[]
    Pc::T   = Float64[]
    a::T    = Float64[]
    b::T    = Float64[]
    c::T    = Float64[]
    Pt::T   = Float64[]
end

Base.@kwdef struct Kiss2023{T} <: AbstractPlasticity
    C::T    = Float64[]
    ϕ::T    = Float64[]
    ψ::T    = Float64[]
    ηvp::T  = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
    σPt::T  = Float64[]
    δPt::T  = Float64[]
    P1::T   = Float64[]
    τ1::T   = Float64[]
    P2::T   = Float64[]
    τ2::T   = Float64[]
end

abstract type AbstractAnisotropy end

struct Isotropic <: AbstractAnisotropy end   # default: anisotropy off

Base.@kwdef struct Anisotropy{T,V} <: AbstractAnisotropy
    mv::T = Float64[]   # viscosity contrast η₂/η₁ (1: isotropic)
    me::T = Float64[]   # shear modulus contrast G₂/G₁ (1: isotropic)
    α::T  = Float64[]   # volume fraction of the strong layer, 0 < α < 1
    δv::T = Float64[]   # viscous anisotropy factor ηn/ηs = (1-α+α mv)(1-α+α/mv), output of preprocess!
    δe::T = Float64[]   # elastic anisotropy factor (1-α+α me)(1-α+α/me), output of preprocess!
    n1::T = Float64[]   # power-law exponent of layer 1 (weak), 1: linear
    n2::T = Float64[]   # power-law exponent of layer 2 (strong), 1: linear
    r::T  = Float64[]   # shape exponent of the anisotropic yield function f, 0 < r ≤ 4 (power-law layers only)
    ϵv::T = Float64[]   # regularisation of the 1/n root of the shared viscous normal strain rate, relative to the viscous
                        # strain-rate magnitude (power-law layers only; 0: none, the global Newton may stall)
    dv::V = [0.0, 1.0]
end

# Phase p is anisotropic: 2 virtual phases with their own old stresses (per-phase stress framework, Rheology_Voigt.jl)
@inline is_anisotropic(::Isotropic, materials, p)   = false
@inline is_anisotropic(a::Anisotropy, materials, p) = !(isone(a.mv[p]) && isone(a.me[p]) && a.n1[p] == a.n2[p])
@inline is_anisotropic(materials, p) = hasproperty(materials, :anisotropy) && is_anisotropic(materials.anisotropy, materials, p)

Base.@kwdef struct Materials{T,P<:AbstractPlasticity,A<:AbstractAnisotropy}
    g::T    = [0.0, 0.0]
    ρ::T    = Float64[]
    n::T    = Float64[]
    η0::T   = Float64[]
    Ea::T   = Float64[]   # activation energy [J/mol], 0: no temperature dependence
    Tref::T = Float64[]   # reference temperature [K] at which η = η0
    ξ0::T   = Float64[]
    G::T    = Float64[]
    β::T    = Float64[]
    B::T    = Float64[]
    plasticity::P          = NoPlasticity()
    anisotropy::A          = Isotropic()
    compressible::Bool     = false
    phase_avg::Symbol      = :harmonic
    mixing::Symbol         = :default        # interface cells: :default (each phase sees the cell strain rate, shared τ0), :voigt (iso-strain rate, τ0 per phase), :reuss (iso-stress), :isowork (iso-work rate), :markers, :markers_stress, :markers_work (rheology on the markers: iso-strain, iso-stress, iso-work between phases; needs TensorMarkers)
    plastic_strain::Symbol = :mixed          # strain hardening: :mixed (one εp per cell, += Δt Σφᵢλ̇ᵢ, seen by all phases), :phase (εpᵢ += Δt λ̇ᵢ per phase)
    pressure::Symbol       = :shared         # old pressure: :shared (one P0 per cell), :phase (P0ᵢ per phase; only with mixing = :voigt or :isowork)
end

Base.@kwdef struct Materials_TwoPhases{T,P<:AbstractPlasticity}
    g::T               = [0.0, 0.0]
    ρs::T              = Float64[]
    ρf::T              = Float64[]
    n::T               = Float64[]
    η0::T              = Float64[]
    Ea::T              = Float64[]   # activation energy [J/mol], 0: no temperature dependence
    Tref::T            = Float64[]   # reference temperature [K] at which η = η0
    ξ0::T              = Float64[]
    G::T               = Float64[]
    β::T               = Float64[]
    Ks::T              = Float64[]
    KΦ0::T             = Float64[]   # KΦ = KΦ0 Φ^m (same exponent as the bulk viscosity)
    Kf::T              = Float64[]
    Φ0::T              = Float64[]
    m::T               = Float64[]
    n_CK::T            = Float64[]
    k_ηf0::T           = Float64[]
    B::T               = Float64[]
    plasticity::P      = NoPlasticity()
    linearizeΦ::Bool   = false
    oneway::Bool       = false
    single_phase::Bool = false
    conservative::Bool = false
    compressible::Bool = false
    phase_avg::Symbol  = :harmonic
end

struct NoPlasticity <: AbstractPlasticity end

initialize(::Type{VonMises}, n::Integer) = VonMises(
    C    = fill(1e50, n),
    cosϕ = ones(n),
    ηvp  = zeros(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{MohrCoulomb}, n::Integer) = MohrCoulomb(
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    θt   = zeros(n),
    Pt   = fill(-Inf, n),
    Δϕ   = zeros(n),
    ΔC   = zeros(n),
    εh   = ones(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = ones(n)
)

initialize(::Type{DruckerPrager}, n::Integer) = DruckerPrager(
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{DruckerPragerCap}, n::Integer) = DruckerPragerCap(
    C    = fill(1e50, n),
    Pt   = zeros(n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n),
    k    = zeros(n),
    kq   = zeros(n),
    c    = zeros(n),
    a    = zeros(n),
    b    = zeros(n),
    py   = zeros(n),
    Ry   = zeros(n),
    pd   = zeros(n),
    τd   = zeros(n),
    pq   = zeros(n),
    Δϕ   = zeros(n),
    ΔC   = zeros(n),
    εh   = ones(n),
    aRS  = zeros(n),
    bRS  = zeros(n),
    LRS  = ones(n),
    V0RS = ones(n),
    DRS  = ones(n),
    ηRS  = zeros(n),
)

initialize(::Type{DruckerPrager1}, n::Integer) = DruckerPrager1(
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{DruckerHyperbolic}, n::Integer) = DruckerHyperbolic(
    Pt   = zeros(n),
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{DruckerAniso}, n::Integer) = DruckerAniso(
    δ    = ones(n),
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{Golchin2021}, n::Integer) = Golchin2021(
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n),
    M    = zeros(n),
    N    = zeros(n),
    Pc   = zeros(n),
    a    = zeros(n),
    b    = zeros(n),
    c    = zeros(n),
    Pt   = zeros(n),
)

initialize(::Type{Kiss2023}, n::Integer) = Kiss2023(
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n),
    Pt   = zeros(n),
    δPt  = zeros(n),
    P1   = zeros(n),
    τ1   = zeros(n),
    P2   = zeros(n),
    τ2   = zeros(n),
)

initialize(::Type{Tensile}, n::Integer) = Tensile(
    Pt   = zeros(n),
    C    = fill(1e50, n),
    ϕ    = zeros(n),
    ψ    = zeros(n),
    ηvp  = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n),
)

initialize(::Type{NoPlasticity}, ::Integer) = NoPlasticity()

initialize(::Type{Isotropic}, ::Integer) = Isotropic()

initialize(::Type{Anisotropy}, n::Integer) = Anisotropy(mv = ones(n), me = ones(n), α = fill(0.5, n), δv = ones(n), δe = ones(n),
                                                        n1 = ones(n), n2 = ones(n), r = fill(3.0, n), ϵv = fill(1e-2, n))

function initialize_materials(nphases::Integer;
    plasticity         = NoPlasticity(),
    anisotropy         = Isotropic(),
    compressible::Bool = false,
    phase_avg::Symbol  = :harmonic,
    mixing::Symbol     = :default,
    plastic_strain::Symbol = :mixed,
    pressure::Union{Symbol,Nothing} = nothing)
    mixing in (:default, :voigt, :reuss, :isowork, :markers, :markers_stress, :markers_work) ||
        error("mixing must be :default, :voigt, :reuss, :isowork, :markers, :markers_stress or :markers_work (got :$mixing)")
    plastic_strain in (:mixed, :phase) || error("plastic_strain must be :mixed or :phase (got :$plastic_strain)")
    # Per-phase old pressure: default for the schemes with per-phase stresses (Voigt, iso-work)
    pressure = something(pressure, mixing in (:voigt, :isowork) ? :phase : :shared)
    pressure in (:shared, :phase) || error("pressure must be :shared or :phase (got :$pressure)")
    pressure === :phase && !(mixing in (:voigt, :isowork)) && error("pressure = :phase requires mixing = :voigt or :isowork")
    P = plasticity isa Type ? plasticity : typeof(plasticity)
    A = anisotropy isa Type ? anisotropy : typeof(anisotropy)
    A <: Anisotropy && !(mixing in (:voigt, :isowork)) &&
        error("anisotropy = Anisotropy requires mixing = :voigt or :isowork (per-phase old stresses, got :$mixing)")
    return Materials(
        ρ            = ones(nphases),
        n            = ones(nphases),
        η0           = ones(nphases),
        Ea           = zeros(nphases),
        Tref         = ones(nphases),
        ξ0           = 1e50 * ones(nphases),
        G            = 1e50 * ones(nphases),
        β            = 1e-50 * ones(nphases),
        B            = ones(nphases),
        plasticity   = initialize(P, nphases),
        anisotropy   = initialize(A, nphases),
        compressible = compressible,
        phase_avg    = phase_avg,
        mixing       = mixing,
        plastic_strain = plastic_strain,
        pressure       = pressure
    )
end

function initialize_materials_TwoPhases(nphases::Integer;
    plasticity         = NoPlasticity(),
    compressible::Bool = true,
    single_phase::Bool = false,
    oneway::Bool       = false,
    linearizeΦ::Bool   = false,
    conservative::Bool = false,
    phase_avg::Symbol  = :harmonic,
    g::Vector{Float64} = [0.0, 0.0])
    P = plasticity isa Type ? plasticity : typeof(plasticity)
    return Materials_TwoPhases(
        g            = g,
        ρs           = ones(nphases),
        ρf           = ones(nphases),
        n            = ones(nphases),
        η0           = ones(nphases),
        Ea           = zeros(nphases),
        Tref         = ones(nphases),
        ξ0           = 1e2 * ones(nphases),
        G            = 1e50 * ones(nphases),
        Ks           = 1e50 * ones(nphases),
        KΦ0          = 1e50 * ones(nphases),
        Kf           = 1e50 * ones(nphases),
        Φ0           = 1e-2 * ones(nphases),
        m            = zeros(nphases),
        n_CK         = 3.0*ones(nphases),
        k_ηf0        = 2.220394736842105*ones(nphases),
        B            = ones(nphases),
        plasticity   = initialize(P, nphases),
        compressible = compressible,
        oneway       = oneway,
        linearizeΦ   = linearizeΦ,
        single_phase = single_phase,
        conservative = conservative,
        phase_avg    = phase_avg
    )
end

function preprocess!(dp::DruckerPrager)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
    @. dp.cosψ = cosd(dp.ψ)
end

function preprocess!(mc::MohrCoulomb)
    @. mc.cosϕ = cosd(mc.ϕ)
    @. mc.sinϕ = sind(mc.ϕ)
    @. mc.sinψ = sind(mc.ψ)
    @. mc.cosψ = cosd(mc.ψ)
end

function preprocess!(dp::DruckerPragerCap)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
    @. dp.cosψ = cosd(dp.ψ)
    @. dp.k    = dp.sinϕ
    @. dp.kq   = dp.sinψ
    @. dp.c    = dp.C*dp.cosϕ
    @. dp.a    = sqrt(1.0 + dp.k^2)
    @. dp.b    = sqrt(1.0 + dp.kq^2)
    @. dp.py   = (dp.Pt + dp.c/dp.a)/(1-dp.k/dp.a)
    @. dp.Ry   = dp.py - dp.Pt
    @. dp.pd   = dp.py - dp.Ry*dp.k/dp.a
    @. dp.τd   = dp.k*dp.pd + dp.c
    @. dp.pq   = dp.pd + dp.kq*dp.τd
end

function preprocess!(dp::DruckerPrager1)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
    @. dp.cosψ = cosd(dp.ψ)
end

function preprocess!(dh::DruckerHyperbolic)
    @. dh.cosϕ = cosd(dh.ϕ)
    @. dh.sinϕ = sind(dh.ϕ)
    @. dh.sinψ = sind(dh.ψ)
    @. dh.cosψ = cosd(dh.ψ)
end

function preprocess!(da::DruckerAniso)
    @. da.cosϕ = cosd(da.ϕ)
    @. da.sinϕ = sind(da.ϕ)
    @. da.sinψ = sind(da.ψ)
    @. da.cosψ = cosd(da.ψ)
end

function preprocess!(g::Golchin2021)
    @. g.cosϕ = cosd(g.ϕ)
    @. g.sinϕ = sind(g.ϕ)
    @. g.sinψ = sind(g.ψ)
    @. g.cosψ = cosd(g.ψ)
    @. g.M    = 3 * sind(g.ϕ) / (3 - sind(g.ϕ))
    @. g.N    = 3 * sind(g.ψ) / (3 - sind(g.ψ))

end

function preprocess!(k::Kiss2023)
    @. k.cosϕ = cosd(k.ϕ)
    @. k.sinϕ = sind(k.ϕ)
    @. k.sinψ = sind(k.ψ)
    @. k.P1   = (k.Pt - k.δPt)
    @. k.τ1   = -k.δPt
    @. k.P2   = -(-k.Pt - k.C * cosd(k.ϕ)) / (1.0 - sind(k.ϕ))
    @. k.τ2   = k.P2 - k.Pt
end

function preprocess!(t::Tensile)
    @. t.cosϕ = cosd(t.ϕ)
    @. t.sinϕ = sind(t.ϕ)
    @. t.sinψ = sind(t.ψ)
    @. t.cosψ = cosd(t.ψ)
end

function preprocess!(mat::Materials)
    @. mat.B = (2 * mat.η0)^(-mat.n)
    preprocess!(mat.plasticity)
    preprocess!(mat.anisotropy, mat)
end

  # This is same as above, could be Union ? 
function preprocess!(mat::Materials_TwoPhases)
    @. mat.B = (2 * mat.η0)^(-mat.n)
    preprocess!(mat.plasticity)
end

preprocess!(::VonMises) = nothing

preprocess!(::NoPlasticity) = nothing

preprocess!(::Isotropic, mat) = nothing

function preprocess!(a::Anisotropy, mat)
    all(α -> 0 < α < 1, a.α) || error("Anisotropy: the strong-layer fractions α must be in (0, 1) (got $(a.α))")
    all(>(0), a.mv) && all(>(0), a.me) || error("Anisotropy: the contrasts mv, me must be > 0")
    all(>(0), a.n1) && all(>(0), a.n2) || error("Anisotropy: the layer power-law exponents n1, n2 must be > 0")
    all(r -> 0 < r <= 4, a.r) || error("Anisotropy: the shape exponent r must be in (0, 4] (got $(a.r))")
    all(>=(0), a.ϵv) || error("Anisotropy: the regularisation ϵv must be ≥ 0 (got $(a.ϵv))")
    @. a.δv = (1 - a.α + a.α * a.mv) * (1 - a.α + a.α / a.mv)
    @. a.δe = (1 - a.α + a.α * a.me) * (1 - a.α + a.α / a.me)
    return nothing
end

preprocess(x) = (preprocess!(x); x)
