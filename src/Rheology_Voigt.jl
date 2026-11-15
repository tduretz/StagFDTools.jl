# Iso-strain rate (Voigt) mixing of the phases in a cell (materials.mixing = :voigt), and the per-phase stress
# framework also used by the iso-work rate mixing (materials.mixing = :isowork, Rheology_IsoWork.jl: phase i sees aᵢε̇)
#
# All phases see the cell strain rate ε̇, but each phase i keeps its own old stress τ0ᵢ and shear modulus Gᵢ:
#     τᵢ = 2ηᵢ ϵ̇ᵢ,   ϵ̇ᵢ = ε̇ + τ0ᵢ/(2GᵢΔt),   ηᵢ = τIIᵢ/(2ϵ̇IIᵢ)   (full VEP update of phase i)
#     τ  = Σᵢ φᵢ τᵢ = 2η̄ ε̇ + Σᵢ φᵢ ηᵢ τ0ᵢ/(GᵢΔt),   η̄ = Σᵢ φᵢ ηᵢ
# τ is in general not parallel to a single effective strain rate. To keep the momentum residual of Stokes.jl
# (τ = 2η̄ (ε̇ + τ0/(2GΔt)) with the grid G and the Picard operator 𝐷 = 2η̄) exact, the tangent operator overwrites
# τ0 (xx, yy at centroids, xy at vertices) with the equivalent old stress
#     τ0_eq = (τ - 2η̄ ε̇) GΔt / η̄
# The physical per-phase old stresses live in τ0.ph, the per-phase new stresses (and ηᵢ) in τ.ph (see Allocs.jl).
# Per-phase stresses are stored for all phases, also where a phase is absent (φᵢ = 0: "what phase i would carry here"),
# which makes the marker interpolations well defined.
#
# Markers (TensorMarkers): each marker carries the stress of its own phase. Grid ↔ marker transfers are done per phase,
# weighted by the phase, with the standard JustPIC kernels:
#     P2G: Fᵢ = P2G(mᵢ F) / P2G(mᵢ),        mᵢ = (marker phase == i)
#     G2P: Fᵢ = G2P(φᵢ Fᵢ) / G2P(φᵢ)        for the markers of phase i
# Pressure: shared P0, or per-phase P0ᵢ (materials.pressure = :phase), transferred like the stresses.

# ------------------------------------------------------------------------------------------------------------------
# Local rheology

# Grid compressibility: harmonic average of the phase β (as compute_grid_fields!, default mode)
@inline function grid_β(materials, phase_ratios)
    s = 0.0
    for p in eachindex(materials.β)
        s += phase_ratios[p] / materials.β[p]
    end
    return inv(s)
end

# ε̇: true deviatoric strain rate (ε̇xx, ε̇yy, ε̇xy, P). T0 = (τ0xx, τ0yy, τ0xy) per-phase arrays at the location I.
# Phase p sees the strain rate aₚ ε̇: aₚ = 1 (Voigt) or the iso-work partition (Rheology_IsoWork.jl), see phase_multipliers
function StressVector_Voigt!(ε̇, ε̇kk, P0, T0, I, materials, phase_ratios, Δ, Tk=nothing, εp=0.0, Ω=0.0)
    nphases = length(materials.n)
    eps0 = 1e-17
    T = promote_type(eltype(ε̇), typeof(P0))
    τxx, τyy, τxy, P_average = zero(T), zero(T), zero(T), zero(T)
    η_average, λ̇_average = zero(T), zero(T)
    βbar = grid_β(materials, phase_ratios)
    A = phase_multipliers(ε̇, ε̇kk, P0, T0, I, materials, phase_ratios, Δ, Tk, εp, Ω)
    for p = 1:nphases
        φ = phase_ratios[p]
        φ > eps() || continue
        τp, Pp, λ̇p, ηp = voigt_phase(ε̇, ε̇kk, P0, T0, I, materials, p, Δ, Tk, εp, Ω, eps0, A[p], βbar)
        τxx += φ * τp[1]
        τyy += φ * τp[2]
        τxy += φ * τp[3]
        P_average += φ * Pp
        η_average += φ * ηp
        λ̇_average += φ * λ̇p
    end
    τII = sqrt((τxx^2 + τyy^2 + (-τxx - τyy)^2) / 2 + τxy^2)
    return SVector{4}(τxx, τyy, τxy, P_average), η_average, λ̇_average, τII
end

# a: strain rate multiplier of phase p (its strain rate is a ε̇; 1: Voigt)
# βbar: grid compressibility (harmonic average of the phase β, as compute_grid_fields!), for the per-phase pressure
# T0 = (τ0xx, τ0yy, τ0xy, P0) per-phase old stresses and pressure
@inline function voigt_phase(ε̇, ε̇kk, P0, T0, I, materials, p, Δ, Tk, εp, Ω, eps0=1e-17, a=1.0, βbar=1.0)
    # Pressure: shared (trial = cell trial pressure ε̇[4]), or per phase (materials.pressure = :phase):
    # phase p has its own old pressure P0ₚ and the same volumetric strain rate as the cell:
    #     Pₚ_trial = P0ₚ - KₚΔt ∇⋅v = P0ₚ + (βbar/βₚ)(P_trial - P0),   Σφₚ Pₚ_trial = P_trial if P0 = Σφₚ P0ₚ
    P0p, Ptr = if _phase_pressure(materials)
        P0ₚ = T0[4][p][I]
        P0ₚ, (materials.compressible ? P0ₚ + βbar / materials.β[p] * (ε̇[4] - P0) : one(P0ₚ) * ε̇[4])
    else
        P0, ε̇[4]
    end
    G = materials.G[p]
    _2GΔt = inv(2 * G * Δ.t)
    ϵ̇xx = a * ε̇[1] + T0[1][p][I] * _2GΔt
    ϵ̇yy = a * ε̇[2] + T0[2][p][I] * _2GΔt
    ϵ̇xy = a * ε̇[3] + T0[3][p][I] * _2GΔt
    ϵ̇II = sqrt((ϵ̇xx^2 + ϵ̇yy^2 + (-ϵ̇xx - ϵ̇yy)^2) / 2 + ϵ̇xy^2 + eps0^2) # regularised inside: finite derivative at ϵ̇ = 0 (iso-work, a = 0)
    τII, Pp, λ̇p, ε̇plp = phase_vep(ϵ̇II, Ptr, ε̇kk, P0p, G, materials, p, Δ, Tk, _εp(εp, p), Ω)
    η = τII / (2 * ϵ̇II)
    return SVector{3}(2η * ϵ̇xx, 2η * ϵ̇yy, 2η * ϵ̇xy), Pp, λ̇p, η, ε̇plp
end

# Called by TangentOperator! (Stokes.jl) at a centroid (loc = :c) or vertex (loc = :v) when materials.mixing is :voigt or :isowork.
# Returns what the default branch gets from StressVector!, writes τ0_eq and the per-phase stresses.
# ε̇plph: nothing, or per-phase fields receiving the per-phase plastic strain rates ε̇IIplᵢ (plastic strain accumulation)
function TangentOperator_Voigt!(τ0, τ, i, j, loc::Symbol, ε̇vec, ε̇kk, P0, Ggrid, materials, phase_ratios, Δ, Tk, εp, Ω, ε̇plph=nothing)
    I = CartesianIndex(i, j)
    T0 = loc === :c ? (τ0.ph.xx_c, τ0.ph.yy_c, τ0.ph.xy_c, τ0.ph.P_c) : (τ0.ph.xx_v, τ0.ph.yy_v, τ0.ph.xy_v, τ0.ph.P_v)
    τ_vec, η_local, λ̇_local, τII_local = StressVector_Voigt!(ε̇vec, ε̇kk, P0, T0, I, materials, phase_ratios, Δ, Tk, εp, Ω)
    _, jac = fd_jacobian_first(StressVector_Voigt!, ε̇vec, ε̇kk, P0, T0, I, materials, phase_ratios, Δ, Tk, εp, Ω)

    # Equivalent old stress for the momentum residual: 2η̄(ε̇ + τ0_eq/(2GΔt)) = τ
    s = Ggrid * Δ.t / η_local
    if loc === :c
        τ0.xx[I] = (τ_vec[1] - 2 * η_local * ε̇vec[1]) * s
        τ0.yy[I] = (τ_vec[2] - 2 * η_local * ε̇vec[2]) * s
    else
        τ0.xy[I] = (τ_vec[3] - 2 * η_local * ε̇vec[3]) * s
    end

    # Per-phase stresses (all phases, also absent ones: a = 1), for the markers
    Tn = loc === :c ? (τ.ph.xx_c, τ.ph.yy_c, τ.ph.xy_c, τ.ph.η_c, τ.ph.P_c) : (τ.ph.xx_v, τ.ph.yy_v, τ.ph.xy_v, τ.ph.η_v, τ.ph.P_v)
    A = phase_multipliers(ε̇vec, ε̇kk, P0, T0, I, materials, phase_ratios, Δ, Tk, εp, Ω)
    βbar = grid_β(materials, phase_ratios)
    for p in eachindex(materials.n)
        τp, Pp, _, ηp, ε̇plp = voigt_phase(ε̇vec, ε̇kk, P0, T0, I, materials, p, Δ, Tk, εp, Ω, 1e-17, phase_ratios[p] > eps() ? A[p] : 1.0, βbar)
        ε̇plph === nothing || (ε̇plph[p][I] = phase_ratios[p] > eps() ? ε̇plp : 0.0)
        Tn[1][p][I] = τp[1]
        Tn[2][p][I] = τp[2]
        Tn[3][p][I] = τp[3]
        Tn[4][p][I] = ηp
        Tn[5][p][I] = Pp
    end
    return τ_vec, jac, η_local, λ̇_local, τII_local
end

# Grid-only runs (no stress markers): per-phase old stresses = per-phase new stresses of the last step
function update_old_stress_Voigt!(a)
    for k in (:xx_c, :yy_c, :xy_c, :xx_v, :yy_v, :xy_v, :P_c, :P_v), p in eachindex(a.τ0.ph[k])
        a.τ0.ph[k][p] .= a.τ.ph[k][p]
    end
    return nothing
end

# ------------------------------------------------------------------------------------------------------------------
# Marker transfers

struct VoigtWork{M,G}
    mf::M     # marker scratch fields
    φc::Vector{G}
    φv::Vector{G}
    num_c::G
    den_c::G
    mix_c::G
    num_v::G
    den_v::G
    mix_v::G
    D::NamedTuple
    D_sub::NamedTuple
    D_P::Vector{G}
end

const voigt_work_cache = Ref{Any}(nothing)
const voigt_work_key = Ref{UInt}(0)

function voigt_work(a, particles, nphases)
    key = objectid(particles)
    if voigt_work_cache[] === nothing || voigt_work_key[] != key
        sc, sv = size(a.τ0.xx), size(a.τ0.xy)
        phc() = [zeros(sc...) for _ in 1:nphases]
        phv() = [zeros(sv...) for _ in 1:nphases]
        voigt_work_cache[] = VoigtWork(cell_fields(particles, :m, :f, :a, :b, :c),
            phc(), phv(), zeros(sc...), zeros(sc...), zeros(sc...), zeros(sv...), zeros(sv...), zeros(sv...),
            (xx_c=phc(), yy_c=phc(), xy_v=phv()), (xx_c=phc(), yy_c=phc(), xy_v=phv()), phc())
        voigt_work_key[] = key
    end
    w = voigt_work_cache[]
    for p in 1:nphases
        w.φc[p] .= getindex.(a.phase_ratios.c, p)
        w.φv[p] .= getindex.(a.phase_ratios.v, p)
    end
    return w
end

@inline marker_phase(phases, ip, i, j) = round(Int, @index(phases[ip, i, j]))

# Loop over active markers: f(ip, i, j)
function foreach_marker(f::F, particles) where {F}
    (; index) = particles
    Threads.@threads for j in axes(index, 2)
        for i in axes(index, 1), ip in cellaxes(index)
            @index(index[ip, i, j]) || continue
            f(ip, i, j)
        end
    end
    return nothing
end

_p2g!(F, Fp, particles, loc) = loc === :c ? particle2centroid!(F, Fp, particles) : particle2grid!(F, Fp, particles)
_g2p!(Fp, F, particles, loc) = loc === :c ? centroid2particle!(Fp, F, particles) : grid2particle!(Fp, F, particles)

# Per-phase P2G: Fs[p] = average over the markers of phase p (fallback where there are none)
function masked_p2g!(Fs, Fp, fallback, phases, particles, w, loc)
    num, den = loc === :c ? (w.num_c, w.den_c) : (w.num_v, w.den_v)
    m, f = w.mf.m, w.mf.f
    for p in eachindex(Fs)
        foreach_marker(particles) do ip, i, j
            isp = marker_phase(phases, ip, i, j) == p
            @index m[ip, i, j] = isp ? 1.0 : 0.0
            @index f[ip, i, j] = isp ? @index(Fp[ip, i, j]) : 0.0
        end
        _p2g!(num, f, particles, loc)
        _p2g!(den, m, particles, loc)
        @. Fs[p] = ifelse(den > 1e-12, num / den, fallback)
    end
    return nothing
end

# Per-phase G2P: markers of phase p receive Fs[p] weighted by φp (unweighted where φp vanishes around the marker)
function masked_g2p!(Fp, Fs, φs, phases, particles, w, loc)
    num = loc === :c ? w.num_c : w.num_v
    a, b, c = w.mf.a, w.mf.b, w.mf.c
    for p in eachindex(Fs)
        @. num = φs[p] * Fs[p]
        _g2p!(a, num, particles, loc)
        _g2p!(b, φs[p], particles, loc)
        _g2p!(c, Fs[p], particles, loc)
        foreach_marker(particles) do ip, i, j
            marker_phase(phases, ip, i, j) == p || return
            bb = @index b[ip, i, j]
            @index Fp[ip, i, j] = bb > 1e-12 ? @index(a[ip, i, j]) / bb : @index(c[ip, i, j])
        end
    end
    return nothing
end

function zero_gradient!(A)
    A[1,   :] .= A[2, :]
    A[end, :] .= A[end-1, :]
    A[:,   1] .= A[:, 2]
    A[:, end] .= A[:, end-1]
    return nothing
end

# Markers → grid: mixture (as stress_ToGrid!) and per-phase old stresses at centroids and vertices
function stress_ToGrid_Voigt!(tm::TensorMarkers, a, particles, phases, nphases)
    stress_ToGrid!(tm, a, particles)
    w = voigt_work(a, particles, nphases)
    ph = a.τ0.ph
    for (Fp, kc, kv) in ((tm.p.τxx, :xx_c, :xx_v), (tm.p.τyy, :yy_c, :yy_v), (tm.p.τxy, :xy_c, :xy_v))
        particle2centroid!(w.mix_c, Fp, particles)
        particle2grid!(w.mix_v, Fp, particles)
        zero_gradient!(w.mix_c)
        zero_gradient!(w.mix_v)
        masked_p2g!(ph[kc], Fp, w.mix_c, phases, particles, w, :c)
        masked_p2g!(ph[kv], Fp, w.mix_v, phases, particles, w, :v)
        foreach(zero_gradient!, ph[kc])
        foreach(zero_gradient!, ph[kv])
    end
    # Per-phase old pressure (materials.pressure = :phase; filled in all cases, cheap compared to the solve)
    particle2grid!(w.mix_v, tm.p.P, particles)
    zero_gradient!(w.mix_v)
    masked_p2g!(ph.P_c, tm.p.P, a.Pt0, phases, particles, w, :c)
    masked_p2g!(ph.P_v, tm.p.P, w.mix_v, phases, particles, w, :v)
    foreach(zero_gradient!, ph.P_c)
    foreach(zero_gradient!, ph.P_v)
    return nothing
end

# Per-phase pressure: cell old pressure = Σφᵢ P0ᵢ, so that the per-phase trial pressures add up to the cell trial pressure
# (called at the start of the solve, with the phase ratios of the step)
function sync_old_pressure_Voigt!(a)
    P0ph = a.τ0.ph.P_c
    # First use (per-phase pressures never set, e.g. step 1 or a grid-only run): all phases start from the cell pressure
    if all(P -> all(iszero, P), P0ph) && !all(iszero, a.Pt0)
        for p in eachindex(P0ph)
            P0ph[p] .= a.Pt0
            Pv = a.τ0.ph.P_v[p]
            for j in axes(Pv, 2), i in axes(Pv, 1)
                ic, jc = clamp(i, 2, size(a.Pt0, 1)), clamp(j, 2, size(a.Pt0, 2))
                Pv[i, j] = 0.25 * (a.Pt0[ic-1, jc-1] + a.Pt0[ic, jc-1] + a.Pt0[ic-1, jc] + a.Pt0[ic, jc])
            end
        end
        return nothing
    end
    for I in CartesianIndices(a.Pt0)
        pr = a.phase_ratios.c[I]
        s = 0.0
        for p in eachindex(P0ph)
            s += pr[p] * P0ph[p][I]
        end
        sum(pr) > 0.5 && (a.Pt0[I] = s)
    end
    return nothing
end

# Grid → markers after the solve: per-phase increments (+ per-phase subgrid diffusion), rotation, update
function stress_ToParticles_Voigt!(a, tm::TensorMarkers, materials, particles, phases, Δ; d=1.0, rotation::Symbol=:upper_convected)
    nphases = length(materials.n)
    w       = voigt_work(a, particles, nphases)
    compute_vorticity!(a, Δ)
    τ0, τ = a.τ0.ph, a.τ.ph

    # Per-phase grid increments
    for p in 1:nphases
        @. w.D.xx_c[p] = τ.xx_c[p] - τ0.xx_c[p]
        @. w.D.yy_c[p] = τ.yy_c[p] - τ0.yy_c[p]
        @. w.D.xy_v[p] = τ.xy_v[p] - τ0.xy_v[p]
    end

    # Subgrid diffusion, per phase: markers relax towards the grid old stress of their phase with their phase's Maxwell time
    masked_g2p!(tm.t.τ0xx, τ0.xx_c, w.φc, phases, particles, w, :c)
    masked_g2p!(tm.t.τ0yy, τ0.yy_c, w.φc, phases, particles, w, :c)
    masked_g2p!(tm.t.τ0xy, τ0.xy_v, w.φv, phases, particles, w, :v)
    masked_g2p!(tm.t.η, τ.η_c, w.φc, phases, particles, w, :c)
    dΔt = -d * Δ.t
    foreach_marker(particles) do ip, i, j
        p = marker_phase(phases, ip, i, j)
        G = materials.G[p]
        η = @index tm.t.η[ip, i, j]
        inv_ηM = max(inv(η) - inv(G * Δ.t), 0.0) # viscous(-plastic) part only, see stress_subgrid_diffusion!
        diff = -expm1(dΔt * G * inv_ηM)
        @index tm.Δ_sub.τxx[ip, i, j] = (@index(tm.t.τ0xx[ip, i, j]) - @index(tm.p.τxx[ip, i, j])) * diff
        @index tm.Δ_sub.τyy[ip, i, j] = (@index(tm.t.τ0yy[ip, i, j]) - @index(tm.p.τyy[ip, i, j])) * diff
        @index tm.Δ_sub.τxy[ip, i, j] = (@index(tm.t.τ0xy[ip, i, j]) - @index(tm.p.τxy[ip, i, j])) * diff
    end
    w.mix_c .= 0.0
    w.mix_v .= 0.0
    masked_p2g!(w.D_sub.xx_c, tm.Δ_sub.τxx, w.mix_c, phases, particles, w, :c)
    masked_p2g!(w.D_sub.yy_c, tm.Δ_sub.τyy, w.mix_c, phases, particles, w, :c)
    masked_p2g!(w.D_sub.xy_v, tm.Δ_sub.τxy, w.mix_v, phases, particles, w, :v)
    for p in 1:nphases
        w.D.xx_c[p] .-= w.D_sub.xx_c[p]
        w.D.yy_c[p] .-= w.D_sub.yy_c[p]
        w.D.xy_v[p] .-= w.D_sub.xy_v[p]
    end

    # Per-phase increments to the markers of each phase, + subgrid increments
    masked_g2p!(tm.Δ.τxx, w.D.xx_c, w.φc, phases, particles, w, :c)
    masked_g2p!(tm.Δ.τyy, w.D.yy_c, w.φc, phases, particles, w, :c)
    masked_g2p!(tm.Δ.τxy, w.D.xy_v, w.φv, phases, particles, w, :v)
    if _phase_pressure(materials)
        for p in 1:nphases
            @. w.D_P[p] = τ.P_c[p] - τ0.P_c[p]
        end
        masked_g2p!(tm.Δ.P, w.D_P, w.φc, phases, particles, w, :c)
    else
        a.D.Pt .= a.Pt .- a.Pt0
        centroid2particle!(tm.Δ.P, a.D.Pt, particles)
    end
    add_subgrid_increments!(tm, particles)

    # Rotation with the new stress of the marker's phase
    masked_g2p!(tm.t.τxx, τ.xx_c, w.φc, phases, particles, w, :c)
    masked_g2p!(tm.t.τyy, τ.yy_c, w.φc, phases, particles, w, :c)
    masked_g2p!(tm.t.τxy, τ.xy_v, w.φv, phases, particles, w, :v)
    centroid2particle!(tm.t.P, a.Pt, particles)
    grid2particle!(tm.t.ω, a.ω.xy_v, particles)
    centroid2particle!(tm.t.ε̇xx, a.ε̇.xx, particles)
    centroid2particle!(tm.t.ε̇yy, a.ε̇.yy, particles)
    grid2particle!(tm.t.ε̇xy, a.ε̇.xy, particles)
    rotation_increments!(tm, particles, Δ.t; rotation)
    update_stress_markers!(tm, particles)
    return nothing
end

# ------------------------------------------------------------------------------------------------------------------
# Plastic strain carried by the markers (all schemes with TensorMarkers, called by main_loop)
#
# εp = ∫ε̇II_pl dt, ε̇II_pl: invariant of the plastic part of the deviatoric strain rate of the phase (phase_vep).
# Grid schemes: after the solve, every marker gets Δt ε̇II_pl of its own phase (materials.plastic_strain = :phase) or
# of the grid point (:mixed, Σᵢφᵢε̇II_plᵢ); after advection the grid εp is the average of the markers (per phase for
# :phase). Marker rheology (mixing = :markers*): εp is integrated on each marker (stress_ToParticles_Markers!), only
# the grid → output part (plastic_strain_ToGrid!) is used.

function plastic_strain_ToParticles!(a, tm::TensorMarkers, materials, particles, phases, Δ)
    nphases = length(materials.n)
    w       = voigt_work(a, particles, nphases)
    rate    = w.mf.m # marker plastic strain rate (masked_g2p! uses a, b, c as work arrays)
    ε̇pl     = a.εp_ph.ε̇pl.c
    if _plastic_strain(materials) === :phase
        masked_g2p!(rate, ε̇pl, w.φc, phases, particles, w, :c)
    else
        for I in eachindex(w.mix_c)
            pr = a.phase_ratios.c[I]
            w.mix_c[I] = sum(pr[p] * ε̇pl[p][I] for p in 1:nphases)
        end
        centroid2particle!(rate, w.mix_c, particles)
    end
    Δt = Δ.t
    foreach_marker(particles) do ip, i, j
        @index tm.p.εp[ip, i, j] = @index(tm.p.εp[ip, i, j]) + Δt * max(@index(rate[ip, i, j]), 0.0)
    end
    return nothing
end

# Markers → grid: εp of the grid points (mean of the markers) and, per phase, of the markers of each phase
function plastic_strain_ToGrid!(tm::TensorMarkers, a, materials, particles, phases)
    nphases = length(materials.n)
    w       = voigt_work(a, particles, nphases)
    particle2centroid!(a.εp.c, tm.p.εp, particles)
    particle2grid!(a.εp.v, tm.p.εp, particles)
    zero_gradient!(a.εp.c)
    zero_gradient!(a.εp.v)
    if _plastic_strain(materials) === :phase || _mixing(materials) in (:markers, :markers_stress, :markers_work)
        εph = a.εp_ph.εp
        masked_p2g!(εph.c, tm.p.εp, a.εp.c, phases, particles, w, :c)
        masked_p2g!(εph.v, tm.p.εp, a.εp.v, phases, particles, w, :v)
        foreach(zero_gradient!, εph.c)
        foreach(zero_gradient!, εph.v)
    end
    return nothing
end
