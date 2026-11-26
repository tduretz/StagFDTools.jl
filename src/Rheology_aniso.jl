# Rotation matrix
function 𝐑(θ)
    c, s = cos(θ), sin(θ)
    return @SMatrix([ c^2   s^2   2c*s;
                      s^2   c^2  -2c*s;
                     -c*s   c*s   c^2-s^2])
end

# (N + a)^q - a^q for N ≥ 0, without cancellation when N ≪ a: a^q expm1(q log1p(N/a)); ≥ 0 also in floating point
@inline powdiff(N, a, q) = (N <= a && a > 0) ? a^q * expm1(q * log1p(N / a)) : (N + a)^q - a^q

# Anisotropic yield function, N = ½(τxx²+τyy²), S = ½(τxy²+τyx²), a = δlin S:
# (N + a)^r + (δ²ʳ - δlin^r) S^r = [(N + a)^r - a^r] + δ²ʳ S^r
# The bracket is evaluated with powdiff: the plain difference jumps by one rounding step when the layer normal stresses
# vanish (N ≪ a, e.g. fabric at 45° to pure shear), which stalls the local Newton
function f(τ, δlin, r, δ²ʳ)
    N = 0.5 * (τ[1]^2 + τ[2]^2)
    S = 0.5 * (τ[3]^2 + τ[4]^2)
    return (powdiff(N, δlin * S, r) + δ²ʳ * S^r)^(1/(2r))
end

# per phase stress update
τ_i(ηve, ε̇, τ0, χ) = 2ηve * ε̇ + τ0 * χ

# signed power: keeps the sign of the per-phase stresses / strain rates
spow(a, p) = sign(a)*abs(a)^p

# signed power with a finite slope at a = 0 (p < 1): a (a² + ϵ²)^((p-1)/2), = spow for ϵ = 0 or p = 1
spow_reg(a, p, ϵ) = a * (a^2 + ϵ^2)^((p - 1)/2)

# Potential in known directions: normal and transverse
𝐘_transverse(τxy, α1, α2, B1, B2, n1, n2) = α1 * τxy^(n1+1) / ((n1+1)*B1^n1) + α2 * τxy^(n2+1) / ((n2+1)*B2^n2)
      𝐘_normal(ε̇, α1, α2, B1, B2, n1, n2) = α1*B1 * ε̇^((n1+1)/n1) / (n1+1) + α2*B2 * ε̇^((n2+1)/(n2)) / (n2+1)

function minimise_shape(x, α1, α2, B1, B2, n1, n2, r)

    # Unpack Unknowns
    τ = @SVector([x[1], x[2], x[3], x[4]])
    E = x[5]
    τt = x[6]

    # Find strong direction from E: the choice of uising E is to avoid other two
    # unknowns: the two normal stresses -> we use the fact that they share the same
    # normal strain rate and compute taun from that
    τ1i = B1*E^(1/n1)
    τ2i = B2*E^(1/n2)
    τn  = α1*τ1i + α2*τ2i

    # Compute anisotropic coefficients 
    δlin = τn*(α1/τ1i + α2/τ2i)
    δ = τn/τt
    δ²ʳ = δ^(2r)

    return  f(τ, δlin, r, δ²ʳ) - τn
end

# Value and analytic gradient of minimise_shape w.r.t. x = (τ1, τ2, τ3, τ4, E, τt) (reference definition: minimise_shape).
# Replaces the AD gradient nested in the AD jacobian of aniso_redidual
function minimise_shape_grad(x, α1, α2, B1, B2, n1, n2, r)
    τ1, τ2, τ3, τ4, E, τt = x[1], x[2], x[3], x[4], x[5], x[6]

    # Endmember stresses at the shared normal strain rate E and their E-derivatives
    τ1i  = B1*E^(1/n1);  dτ1i = τ1i/(n1*E)
    τ2i  = B2*E^(1/n2);  dτ2i = τ2i/(n2*E)
    τn   = α1*τ1i + α2*τ2i
    dτn  = α1*dτ1i + α2*dτ2i

    # Anisotropic coefficients
    w     = α1/τ1i + α2/τ2i
    δlin  = τn*w
    dδlin = dτn*w - τn*(α1*dτ1i/τ1i^2 + α2*dτ2i/τ2i^2)
    D     = (τn/τt)^(2r)                         # δ²ʳ

    # f = F^(1/2r), F = [(N + a)^r - a^r] + D S^r, a = δlin S
    N     = 0.5*(τ1^2 + τ2^2)
    S     = 0.5*(τ3^2 + τ4^2)
    a     = δlin*S
    Sr    = S^r
    F     = powdiff(N, a, r) + D*Sr
    fv    = F^(1/(2r))
    c     = fv/(2r*F)                            # df/dF
    dFdN  = r*(N + a)^(r - 1)
    dFda  = r*powdiff(N, a, r - 1)               # r[(N + a)^(r-1) - a^(r-1)]
    dFdS  = dFda*δlin + D*r*S^(r - 1)
    dFdE  = dFda*S*dδlin + Sr*2r*D*dτn/τn
    dFdτt = -Sr*2r*D/τt

    G  = fv - τn
    ∂G = SVector(c*dFdN*τ1, c*dFdN*τ2, c*dFdS*τ3, c*dFdS*τ4, c*dFdE - dτn, c*dFdτt)
    return G, ∂G
end

# Value of a (possibly nested) Dual, and of static arrays of them
_primal(x::Real) = x
_primal(x::StagFDTools.Dual) = _primal(StagFDTools.value(x))
_primal(x::StaticArray) = map(_primal, x)

function aniso_redidual(x, 𝐃el, ε̇′, τ0′, τ0_1, τ0_2, α1, α2, B1, B2, n1, n2, ηe1, ηe2, r, ϵv)

    # Unpack the unknowns for clarity: the endmember pair is solved in log space,
    # since E enters as E^(1/n) (infinite slope at 0) -> Newton overshoots to E < 0
    En  = exp(x[5])
    τt  = exp(x[6])
    τ_1 = x[7]
    τ_2 = x[8]

    # Per-phase viscous strain rates in the normal direction
    ε̇v1 = spow(τ_1/B1, n1)
    ε̇v2 = spow(τ_2/B2, n2)

    # Bulk viscous normal strain rate: stiffness-weighted average of the phases
    ε̇vn = (α1*ηe1*ε̇v1 + α2*ηe2*ε̇v2) / (α1*ηe1 + α2*ηe2)

    # Correction of the normal stress seen by the viscous potential: stress carried at ε̇vn
    # if the phases shared the viscous strain rate, minus the stress the phases carry.
    # The 1/n root has an infinite slope at ε̇vn = 0 (layer normal stresses changing sign), where the global Newton
    # stalls -> regularised at ϵv times the viscous strain-rate magnitude E (caps the viscosity of this term)
    ϵr  = ϵv * En
    Δτn = α1*B1*spow_reg(ε̇vn, 1/n1, ϵr) + α2*B2*spow_reg(ε̇vn, 1/n2, ϵr) - (α1*τ_1 + α2*τ_2)
    τv  = @SVector([x[1] + Δτn, x[2] - Δτn, x[3], x[4], En, τt])

    # Minimise the shape potential
    G, ∂G∂x = minimise_shape_grad(τv, α1, α2, B1, B2, n1, n2, r)

    # Minimise the value of the potential at endmember directions
    # (in log form: same root, but ~linear in the log unknowns instead of exp((n+1)*x))
    𝚿 = log(𝐘_transverse(τt, α1, α2, B1, B2, n1, n2)) - log(𝐘_normal(En, α1, α2, B1, B2, n1, n2))

    # Magnitude of power-law: algebra of derivatives
    dPdE   = α1*B1*En^(1/n1)/n1 + α2*B2*En^(1/n2)/n2
    dτtdE  = dPdE / (α1*(τt/B1)^n1 + α2*(τt/B2)^n2)
    dGdE   = ∂G∂x[5] + ∂G∂x[6]*dτtdE
    B_bulk =  -2*dPdE/dGdE
    ε̇pwl   = B_bulk * @SVector([∂G∂x[1], ∂G∂x[2], ∂G∂x[3], ∂G∂x[4]])

    return @SVector([x[1] - 𝐃el[1,1] * (ε̇′[1] - ε̇pwl[1]) - τ0′[1],
                     x[2] - 𝐃el[2,2] * (ε̇′[2] - ε̇pwl[2]) - τ0′[2],
                     x[3] - 𝐃el[3,3] * (ε̇′[3] - ε̇pwl[3]) - τ0′[3],
                     x[4] - 𝐃el[4,4] * (ε̇′[4] - ε̇pwl[4]) - τ0′[4],
                     G, 𝚿,
                     (τ_1 - τ0_1)/(2ηe1) + ε̇v1 - ε̇′[1],
                     (τ_2 - τ0_2)/(2ηe2) + ε̇v2 - ε̇′[1]])
end

# Power-law viscoelasticity of the laminate: layers 1 (weak) and 2 (strong), volume fractions 1-α2 and α2, τ = Bᵢ ε̇^(1/nᵢ).
# η: composite normal consistency (at unit strain rate), mv = B₂/B₁, G: normal (arithmetic) shear modulus, me = G₂/G₁,
# r: shape exponent of the yield function f. Returns the composite stress (x-y frame), the new normal stresses of
# layers 1 and 2 (layer frame) and a secant visco-elastic normal viscosity.
function aniso_ViscoElastic_powerlaw(ε̇, τ0, τ0_2, G, η, δe, me, mv, n1, n2, r, ϵv, α2, Δt, nd)
    # Rotate
    θ   = atan(-nd[1], nd[2])
    ε̇r  = 𝐑(θ) * ε̇
    τ0r = 𝐑(θ) * τ0

    # 4-component layer-frame vectors (xx, yy, xy, yx) used by the residual
    ε̇′  = @SVector([ε̇r[1],  ε̇r[2],  ε̇r[3],  ε̇r[3]])
    τ0′ = @SVector([τ0r[1], τ0r[2], τ0r[3], τ0r[3]])

    # Layer elastic viscosities (ηe₂ from the composite normal value; ηe₁ = ηe₂/me) and elastic stiffness (layer frame)
    α1  = 1 - α2
    ηe  = G * Δt
    ηe2 = ηe / (α1 / me + α2)
    ηe1 = ηe2 / me
    𝐃el = @SMatrix([2ηe 0. 0. 0.; 0. 2ηe 0. 0.; 0. 0. 2ηe/δe 0.; 0. 0. 0. 2ηe/δe])

    # Layer consistencies (B = 2η for n = 1)
    B2  = 2η / (α1 / mv + α2)
    B1  = B2 / mv

    # Old normal stress of layer 1
    τ0_1 = (τ0′[1] - α2 * τ0_2) / α1

    # Parameters of the residual (Duals when called by the tangent operator) and their values: the Newton iterations run
    # on values, the derivatives follow from the implicit function theorem at the end
    p  = (𝐃el, ε̇′, τ0′, τ0_1, τ0_2, α1, α2, B1, B2, n1, n2, ηe1, ηe2, r, ϵv)
    pv = map(_primal, p)
    𝐃elv, ε̇′v, τ0′v, τ0_1v, τ0_2v, α1v, α2v, B1v, B2v, n1v, n2v, ηe1v, ηe2v = pv

    # Iter params
    max_iter  = 20
    tolerance = 1e-11
    converged = false

    # Unknowns block w initial guesses (all from upper bounds) -----------------------------------
    # From the effective strain rate ε̇′ + τ0′/𝐃el (Maxwell trial): right sign and magnitude also when the old stress
    # relaxes against a small or reversed strain rate (with ε̇′ alone the guess has the wrong sign and Newton fails)
    ε̇g  = @SVector([ε̇′v[i] + τ0′v[i]/𝐃elv[i,i] for i in 1:4])
    E0  = invII(ε̇g)
    τn0 = α1v*B1v*E0^(1/n1v) + α2v*B2v*E0^(1/n2v)
    # each phase alone gives an upper bound -> Newton converges monotonically from above
    R   = 𝐘_normal(E0, α1v, α2v, B1v, B2v, n1v, n2v)
    τxy = min(((n1v+1)*B1v^n1v*R/α1v)^(1/(n1v+1)), ((n2v+1)*B2v^n2v*R/α2v)^(1/(n2v+1)))
    # Maxwell: elastic or viscous part alone gives an upper bound of the stress magnitude
    τv0 = @SVector([τn0*ε̇g[1]/E0, τn0*ε̇g[2]/E0, τxy*ε̇g[3]/E0, τxy*ε̇g[4]/E0])
    τe0 = @SVector([𝐃elv[i,i]*ε̇g[i] for i in 1:4])
    τ00 = @SVector([abs(τv0[i]) < abs(τe0[i]) ? τv0[i] : τe0[i] for i in 1:4])
    # scale the endmember pair to the stress level of the guess (n_min keeps E0 from above)
    s    = invII(τ00) / invII(τv0)
    E0   = E0*s^min(n1v, n2v)
    τxy  = τxy*s
    a1   = ε̇′v[1] + τ0_1v/(2ηe1v)
    a2   = ε̇′v[1] + τ0_2v/(2ηe2v)
    τ_10 = sign(a1)*min(B1v*abs(a1)^(1/n1v), 2ηe1v*abs(a1))
    τ_20 = sign(a2)*min(B2v*abs(a2)^(1/n2v), 2ηe2v*abs(a2))

    # Unknown vector
    x    = @SVector([τ00[1], τ00[2], τ00[3], τ00[4], log(E0), log(τxy), τ_10, τ_20])

    # Newton iterations on values
    𝑟, ∂x∂r = fd_value_and_jacobian(aniso_redidual, x, pv...)
    for _ in 1:max_iter
        if norm(𝑟) < tolerance
            converged = true
            break
        end
        δx = ∂x∂r \ 𝑟

        # Convergence check
        ω = 1.0
        for _ in 1:30
            norm(aniso_redidual(x - ω*δx, pv...)) < norm(𝑟) && break
            ω /= 2
        end
        x -= ω*δx
        𝑟, ∂x∂r = fd_value_and_jacobian(aniso_redidual, x, pv...)
    end
    converged || @warn "stress_solver did not converge"

    # Implicit function theorem: one Newton correction with the parameters p (Duals in the tangent operator) at the
    # converged x gives the exact first derivatives of x, without differentiating through the iterations
    xd = x - ∂x∂r \ aniso_redidual(x, p...)

    # De-Rotate (back to 3 components: xx, yy, xy)
    τ′ = @SVector([xd[1], xd[2], xd[3]])
    τ  = 𝐑(-θ) * τ′

    # Updated per-phase stresses in the normal direction
    τ_1 = xd[7]
    τ_2 = xd[8]

    # Secant visco-elastic normal viscosity: viscous normal secant viscosity at the converged E, in series with ηe
    E   = exp(xd[5])
    ηvn = (α1*B1*E^(1/n1) + α2*B2*E^(1/n2)) / (2E)
    ηve = inv(1/ηvn + 1/ηe)

    return τ, τ_1, τ_2, ηve
end
function aniso_ViscoElastic(ε̇, τ0, τ0_2, G, η, δe, δv, me, mv, α2, Δt, n)

    # Rotate
    θ   = atan(-n[1], n[2])
    ε̇′  = 𝐑(θ) * ε̇
    τ0′ = 𝐑(θ) * τ0

    # Layer visco-elastic viscosities (η₂, ηe₂ from the composite normal values; η₁ = η₂/mv, ηe₁ = ηe₂/me)
    α1   = 1 - α2
    ηe   = G * Δt
    ηe2  = ηe / (α1 / me + α2)
    ηve2 = inv(1 / (η / (α1 / mv + α2)) + 1 / ηe2)
    ηve1 = inv(mv * (α1 / mv + α2) / η + me / ηe2)
    ηve  = α1 * ηve1 + α2 * ηve2
    mve  = ηve2 / ηve1
    δve  = (α1 + α2 * mve) * (α1 + α2 / mve)
    𝐃ve  = @SMatrix([2ηve 0. 0.; 0. 2ηve 0.; 0. 0. 2ηve/δve])

    # Old stress coefficients
    χs = (ηve / δve) / (ηe / δe)
    χ2 = ηve2 / ηe2
    χ1 = χ2 * me / mve

    # Solve
    τ0_bulk = τ0′[1]
    τ0_s    = τ0′[3]
    τ0_1    = (τ0_bulk - α2 * τ0_2) / α1

    # Update per-phase stresses
    τ_2 = τ_i(ηve2, ε̇′[1], τ0_2, χ2)
    τ0n = α1 * τ0_1 * χ1 + α2 * τ0_2 * χ2
    τ0′ = @SVector([τ0n, -τ0n, χs * τ0_s])
    τ′  = 𝐃ve * ε̇′ + τ0′

    # De-Rotate
    τ = 𝐑(-θ) * τ′

    return τ, τ_2, ηve
end

function LocalRheology_aniso(ε̇p, τ0p, τ0v, nd, Ptr, materials, p, Δ)
    an = materials.anisotropy
    α2 = an.α[p]
    θ  = atan(-nd[1], nd[2])
    τ0_2 = (𝐑(θ) * τ0v[2])[1]
    if isone(an.n1[p]) && isone(an.n2[p])   # linear layers
        τ, τ_2, ηve = aniso_ViscoElastic(ε̇p, τ0p, τ0_2, materials.G[p], materials.η0[p], an.δe[p], an.δv[p], an.me[p], an.mv[p], α2, Δ.t, nd)
        τ′  = 𝐑(θ) * τ
        τ_1 = (τ′[1] - α2 * τ_2) / (1 - α2)
    else                                    # power-law layers
        τ, τ_1, τ_2, ηve = aniso_ViscoElastic_powerlaw(ε̇p, τ0p, τ0_2, materials.G[p], materials.η0[p], an.δe[p], an.me[p], an.mv[p],
                                                       an.n1[p], an.n2[p], an.r[p], an.ϵv[p], α2, Δ.t, nd)
        τ′  = 𝐑(θ) * τ
    end
    Rᵀ  = 𝐑(-θ)
    τv  = SVector(Rᵀ * SVector(τ_1, -τ_1, τ′[3]), Rᵀ * SVector(τ_2, -τ_2, τ′[3]))
    z   = zero(ηve)
    return τ, Ptr, z, ηve, z, τv
end

has_anisotropy(materials) = any(p -> is_anisotropic(materials, p), eachindex(materials.n))

# Unit director at a grid point: value of the director field, or the initial director of the materials where it is not set (0)
@inline function director(nx, ny, an::Anisotropy)
    if iszero(nx) && iszero(ny)
        nx, ny = an.dv[1], an.dv[2]
    end
    s = inv(sqrt(nx^2 + ny^2))
    return SVector(nx * s, ny * s)
end
