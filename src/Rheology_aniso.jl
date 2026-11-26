# Rotation matrix
function 𝐑(θ)
    c, s = cos(θ), sin(θ)
    return @SMatrix([ c^2   s^2   2c*s;
                      s^2   c^2  -2c*s;
                     -c*s   c*s   c^2-s^2])
end

# per phase stress update
τ_i(ηve, ε̇, τ0, χ) = 2ηve * ε̇ + τ0 * χ

# Linear viscoelasticity of the laminate: layers 1 (weak) and 2 (strong), volume fractions 1-α2 and α2.
# Layer frame (x' along the layers, y' along the director n): normal components iso-strain rate (each layer has its own old
# normal stress), shear iso-stress (one old shear stress).
# τ0: old stress of the composite, τ0_2: old normal stress of layer 2 (layer frame). η, G: normal (arithmetic) values of the
# composite, mv = η₂/η₁, me = G₂/G₁, δe = (1-α2+α2 me)(1-α2+α2/me). δv is not used: the visco-elastic factor δve follows
# from mv, me. Returns the composite stress, the new normal stress of layer 2 (layer frame) and ηve.
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

# Anisotropic phase p, called by voigt_phase (Rheology_Voigt.jl). ε̇p: deviatoric strain rate of the phase, τ0p: old stress of
# the phase (composite), τ0v: old stresses of the 2 layers (x-y frame), nd: unit director, Ptr: trial pressure.
# Returns τp, Pp, λ̇p, η, ε̇plp, τv (see voigt_phase); τv: stresses of the 2 layers (x-y frame), normal parts from τ_2 and the
# composite, shear part = composite (iso-stress).
function LocalRheology_aniso(ε̇p, τ0p, τ0v, nd, Ptr, materials, p, Δ)
    an = materials.anisotropy
    α2 = an.α[p]
    θ  = atan(-nd[1], nd[2])
    τ0_2 = (𝐑(θ) * τ0v[2])[1]
    τ, τ_2, ηve = aniso_ViscoElastic(ε̇p, τ0p, τ0_2, materials.G[p], materials.η0[p], an.δe[p], an.δv[p], an.me[p], an.mv[p], α2, Δ.t, nd)
    τ′  = 𝐑(θ) * τ
    τ_1 = (τ′[1] - α2 * τ_2) / (1 - α2)
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

# ------------------------------------------------------------------------------------------------------------------
# Fabric rotation with finite strain

# Director n (unit normal to the layers) after Δt: ṅ = Wⁿ n, Wⁿ = W - (Dλ - λD), λ = n⊗n, i.e. ṅ = W n - D n + (n⋅Dn) n
# (normal to a material line). W = ½(L - Lᵀ), D = ½(L + Lᵀ), Lᵢⱼ = ∂vᵢ/∂xⱼ; explicit update, renormalised.
function DirectorVector(n0, W, D, Δt)

    # Store previous director
    n₁ = n0[1]
    n₂ = n0[2]
    λ = @SMatrix([n₁^2 n₁*n₂; n₂*n₁ n₂^2])

    # Spin of the director n
    Wⁿ = W - (D * λ - λ * D)

    # New director
    n = (Wⁿ * n0) .* Δt + n0
    n_norm = sqrt(n[1]^2 + n[2]^2)
    n = n ./ n_norm

    return n
end

# DirectorVector applied in substeps of Δt (W, D fixed) so that Δtₛ (|ω| + |D|) ≤ 0.05: the explicit update stays accurate
# where the rotation is fast
@inline function rotate_director(nx, ny, Dxx, Dyy, Dxy, ω, an, Δt)
    n    = director(nx, ny, an)
    W    = @SMatrix([0 ω; -ω 0])
    D    = @SMatrix([Dxx Dxy; Dxy Dyy])
    nsub = max(1, ceil(Int, Δt * (abs(ω) + sqrt(Dxx^2 + Dyy^2 + 2 * Dxy^2)) / 0.05))
    for _ in 1:nsub
        n = DirectorVector(n, W, D, Δt / nsub)
    end
    return n
end

# Volume fraction of the anisotropic phases at a grid point (pr: phase ratios)
@inline aniso_fraction(pr, materials) = sum(pr[p] for p in eachindex(pr) if is_anisotropic(materials, p); init=0.0)

# Rotate the director field (a.τ0.nd) with the flow of the converged step, inner centroids and vertices that contain an
# anisotropic phase. D: deviatoric strain rate of the step (the isotropic part does not rotate n), W = [0 ω; -ω 0],
# ω = ½(∂vx/∂y - ∂vy/∂x). The values interpolated from the neighbouring grid points are weighted by their volume fraction of
# anisotropic phases: the fabric at the edge of an anisotropic body does not see the (much faster) flow of the other phases.
# Grid only: the director field is not advected.
function update_director!(a, materials, Δ)
    compute_vorticity!(a, Δ)
    an, nd, ε̇, ω = materials.anisotropy, a.τ0.nd, a.ε̇, a.ω
    prc, prv = a.phase_ratios.c, a.phase_ratios.v
    # Centroids: ε̇xy, ω from the 4 vertices
    for j in axes(nd.x_c, 2)[2:end-1], i in axes(nd.x_c, 1)[2:end-1]
        aniso_fraction(prc[i, j], materials) > 0 || continue
        sw = sD = sω = 0.0
        for (iv, jv) in ((i, j), (i+1, j), (i, j+1), (i+1, j+1))
            w   = aniso_fraction(prv[iv, jv], materials)
            sw += w
            sD += w * ε̇.xy[iv, jv]
            sω += w * ω.xy_v[iv, jv]
        end
        Dxy, ωc = sw > 0 ? (sD / sw, sω / sw) : (0.0, 0.0)
        n = rotate_director(nd.x_c[i, j], nd.y_c[i, j], ε̇.xx[i, j], ε̇.yy[i, j], Dxy, ωc, an, Δ.t)
        nd.x_c[i, j], nd.y_c[i, j] = n[1], n[2]
    end
    # Vertices: ε̇xx, ε̇yy from the 4 centroids (ghost centroids filled by fill_solution_ghosts!)
    for j in axes(nd.x_v, 2)[2:end-1], i in axes(nd.x_v, 1)[2:end-1]
        aniso_fraction(prv[i, j], materials) > 0 || continue
        sw = sxx = syy = 0.0
        for (ic, jc) in ((i-1, j-1), (i, j-1), (i-1, j), (i, j))
            w    = aniso_fraction(prc[ic, jc], materials)
            sw  += w
            sxx += w * ε̇.xx[ic, jc]
            syy += w * ε̇.yy[ic, jc]
        end
        Dxx, Dyy = sw > 0 ? (sxx / sw, syy / sw) : (0.0, 0.0)
        n = rotate_director(nd.x_v[i, j], nd.y_v[i, j], Dxx, Dyy, ε̇.xy[i, j], ω.xy_v[i, j], an, Δ.t)
        nd.x_v[i, j], nd.y_v[i, j] = n[1], n[2]
    end
    return nothing
end
