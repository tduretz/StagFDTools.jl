@inline mynorm(x) = sum(xi^2 for xi in x)

bulk_viscosity(ϕ, η0, m) = η0 * abs(ϕ)^m

function PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δt)
    ηΦ = bulk_viscosity(Φ, ηΦ0, m)
    dPtdt = (Pt - Pt0) / Δt
    dPfdt = (Pf - Pf0) / Δt
    dΦdt = ((dPfdt - dPtdt) / KΦ + (Pf - Pt) / ηΦ + λ̇ * sinψ) * 1
    return dΦdt, ηΦ
end

function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δt)
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δt)[1]
    r = Φ - (Φ0 + dΦdt * Δt)
    return r
end

function Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δt)

    dΦdt, ηΦ = PorosityRate(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δt)
    Φ = Φ0 + dΦdt * Δt
    r0 = 1.0
    for iter in 1:2
        r, dresdΦ = ad_value_and_derivative(PorosityResidual, Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δt)
        if iter == 1
            r0 = abs(r) + 1.0e-10
        end
        # @show iter, abs(r), abs(r)/r0
        # if min(abs(r), abs(r)/r0 ) < 1e-10 break end
        Φ -= r / dresdΦ
    end
    dΦdt, ηΦ = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δt)
    return Φ, dΦdt, ηΦ
end

function ΔP_Trial(x, Pt, Pf, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, sinψ, Δt)

    Pt, Pf = x[1], x[2]

    # Porosity rate
    dPtdt = (Pt - Pt0) / Δt
    dPfdt = (Pf - Pf0) / Δt
    dlnρfdt = dPfdt / Kf
    # dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    Φ, dΦdt = Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ, m, λ̇, sinψ, Δt)
    dPsdt = ((Pt - Φ * Pf) / (1 - Φ) - (Pt0 - Φ0 * Pf0) / (1 - Φ0)) / Δt
    # dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1 / Ks * (dPsdt)


    # Ps     = (Pt - phi*Pf)/(1-phi)
    # dPsdt = (dPtdt - phi*dPfdt) /(1-phi)
    # # dPsdt = ((Pt - phi*Pf)/(1-phi) - (Pt0 - phi0*Pf0)/(1-phi0))/dt
    # # dPsdt = dphidt*(Pt - Pf*phi)/(1-phi)**2 + (dPtdt - phi*dPfdt - 0*Pf*dphidt) / (1 - phi)
    # dlnrhosdt = elastic * 1/K_s * ( dPsdt )

    return @SVector [
        dlnρsdt - dΦdt / (1 - Φ),
        (Φ * dlnρfdt + dΦdt) / ηΦ,
    ]
end

function ΔP(Pt_trial, Pf_trial, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, sinψ, Δt)

    x = @SVector[0.0, 0.0]
    r0 = 1.0
    tol = 1.0e-13

    for iter in 1:10
        R, J = ad_value_and_jacobian(ΔP_Trial, x, Pt_trial, Pf_trial, 0 * divVs, 0 * divqD, λ̇, 0 * Pt0, 0 * Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, sinψ, Δt)
        x = x .- J \ R
        nr = mynorm(R)
        if iter == 1 && nr > 1.0e-17
            r0 = nr
        end
        r = nr / r0
        if r < tol
            break
        end
    end
    return x[1], x[2]
end


# function residual_two_phase_P(x, ηve, Δt, ε̇II_eff, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )

#     τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
# α1 = single_phase ? 0.0 : 1.0

# # Pressure corrections
# # ΔPt = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
# # ΔPf = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)

# # Pressure corrections
# ΔPt_1, ΔPf = ΔP(Pt_trial, Pf_trial, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ, m,  KΦ, Ks, Kf, sinψ, Δt)

# # Check yield

# f = if single_phase
#         τII - C*cosϕ - Pt*sinϕ
#     else
#         F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)
#     end

# ΔPt = if single_phase
#     Ks .* sinψ .* Δt .* λ̇
#     else
#         ΔPt_1
#     end

# return @SVector [
#     ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
#     Pt - (Pt_trial + ΔPt),
#     Pf - (Pf_trial + ΔPf),
#     f,
# ]

function residual_two_phase_P(x::SVector{N, D}, ηve, Δt, ε̇II_eff, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase) where {N, D}
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    # α1 = single_phase ? D(0.0) : D(1.0)
    return @SVector [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
end

function LocalRheology_P(ε̇::SVector{N, D}, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ) where {N, D}

    # Effective strain rate & pressure
    ε̇II_eff = invII(ε̇)
    Pt = ε̇[4]
    Pf = ε̇[5]

    # Parameters
    ϵ = 1.0e-10 # tolerance
    n = materials.n[phases]
    m = materials.m[phases]
    η0 = materials.ηs0[phases]
    # B    = materials.B[phases]
    G = materials.G[phases]
    C = materials.C[phases]
    ηΦ = materials.ηΦ0[phases]
    KΦ = materials.KΦ[phases]
    Ks = materials.Ks[phases]
    Kf = materials.Kf[phases]

    ηvp = materials.ηvp[phases]
    sinψ = materials.sinψ[phases]
    sinϕ = materials.sinϕ[phases]
    cosϕ = materials.cosϕ[phases]

    # ηvep, λ̇, Pt, Pf, τII, Φ, f  = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    α1 = materials.single_phase ? 0.0 : 1.0

    # Initial guess
    η = (η0 .* ε̇II_eff .^ (1 ./ n .- 1.0))
    ηve = inv(1 / η + 1 / (G * Δ.t))
    τII = 2 * ηve * ε̇II_eff
    ηvep = ηve

    # Trial porosity
    # Φ = 0.1
    # Φ = (KΦ * Δ.t * (Pf - Pt) + KΦ * Φ0 * ηΦ + ηΦ * (Pf - Pf0 - Pt + Pt0)) / (KΦ * ηΦ)
    Φ = if materials.single_phase
        zero(D)
    else
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ, m, 0.0, 0.0, Δ.t)[1]
    end

    # Check yield
    λ̇ = zero(D)

    # # # # f        = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, 0.0)
    # # # # if f>0
    # # # #     λ̇ = f / (KΦ .* Δ.t * sinϕ * sinψ + ηve + ηvp)
    # # # #     f  = τII - λ̇*ηve - C*cosϕ - (Pt + KΦ .* Δ.t * sinψ * λ̇)*sinϕ
    # # # #     # @show f, λ̇
    # # # #     # error()

    # # # #     τII = τII - λ̇*ηve
    # # # #     Pt  = Pt + KΦ .* Δ.t * sinψ * λ̇
    # # # # end

    # #############################

    f = F(τII, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    x = @SVector [τII, Pt, Pf, λ̇]

    nr = 1.0
    nr0 = 1.0
    tol = 1.0e-10

    # Return mapping
    if f > D(-1.0e-13)
        # This is the proper return mapping with plasticity
        for iter in 1:10
            R, J = ad_value_and_jacobian(residual_two_phase_P, x, ηve, Δ.t, ε̇II_eff, Pt, Pf, divVs, divqD, Φ, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, materials.single_phase)
            x -= J \ R
            nr = mynorm(R)
            if iter == 1
                nr0 = nr
            end
            r = nr / nr0
            r < tol && break
        end
    end

    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]

    Φ = if materials.single_phase
        zero(D)
    else
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ, m, λ̇, sinψ, Δ.t)[1]
    end

    #############################

    # Effective viscosity
    ηvep = τII / (2 * ε̇II_eff)

    f = F(τII, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    return ηvep, λ̇, Pt, Pf, τII, Φ, f
end


function StressVector_P!(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
    τ = @SVector(
        [
            2 * η * ε̇[1],
            2 * η * ε̇[2],
            2 * η * ε̇[3],
            Pt,
            Pf,
        ]
    )
    return τ, η, λ̇, τII, Φ, f
end

function StressVector_P2!(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
    τ = @SVector(
        [
            2 * η * ε̇[1],
            2 * η * ε̇[2],
            2 * η * ε̇[3],
            Pt,
            Pf,
        ]
    )
    return τ
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)

    _ones = @SVector ones(5)
    Δt = Δ.t

    ########################### Loop over centroids ###########################
    for j in 2:(size(ε̇.xx, 2) - 1), i in 2:(size(ε̇.xx, 1) - 1)

        # Local arrays
        Vx_loc = SMatrix{2, 3}(V.x[ii, jj] for ii in i:(i + 1),   jj in j:(j + 2))
        Vy_loc = SMatrix{3, 2}(V.y[ii, jj] for ii in i:(i + 2),   jj in j:(j + 1))
        bcx = SMatrix{2, 3}(BC.Vx[ii, jj] for ii in i:(i + 1),   jj in j:(j + 2))
        bcy = SMatrix{3, 2}(BC.Vy[ii, jj] for ii in i:(i + 2),   jj in j:(j + 1))
        typex = SMatrix{2, 3}(type.Vx[ii, jj] for ii in i:(i + 1),   jj in j:(j + 2))
        typey = SMatrix{3, 2}(type.Vy[ii, jj] for ii in i:(i + 2),   jj in j:(j + 1))
        τxy0 = SMatrix{2, 2}(τ0.xy[ii, jj] for ii in i:(i + 1),   jj in j:(j + 1))
        Φ0_loc = SMatrix{3, 3}(Φ0.c[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        Pf_loc = SMatrix{3, 3}(P.f[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        Pf0_loc = SMatrix{3, 3}(P0.f[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        Pt_loc = SMatrix{3, 3}(P.t[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        Pt0_loc = SMatrix{3, 3}(P0.t[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        typept = SMatrix{3, 3}(type.Pt[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        bcpt = SMatrix{3, 3}(BC.Pt[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        typepf = SMatrix{3, 3}(type.Pf[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        bcpf = SMatrix{3, 3}(BC.Pf[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))
        phc = SMatrix{3, 3}(phases.c[ii, jj] for ii in (i - 1):(i + 1), jj in (j - 1):(j + 1))

        k_ηf0 = materials.k_ηf0[phc]
        ηΦ = materials.ηΦ0[phc]
        KΦ = materials.KΦ[phc]
        n = materials.n_CK[phc] # Carman-Kozeny
        m = materials.m[phc]

        # Density for Darcy flux
        ρfgC = SMatrix{3, 3}(@. materials.ρf[phc] * materials.g[2])
        ρfg = SVector{2, Float64}(1 / 2 * (ρfgC[2, j] + ρfgC[2, j + 1]) for j in 1:2)

        # BCs
        Vx = SetBCVx1(Vx_loc, typex, bcx, Δ)
        Vy = SetBCVy1(Vy_loc, typey, bcy, Δ)
        Pf = SetBCPf1(Pf_loc, typepf, bcpf, Δ, ρfg)
        Pt = SetBCPf1(Pt_loc, typept, bcpt, Δ, ρfg)
        Pf0 = SetBCPf1(Pf0_loc, typepf, bcpf, Δ, ρfg)
        Pt0 = SetBCPf1(Pt0_loc, typepf, bcpf, Δ, ρfg)

        # Porosity
        Φ_loc = if materials.linearizeϕ
            SMatrix{3, 3, Float64}(Φ0_loc)
        else
            SMatrix{3, 3, Float64}(Porosity(Φ0_loc[i, j], Pt[i, j], Pf[i, j], Pt0[i, j], Pf0[i, j], KΦ[i, j], ηΦ[i, j], m[i, j], 0.0, 0.0, Δ.t)[1] for i in 1:3, j in 1:3)
        end

        # Kinematics
        Dxx = ∂x_inn(Vx) / Δ.x
        Dyy = ∂y_inn(Vy) / Δ.y
        Dxy = ∂y(Vx) / Δ.y
        Dyx = ∂x(Vy) / Δ.x

        # Strain rate
        Dkk = Dxx .+ Dyy
        ε̇xx = @. Dxx - Dkk ./ 3
        ε̇yy = @. Dyy - Dkk ./ 3
        ε̇xy = @. (Dxy + Dyx) ./ 2
        ε̇̄xy = av(ε̇xy)

        # Darcy flux
        k_μ_xx = SMatrix{3, 3, Float64}(@.  k_ηf0 * max.(Φ_loc, 1.0e-6) .^ n)
        kx_μ_xx = SVector{2, Float64}(@. (k_μ_xx[i, 2] + k_μ_xx[i + 1, 2]) / 2 for i in 1:2)
        k_μ_yy = SMatrix{3, 3, Float64}(@.  k_ηf0 * max.(Φ_loc, 1.0e-6) .^ n)
        ky_μ_yy = SVector{2, Float64}(@. (k_μ_yy[2, j] + k_μ_yy[2, j + 1]) / 2 for j in 1:2)
        ∂Pf∂x = SVector{2, Float64}(@. (Pf[i + 1, 2] - Pf[i, 2]) / Δ.x for i in 1:2)
        ∂Pf∂y = SVector{2, Float64}(@. (Pf[2, j + 1] - Pf[2, j]) / Δ.y for j in 1:2)
        qDx = SVector{2, Float64}(- kx_μ_xx .* ∂Pf∂x)
        qDy = SVector{2, Float64}(- ky_μ_yy .* ∂Pf∂y - ρfg)
        divqD = ((qDx[2] - qDx[1]) / Δ.x + (qDy[2] - qDy[1]) / Δ.y)[1]

        # Visco-elasticity
        G = materials.G[phases.c[i, j]]
        τ̄xy0 = av(τxy0)
        ε̇vec = @SVector([ε̇xx[1] + τ0.xx[i, j] / (2 * G[1] * Δ.t), ε̇yy[1] + τ0.yy[i, j] / (2 * G[1] * Δ.t), ε̇̄xy[1] + τ̄xy0[1] / (2 * G[1] * Δ.t), P.t[i, j], P.f[i, j]])
        τ0_loc = @SVector([τ0.xx[i, j], τ0.yy[i, j], τ̄xy0[1]])

        ##################################

        # Tangent operator used for Newton Linearisation
        τ_vec, jac = ad_value_and_jacobian(StressVector_P2!, ε̇vec, Dkk[1], divqD, P0.t[i, j], P0.f[i, j], Φ0.c[i, j], τ0_loc, materials, phases.c[i, j], Δ)
        η_local, Pt1, Pf1, λ̇_local, τII_local, Φ_local, f_local = LocalRheology_P(ε̇vec, Dkk[1], divqD, P0.t[i, j], P0.f[i, j], Φ0.c[i, j], τ0_loc, materials, phases.c[i, j], Δ)
        @views 𝐷_ctl.c[i, j] .= jac

        #################################

        # Tangent operator used for Picard Linearisation
        𝐷.c[i, j] .= diagm(2 * η_local * _ones)
        𝐷.c[i, j][4, 4] = 1
        𝐷.c[i, j][5, 5] = 1

        ##################################

        # Update stress
        τ.xx[i, j] = τ_vec[1]
        τ.yy[i, j] = τ_vec[2]
        τ.II[i, j] = τII_local
        τ.f[i, j] = f_local
        ε̇.xx[i, j] = ε̇xx[1]
        ε̇.yy[i, j] = ε̇yy[1]
        ε̇.II[i, j] = invII(@SVector([ε̇xx[1], ε̇yy[1], ε̇̄xy[1]]))
        λ̇.c[i, j] = λ̇_local
        Φ.c[i, j] = Φ_local
        η.c[i, j] = η_local
        if λ̇.c[i, j] > 0
            ΔP.t[i, j] = (τ_vec[4] - P.t[i, j])
            ΔP.f[i, j] = (τ_vec[5] - P.f[i, j])
        end
    end

    # Need a lazy copy at ghost boundaries in case of stress BC along that boundary
    for i in axes(ε̇.xx, 1)
        if type.Vy[i + 1, 1] == :Neumann_normal
            𝐷.c[i, 1] = 𝐷.c[i, 2]
        end
        if type.Vy[i + 1, end] == :Neumann_normal
            𝐷.c[i, end] = 𝐷.c[i, end - 1]
        end
    end

    for j in axes(ε̇.xx, 2)
        if type.Vx[1, j + 1] == :Neumann_normal
            𝐷.c[1, j] = 𝐷.c[2, j]
        end
        if type.Vx[end, j + 1] == :Neumann_normal
            𝐷.c[end, j] = 𝐷.c[end - 1, j]
        end
    end

    # ########################### Loop over vertices ###########################
    for j in 3:(size(ε̇.xy, 2) - 2), i in 3:(size(ε̇.xy, 1) - 2)
        Vx_loc = SMatrix{3, 2}(V.x[ii, jj] for ii in (i - 1):(i + 1),   jj in (j - 1 + 1):(j + 1))
        Vy_loc = SMatrix{2, 3}(V.y[ii, jj] for ii in (i - 1 + 1):(i + 1), jj in (j - 1):(j + 1))
        bcx = SMatrix{3, 2}(BC.Vx[ii, jj] for ii in (i - 1):(i + 1),   jj in (j - 1 + 1):(j + 1))
        bcy = SMatrix{2, 3}(BC.Vy[ii, jj] for ii in (i - 1 + 1):(i + 1), jj in (j - 1):(j + 1))
        typex = SMatrix{3, 2}(type.Vx[ii, jj] for ii in (i - 1):(i + 1),   jj in (j - 1 + 1):(j + 1))
        typey = SMatrix{2, 3}(type.Vy[ii, jj] for ii in (i - 1 + 1):(i + 1), jj in (j - 1):(j + 1))
        τxx0 = SMatrix{2, 2}(τ0.xx[ii, jj] for ii in (i - 1):(i + 0),   jj in (j - 1):(j + 0))
        τyy0 = SMatrix{2, 2}(τ0.yy[ii, jj] for ii in (i - 1):(i + 0),   jj in (j - 1):(j + 0))
        Φ0_loc = SMatrix{4, 4}(Φ0.c[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        Pt0_loc = SMatrix{4, 4}(P0.t[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        Pf0_loc = SMatrix{4, 4}(P0.f[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        Pf_loc = SMatrix{4, 4}(P.f[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        Pt_loc = SMatrix{4, 4}(P.t[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        typept = SMatrix{4, 4}(type.Pt[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        bcpt = SMatrix{4, 4}(BC.Pt[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        typepf = SMatrix{4, 4}(type.Pf[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        bcpf = SMatrix{4, 4}(BC.Pf[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))
        phc = SMatrix{4, 4}(phases.c[ii, jj] for ii in (i - 2):(i + 1),   jj in (j - 2):(j + 1))

        k_ηf0 = materials.k_ηf0[phc]
        ηΦ = materials.ηΦ0[phc]
        KΦ = materials.KΦ[phc]
        n = materials.n_CK[phc] # Carman-Kozeny
        m = materials.m[phc]    # Carman-Kozeny

        # Fluid density
        ρfgC = SMatrix{4, 4}(@. materials.ρf[phc] * materials.g[2])
        ρfg = SMatrix{2, 3, Float64}(1 / 2 * (ρfgC[i + 1, j] + ρfgC[i + 1, j + 1]) for i in 1:2, j in 1:3)

        # Set BCs
        Vx = SetBCVx1(Vx_loc, typex, bcx, Δ)
        Vy = SetBCVy1(Vy_loc, typey, bcy, Δ)
        Pf = SetBCPf1(Pf_loc, typepf, bcpf, Δ, ρfg)
        Pt = SetBCPf1(Pt_loc, typept, bcpt, Δ, ρfg)
        Pf0 = SetBCPf1(Pf0_loc, typepf, bcpf, Δ, ρfg)
        Pt0 = SetBCPf1(Pt0_loc, typept, bcpt, Δ, ρfg)

        # Porosity
        Φ_loc = if materials.linearizeϕ
            SMatrix{4, 4, Float64}(@. Φ0_loc)
        else
            SMatrix{4, 4, Float64}(Porosity(Φ0_loc[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ηΦ[ii], m[ii], 0.0, 0.0, Δt)[1] for ii in eachindex(Φ0_loc))
        end

        # Kinematics
        Dxx = ∂x(Vx) / Δ.x
        Dyy = ∂y(Vy) / Δ.y
        Dxy = ∂y_inn(Vx) / Δ.y
        Dyx = ∂x_inn(Vy) / Δ.x

        # Strain rate
        Dkk = @. Dxx + Dyy
        ε̇xx = @. Dxx - Dkk / 3
        ε̇yy = @. Dyy - Dkk / 3
        ε̇xy = @. (Dxy + Dyx) / 2
        ε̇̄xx = av(ε̇xx)
        ε̇̄yy = av(ε̇yy)

        # Darcy flux
        k_μ_xx = SMatrix{4, 4, Float64}(@.  k_ηf0 * max.(Φ_loc, 1.0e-6) .^ n)
        kx_μ_xx = SMatrix{3, 2, Float64}((k_μ_xx[i, j + 1] + k_μ_xx[i + 1, j + 1]) / 2 for i in 1:3, j in 1:2)
        k_μ_yy = SMatrix{4, 4, Float64}(@.  k_ηf0 * max.(Φ_loc, 1.0e-6) .^ n)
        ky_μ_yy = SMatrix{2, 3, Float64}((k_μ_yy[i + 1, j] + k_μ_yy[i + 1, j + 1]) / 2 for i in 1:2, j in 1:3)
        ∂Pf∂x = SMatrix{3, 2, Float64}((Pf[i + 1, j + 1] - Pf[i, j + 1]) / Δ.x for i in 1:3, j in 1:2)
        ∂Pf∂y = SMatrix{2, 3, Float64}((Pf[i + 1, j + 1] - Pf[i + 1, j]) / Δ.y for i in 1:2, j in 1:3)
        qDx = SMatrix{3, 2, Float64}(- kx_μ_xx .* ∂Pf∂x)
        qDy = SMatrix{2, 3, Float64}(- ky_μ_yy .* ∂Pf∂y - ρfg)
        divqD = ∂x(qDx) / Δ.x .+ ∂y(qDy) / Δ.y
        divqD̄ = av(divqD)[1]

        # Visco-elasticity
        G = materials.G[phases.v[i, j]]
        τ̄xx0 = av(τxx0)
        τ̄yy0 = av(τyy0)
        P̄t = av(Pt)
        P̄f = av(Pf)
        ε̇vec = @SVector([ε̇̄xx[1] + τ̄xx0[1] / (2 * G[1] * Δ.t), ε̇̄yy[1] + τ̄yy0[1] / (2 * G[1] * Δ.t), ε̇xy[1] + τ0.xy[i, j] / (2 * G[1] * Δ.t), P̄t[1], P̄f[1]])
        τ0_loc = @SVector([τ̄xx0[1], τ̄yy0[1], τ0.xy[i, j]])

        D̄kk = av(Dkk)
        ϕ̄0 = av(Φ0_loc)
        P̄t0 = av(Pt0)
        P̄f0 = av(Pf0)

        ##################################

        # Tangent operator used for Newton Linearisation
        τ_vec, jac = ad_value_and_jacobian(StressVector_P2!, ε̇vec, D̄kk[1], divqD̄, P̄t0[1], P̄f0[1], ϕ̄0[1], τ0_loc, materials, phases.v[i, j], Δ)
        η_local, Pt1, Pf1, λ̇_local, τII_local, Φ_local, f_local = LocalRheology_P(ε̇vec, D̄kk[1], divqD̄, P̄t0[1], P̄f0[1], ϕ̄0[1], τ0_loc, materials, phases.v[i, j], Δ)
        @views 𝐷_ctl.v[i, j] .= jac

        ##################################

        # Tangent operator used for Picard Linearisation
        𝐷.v[i, j] .= diagm(2 * η_local * _ones)
        𝐷.v[i, j][4, 4] = 1
        𝐷.v[i, j][5, 5] = 1

        # Update stress
        τ.xy[i, j] = τ_vec[3]
        ε̇.xy[i, j] = ε̇xy[1]
        λ̇.v[i, j] = λ̇_local
        η.v[i, j] = η_local
    end

    # !!!!!! Cheap copy edges
    # This crap is necessary becaus eof the vertex CTL loop is such
    # for j=3:size(ε̇.xy,2)-2, i=3:size(ε̇.xy,1)-2
    for j in 2:(size(ε̇.xy, 2) - 1)
        i = 2
        @views 𝐷_ctl.v[i, j] .= 𝐷_ctl.v[3, j]
        @views 𝐷.v[i, j] .= 𝐷.v[3, j]
        i = size(ε̇.xy, 1) - 1
        @views 𝐷_ctl.v[i, j] .= 𝐷_ctl.v[end - 2, j]
        @views 𝐷.v[i, j] .= 𝐷.v[end - 2, j]
    end

    for i in 2:(size(ε̇.xy, 1) - 1)
        j = 2
        @views 𝐷_ctl.v[i, j] .= 𝐷_ctl.v[i, 3]
        @views 𝐷.v[i, j] .= 𝐷.v[i, 3]
        j = size(ε̇.xy, 2) - 1
        @views 𝐷_ctl.v[i, j] .= 𝐷_ctl.v[i, end - 2]
        @views 𝐷.v[i, j] .= 𝐷.v[i, end - 2]
    end
    return
end
