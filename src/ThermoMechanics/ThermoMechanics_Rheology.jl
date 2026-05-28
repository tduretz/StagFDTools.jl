function LocalRheology(ε̇, materials, phases, Δ)

    # Effective strain rate & pressure
    ε̇II = sqrt.((ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1] - ε̇[2])^2) / 2 + ε̇[3]^2) + 1.0e-14
    Pt = ε̇[4]
    T = ε̇[5]

    # Parameters
    ϵ = 1.0e-10 # tolerance
    n = materials.n[phases]
    η0 = materials.ηs0[phases]
    # B    = materials.B[phases]
    G = materials.G[phases]
    # C    = materials.C[phases]

    # ϕ    = materials.ϕ[phases]
    # ψ    = materials.ψ[phases]

    # ηvp  = materials.ηvp[phases]
    # sinψ = materials.sinψ[phases]
    # sinϕ = materials.sinϕ[phases]
    # cosϕ = materials.cosϕ[phases]

    # β    = materials.β[phases]
    # comp = materials.compressible

    # Initial guess
    η = (η0 .* ε̇II .^ (1 ./ n .- 1.0))[1]
    ηvep = inv(1 / η + 1 / (G * Δ.t))
    # ηvep = G*Δ.t

    τII = 2 * ηvep * ε̇II

    # # Visco-elastic powerlaw
    # for it=1:20
    #     r      = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
    #     # @show abs(r)
    #     (abs(r)<ϵ) && break
    #     ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τII, G, Δ.t, B, n)
    #     ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
    #     τII     += ∂τII∂ε̇II*r
    # end
    # isnan(τII) && error()

    # # Viscoplastic return mapping
    λ̇ = 0.0
    # if materials.plasticity === :DruckerPrager
    #     τII, P, λ̇ = DruckerPrager(τII, P, ηvep, comp, β, Δ.t, C, cosϕ, sinϕ, sinψ, ηvp)
    # elseif materials.plasticity === :tensile
    #     τII, P, λ̇ = Tensile(τII, P, ηvep, comp, β, Δ.t, materials.σT[phases], ηvp)
    # elseif materials.plasticity === :Kiss2023
    #     τII, P, λ̇ = Kiss2023(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp, materials.σT[phases], materials.δσT[phases], materials.P1[phases], materials.τ1[phases], materials.P2[phases], materials.τ2[phases])
    # end

    # Effective viscosity
    ηvep = τII / (2 * ε̇II)

    return ηvep, λ̇, Pt, τII, T
end

function StressVector!(ε̇, materials, phases, Δ)
    η, λ̇, Pt, τII, T = LocalRheology(ε̇, materials, phases, Δ)
    τ = @SVector(
        [
            2 * η * ε̇[1],
            2 * η * ε̇[2],
            2 * η * ε̇[3],
            Pt,
            T,
        ]
    )
    return τ, η, λ̇, τII
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, T, P, ΔP, type, BC, materials, phases, Δ)

    _ones = @SVector ones(5)
    # Dzz   = materials.Dzz
    OOP = materials.OOP

    # Loop over centroids
    for j in 2:(size(ε̇.xx, 2) - 1), i in 2:(size(ε̇.xx, 1) - 1)
        # if (i==1 && j==1) || (i==size(ε̇.xx,1) && j==1) || (i==1 && j==size(ε̇.xx,2)) || (i==size(ε̇.xx,1) && j==size(ε̇.xx,2))
        #     # Avoid the outer corners - nothing is well defined there ;)
        # else
        Vx = SMatrix{2, 3}(V.x[ii, jj] for ii in i:(i + 1),   jj in j:(j + 2))
        Vy = SMatrix{3, 2}(V.y[ii, jj] for ii in i:(i + 2),   jj in j:(j + 1))
        bcx = SMatrix{2, 3}(BC.Vx[ii, jj] for ii in i:(i + 1),   jj in j:(j + 2))
        bcy = SMatrix{3, 2}(BC.Vy[ii, jj] for ii in i:(i + 2),   jj in j:(j + 1))
        typex = SMatrix{2, 3}(type.Vx[ii, jj] for ii in i:(i + 1),   jj in j:(j + 2))
        typey = SMatrix{3, 2}(type.Vy[ii, jj] for ii in i:(i + 2),   jj in j:(j + 1))
        τxy0 = SMatrix{2, 2}(τ0.xy[ii, jj] for ii in i:(i + 1),   jj in j:(j + 1))

        Vx = SetBCVx1(Vx, typex, bcx, Δ)
        Vy = SetBCVy1(Vy, typey, bcy, Δ)

        Dxx = ∂x_inn(Vx) / Δ.x
        Dyy = ∂y_inn(Vy) / Δ.y
        Dxy = ∂y(Vx) / Δ.y
        Dyx = ∂x(Vy) / Δ.x

        Dzz = 0.5 * (Dxx .+ Dyy) * OOP
        Dkk = Dxx .+ Dyy .+ Dzz
        ε̇xx = @. Dxx - Dkk ./ 3
        ε̇yy = @. Dyy - Dkk ./ 3
        ε̇zz = @. Dzz - Dkk ./ 3
        ε̇xy = @. (Dxy + Dyx) ./ 2
        ε̇̄xy = av(ε̇xy)

        # Visco-elasticity
        G = materials.G[phases.c[i, j]]
        τ̄xy0 = av(τxy0)
        ε̇vec = @SVector([ε̇xx[1] + τ0.xx[i, j] / (2 * G[1] * Δ.t), ε̇yy[1] + τ0.yy[i, j] / (2 * G[1] * Δ.t), ε̇̄xy[1] + τ̄xy0[1] / (2 * G[1] * Δ.t), P.t[i, j], T.c[i, j]])

        # Tangent operator used for Newton Linearisation
        stress_state, τ_vec, jac = ad_value_and_jacobian_first(StressVector!, ε̇vec, materials, phases.c[i, j], Δ)
        _, η_local, λ̇_local, τII_local = stress_state

        @views 𝐷_ctl.c[i, j] .= jac

        # Tangent operator used for Picard Linearisation
        𝐷.c[i, j] .= diagm(2 * η_local * _ones)
        𝐷.c[i, j][4, 4] = 1
        𝐷.c[i, j][5, 5] = 1

        # Update stress
        τ.xx[i, j] = τ_vec[1]
        τ.yy[i, j] = τ_vec[2]
        τ.II[i, j] = τII_local
        ε̇.xx[i, j] = ε̇xx[1]
        ε̇.yy[i, j] = ε̇yy[1]
        λ̇.c[i, j] = λ̇_local
        η.c[i, j] = η_local
        ΔP.t[i, j] = (τ_vec[4] - P.t[i, j])
        # end
    end

    # Loop over vertices
    for j in 2:(size(ε̇.xy, 2) - 1), i in 2:(size(ε̇.xy, 1) - 1)
        Vx = SMatrix{3, 2}(V.x[ii, jj] for ii in (i - 1):(i + 1), jj in j:(j + 1))
        Vy = SMatrix{2, 3}(V.y[ii, jj] for ii in i:(i + 1),   jj in (j - 1):(j + 1))
        bcx = SMatrix{3, 2}(BC.Vx[ii, jj] for ii in (i - 1):(i + 1), jj in j:(j + 1))
        bcy = SMatrix{2, 3}(BC.Vy[ii, jj] for ii in i:(i + 1),   jj in (j - 1):(j + 1))
        typex = SMatrix{3, 2}(type.Vx[ii, jj] for ii in (i - 1):(i + 1), jj in j:(j + 1))
        typey = SMatrix{2, 3}(type.Vy[ii, jj] for ii in i:(i + 1),   jj in (j - 1):(j + 1))
        τxx0 = SMatrix{2, 2}(τ0.xx[ii, jj] for ii in (i - 1):i,   jj in (j - 1):j)
        τyy0 = SMatrix{2, 2}(τ0.yy[ii, jj] for ii in (i - 1):i,   jj in (j - 1):j)
        τzz0 = SMatrix{2, 2}(τ0.zz[ii, jj] for ii in (i - 1):i,   jj in (j - 1):j)
        Pt = SMatrix{2, 2}(P.t[ii, jj] for ii in (i - 1):i,   jj in (j - 1):j)
        Tc = SMatrix{2, 2}(T.c[ii, jj] for ii in (i - 1):i,   jj in (j - 1):j)

        Vx = SetBCVx1(Vx, typex, bcx, Δ)
        Vy = SetBCVy1(Vy, typey, bcy, Δ)

        Dxx = ∂x(Vx) / Δ.x
        Dyy = ∂y(Vy) / Δ.y
        Dxy = ∂y_inn(Vx) / Δ.y
        Dyx = ∂x_inn(Vy) / Δ.x

        Dzz = 0.5 * (Dxx .+ Dyy) * OOP
        Dkk = @. Dxx + Dyy + Dzz
        ε̇xx = @. Dxx - Dkk / 3
        ε̇yy = @. Dyy - Dkk / 3
        ε̇zz = @. Dzz - Dkk / 3
        # @show ε̇zz
        ε̇xy = @. (Dxy + Dyx) / 2
        ε̇̄xx = av(ε̇xx)
        ε̇̄yy = av(ε̇yy)
        ε̇̄zz = av(ε̇zz)

        # Visco-elasticity
        G = materials.G[phases.v[i, j]]
        τ̄xx0 = av(τxx0)
        τ̄yy0 = av(τyy0)
        P̄t = av(Pt)
        T̄ = av(Tc)
        ε̇vec = @SVector([ε̇̄xx[1] + τ̄xx0[1] / (2 * G[1] * Δ.t), ε̇̄yy[1] + τ̄yy0[1] / (2 * G[1] * Δ.t), ε̇xy[1] + τ0.xy[i, j] / (2 * G[1] * Δ.t), P̄t[1], T̄[1]])

        # Tangent operator used for Newton Linearisation
        stress_state, τ_vec, jac = ad_value_and_jacobian_first(StressVector!, ε̇vec, materials, phases.v[i, j], Δ)
        _, η_local, λ̇_local, _ = stress_state

        @views 𝐷_ctl.v[i, j] .= jac

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
    return
end
