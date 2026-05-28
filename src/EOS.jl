function DensityExponential(T, P, materials, phase)
    ρr = materials.ρr[phase]
    α = materials.α[phase]
    K = materials.K[phase]
    ρ = ρr * exp(P / K - α * T)
    return ρ
end

function P_mechanical(V, materials, phase)
    # Birch-Murnaghan EOS
    V0 = materials.V0[phase]
    K0 = materials.K[phase] / 1.0e9
    Kp = materials.∂K∂P[phase]
    # Kpp = materials.Kpp[phase]
    f = ((V0 / V)^(2 / 3) - 1) / 2
    # P = 3*K0*f*(1+2*f)^(5/2) * (1 + 3/2*(Kp -4)*f + 3/2*(K0*Kpp + (Kp - 4)*(Kp-3) + 35/9)*f^2 )
    P = 3 * K0 * f * (1 + 2 * f)^(5 / 2) * (1 + 3 / 2 * (Kp - 4) * f)
    return P
end

function U_Einstein(T, θE, R)
    return R * θE / (exp(θE / T) - 1)
end

function P_thermal(V, T, materials, phase)
    γ0 = materials.γ0[phase]
    θE = materials.θE[phase]
    T0 = materials.T0[phase]
    V0 = materials.V0[phase]
    q = materials.q[phase]
    Natom = materials.Natom[phase]
    R = materials.R
    γ = γ0 * (V / V0)^q
    sca = 1.0e-3 # 1e6/1e9 : (m3 -> cm3) / (GPa -> Pa)
    P = sca * 3 * Natom * γ / (V) * (U_Einstein(T, θE, R) - U_Einstein(T0, θE, R))
    return P
end

function residual(V, P, T, materials, phase)
    return P - P_mechanical(V, materials, phase) - P_thermal(V, T, materials, phase)
end

function DensityBirchMurnaghanEinstein(T, P, materials, phase)

    P /= 1.0e9
    ρr = materials.ρr[phase]
    V0 = materials.V0[phase]

    niter = 20
    tol = 1.0e-12
    iter = 0

    # Initial guess
    V = V0
    r0 = 1.0
    iter = 0
    err = 1.0

    # for iter=1:niter
    #     # Evaluate the function and the Jacobian: r, ∂r∂V
    #     r, dresdV = ad_value_and_derivative(residual, V, P, T, materials, phase)
    #     @show r
    #     if iter==1 r0 = r end
    #     err         = abs(r/r0)
    #     if err<tol break end
    #     # Newton update
    #     V = V - r / dresdV
    # end

    while (iter < niter && err > tol)
        iter += 1
        # Evaluate the function and the Jacobian: r, ∂r∂V
        r, dresdV = ad_value_and_derivative(residual, V, P, T, materials, phase)
        if iter == 1
            r0 = r
        end
        err = abs(r / r0)
        # @show r, dresdV
        # Newton update
        V -= r / dresdV
    end
    ρ = ρr * V0 / V
    return ρ
end
