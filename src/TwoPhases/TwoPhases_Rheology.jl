invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + 1/2*(-x[1]-x[2])^2 + x[3]^2) 

function StrainRateTrial(τII, Pt, Pf, ηve, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δt)
    ε̇II_trial = τII/2/ηve
    return ε̇II_trial
end

F(τ, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α) = τ - (1-Φ)*C*cosϕ - (Pt - α*Pf)*sinϕ  - λ̇*ηvp 

function residual_two_phase_trial(x, divVs, divqD, Δt, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf)
     
    Pt, Pf, Φ = x[1], x[2], x[3]

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ

    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    f_sol = dlnρsdt   - dΦdt/(1-Φ) +   divVs
    f_liq = (Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD)/ηΦ

    return @SVector [ 
        f_sol,
        f_liq,
        Φ    - (Φ0 + dΦdt*Δt),
    ]
end

function residual_two_phase(x, ηve, Δt, ε̇II_eff, Pt_trial, Pf_trial, Φ_trial, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )
     
    # eps   = -1e-20
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]
    single_phase ? α1 = 0.0 : α1 = 1.0 

    # Pressure corrections
    ΔPt = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
    ΔPf = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    
    # Check yield
    f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sinψ#*(f>=eps)

    if single_phase
        f   = τII - C*cosϕ - Pt*sinϕ  
        ΔPt = Ks .* sinψ .* Δt .* λ̇
    end

    return @SVector [ 
        ε̇II_eff   -  τII/2/ηve - λ̇/2,#*(f>=eps),
        Pt - (Pt_trial + ΔPt),
        Pf - (Pf_trial + ΔPf),
        f, #(f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps),
        Φ    - (Φ0 + dΦdt*Δt),
    ]
end

function residual_two_phase_div(x, ηve, Δt, ε̇II_eff, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )
     
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]
     α1 = single_phase ? 0.0 : 1.0 

    # Check yield
    f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sinψ

    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    f_sol = dlnρsdt   - dΦdt/(1-Φ) +   divVs
    f_liq = (Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD)/ηΦ
    f_por = Φ  - (Φ0 + dΦdt*Δt)

    if single_phase
        f     = τII - C*cosϕ - Pt*sinϕ  
        f_sol = (Pt - Pt0)/(Ks*Δt) - λ̇*sinψ + divVs
    end

    return @SVector [ 
        ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        f_sol,
        f_liq,
        f,
        f_por,
    ]
end

function LocalRheology(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt       = ε̇[4]
    Pf       = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    η0   = materials.η0[phases]
    # B    = materials.B[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]
    ηΦ   = materials.ξ0[phases]
    KΦ   = materials.KΦ[phases]
    Ks   = materials.Ks[phases]
    Kf   = materials.Kf[phases]

    ηvp  = materials.ηvp[phases]
    sinψ = materials.sinψ[phases]    
    sinϕ = materials.sinϕ[phases] 
    cosϕ = materials.cosϕ[phases]  
    
    α1 = materials.single_phase ? 0.0 : 1.0 

    # Initial guess
    η    = (η0 .* ε̇II_eff.^(1 ./ n .- 1.0 ))[1]
    ηve  = inv(1/η + 1/(G*Δ.t))
    τII  = 2*ηve*ε̇II_eff

    # Trial porosity
    Φ = (KΦ .* Δ.t .* (Pf - Pt) + KΦ .* Φ0 .* ηΦ + ηΦ .* (Pf - Pf0 - Pt + Pt0)) ./ (KΦ .* ηΦ)

    # Check yield
    λ̇  = 0.0

    # f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, 0.0)
    # if f>0
    #     λ̇ = f / (KΦ .* Δ.t * sinϕ * sinψ + ηve + ηvp)
    #     f  = τII - λ̇*ηve - C*cosϕ - (Pt + KΦ .* Δ.t * sinψ * λ̇)*sinϕ
    #     # @show f, λ̇
    #     # error()

    #     τII = τII - λ̇*ηve
    #     Pt  = Pt + KΦ .* Δ.t * sinψ * λ̇
    # end

    #############################

    f_trial  = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    x = @MVector ([τII, Pt, Pf, 0.0, Φ])

    # Return mapping
    if f_trial>-1e-13 

        # This is the proper return mapping with plasticity
        r0  = 1.0
        tol = 1e-10

        for iter=1:10
            R, J = ad_value_and_jacobian(residual_two_phase, x, ηve, Δ.t, ε̇II_eff, Pt, Pf, Φ, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, materials.single_phase)
            x .-= J \ R
            if iter==1 
                r0 = norm(R)
            end
            r = norm(R)/r0

            R = residual_two_phase( x, (ηve), (Δ.t), (ε̇II_eff), (Pt), (Pf), (Φ), (Pt0), (Pf0), (Φ0), (ηΦ), (KΦ), (Ks), (Kf), (C), (cosϕ), (sinϕ), (sinψ), (ηvp), (materials.single_phase))

            # @show iter, J.val
            # @show R
            # @show (x[1], x[2], x[3], 0.0, C, cosϕ, sinϕ, x[4], ηvp, 0.0)
            # @show F(x[1], x[2], x[3], 0.0, C, cosϕ, sinϕ, x[4], ηvp, 0.0)
   
            if r<tol
                break
            end
        end

    end

    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    #############################

    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    if materials.single_phase
        Φ = 0.0
    end

    f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    return ηvep, λ̇, Pt, Pf, τII, Φ, f
end











function residual_two_phase_div_pressure(x, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ηΦ, Δt)
    
    Pt, Pf, Φ = x[1], x[2], x[3]
    
    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ

    # Equations of states
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    return @SVector([ 
        dlnρsdt   - dΦdt/(1-Φ) +   divVs
        (Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD)/ηΦ
        Φ  - (Φ0 + dΦdt*Δt)
    ])
end

function Pressures(div, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ηΦ, Δt)

    divVs = div[1]
    divqD = div[2]
   
    x = @MVector[Pt0, Pf0, Φ0]

    # This is the proper return mapping with plasticity
    r0  = 1.0
    tol = 1e-13

    for iter=1:10
        R, J = ad_value_and_jacobian(residual_two_phase_div_pressure, x, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ηΦ, Δt)
        x .-= J \ R
        if iter==1 
            r0 = norm(R)
        end
        r = norm(R)/r0
        # @show iter, r
        if r<tol
            break
        end
    end

    return @SVector[x[1], x[2], x[3]] 
end

function LocalRheology_div(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    divVs    = ε̇[4]
    divqD    = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    η0   = materials.η0[phases]
    # B    = materials.B[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]
    ηΦ   = materials.ξ0[phases]
    KΦ   = materials.KΦ[phases]
    Ks   = materials.Ks[phases]
    Kf   = materials.Kf[phases]

    ηvp  = materials.ηvp[phases]
    sinψ = materials.sinψ[phases]    
    sinϕ = materials.sinϕ[phases] 
    cosϕ = materials.cosϕ[phases]  
    
    α1 = materials.single_phase ? 0.0 : 1.0 

    # Initial guess
    η    = (η0 .* ε̇II_eff.^(1 ./ n .- 1.0 ))[1]
    ηve  = inv(1/η + 1/(G*Δ.t))
    τII  = 2*ηve*ε̇II_eff

    div = @SVector[divVs, divqD]
    x = Pressures(div, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ηΦ, Δ.t)
    Pt, Pf, Φ = x[1], x[2], x[3]

    #############################
    λ̇ = 0.0

    f_trial  = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    x = @MVector ([τII, Pt, Pf, 0.0, Φ])

    # Return mapping
    if f_trial>-1e-13 

        # This is the proper return mapping with plasticity
        r0  = 1.0
        tol = 1e-10

        for iter=1:10
            R, J = ad_value_and_jacobian(residual_two_phase_div, x, ηve, Δ.t, ε̇II_eff, divVs, divqD, Φ, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, materials.single_phase)
            x .-= J \ R
            if iter==1 
                r0 = norm(R)
            end
            r = norm(R)/r0

            R = residual_two_phase( x, (ηve), (Δ.t), (ε̇II_eff), (Pt), (Pf), (Φ), (Pt0), (Pf0), (Φ0), (ηΦ), (KΦ), (Ks), (Kf), (C), (cosϕ), (sinϕ), (sinψ), (ηvp), (materials.single_phase))

            # @show iter, J.val
            # @show R
            # @show (x[1], x[2], x[3], 0.0, C, cosϕ, sinϕ, x[4], ηvp, 0.0)
            # @show F(x[1], x[2], x[3], 0.0, C, cosϕ, sinϕ, x[4], ηvp, 0.0)
   
            if r<tol
                break
            end
        end

    end

    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    #############################

    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    if materials.single_phase
        Φ = 0.0
    end

    f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)
    
    return ηvep, λ̇, Pt, Pf, τII, Φ, f
end

function StressVector!(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ) 
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
    return τ, η, λ̇, τII, Φ, f
end

function StressVector_div!(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ) 
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_div(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
    return τ, η, λ̇, τII, Φ, f
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)

    _ones = @SVector ones(5)

    D_test = @MMatrix zeros(5,5)
    C           = @MMatrix zeros(5,5)
    C[diagind(C)] .= 1.0

    # Loop over centroids
    # @show "CENTROIDS"
    for j=2:size(ε̇.xx,2)-1, i=2:size(ε̇.xx,1)-1
 
        Vx     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
        Vy     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
        bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        τxy0   = SMatrix{2,2}(    τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pf_loc = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typepf = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcpf   = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # BCs
        Vx  = SetBCVx1(Vx, typex, bcx, Δ)
        Vy  = SetBCVy1(Vy, typey, bcy, Δ)
        Pf  = SetBCPf1(Pf_loc, typepf, bcpf, Δ)

        # Kinematics
        Dxx = ∂x_inn(Vx) / Δ.x 
        Dyy = ∂y_inn(Vy) / Δ.y 
        Dxy = ∂y(Vx) / Δ.y
        Dyx = ∂x(Vy) / Δ.x
        
        Dkk = Dxx .+ Dyy
        ε̇xx = @. Dxx - Dkk ./ 3
        ε̇yy = @. Dyy - Dkk ./ 3
        ε̇xy = @. (Dxy + Dyx) ./ 2
        ε̇̄xy = av(ε̇xy)

        qDx   = materials.k_ηf0[1] .*  ∂x_inn(Pf) / Δ.x 
        qDy   = materials.k_ηf0[1] .*  ∂y_inn(Pf) / Δ.y
        divqD = (∂x(qDx) + ∂y(qDy))[1]
       
        # Visco-elasticity
        G      = materials.G[phases.c[i,j]]
        τ̄xy0   = av(τxy0)
        ε̇vec   = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), P.t[i,j], P.f[i,j]])
        τ0_loc = @SVector([τ0.xx[i,j], τ0.yy[i,j], τ̄xy0[1]])

        ##################################

        # Tangent operator used for Newton Linearisation
        stress_state, τ_vec, jac = ad_value_and_jacobian_first(StressVector!, ε̇vec, Dkk[1], divqD, P0.t[i,j], P0.f[i,j], Φ0.c[i,j], τ0_loc, materials, phases.c[i,j], Δ)
        _, η_local, λ̇_local, τII_local, Φ_local, f_local = stress_state

        @views 𝐷_ctl.c[i,j] .= jac

        ##################################

        # Pressure block
        KΦ      = materials.KΦ[phases.c[i,j]]
        Ks      = materials.Ks[phases.c[i,j]]
        Kf      = materials.Kf[phases.c[i,j]]
        ηΦ      = materials.ξ0[phases.c[i,j]]
        x = @SVector[Dkk[1], divqD]
        Jp = ad_jacobian(Pressures, x, P0.t[i,j], P0.f[i,j], Φ0.c[i,j], KΦ, Ks, Kf, ηΦ, Δ.t)

        @views C[4:5,4:5] .=  inv(Jp[1:2,1:2])

        ε̇vec   = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Dkk[1], divqD])
        _, _, jac2 = ad_value_and_jacobian_first(StressVector_div!, ε̇vec, Dkk[1], divqD, P0.t[i,j], P0.f[i,j], Φ0.c[i,j], τ0_loc, materials, phases.c[i,j], Δ)

        @views D_test .= jac2

        𝐷_ctl.c[i,j] .= D_test * C

        # display(𝐷_ctl.c[i,j])
        # display(D_test * C)
        # error()

        # Derr = (𝐷_ctl.c[i,j] .- D_test * C)
        # if norm(Derr)>1e-10
        #     display(𝐷_ctl.c[i,j])
        #     display(D_test * C)
        #     display(Derr)
        #     error()
        # end

        ##################################

        # Tangent operator used for Picard Linearisation
        𝐷.c[i,j] .= diagm(2 * η_local * _ones)
        𝐷.c[i,j][4,4] = 1
        𝐷.c[i,j][5,5] = 1

        ##################################

        # Update stress
        τ.xx[i,j] = τ_vec[1]
        τ.yy[i,j] = τ_vec[2]
        τ.II[i,j] = τII_local
        τ.f[i,j]  = f_local
        ε̇.xx[i,j] = ε̇xx[1]
        ε̇.yy[i,j] = ε̇yy[1]
        ε̇.II[i,j] = invII( @SVector([ε̇xx[1], ε̇yy[1], ε̇̄xy[1]]) )
        λ̇.c[i,j]  = λ̇_local
        Φ.c[i,j]  = Φ_local
        η.c[i,j]  = η_local
        ΔP.t[i,j] = τ_vec[4] - P.t[i,j]
        ΔP.f[i,j] = τ_vec[5] - P.f[i,j]
    end

    # Loop over vertices
    # @show "VERTICES" 
    for j=3:size(ε̇.xy,2)-2, i=3:size(ε̇.xy,1)-2
        Vx      = SMatrix{3,2}(      V.x[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
        Vy      = SMatrix{2,3}(      V.y[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
        bcx     = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
        bcy     = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
        typex   = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
        typey   = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
        τxx0    = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        τyy0    = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pt      = SMatrix{2,2}(      P.t[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pf      = SMatrix{2,2}(      P.f[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Φ0_loc  = SMatrix{2,2}(     Φ0.c[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pt0_loc = SMatrix{2,2}(     P0.t[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pf0_loc = SMatrix{2,2}(     P0.f[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)

        Pfex    = SMatrix{4,4}(      P.f[ii,jj] for ii in i-2:i+1,   jj in j-2:j+1)
        typepf  = SMatrix{4,4}(  type.Pf[ii,jj] for ii in i-2:i+1,   jj in j-2:j+1)
        bcpf    = SMatrix{4,4}(    BC.Pf[ii,jj] for ii in i-2:i+1,   jj in j-2:j+1)

        Vx     = SetBCVx1(Vx, typex, bcx, Δ)
        Vy     = SetBCVy1(Vy, typey, bcy, Δ)
        Pf     = SetBCPf1(Pfex, typepf, bcpf, Δ)

        Dxx    = ∂x(Vx) / Δ.x
        Dyy    = ∂y(Vy) / Δ.y
        Dxy    = ∂y_inn(Vx) / Δ.y
        Dyx    = ∂x_inn(Vy) / Δ.x

        Dkk   = @. Dxx + Dyy
        ε̇xx   = @. Dxx - Dkk / 3
        ε̇yy   = @. Dyy - Dkk / 3
        ε̇xy   = @. (Dxy + Dyx) /2
        ε̇̄xx   = av(ε̇xx)
        ε̇̄yy   = av(ε̇yy)

        qDx   = materials.k_ηf0[1] .*  ∂x_inn(Pf) / Δ.x 
        qDy   = materials.k_ηf0[1] .*  ∂y_inn(Pf) / Δ.y
        divqD = (∂x(qDx) + ∂y(qDy))
        
        divqD̄ = av(divqD)[1]
        
        # Visco-elasticity
        G     = materials.G[phases.v[i,j]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄t    = av(  Pt)
        P̄f    = av(  Pf)
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), P̄t[1], P̄f[1]])
        τ0_loc  = @SVector([τ̄xx0[1], τ̄yy0[1], τ0.xy[i,j]])

        D̄kk   = av(Dkk)
        ϕ̄0    = av(Φ0_loc)
        P̄t0   = av(Pt0_loc)
        P̄f0   = av(Pf0_loc)

        ##################################

        # Tangent operator used for Newton Linearisation
        stress_state, τ_vec, jac = ad_value_and_jacobian_first(StressVector!, ε̇vec, D̄kk[1], divqD̄, P̄t0[1], P̄f0[1], ϕ̄0[1], τ0_loc, materials, phases.v[i,j], Δ)
        _, η_local, λ̇_local, _, _, _ = stress_state

        @views 𝐷_ctl.v[i,j] .= jac

        ##################################

        # Pressure block
        KΦ      = materials.KΦ[phases.v[i,j]]
        Ks      = materials.Ks[phases.v[i,j]]
        Kf      = materials.Kf[phases.v[i,j]]
        ηΦ      = materials.ξ0[phases.v[i,j]]
        x = @SVector[D̄kk[1], divqD̄]
        Jp = ad_jacobian(Pressures, x, P̄t0[1], P̄f0[1], ϕ̄0[1], KΦ, Ks, Kf, ηΦ, Δ.t)

        @views C[4:5,4:5] .=  inv(Jp[1:2,1:2])

        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), D̄kk[1], divqD̄])
        _, _, jac2 = ad_value_and_jacobian_first(StressVector_div!, ε̇vec, D̄kk[1], divqD̄, P̄t0[1], P̄f0[1], ϕ̄0[1], τ0_loc, materials, phases.v[i,j], Δ)

        @views D_test .= jac2

        # Derr = (𝐷_ctl.v[i,j] .- D_test * C)
        # if norm(Derr)>1e-10
        #     display(𝐷_ctl.v[i,j])
        #     display(D_test * C)
        #     display(Derr)
        #     error()
        # end

        𝐷_ctl.v[i,j] .= D_test * C

        ##################################

        # Tangent operator used for Picard Linearisation
        𝐷.v[i,j] .= diagm(2 * η_local * _ones)
        𝐷.v[i,j][4,4] = 1
        𝐷.v[i,j][5,5] = 1

        # Update stress
        τ.xy[i,j] = τ_vec[3]
        ε̇.xy[i,j] = ε̇xy[1]
        λ̇.v[i,j]  = λ̇_local
        η.v[i,j]  = η_local
    end
end
