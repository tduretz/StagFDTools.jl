using MuladdMacro

@inline mynorm(x) = sum(xi^2 for xi in x)

# bulk_viscosity(ϕ, η0, m) = η0*abs(ϕ)^m
@inline bulk_viscosity(ϕ::T, η0, m) where T = iszero(m) ? T(η0) : η0*abs(ϕ)^m

# Trial VE
@inline function PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)  
    ηΦ      = bulk_viscosity(Φ, ξ0, m)
    dPtdt   = @muladd (Pt - Pt0) / Δt
    dPfdt   = @muladd (Pf - Pf0) / Δt
    dΦdt    = @muladd ((dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ)
    return dΦdt, ηΦ
end

# Corrected VEP
@inline function PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)  
    ηΦ      = bulk_viscosity(Φ, ξ0, m)
    dPtdt   = @muladd (Pt - Pt0) / Δt
    dPfdt   = @muladd (Pf - Pf0) / Δt
    P_eff   = Pt - Pf
    ∂Q∂p    = ForwardDiff.derivative( P_eff -> Q(pl, τII, P_eff, 0.0, λ̇, ph), P_eff)
    # ∂Q∂p    = ForwardDiff.derivative( P_eff -> (τII, P_eff, 0.0, pl.C, pl.cosϕ, pl.sinψ, λ̇, pl.ηvp), P_eff)
    dΦdt    = @muladd ((dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ - λ̇*∂Q∂p)
    return dΦdt, ηΦ
end

# Trial VE
@inline function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, Δt) 
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)[1] 
    r    = @muladd Φ - (Φ0  + dΦdt * Δt)  
    return r 
end

# Corrected VEP
@inline function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt) 
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1] 
    r    = @muladd Φ - (Φ0  + dΦdt * Δt)  
    return r 
end

# Trial VE
@inline function Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt) 

    dΦdt, ηΦ = PorosityRate(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)
    Φ        = Φ0  + dΦdt * Δt
    if iszero(m)
        return Φ, dΦdt, ηΦ
    end

    r0       = one(Φ)  # typed to match Φ so r0 doesn't change type after first iter
    for iter=1:10
        r, dresdΦ = ad_value_and_derivative(PorosityResidual, Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)
        if iter==1 r0 = abs(r) + 1e-10 end
        # @show iter, abs(r), abs(r)/r0
        if min(abs(r), abs(r)/r0 ) < 1e-10 break end
        Φ    -=  r / dresdΦ
    end
    dΦdt, ηΦ = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δt)
    return Φ, dΦdt, ηΦ 
end

# Corrected VEP
@inline function Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt) 

    dΦdt, ηΦ = PorosityRate(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)
    Φ        = Φ0  + dΦdt * Δt
    if iszero(m)
        return Φ, dΦdt, ηΦ
    end

    r0       = one(Φ)  # typed to match Φ so r0 doesn't change type after first iter
    for iter=1:10
        r, dresdΦ = ad_value_and_derivative(PorosityResidual, Φ, Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)
        if iter==1 r0 = abs(r) + 1e-10 end
        # @show iter, abs(r), abs(r)/r0
        if min(abs(r), abs(r)/r0 ) < 1e-10 break end
        Φ    -=  r / dresdΦ
    end
    dΦdt, ηΦ = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)
    return Φ, dΦdt, ηΦ 
end

function ΔP_residual(x, Φ, Pt, Pf, divVs, divqD, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇, Δt )

    Pt, Pf = x[1], x[2]

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dlnρfdt = dPfdt / Kf
    # dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    # Φ, dΦdt = Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ, m, λ̇, sinψ, Δt)  
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1]  
    dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    # dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1/Ks * ( dPsdt ) 

    # Ps     = (Pt - phi*Pf)/(1-phi) 
    # dPsdt = (dPtdt - phi*dPfdt) /(1-phi)
    # # dPsdt = ((Pt - phi*Pf)/(1-phi) - (Pt0 - phi0*Pf0)/(1-phi0))/dt
    # # dPsdt = dphidt*(Pt - Pf*phi)/(1-phi)**2 + (dPtdt - phi*dPfdt - 0*Pf*dphidt) / (1 - phi)
    # dlnrhosdt = elastic * 1/K_s * ( dPsdt ) 

    return @SVector [ 
        dlnρsdt   - dΦdt/(1-Φ),
        Φ*dlnρfdt + dΦdt      , 
    ]
end

function ΔP(Pt_trial, Pf_trial, divVs, divqD, Φ, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇::Tλ, Δt) where Tλ

    x   = @SVector[zero(Tλ), zero(Tλ)]  # typed to match λ̇ so J\R doesn't change x's type
    r0  = one(Tλ)
    tol = 1e-13

    for iter=1:10
        R, J = ad_value_and_jacobian(ΔP_residual, x, Φ, Pt_trial, Pf_trial, 0 * divVs, 0 * divqD, 0 * Pt0, 0 * Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇, Δt)
        x  = x .- J \ R
        nr = mynorm(R)
        if iter==1 && nr>1e-17
            r0 = nr
        end
        r = nr/r0
        if r<tol
            break
        end
    end
    return x[1], x[2]
end


function residual_two_phase_P(x, ηve, Δt, ε̇II_eff, τII_trial, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, single_phase )
     
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    α1 = single_phase ? 0.0 : 1.0 

    Pe = if single_phase
        Pt
    else
         Pt .- Pf
       
    end

    ∂Q∂τ  = ForwardDiff.derivative( τII -> F(pl, τII, Pe, 0.0,  λ̇, ph), τII )

    # Pressure corrections: closed form
    # ΔPt_1 = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
    # ΔPf   = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    
    # Pressure corrections: numerics (nested AD)
    ΔPt_1, ΔPf = ΔP(Pt_trial, Pf_trial, divVs, divqD, Φ, Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, τII, pl, ph, λ̇, Δt)

    # Check yield
    fy =  F(pl, τII, Pe, 0.0, λ̇, ph)

    ΔPt = if single_phase
        Ks .* sinψ .* Δt .* λ̇
        else
            ΔPt_1
        end
    
    dΦdt = PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δt)[1]  
    fΦ   =  @muladd Φ - (Φ0  + dΦdt * Δt)  

    return @SVector [ 
        # ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        τII - (τII_trial - ηve*λ̇*∂Q∂τ),
        Pt - (Pt_trial + ΔPt),
        Pf - (Pf_trial + ΔPf),
        fy, 
        fΦ,
    ]
end

function LocalRheology_P(ε̇::SVector{N, D}, divVs, divqD, Pt0, Pf0, Φ0, materials, ph, Δ) where {N, D}

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt = ε̇[4]
    Pf = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[ph]
    m    = materials.m[ph]
    η0   = materials.η0[ph]
    G    = materials.G[ph]
    ξ0   = materials.ξ0[ph]
    KΦ   = materials.KΦ[ph]
    Ks   = materials.Ks[ph]
    Kf   = materials.Kf[ph]

    pl   = materials.plasticity

    # C    = materials.plasticity.C[ph]
    # ηvp  = materials.plasticity.ηvp[ph]
    # sinψ = materials.plasticity.sinψ[ph]    
    # sinϕ = materials.plasticity.sinϕ[ph] 
    # cosϕ = materials.plasticity.cosϕ[ph] 
    
    # pl = (C   =C   ,
    #       ηvp =ηvp ,
    #       sinψ=sinψ,
    #       sinϕ=sinϕ,
    #       cosϕ=cosϕ,)

    # ηvep, λ̇, Pt, Pf, τII, Φ, f  = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    
    α1 = materials.single_phase ? zero(D) : one(D)

    # Initial guess
    η         = η0 * ε̇II_eff^(1 / n - 1 )
    ηve       = inv(1/η + 1/(G*Δ.t))
    τII       = 2*ηve*ε̇II_eff
    ηvep      = ηve

    Φ = if materials.single_phase
        zero(D)
    else
        # Trial porosity: closed form
        # Φ = (KΦ * Δ.t * (Pf - Pt) + KΦ * Φ0 * ξ0 + ξ0 * (Pf - Pf0 - Pt + Pt0)) / (KΦ * ξ0)
    
        # Trial porosity: numerics (nested AD)
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, Δ.t)[1]
    end

    # Check yield
    λ̇  = zero(D)

    #############################

    Peff =  Pt - Pf
    f    = F(pl, τII, Peff, Φ, λ̇, ph)

    x = @SVector [τII, Pt, Pf, λ̇, Φ]
    x0 =copy(x)
    plastic_correction = false

    nr   = D(1.0)
    nr0  = D(1.0)
    tol  = D(1e-10)

    # Return mapping
    if f > D(-1e-13)
        plastic_correction = true
        # This is the proper return mapping with plasticity
        for iter=1:20
            R, J = fd_value_and_jacobian(residual_two_phase_P, x, ηve, Δ.t, ε̇II_eff, τII,       Pt,       Pf,       divVs, divqD, Φ,       Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, materials.single_phase)
            x   -= J \ R
            nr   = mynorm(R)
            if iter==1 
                nr0 = nr
            end
            if iter==20
                error("Local iteration failed: nr=$(nr) nr0=$(nr0) f = $(f) x0 = $(x0) ")
            end
            nr/nr0 < tol && break
        end
    end

    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    Φ = if materials.single_phase
        zero(D)
    else
        Φ
    end

    dΦdt = if materials.single_phase
        zero(D)
    else
        PorosityRate(Φ, Pt, Pf, Pt0, Pf0, KΦ, ξ0, m, τII, pl, ph, λ̇, Δ.t)[1]  
    end

    # EOS
    dPtdt   = (Pt - Pt0) / Δ.t
    dPfdt   = (Pf - Pf0) / Δ.t
    dlnρfdt = dPfdt / Kf
    # dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1/Ks * ( dPsdt ) 

    #############################

    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    return ηvep, λ̇, Pt, Pf, τII, Φ, f, dlnρsdt, dlnρfdt 
end

@inline function StressVector_P!(ε̇::SVector{N, T}, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) where {N,T}
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
    return τ
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, ε̇, λ̇, η, V, P, ΔP, Φ, ρ, old, div_Vs, div_qD, type, BC, materials, phases, rheo, Δ)

    _ones = @SVector ones(5)
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo
    τ0, P0, Φ0, ρ0 = old 
    invΔx, invΔy, Δt = 1 / Δ.x, 1 / Δ.y, Δ.t


    ########################### Loop over centroids ###########################
    Threads.@threads for j=2:size(ε̇.xx,2)-1
        for i=2:size(ε̇.xx,1)-1
            # Local arrays
            Vx_loc    = SMatrix{2,3}(         V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
            Vy_loc    = SMatrix{3,2}(         V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
            bcx       = SMatrix{2,3}(       BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
            bcy       = SMatrix{3,2}(       BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
            typex     = SMatrix{2,3}(     type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
            typey     = SMatrix{3,2}(     type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
            τxy0      = SMatrix{2,2}(       τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)
            Φ0_loc    = SMatrix{3,3}(        Φ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pf_loc    = SMatrix{3,3}(         P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pf0_loc   = SMatrix{3,3}(        P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pt_loc    = SMatrix{3,3}(         P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pt0_loc   = SMatrix{3,3}(        P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typept    = SMatrix{3,3}(     type.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcpt      = SMatrix{3,3}(       BC.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typepf    = SMatrix{3,3}(     type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcpf      = SMatrix{3,3}(       BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            k_ηf0_loc = SMatrix{3,3}(     k_ηf0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ηΦ_loc    = SMatrix{3,3}(        ξ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            KΦ_loc    = SMatrix{3,3}(        KΦ.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            n_loc     = SMatrix{3,3}(      n_CK.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            m_loc     = SMatrix{3,3}(         m.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ρfi_loc   = SMatrix{3,3}(       ρfi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

            # Density for Darcy flux
            ρfgC   = SMatrix{3,3}( @. ρfi_loc * materials.g[2] )
            ρfg    = SVector{2}( 1/2 * (ρfgC[2,j] + ρfgC[2,j+1]) for j=1:2 )

            # BCs
            Vx  = SetBCVx1(Vx_loc, typex, bcx, Δ)
            Vy  = SetBCVy1(Vy_loc, typey, bcy, Δ)
            Pf  = SetBCPf1(Pf_loc,  typepf, bcpf, Δ, ρfg)
            Pt  = SetBCPf1(Pt_loc,  typept, bcpt, Δ, ρfg)
            Pf0 = SetBCPf1(Pf0_loc, typepf, bcpf, Δ, ρfg)
            Pt0 = SetBCPf1(Pt0_loc, typepf, bcpf, Δ, ρfg)

            # Porosity
            Φ_loc = if materials.linearizeΦ
                        SMatrix{3,3}( Φ0_loc ) 
                    else
                        SMatrix{3,3}( Porosity(Φ0_loc[i,j], Pt[i,j], Pf[i,j], Pt0[i,j], Pf0[i,j], KΦ_loc[i,j], ηΦ_loc[i,j], m_loc[i,j], Δ.t )[1] for i=1:3, j=1:3)
            end 

            # Interp Vy -> Vx, Vx - > Vy
            V̄y = SMatrix{2,1}(av2D(Vy))
            V̄x = SMatrix{1,2}(av2D(Vx))

            # More averages
            τ0xx = τ0.xx[i, j]
            τ0yy = τ0.yy[i, j]
            τ0xy = av(τxy0)[1]

            # Velocity gradient - centroids
            Dxx = (∂x(Vx) * invΔx)[1,2]
            Dxy = (∂y(V̄x) * invΔy)[1]
            Dyy = (∂y(Vy) * invΔy)[2,1]
            Dyx = (∂x(V̄y) * invΔx)[1]

            # Deviatoric strain rate
            ε̇xx, ε̇yy, ε̇xy, ε̇kk = deviatoric_strain_rate(Dxx, Dxy, Dyx, Dyy)

            # Effective visco-elastic strain rate
            _2GΔt = inv(2 * G.c[i, j] * Δ.t)
            ϵ̇xx, ϵ̇yy, ϵ̇xy = effective_strain_rate(ε̇xx, ε̇yy, ε̇xy, τ0xx, τ0yy, τ0xy, _2GΔt)
            ε̇vec = SVector{5}(ϵ̇xx, ϵ̇yy, ϵ̇xy, P.t[i, j], P.f[i,j])

            # Darcy flux
            k_μ_xx  = SMatrix{3,3, Float64}( @.  k_ηf0_loc * max.(Φ_loc, 1e-6).^n_loc  )
            kx_μ_xx = SVector{2,   Float64}( @. (k_μ_xx[i,2] + k_μ_xx[i+1,2]) / 2 for i=1:2 )
            k_μ_yy  = k_μ_xx
            ky_μ_yy = SVector{2,   Float64}( @. (k_μ_yy[2,j] + k_μ_yy[2,j+1]) / 2 for j=1:2 )
            ∂Pf∂x   = SVector{2,   Float64}( @. (Pf[i+1,2] - Pf[i,2] ) / Δ.x for i=1:2 )
            ∂Pf∂y   = SVector{2,   Float64}( @. (Pf[2,j+1] - Pf[2,j] ) / Δ.y for j=1:2 )
            qDx     = SVector{2,   Float64}( - kx_μ_xx .*  ∂Pf∂x       ) 
            qDy     = SVector{2,   Float64}( - ky_μ_yy .*  ∂Pf∂y - ρfg ) 
            divqD   = ((qDx[2] - qDx[1]) / Δ.x + (qDy[2] - qDy[1]) / Δ.y)
            
            ##################################

            # # TODO: adapt to phase ratios
            # # Tangent operator used for Newton Linearisation
            τ_vec, jac = fd_value_and_jacobian(StressVector_P!, ε̇vec, ε̇kk, divqD, P0.t[i,j], P0.f[i,j], Φ0.c[i,j], materials, phases.c[i,j], Δ)
            η_local, λ̇_local, Pt1, Pf1, τII_local, Φ_local, f_local, dlnρsdt, dlnρfdt  = LocalRheology_P(ε̇vec, ε̇kk, divqD, P0.t[i,j], P0.f[i,j], Φ0.c[i,j], materials, phases.c[i,j], Δ)
            @views 𝐷_ctl.c[i,j] .= jac

            ##################################

            # Tangent operator used for Picard Linearisation
            𝐷.c[i,j] .= diagm(2 * η_local * _ones)
            𝐷.c[i,j][4,4] = 1
            𝐷.c[i,j][5,5] = 1

            ##################################

            # Update stress
            τ.xx[i,j]     = τ_vec[1]
            τ.yy[i,j]     = τ_vec[2]
            τ.II[i,j]     = τII_local
            τ.f[i,j]      = f_local
            ε̇.xx[i,j]     = ε̇xx[1]
            ε̇.yy[i,j]     = ε̇yy[1]
            ε̇.II[i,j]     = sqrt(1 / 2 * (ε̇xx^2 + ε̇yy^2) + ε̇xy^2)
            λ̇.c[i,j]      = λ̇_local
            Φ.c[i,j]      = Φ_local
            η.c[i,j]      = η_local
            Φ.c[i,j]      = Φ_local
            η.c[i,j]      = η_local
            ρ.s[i,j]      = ρ0.s[i,j] * (1 + dlnρsdt * Δ.t)
            ρ.f[i,j]      = ρ0.f[i,j] * (1 + dlnρfdt * Δ.t)
            div_Vs.c[i,j] = ε̇kk
            div_qD.c[i,j] = divqD
            if  λ̇.c[i,j] > 0
                ΔP.t[i,j] =  (τ_vec[4] - P.t[i,j])
                ΔP.f[i,j] =  (τ_vec[5] - P.f[i,j])
            end
        end
    end

    # Mess with boundaries -  cheap copy !!!
    for j=2:size(div_Vs.c,2)-1
        div_Vs.c[  1, j] = div_Vs.c[    2, j]
        div_Vs.c[end, j] = div_Vs.c[end-1, j]
    end
    for i=1:size(div_Vs.c,1)-0
        div_Vs.c[i,   1] = div_Vs.c[i,     2]
        div_Vs.c[i, end] = div_Vs.c[i, end-1]
    end

    # Need a cheap copy at ghost boundaries in case of stress BC along that boundary
    for i in axes(ε̇.xx, 1)
        if type.Vy[i+1, 1] == :Neumann_normal
            𝐷.c[i, 1] = 𝐷.c[i, 2]
        end
        if type.Vy[i+1, end] == :Neumann_normal
            𝐷.c[i, end] = 𝐷.c[i, end-1]
        end
    end

    for j in axes(ε̇.xx, 2)
        if type.Vx[1, j+1] == :Neumann_normal
            𝐷.c[1, j] = 𝐷.c[2, j]
        end
        if type.Vx[end, j+1] == :Neumann_normal
            𝐷.c[end, j] = 𝐷.c[end-1,j]
        end
    end

    ########################### Loop over vertices ###########################
    Threads.@threads for j=2:size(ε̇.xy,2)-1
        for i=2:size(ε̇.xy,1)-1
            Vx_loc    = SMatrix{3,2}(        V.x[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
            Vy_loc    = SMatrix{2,3}(        V.y[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
            bcx       = SMatrix{3,2}(      BC.Vx[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
            bcy       = SMatrix{2,3}(      BC.Vy[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
            typex     = SMatrix{3,2}(    type.Vx[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
            typey     = SMatrix{2,3}(    type.Vy[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
            τxx0      = SMatrix{2,2}(      τ0.xx[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            τyy0      = SMatrix{2,2}(      τ0.yy[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            Φ0_loc    = SMatrix{2,2}(       Φ0.c[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            Pt0_loc   = SMatrix{2,2}(       P0.t[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            Pf0_loc   = SMatrix{2,2}(       P0.f[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            Pf_loc    = SMatrix{2,2}(        P.f[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            Pt_loc    = SMatrix{2,2}(        P.t[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            typept    = SMatrix{2,2}(    type.Pt[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            bcpt      = SMatrix{2,2}(      BC.Pt[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            typepf    = SMatrix{2,2}(    type.Pf[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            bcpf      = SMatrix{2,2}(      BC.Pf[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
            ρfi_loc   = SMatrix{2,2}(      ρfi.c[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)

            # Fluid density
            ρfgC   = SMatrix{2,2}( @. ρfi_loc * materials.g[2] )
            ρfg    = SMatrix{2, 1, Float64}(1/2 * (ρfgC[i,j] + ρfgC[i,j+1]) for i=1:2, j=1:1)

            # Set BCs
            Vx  = SetBCVx1(Vx_loc,  typex, bcx, Δ)
            Vy  = SetBCVy1(Vy_loc,  typey, bcy, Δ)
            Pf  = SetBCPf1(Pf_loc,  typepf, bcpf, Δ, ρfg)
            Pt  = SetBCPf1(Pt_loc,  typept, bcpt, Δ, ρfg)
            Pf0 = SetBCPf1(Pf0_loc, typepf, bcpf, Δ, ρfg)
            Pt0 = SetBCPf1(Pt0_loc, typept, bcpt, Δ, ρfg)

            # Interp Vy -> Vx, Vx - > Vy
            V̄y = SMatrix{1,2}(av2D(Vy))
            V̄x = SMatrix{2,1}(av2D(Vx))

            # More averages
            τ0xx = av(τxx0)[1]
            τ0yy = av(τyy0)[1]
            τ0xy = τ0.xy[i, j]
            P̄t   = av(Pt)[1]
            P̄f   = av(  Pf)[1]
            P̄t0  = av(Pt0)[1]
            P̄f0  = av(Pf0)[1]
            ϕ̄0   = av(Φ0_loc)[1]

            # Velocity gradient - centroids
            Dxx = (∂x(V̄x) * invΔx)[1]
            Dxy = (∂y(Vx) * invΔy)[2,1]
            Dyy = (∂y(V̄y) * invΔy)[1]
            Dyx = (∂x(Vy) * invΔx)[1,2]

            # Deviatoric strain rate
            ε̇xx, ε̇yy, ε̇xy, ε̇kk = deviatoric_strain_rate(Dxx, Dxy, Dyx, Dyy)

            # Effective visco-elastic strain rate
            _2GΔt = inv(2 * G.v[i, j] * Δ.t)
            ϵ̇xx, ϵ̇yy, ϵ̇xy = effective_strain_rate(ε̇xx, ε̇yy, ε̇xy, τ0xx, τ0yy, τ0xy, _2GΔt)
            ε̇vec = SVector{5}(ϵ̇xx, ϵ̇yy, ϵ̇xy, P̄t, P̄f)

            # Darcy flux divergence
            divqD̄   = 0.25*(div_qD.c[i-1,j-1] + div_qD.c[i,j-1] + div_qD.c[i-1,j] + div_qD.c[i,j])

            ##################################

            # TODO: adapt to phase ratios
            # Tangent operator used for Newton Linearisation
            τ_vec, jac = fd_value_and_jacobian(StressVector_P!, ε̇vec, ε̇kk, divqD̄, P̄t0, P̄f0, ϕ̄0, materials, phases.v[i,j], Δ)
            η_local, λ̇_local, Pt1, Pf1, τII_local, Φ_local, f_local, dlnρsdt, dlnρfdt  = LocalRheology_P(ε̇vec, ε̇kk, divqD̄, P̄t0, P̄f0, ϕ̄0, materials, phases.v[i,j], Δ)
            @views 𝐷_ctl.v[i,j] .= jac

            ##################################

            # Tangent operator used for Picard Linearisation
            𝐷.v[i,j]     .= diagm(2 * η_local * _ones)
            𝐷.v[i,j][4,4] = 1
            𝐷.v[i,j][5,5] = 1

            ##################################

            # Update stress
            τ.xy[i,j]     = τ_vec[3]
            ε̇.xy[i,j]     = ε̇xy
            λ̇.v[i,j]      = λ̇_local
            η.v[i,j]      = η_local
            div_qD.v[i,j] = divqD̄   
            
        end
    end
end
