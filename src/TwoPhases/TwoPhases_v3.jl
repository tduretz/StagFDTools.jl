struct Fields{Tx,Ty,Tp,Tpf}
    Vx::Tx
    Vy::Ty
    Pt::Tp
    Pf::Tpf
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 5 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
    i == 4 && return x.Pf
end

function Ranges(nc)     
    return (inx_Vx = 2:nc.x+2, iny_Vx = 3:nc.y+2, inx_Vy = 3:nc.x+2, iny_Vy = 2:nc.y+2, inx_c = 2:nc.x+1, iny_c = 2:nc.y+1, inx_v = 2:nc.x+2, iny_v = 2:nc.y+2, size_x = (nc.x+3, nc.y+4), size_y = (nc.x+4, nc.y+3), size_c = (nc.x+2, nc.y+2), size_v = (nc.x+3, nc.y+3))
end

function SMomentum_x_Generic(Vx_loc, Vy_loc, Pt,    Pf,     ΔP,      τ0,    G_loc, 𝐷, materials, type,      bcv,    Δ)
    
    invΔx, invΔy, BC_sym = 1 / Δ.x, 1 / Δ.y, 1.0

    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    # Interp Vy -> Vx, Vx - > Vy
    V̄y = av2D(Vy)
    V̄x = av2D(Vx)

    # More averages
    P̄f     = SVector(av(Pf))
    P̄t     = SVector(av(Pt))
    τ0xx_c = SVector{2}(τ0.xx[i, 2] for i = 1:2)
    τ0yy_c = SVector{2}(τ0.yy[i, 2] for i = 1:2)
    τ0xy_c = SVector(av(τ0.xy))
    τ0xx_v = SVector(av(τ0.xx))
    τ0yy_v = SVector(av(τ0.yy))
    τ0xy_v = SVector{2}(τ0.xy[2, i] for i = 1:2)

    # Velocity gradient - centroids
    ∂Vx∂x = ∂x(Vx) .* invΔx
    Dxx_c = SVector{2}(∂Vx∂x[i, 2] for i = 1:2)
    ∂V̄x∂y = (∂y(V̄x) * invΔy)
    Dxy_c = SVector{2}(∂V̄x∂y[i] for i = 1:2)
    ∂Vy∂y = ∂y(Vy) * invΔy
    Dyy_c = SVector{2}(∂Vy∂y[i, 2] for i = 2:3)
    ∂V̄y∂x = ∂x(V̄y) * invΔx
    Dyx_c = SVector{2}(∂V̄y∂x[i, 2] for i = 1:2)

    # Velocity gradient - vertices
    ∂V̄x∂x = ∂x(V̄x) * invΔx
    Dxx_v = SVector{2}(∂V̄x∂x[i] for i = 1:2)
    ∂Vx∂y = ∂y(Vx) * invΔy
    Dxy_v = SVector{2}(∂Vx∂y[2, i] for i = 1:2)
    ∂V̄y∂y = ∂y(V̄y) * invΔy
    Dyy_v = SVector{2}(∂V̄y∂y[2, i] for i = 1:2)
    ∂Vy∂x = ∂x(Vy) * invΔx
    Dyx_v = SVector{2}(∂Vy∂x[2, i] for i = 2:3)
    # Deviatoric strain rate
    ε̇xx_c, ε̇yy_c, ε̇xy_c, ε̇kk_c = deviatoric_strain_rate(Dxx_c, Dxy_c, Dyx_c, Dyy_c)
    ε̇xx_v, ε̇yy_v, ε̇xy_v, ε̇kk_v = deviatoric_strain_rate(Dxx_v, Dxy_v, Dyx_v, Dyy_v)
    # Effective visco-elastic strain rate
    Gc = SVector{2}(G_loc.c[i, 1] for i = 1:2)
    Gv = SVector{2}(G_loc.v[1, i] for i = 1:2)
    _2GΔt_c = @. inv(2 * Gc * Δ.t)
    _2GΔt_v = @. inv(2 * Gv * Δ.t)
    ϵ̇xx_c, ϵ̇yy_c, ϵ̇xy_c = effective_strain_rate(ε̇xx_c, ε̇yy_c, ε̇xy_c, τ0xx_c, τ0yy_c, τ0xy_c, _2GΔt_c)
    ϵ̇xx_v, ϵ̇yy_v, ϵ̇xy_v = effective_strain_rate(ε̇xx_v, ε̇yy_v, ε̇xy_v, τ0xx_v, τ0yy_v, τ0xy_v, _2GΔt_v)

    # Corrected pressure
    comp = materials.compressible
    Ptc  = SVector{2}(Pt[i, 2] + comp * ΔP[i] for i = 1:2)

    # Stress
    σxx = SVector{2}(
        (𝐷.c[i][1,1] - 𝐷.c[i][4,1]) * ϵ̇xx_c[i] + (𝐷.c[i][1,2] - 𝐷.c[i][4,2]) * ϵ̇yy_c[i] + (𝐷.c[i][1,3] - 𝐷.c[i][4,3]) * ϵ̇xy_c[i] + (𝐷.c[i][1,4] + (1 - 𝐷.c[i][4,4])) * Pt[i,2] + 𝐷.c[i][1,5] * Pf[i,2] - Ptc[i]
        for i in 1:2
    )
    τxy = SVector{2}(
        𝐷.v[i][3,1] * ϵ̇xx_v[i] + 𝐷.v[i][3,2] * ϵ̇yy_v[i] + 𝐷.v[i][3,3] * ϵ̇xy_v[i] + 𝐷.v[i][3,4] * P̄t[i] + 𝐷.v[i][3,5] * P̄f[i]
        for i in 1:2
    )

    # Apply normal stress BC 
    if type.x[1,2] == :normal_stress
        σxx = SVector{2}([2*bcv.x[2,2]-σxx[2] σxx[2]])
        BC_sym = 1 / 2 
    end
    if type.x[end,2] == :normal_stress
        σxx = SVector{2}([σxx[1] 2*bcv.x[end-1,2]-σxx[1] ])
        BC_sym = 1 / 2 
    end

    # Residual
    fx =  (σxx[2] - σxx[1]) * invΔx
    fx += (τxy[2] - τxy[1]) * invΔy
    fx *= -Δ.x * Δ.y
    fx *= BC_sym

    return fx
end

function SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP,     Pt0,     Pf0,     Φ0,     τ0,     G_loc, rheo    , 𝐷, materials, type, bcv, Δ)

    invΔx, invΔy, BC_sym = 1 / Δ.x, 1 / Δ.y, 1.0 

    ξ0, KΦ, m, ρs, ρf = rheo
 
    # BC
    Vx   = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy   = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)
    # @show ρf
    # @show materials.g[2]
    # @show ρf .* materials.g[2]
    ρ0fg = ρf .* materials.g[2]
    Pt   = SetBCPt1(Pt_loc, type.pt, bcv.pt, Δ, ρ0fg)
    Pf   = SetBCPf1(Pf_loc, type.pf, bcv.pf, Δ, ρ0fg)

    # Interp Vy -> Vx, Vx - > Vy
    V̄y = av2D(Vy)   # 2, 2
    V̄x = av2D(Vx)   # 3, 3

    # # More averages
    P̄t     = SVector(av(Pt))
    P̄f     = SVector(av(Pf))
    τ0xx_c = SVector{2}(τ0.xx[2, i] for i = 1:2)
    τ0yy_c = SVector{2}(τ0.yy[2, i] for i = 1:2)
    τ0xy_c = SVector(av(τ0.xy))
    τ0xx_v = SVector(av(τ0.xx))
    τ0yy_v = SVector(av(τ0.yy))
    τ0xy_v = SVector{2}(τ0.xy[i, 2] for i = 1:2)

    # Velocity gradient - centroids
    ∂Vx∂x = ∂x(Vx) * invΔx
    Dxx_c = SVector{2}(∂Vx∂x[2, i] for i = 2:3)
    ∂V̄x∂y = ∂y(V̄x) * invΔy
    Dxy_c = SVector{2}(∂V̄x∂y[2, i] for i = 1:2)
    ∂Vy∂y = ∂y(Vy) * invΔy
    Dyy_c = SVector{2}(∂Vy∂y[2, i] for i = 1:2)
    ∂V̄y∂x = ∂x(V̄y) * invΔx
    Dyx_c = SVector{2}(∂V̄y∂x[i] for i = 1:2)

    # Velocity gradient - vertices
    ∂V̄x∂x = ∂x(V̄x) * invΔx
    Dxx_v = SVector{2}(∂V̄x∂x[i, 2] for i = 1:2)
    ∂Vx∂y = ∂y(Vx) * invΔy
    Dxy_v = SVector{2}(∂Vx∂y[i, 2] for i = 2:3)
    ∂V̄y∂y = ∂y(V̄y) * invΔy
    Dyy_v = SVector{2}(∂V̄y∂y[i] for i = 1:2)
    ∂Vy∂x = ∂x(Vy) * invΔx
    Dyx_v = SVector{2}(∂Vy∂x[i, 2] for i = 1:2)

    # Deviatoric strain rate
    ε̇xx_c, ε̇yy_c, ε̇xy_c, ε̇kk_c = deviatoric_strain_rate(Dxx_c, Dxy_c, Dyx_c, Dyy_c)
    ε̇xx_v, ε̇yy_v, ε̇xy_v, ε̇kk_v = deviatoric_strain_rate(Dxx_v, Dxy_v, Dyx_v, Dyy_v)

    # Effective visco-elastic strain rate
    Gc = SVector{2}(G_loc.c[1, i] for i = 1:2)
    Gv = SVector{2}(G_loc.v[i, 1] for i = 1:2)
    _2GΔt_c = @. inv(2 * Gc * Δ.t)
    _2GΔt_v = @. inv(2 * Gv * Δ.t)
    ϵ̇xx_c, ϵ̇yy_c, ϵ̇xy_c = effective_strain_rate(ε̇xx_c, ε̇yy_c, ε̇xy_c, τ0xx_c, τ0yy_c, τ0xy_c, _2GΔt_c)
    ϵ̇xx_v, ϵ̇yy_v, ϵ̇xy_v = effective_strain_rate(ε̇xx_v, ε̇yy_v, ε̇xy_v, τ0xx_v, τ0yy_v, τ0xy_v, _2GΔt_v)

    # Corrected pressure
    comp = materials.compressible
    Ptc  = SVector{2}(  Pt[2, i] + comp * ΔP.t[i] for i = 1:2)
    Ptc0 = SVector{2}( Pt0[2, i]                  for i = 1:2)
    Pfc  = SVector{2}(  Pf[2, i] + comp * ΔP.f[i] for i = 1:2)
    Pfc0 = SVector{2}( Pf0[2, i]                  for i = 1:2)

    # Porosity
    # THIS IF STATEMENT DOES NOT COMPILE WITH ENZYME
    # if materials.linearizeΦ == true
    #     Φ         = @. Φ0 
    # else 
        Φ         = SVector{2}( Porosity(Φ0[ii], Ptc[ii], Pfc[ii], Ptc0[ii], Pfc0[ii], KΦ[ii], ξ0[ii], m[ii], 0., 0., Δ.t)[1] for ii in eachindex(Φ0))
    # end

    # Density
    ρt   = @. (1-Φ) * ρs + Φ * ρf
    ρg   = materials.g[2] * 0.5*(ρt[1] + ρt[2])

    # Stress
    σyy = SVector{2}(
        (𝐷.c[i][2,1] - 𝐷.c[i][4,1]) * ϵ̇xx_c[i] + (𝐷.c[i][2,2] - 𝐷.c[i][4,2]) * ϵ̇yy_c[i] + (𝐷.c[i][2,3] - 𝐷.c[i][4,3]) * ϵ̇xy_c[i] + (𝐷.c[i][2,4] + (1 - 𝐷.c[i][4,4])) * Pt[2,i] + 𝐷.c[i][2,5] * Pf[2,i] - Ptc[i]
        for i in 1:2
    )
    τxy = SVector{2}(
        𝐷.v[i][3,1] * ϵ̇xx_v[i] + 𝐷.v[i][3,2] * ϵ̇yy_v[i] + 𝐷.v[i][3,3] * ϵ̇xy_v[i] + 𝐷.v[i][3,4] * P̄t[i] + 𝐷.v[i][3,5] * P̄f[i]
        for i in 1:2
    )

    # Gravity
    # ρ  = SVector{2}(ρ_loc[1, i] for i = 1:2)
    ρg = materials.g[2] * 0.5 * (ρt[1] + ρt[2])

    # Apply normal stress BC 
    if type.x[2,1] == :normal_stress
        σyy = SVector{2}([2*bcv.y[2,2]-σyy[2] σyy[2]])
        BC_sym = 1 / 2 
    end
    if type.y[2,end] == :normal_stress
        σyy = SVector{2}([σyy[1] 2*bcv.y[2,end-1]-σyy[1] ])
        BC_sym = 1 / 2 
    end

    # Residual
    fy =  (σyy[2] - σyy[1]) * invΔy
    fy += (τxy[2] - τxy[1]) * invΔx
    fy += ρg
    fy *= -Δ.x * Δ.y
    fy *= BC_sym
    
    return fy
end



# function SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP, Pt0, Pf0, Φ0, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
#     invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

#     # BC
#     Vx   = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
#     Vy   = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)
#     ρ0fg = SMatrix{1,2}( materials.ρf[phases.c] .* materials.g[2])
#     Pt   = SetBCPt1(Pt_loc, type.pt, bcv.pt, Δ, ρ0fg)
#     Pf   = SetBCPf1(Pf_loc, type.pf, bcv.pf, Δ, ρ0fg)

#     # Velocity gradient
#     Dxx = ∂x_inn(Vx) * invΔx 
#     Dyy = ∂y(Vy) * invΔy
#     Dxy = ∂y_inn(Vx) * invΔy
#     Dyx = ∂x(Vy) * invΔx

#     # Strain rate
#     ε̇kk = @. Dxx + Dyy
#     ε̇xx = @. Dxx - 1/3*ε̇kk      
#     ε̇yy = @. Dyy - 1/3*ε̇kk      
#     ε̇xy = @. 1/2 * (Dxy + Dyx)

#     # Average vertex to centroid
#     ε̇̄xy  = av(ε̇xy)
#     # Average centroid to vertex
#     ε̇̄xx  = av(ε̇xx)
#     ε̇̄yy  = av(ε̇yy)
#     P̄t   = av( Pt)
#     P̄f   = av( Pf)
#     τ̄0xx = av(τ0.xx)
#     τ̄0yy = av(τ0.yy)
#     τ̄0xy = av(τ0.xy)
    
#     # Effective strain rate
#     Gc   = SVector{2}( materials.G[phases.c])
#     Gv   = SVector{2}( materials.G[phases.v])
#     tmpc = (2*Gc.*Δ.t)
#     tmpv = (2*Gv.*Δ.t)
#     ϵ̇xx  = @. ε̇xx[2,:] + τ0.xx[2,:] / tmpc
#     ϵ̇yy  = @. ε̇yy[2,:] + τ0.yy[2,:] / tmpc
#     ϵ̇̄xy  = @. ε̇̄xy[:]   + τ̄0xy[:]    / tmpc
#     ϵ̇̄xx  = @. ε̇̄xx[:]   + τ̄0xx[:]    / tmpv
#     ϵ̇̄yy  = @. ε̇̄yy[:]   + τ̄0yy[:]    / tmpv
#     ϵ̇xy  = @. ε̇xy[:,2] + τ0.xy[:,2] / tmpv

#     # Corrected pressure
#     comp = materials.compressible
#     Ptc  = SVector{2}( @. Pt[2,:]  + comp * ΔP.t[:] )
#     Ptc0 = SVector{2}( @. Pt0[2,:] )
#     Pfc  = SVector{2}( @. Pf[2,:]  + comp * ΔP.f[:] )
#     Pfc0 = SVector{2}( @. Pf0[2,:] )

#     # Porosity
#     ηΦ      = SVector{2}( materials.ξ0[phases.c])
#     KΦ      = SVector{2}( materials.KΦ[phases.c])
#     m       = SVector{2}( materials.m[phases.c])

#     # THIS IF STATEMENT DOES NOT COMPILE WITH ENZYME
#     # if materials.linearizeΦ == true
#     #     Φ         = @. Φ0 
#     # else 
#         Φ         = SVector{2}( Porosity(Φ0[ii], Ptc[ii], Pfc[ii], Ptc0[ii], Pfc0[ii], KΦ[ii], ηΦ[ii], m[ii], 0., 0., Δ.t)[1] for ii in eachindex(Φ0))
#     # end

#     # Density
#     ρs   = SVector{2}( materials.ρs[phases.c])
#     ρf   = SVector{2}( materials.ρf[phases.c])
#     ρt   = @. (1-Φ) * ρs + Φ * ρf
#     ρg   = materials.g[2] * 0.5*(ρt[1] + ρt[2])

#     # Stress
#     τyy = SVector{2}(
#         (𝐷.c[i][2,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][2,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][2,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][2,4] + (1 - 𝐷.c[i][4,4])) * Pt[2,i] + 𝐷.c[i][2,5] * Pf[2,i]
#         for i in 1:2
#     )
#     τxy = SVector{2}(
#         𝐷.v[i][3,1] * ϵ̇̄xx[i] + 𝐷.v[i][3,2] * ϵ̇̄yy[i] + 𝐷.v[i][3,3] * ϵ̇xy[i] + 𝐷.v[i][3,4] * P̄t[i] + 𝐷.v[i][3,5] * P̄f[i]
#         for i in 1:2
#     )

#     # Residual
#     fy  = ( τyy[2]  -  τyy[1] ) * invΔy
#     fy += ( τxy[2]  -  τxy[1] ) * invΔx
#     fy -= ( Ptc[2]  -  Ptc[1])  * invΔy
#     fy += ρg 
#     fy *= -1 * Δ.x * Δ.y

#     return fy
# end

function Continuity(Vx, Vy, Pt_loc, Pf_loc, old, rheo, materials, type, bcv, Δ)
    Pt0, Pf0, Φ0, ρs0, ρf0 = old
    Ks, KΦ, Kf, ξ0, m, ρsi, ρfi = rheo
    invΔx   = inv(Δ.x)
    invΔy   = inv(Δ.y)
    Δt      = Δ.t

    # Density - currently using reference density fluid density
    ρ0f = ρfi
    ρfg = SVector{2}(
        materials.g[2] * 0.5 * (ρ0f[2,1] + ρ0f[2,2]),
        materials.g[2] * 0.5 * (ρ0f[2,2] + ρ0f[2,3]),
    )   
    Pf   = SetBCPf1(Pf_loc, type.pf, bcv.pf, Δ, ρfg)
    Pt   = SetBCPf1(Pt_loc, type.pt, bcv.pt, Δ, ρfg)

    dPtdt = @. (Pt - Pt0) / Δt
    dPfdt = @. (Pf - Pf0) / Δt
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!
    Φ, dΦdt = if materials.linearizeΦ ||  materials.single_phase
        Φ       = Φ0
        dΦdt    = zeros(Φ)
        Φ, dΦdt 
    else
        Φ       = SMatrix{3, 3}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ξ0[ii], m[ii], 0., 0., Δt)[1] for ii in eachindex(Φ0) )
        dΦdt    = SMatrix{3, 3}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ξ0[ii], m[ii], 0., 0., Δt)[2] for ii in eachindex(Φ0) )
        Φ, dΦdt 
    end

    dPsdt   = @. dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = @. 1/Ks * ( dPsdt )
    # dlnρsdt = SMatrix{3, 3}( @. (1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks) ) # approximation in Yarushina's paper

    # Single phase
    if materials.single_phase
        dPsdt   = dPtdt 
        dlnρsdt = dPsdt / Ks
    end

    divVs   = (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy 
    
    # if materials.oneway
    #     fp      = divVs
    # else
    fp = if materials.conservative === false
        fp = if type.pt[2,2] == :p_eff
            Pt[2,2] - Pf[2,2]
        else
            dlnρsdt[2,2] - dΦdt[2,2] / (1 - Φ[2,2]) + divVs
        end
    else
        # Solid mass / immobile solid mass: ∂ρim∂t  + ∇⋅(q) with q = ρim⋅Vs
        ρim0   = @. (1-Φ0) * ρs0
        # lnρs   = SMatrix{3, 3}( @. log(ρs0) + Δt*dlnρsdt)
        # ρs     = SMatrix{3, 3}( @. exp(lnρs) )
        ρs     = @. ρs0 + ρs0 * Δt*dlnρsdt
        ρim    = @. (1-Φ ) * ρs
        ∂ρim∂t = (ρim[2,2] - ρim0[2,2]) / Δt
        # Brucite paper, Fowler (1985)
        qx = SVector{2}(
            ((ρim[1,2] + ρim[2,2]) * 0.5) * Vx[1,2],
            ((ρim[2,2] + ρim[3,2]) * 0.5) * Vx[2,2],
        )
        
        qy = SVector{2}(
            ((ρim[2,1] + ρim[2,2]) * 0.5) * Vy[2,1],
            ((ρim[2,2] + ρim[2,3]) * 0.5) * Vy[2,2],
        )
        ∂ρim∂t  +  (qx[2] - qx[1]) * invΔx + (qy[2] - qy[1]) * invΔy
    end
    return fp
end

function FluidContinuity(Vx, Vy, Pt_loc, Pf_loc, ΔPf_loc, old, rheo, materials, type, bcv, Δ)
    
    Pt0, Pf0, Φ0, ρs0, ρf0 = old
    Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, kμ, n_CK = rheo
    invΔx   = inv(Δ.x)
    invΔy   = inv(Δ.y)
    Δt      = Δ.t

    # Density - currently explicit in time (= using old fluid density)
    ρ0f  = ρfi
    ρfg  = SVector{2}(materials.g[2] * 0.5 * (ρ0f[2,i] + ρ0f[2,i+1]) for i ∈ 1:2)  
    Pf   = SetBCPf1(Pf_loc, type.pf, bcv.pf, Δ, ρfg)
    Pt   = SetBCPf1(Pt_loc, type.pt, bcv.pt, Δ, ρfg)

    dPtdt   = @. (Pt .- Pt0) / Δt
    dPfdt   = @. (Pf .- Pf0) / Δt
    Φ, dΦdt = if materials.linearizeΦ ||  materials.single_phase
        Φ       = Φ0
        dΦdt    = zeros(Φ0)
        Φ, dΦdt
    else
        Φ       = SMatrix{3, 3}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ξ0[ii], m[ii], 0., 0., Δt)[1] for ii in eachindex(Φ0) )
        dΦdt    = SMatrix{3, 3}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ξ0[ii], m[ii], 0., 0., Δt)[2] for ii in eachindex(Φ0) )
        Φ, dΦdt
    end

    # # if Φ[1]<0 || Φ[2] <0 ||  Φ[3] <0
    # #     @show Φ
    # #     @show Pt
    # #     @show Pf
    # #     @show Pt0
    # #     @show Pf0
    # # end
    
    dPsdt   = @. dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = @. 1/Ks * ( dPsdt )
    dlnρfdt = dPfdt[2,2] / Kf[2,2]

    # Interpolate porosity to velocity nodes
    Φxⁿ = SVector{2}(
        (Φ[1,2]^n_CK[1,2] + Φ[2,2]^n_CK[2,2]) * 0.5,
        (Φ[2,2]^n_CK[2,2] + Φ[3,2]^n_CK[3,2]) * 0.5,
    )
    
    Φyⁿ = SVector{2}(
        (Φ[2,1]^n_CK[2,1] + Φ[2,2]^n_CK[2,2]) * 0.5,
        (Φ[2,2]^n_CK[2,2] + Φ[2,3]^n_CK[2,3]) * 0.5,
    )

    # This allocates? why?
    # Φxⁿ = SVector{2}(0.5 * (Φ[i,2]^n_CK[i,2] + Φ[i+1,2]^n_CK[i+1,2]) for i ∈ 1:2)
    # Φyⁿ = SVector{2}(0.5 * (Φ[2,i]^n_CK[2,i] + Φ[2,i+1]^n_CK[2,i+1]) for i ∈ 1:2)

    # Fluid conductivity
    kμ_xx = SVector{2}(0.5 * (kμ[i+1,2] + kμ[i,2]) for i ∈ 1:2)
    kμ_yy = SVector{2}(0.5 * (kμ[2,i+1] + kμ[2,i]) for i ∈ 1:2)

    # Darcy flux
    qx = SVector{2}( -kμ_xx[i] * Φxⁿ[i] * ( (Pf[i+1,2] - Pf[i,2]) * invΔx          ) for i ∈ 1:2)
    qy = SVector{2}( -kμ_yy[i] * Φyⁿ[i] * (((Pf[2,i+1] - Pf[2,i]) * invΔy) - ρfg[i]) for i ∈ 1:2)

    # Divergence of Darcy flux and solid velocity
    divqD = ( (  qx[2] -   qx[1]) * invΔx + (  qy[2] -   qy[1]) * invΔy)
    divVs = ( (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy) 
    
    fp = if materials.conservative == false
        fp = if materials.oneway
            divqD
        else
            (Φ[2,2]*dlnρfdt + dΦdt[2,2] + Φ[2,2]*divVs + divqD)
        end
    else
        # Total mass: ∂ρt∂t + ∇⋅(q) with q = ρf⋅qD + ρt⋅qD⋅V
        lnρs   = @. log(ρs0) + Δt*dlnρsdt
        ρs     = @. exp(lnρs) 
        lnρf   = @. log(ρf0) + Δt*dlnρsdt
        ρf     = @. exp(lnρf) 
        ρt     = @. (1-Φ ) * ρs  + Φ  * ρf  
        ρt0    = @. (1-Φ0 )* ρs0 + Φ0 * ρf0 
        
        ∂ρt∂t  = (ρt[2,2] - ρt0[2,2]) / Δt
        ρfx    = SVector{2}(0.5 * (ρf[i,2] + ρf[i+1,2]) for i ∈ 1:2)
        ρfy    = SVector{2}(0.5 * (ρf[2,i] + ρf[2,i+1]) for i ∈ 1:2)
        ρtx    = SVector{2}(0.5 * (ρt[i,2] + ρt[i+1,2]) for i ∈ 1:2)
        ρty    = SVector{2}(0.5 * (ρt[2,i] + ρt[2,i+1]) for i ∈ 1:2)
        qρx    = @. ρfx * qx + ρtx * Vx[:,2] # Brucite paper, Fowler (1985)
        qρy    = @. ρfy * qy + ρty * Vy[2,:] # Brucite paper, Fowler (1985)    
        
        if materials.oneway
            ∂ρt∂t  = 0*(ρt[2,2] - ρt0[2,2]) / Δt
            qρx    = @. ρfx * qx # +  0*ρtx * Vx[:,2]    # Brucite paper, Fowler (1985)
            qρy    = @. ρfy * qy # +  0*ρty * Vy[2,:]
        end
        fp = ∂ρt∂t + (qρx[2] - qρx[1]) * invΔx + (qρy[2] - qρy[1]) * invΔy 
    end
    return fp
end

function ResidualMomentum2D_x!(R, V, P, ΔP, old, 𝐷, rheo, materials, number, type, BC, nc, Δ) 
    
    τ0 , P0, ϕ0, ρ0 = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo

    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)

        Pt_loc     = SMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc     = SMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        ΔPt_loc    = SMatrix{2,1}(     ΔP.t[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        ΔPf_loc    = SMatrix{2,1}(     ΔP.t[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )
        Gc_loc     = SMatrix{2,1}(     G.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
        Gv_loc     = SMatrix{1,2}(     G.v[ii, jj] for ii in i-0:i-0, jj in j-1:j-0)
        Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)
        G_loc = (c=Gc_loc, v=Gv_loc)

        if type.Vx[i,j] == :in
            R.x[i,j]   = SMomentum_x_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPt_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, ΔP, old, 𝐷, rheo, materials, num, pattern, type, BC, nc, Δ) 

    τ0 , P0, ϕ0, ρ0 = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo

    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)

        Pt_loc     = SMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc     = SMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        ΔPt_loc    = SMatrix{2,1}(     ΔP.t[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )
      
        Gc_loc     = SMatrix{2,1}(      G.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
        Gv_loc     = SMatrix{1,2}(      G.v[ii, jj] for ii in i-0:i-0, jj in j-1:j-0)
        Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        G_loc      = (c=Gc_loc, v=Gv_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

        if type.Vx[i,j] == :in

            ∂R∂Vx = ad_gradient(Vx_loc -> SMomentum_x_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPt_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ), Vx_loc)
            ∂R∂Vy = ad_gradient(Vy_loc -> SMomentum_x_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPt_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ), Vy_loc)
            ∂R∂Pt = ad_gradient(Pt_loc -> SMomentum_x_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPt_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ), Pt_loc)
            ∂R∂Pf = ad_gradient(Pf_loc -> SMomentum_x_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPt_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ), Pf_loc)
            
            # Vx --- Vx
            Local = SMatrix{3, 3}(num.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1).* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][1][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vx --- Vy
            Local = SMatrix{4, 4}(num.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1) .* pattern[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][2][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vx --- Pt
            Local = SMatrix{2, 3}(num.Pt[ii, jj] for ii in i-1:i, jj in j-2:j) .* pattern[1][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][3][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
            # Vx --- Pf
            Local = SMatrix{2, 3}(num.Pf[ii, jj] for ii in i-1:i, jj in j-2:j) .* pattern[1][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][4][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end 

        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, ΔP, old, 𝐷, rheo, materials, number, type, BC, nc, Δ)                 
    
    τ0 , P0, Φ0, ρ0 = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo
    
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = SMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = SMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        # phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        # phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-1:i-0, jj in j-0:j-0) 
        Pt_loc     = SMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc     = SMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        ΔPt_loc    = SMatrix{1,2}(     ΔP.t[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ΔPf_loc    = SMatrix{1,2}(     ΔP.f[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        Pt0_loc    = SMatrix{3,2}(     P0.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf0_loc    = SMatrix{3,2}(     P0.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Φ0_loc     = SMatrix{1,2}(     Φ0.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        τxx0       = SMatrix{3,2}(    τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τyy0       = SMatrix{3,2}(    τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τxy0       = SMatrix{2,3}(    τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
        Dc         = SMatrix{1,2}(      𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
        Dv         = SMatrix{2,1}(      𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
        bcv_pt     = SMatrix{3,2}(    BC.Pt[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        bcv_pf     = SMatrix{3,2}(    BC.Pf[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        type_pt    = SMatrix{3,2}(  type.Pt[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        type_pf    = SMatrix{3,2}(  type.Pf[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        
        Gc_loc    = SMatrix{1,2}(     G.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
        Gv_loc    = SMatrix{2,1}(     G.v[ii, jj] for ii in i-1:i-0, jj in j-0:j-0)
        ξ0_loc    = SMatrix{1,2}(     ξ0.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        KΦ_loc    = SMatrix{1,2}(     KΦ.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        m_loc     = SMatrix{1,2}(      m.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ρs_loc    = SMatrix{1,2}(    ρsi.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ρf_loc    = SMatrix{1,2}(    ρfi.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )

        G_loc = (c=Gc_loc, v=Gv_loc)
        rheo_loc = (ξ0 = ξ0_loc, KΦ = KΦ_loc, m = m_loc, ρs = ρs_loc, ρf = ρf_loc)

        bcv_loc    = (x=bcx_loc,   y=bcy_loc,   pt=bcv_pt,   pf=bcv_pf)
        type_loc   = (x=typex_loc, y=typey_loc, pt=type_pt,  pf=type_pf)
        # ph_loc     = (c=phc_loc, v=phv_loc)
        ΔP_loc     = (t=ΔPt_loc, f=ΔPf_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

        if type.Vy[i,j] == :in
            R.y[i,j]   = SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP_loc, Pt0_loc, Pf0_loc, Φ0_loc, τ0_loc, G_loc, rheo_loc, D, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, ΔP, old, 𝐷, rheo, materials, num, pattern, type, BC, nc, Δ) 
    
    τ0 , P0, Φ0, ρ0 = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo

    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = SMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = SMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt_loc     = SMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc     = SMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        ΔPt_loc    = @inline SMatrix{1,2}(@inbounds     ΔP.t[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ΔPf_loc    = SMatrix{1,2}(     ΔP.f[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        Pt0_loc    = SMatrix{3,2}(     P0.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf0_loc    = SMatrix{3,2}(     P0.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Φ0_loc     = SMatrix{1,2}(     Φ0.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        τxx0       = @inline SMatrix{3,2}(@inbounds     τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τyy0       = @inline SMatrix{3,2}(@inbounds     τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τxy0       = @inline SMatrix{2,3}(@inbounds     τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
        Dc         = @inline SMatrix{1,2}(@inbounds       𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
        Dv         = @inline SMatrix{2,1}(@inbounds       𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
        bcv_pt     = SMatrix{3,2}(    BC.Pt[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        bcv_pf     = SMatrix{3,2}(    BC.Pf[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        type_pt    = SMatrix{3,2}(  type.Pt[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        type_pf    = SMatrix{3,2}(  type.Pf[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        
        Gc_loc    = SMatrix{1,2}(     G.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
        Gv_loc    = SMatrix{2,1}(     G.v[ii, jj] for ii in i-1:i-0, jj in j-0:j-0)
        ξ0_loc    = SMatrix{1,2}(     ξ0.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        KΦ_loc    = SMatrix{1,2}(     KΦ.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        m_loc     = SMatrix{1,2}(      m.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ρs_loc    = SMatrix{1,2}(    ρsi.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ρf_loc    = SMatrix{1,2}(    ρfi.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )

        G_loc      = (c=Gc_loc, v=Gv_loc)
        rheo_loc   = (ξ0 = ξ0_loc, KΦ = KΦ_loc, m = m_loc, ρs = ρs_loc, ρf = ρf_loc)

        bcv_loc    = (x=bcx_loc,   y=bcy_loc,   pt=bcv_pt,   pf=bcv_pf)
        type_loc   = (x=typex_loc, y=typey_loc, pt=type_pt,  pf=type_pf)
        ΔP_loc     = (t=ΔPt_loc, f=ΔPf_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

        if type.Vy[i,j] == :in

            ∂R∂Vx = ad_gradient(Vx_loc -> SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP_loc, Pt0_loc, Pf0_loc, Φ0_loc, τ0_loc, G_loc, rheo_loc, D, materials, type_loc, bcv_loc, Δ), Vx_loc)
            ∂R∂Vy = ad_gradient(Vy_loc -> SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP_loc, Pt0_loc, Pf0_loc, Φ0_loc, τ0_loc, G_loc, rheo_loc, D, materials, type_loc, bcv_loc, Δ), Vy_loc)
            ∂R∂Pt = ad_gradient(Pt_loc -> SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP_loc, Pt0_loc, Pf0_loc, Φ0_loc, τ0_loc, G_loc, rheo_loc, D, materials, type_loc, bcv_loc, Δ), Pt_loc)
            ∂R∂Pf = ad_gradient(Pf_loc -> SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP_loc, Pt0_loc, Pf0_loc, Φ0_loc, τ0_loc, G_loc, rheo_loc, D, materials, type_loc, bcv_loc, Δ), Pf_loc)

            Local = SMatrix{4, 4}(num.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2).* pattern[2][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][1][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vy --- Vy
            Local = SMatrix{3, 3}(num.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1).* pattern[2][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][2][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vy --- Pt
            # Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
            Local = SMatrix{3, 2}(num.Pt[ii, jj] for ii in i-2:i, jj in j-1:j).* pattern[2][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][3][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
            # Vy --- Pf
            # Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
            Local = SMatrix{3, 2}(num.Pf[ii, jj] for ii in i-2:i, jj in j-1:j).* pattern[2][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][4][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end       
        end
    end
    return nothing
end

function ResidualContinuity2D!(R, V, P, ΔP, old, rheo, materials, number, type, BC, nc, Δ) 
    
    _, P0, ϕ0, ρ0 = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo

    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf         = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt         = SMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)

        typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        typept_loc = SMatrix{3,3}(  type.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typepf_loc = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        bcpt_loc   = SMatrix{3,3}(    BC.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcpf_loc   = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc    = (x=bcx_loc,   y=bcy_loc,   pt=bcpt_loc,   pf=bcpf_loc)
        type_loc   = (x=typex_loc, y=typey_loc, pt=typept_loc, pf=typepf_loc)

        Ks_loc     = SMatrix{3,3}(     Ks.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        KΦ_loc     = SMatrix{3,3}(     KΦ.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Kf_loc     = SMatrix{3,3}(     Kf.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ξ_loc      = SMatrix{3,3}(     ξ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        m_loc      = SMatrix{3,3}(      m.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρsi_loc    = SMatrix{3,3}(    ρsi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρfi_loc    = SMatrix{3,3}(    ρfi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        old_loc    = (Pt = Pt0, Pf=Pf0, ϕ=Φ0, ρs=ρs0, ρf=ρf0 )
        rheo_loc   = (Ks = Ks_loc, KΦ = KΦ_loc, Kf = Kf_loc, ξ = ξ_loc, m = m_loc, ρfi = ρfi_loc, ρsi = ρsi_loc)
        
        R.pt[i,j]  = Continuity(Vx_loc, Vy_loc, Pt, Pf, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ)

    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, ΔP, old, rheo, materials, num, pattern, type, BC, nc, Δ) 
         
    _, P0, ϕ0, ρ0   = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo

    shift    = (x=1, y=1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt_loc     = SMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)

        typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        typept_loc = SMatrix{3,3}(  type.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typepf_loc = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        bcpt_loc   = SMatrix{3,3}(    BC.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcpf_loc   = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc    = (x=bcx_loc,   y=bcy_loc,   pt=bcpt_loc,   pf=bcpf_loc)
        type_loc   = (x=typex_loc, y=typey_loc, pt=typept_loc, pf=typepf_loc)

        Ks_loc     = SMatrix{3,3}(     Ks.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        KΦ_loc     = SMatrix{3,3}(     KΦ.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Kf_loc     = SMatrix{3,3}(     Kf.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ξ_loc      = SMatrix{3,3}(     ξ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        m_loc      = SMatrix{3,3}(      m.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρsi_loc    = SMatrix{3,3}(    ρsi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρfi_loc    = SMatrix{3,3}(    ρfi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        old_loc    = (Pt = Pt0, Pf = Pf0, ϕ = Φ0, ρs = ρs0, ρf = ρf0 )
        rheo_loc   = (Ks = Ks_loc, KΦ = KΦ_loc, Kf = Kf_loc, ξ = ξ_loc, m = m_loc, ρfi = ρfi_loc, ρsi = ρsi_loc)

        ∂R∂Vx = ad_gradient(Vx_loc -> Continuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Vx_loc)
        ∂R∂Vy = ad_gradient(Vy_loc -> Continuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Vy_loc)
        ∂R∂Pt = ad_gradient(Pt_loc -> Continuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Pt_loc)
        ∂R∂Pf = ad_gradient(Pf_loc -> Continuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Pf_loc)

        # # Pt --- Vx
        # Local = SMatrix{2, 3}(num.Vx[ii, jj] for ii in i:i+1, jj in j:j+2).* pattern[3][1]
        # for jj in axes(Local,2), ii in axes(Local,1)
        #     if Local[ii,jj]>0 && num.Pt[i,j]>0
        #         K[3][1][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
        #     end
        # end
        # # Pt --- Vy
        # Local = SMatrix{3, 2}(num.Vy[ii, jj] for ii in i:i+2, jj in j:j+1).* pattern[3][2]
        # for jj in axes(Local,2), ii in axes(Local,1)
        #     if Local[ii,jj]>0 && num.Pt[i,j]>0
        #         K[3][2][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
        #     end
        # end
        # # Pt --- Pt
        # Local = SMatrix{3, 3}(num.Pt[ii, jj] for ii in i-1:i+1, jj in j-1:j+1).* pattern[3][3]
        # for jj in axes(Local,2), ii in axes(Local,1)
        #     if (Local[ii,jj]>0) && num.Pt[i,j]>0
        #         K[3][3][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
        #     end
        # end
        # # Pt --- Pf
        # Local = SMatrix{3, 3}(num.Pf[ii, jj] for ii in i-1:i+1, jj in j-1:j+1).* pattern[3][4]
        # for jj in axes(Local,2), ii in axes(Local,1)
        #     if (Local[ii,jj]>0) && num.Pt[i,j]>0
        #         K[3][4][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
        #     end
        # end
    end
    return nothing
end

function ResidualFluidContinuity2D!(R, V, P, ΔP, old, rheo, materials, number, type, BC, nc, Δ) 
                
    _, P0, ϕ0, ρ0   = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo
    shift    = (x=1, y=1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            Pt_loc     = SMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ΔPf_loc    = SMatrix{3,3}(     ΔP.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
            Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
            kμ_loc     = SMatrix{3,3}(  k_ηf0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
            typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
            typept_loc = SMatrix{3,3}(  type.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typepf_loc = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
            bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
            bcpt_loc   = SMatrix{3,3}(    BC.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcpf_loc   = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcv_loc    = (x=bcx_loc,   y=bcy_loc,   pt=bcpt_loc,   pf=bcpf_loc)
            type_loc   = (x=typex_loc, y=typey_loc, pt=typept_loc, pf=typepf_loc)
            
            Ks_loc     = SMatrix{3,3}(     Ks.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            KΦ_loc     = SMatrix{3,3}(     KΦ.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Kf_loc     = SMatrix{3,3}(     Kf.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ξ_loc      = SMatrix{3,3}(     ξ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            m_loc      = SMatrix{3,3}(      m.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ρsi_loc    = SMatrix{3,3}(    ρsi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ρfi_loc    = SMatrix{3,3}(    ρfi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            n_CK_loc   = SMatrix{3,3}(   n_CK.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            
            old_loc    = (Pt = Pt0, Pf = Pf0, ϕ = Φ0, ρs = ρs0, ρf = ρf0 )
            rheo_loc   = (Ks = Ks_loc, KΦ = KΦ_loc, Kf = Kf_loc, ξ = ξ_loc, m = m_loc, ρfi = ρfi_loc, ρsi = ρsi_loc, kμ = kμ_loc, n_CK = n_CK_loc)

            R.pf[i,j]  = FluidContinuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ)

        end
    end
    return nothing
end

function AssembleFluidContinuity2D!(K, V, P, ΔP, old, rheo, materials, num, pattern, type, BC, nc, Δ) 
              
    _, P0, ϕ0, ρ0 = old
    G, Ks, KΦ, Kf, ξ0, m, ρsi, ρfi, k_ηf0, n_CK = rheo
    shift    = (x=1, y=1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = SMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ΔPf_loc    = SMatrix{3,3}(     ΔP.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) 
        ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)       
        Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        kμ_loc     = SMatrix{3,3}(  k_ηf0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        typept_loc = SMatrix{3,3}(  type.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typepf_loc = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        bcpt_loc   = SMatrix{3,3}(    BC.Pt[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcpf_loc   = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc    = (x=bcx_loc,   y=bcy_loc,   pt=bcpt_loc,   pf=bcpf_loc)
        type_loc   = (x=typex_loc, y=typey_loc, pt=typept_loc, pf=typepf_loc)
        
        Ks_loc     = SMatrix{3,3}(     Ks.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        KΦ_loc     = SMatrix{3,3}(     KΦ.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Kf_loc     = SMatrix{3,3}(     Kf.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ξ_loc      = SMatrix{3,3}(     ξ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        m_loc      = SMatrix{3,3}(      m.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρsi_loc    = SMatrix{3,3}(    ρsi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρfi_loc    = SMatrix{3,3}(    ρfi.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        n_CK_loc   = SMatrix{3,3}(   n_CK.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        
        old_loc    = (Pt = Pt0, Pf=Pf0, ϕ=Φ0, ρs=ρs0, ρf=ρf0 )
        rheo_loc   = (Ks = Ks_loc, KΦ = KΦ_loc, Kf = Kf_loc, ξ = ξ_loc, m = m_loc, ρfi = ρfi_loc, ρsi = ρsi_loc, kμ = kμ_loc, n_CK = n_CK_loc)

        ∂R∂Vx = ad_gradient(Vx_loc -> FluidContinuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Vx_loc)
        ∂R∂Vy = ad_gradient(Vy_loc -> FluidContinuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Vy_loc)
        ∂R∂Pt = ad_gradient(Pt_loc -> FluidContinuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Pt_loc)
        ∂R∂Pf = ad_gradient(Pf_loc -> FluidContinuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPf_loc, old_loc, rheo_loc, materials, type_loc, bcv_loc, Δ), Pf_loc)
             
        # Pf --- Vx
        Local = SMatrix{2, 3}(num.Vx[ii, jj] for ii in i:i+1, jj in j:j+2).* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pf --- Vy
        Local = SMatrix{3, 2}(num.Vy[ii, jj] for ii in i:i+2, jj in j:j+1).* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
        # Pf --- Pt
        Local = SMatrix{3, 3}(num.Pt[ii, jj] for ii in i-1:i+1, jj in j-1:j+1).* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
            end
        end
        # Pf --- Pf
        Local = SMatrix{3, 3}(num.Pf[ii, jj] for ii in i-1:i+1, jj in j-1:j+1).* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
            end
        end
           
    end
    return nothing
end


function UpdatePorosity2D!(R, V, P, P0, Φ, Φ0, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            KΦ        = materials.KΦ[phases.c[i,j]]
            ηΦ        = materials.ξ0[phases.c[i,j]]
            dPtdt     = (P.t[i,j] - P0.t[i,j]) / Δ.t
            dPfdt     = (P.f[i,j] - P0.f[i,j]) / Δ.t
            dΦdt      = (dPfdt - dPtdt)/KΦ + (P.f[i,j] - P.t[i,j])/ηΦ
            Φ.c[i,j]  = Φ0.c[i,j] + dΦdt*Δ.t
        end
    end
    return nothing
end

function ResidualPorosity2D!(R, V, P, P0, Φ, Φ0, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            KΦ        = materials.KΦ[phases.c[i,j]]
            ηΦ        = materials.ξ0[phases.c[i,j]]
            dPtdt     = (P.t[i,j] - P0.t[i,j]) / Δ.t
            dPfdt     = (P.f[i,j] - P0.f[i,j]) / Δ.t
            dΦdt      = (dPfdt - dPtdt)/KΦ + (P.f[i,j] - P.t[i,j])/ηΦ
            R.Φ[i,j]  = Φ.c[i,j] - (Φ0.c[i,j] + dΦdt*Δ.t)
        end
    end
    return nothing
end

function Numbering!(N, type, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, type.Vx[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vx[:,2], dims=1)) > 0

    shift  = (periodic_west) ? 1 : 0 
    # Loop through inner nodes of the mesh
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] != :periodic && i==nc.x+3-1) || type.Vx[i,j] == :constant 
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vx[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1,:] .= N.Vx[end-2,:]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        # South
        N.Vx[:,1] .= N.Vx[:,end-3]
        N.Vx[:,2] .= N.Vx[:,end-2]
        # North
        N.Vx[:,end]   .= N.Vx[:,4]
        N.Vx[:,end-1] .= N.Vx[:,3]
    end
    noisy ? printxy(N.Vx) : nothing

    neq = maximum(N.Vx)

    ############ Numbering Vy ############
    ndof  = 0
    periodic_west  = sum(any(i->i==:periodic, type.Vy[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vy[:,2], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    # Loop through inner nodes of the mesh
    for j=2:nc.y+3-1, i=3:nc.x+4-2

        # Marche avec JAO
        # if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] == :periodic && j==nc.y+3-1)
        
        # Marche avec Rozhko
        # if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] != :periodic && j==nc.y+3-1) || type.Vy[i,j] == :constant 

        # Marche avec ;es deux
        if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] == :periodic && j==nc.y+3-1) || type.Vy[i,j] == :constant 
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vy[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vy[:,1]     .= N.Vy[:,end-2]
        N.Vy[:,end-1] .= N.Vy[:,2]
        N.Vy[:,end]   .= N.Vy[:,3]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        # West
        N.Vy[1,:] .= N.Vy[end-3,:]
        N.Vy[2,:] .= N.Vy[end-2,:]
        # East
        N.Vy[end,:]   .= N.Vy[4,:]
        N.Vy[end-1,:] .= N.Vy[3,:]
    end
    noisy ? printxy(N.Vy) : nothing

    neq = maximum(N.Vy)

    ############ Numbering Pt ############
    # neq_Pt                     = nc.x * nc.y
    # N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)
    ii = 0
    for j=1:nc.y, i=1:nc.x
        if type.Pt[i+1,j+1] != :constant
            ii += 1
            N.Pt[i+1,j+1] = ii
        end
    end

    if periodic_west
        N.Pt[1,:]   .= N.Pt[end-1,:]
        N.Pt[end,:] .= N.Pt[2,:]
    end

    if periodic_south
        N.Pt[:,1]   .= N.Pt[:,end-1]
        N.Pt[:,end] .= N.Pt[:,2]
    end
    noisy ? printxy(N.Pt) : nothing

    neq = maximum(N.Pt)

    ############ Numbering Pf ############

    # neq_Pf                    = nc.x * nc.y
    # N.Pf[2:end-1,2:end-1] .= reshape(1:neq_Pf, nc.x, nc.y)
    ii = 0
    for j=1:nc.y, i=1:nc.x
        if type.Pf[i+1,j+1] != :constant
            ii += 1
            N.Pf[i+1,j+1] = ii
        end
    end

    # Make periodic in x
    for j in axes(type.Pf,2)
        if type.Pf[1,j] === :periodic
            N.Pf[1,j] = N.Pf[end-1,j]
        end
        if type.Pf[end,j] === :periodic
            N.Pf[end,j] = N.Pf[2,j]
        end
    end

    # Make periodic in y
    for i in axes(type.Pf,1)
        if type.Pf[i,1] === :periodic
            N.Pf[i,1] = N.Pf[i,end-1]
        end
        if type.Pf[i,end] === :periodic
            N.Pf[i,end] = N.Pf[i,2]
        end
    end

end

function SetRHS!(r, R, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            r[ind] = R.x[i,j]
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            r[ind] = R.y[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in || type.Pt[i,j] == :p_eff
            ind = number.Pt[i,j] + nVx + nVy
            r[ind] = R.pt[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            r[ind] = R.pf[i,j]
        end
    end
end

function UpdateSolution!(V, P, dx, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            V.x[i,j] += dx[ind] 
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            V.y[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in || type.Pt[i,j] == :p_eff
            ind = number.Pt[i,j] + nVx + nVy
            P.t[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            P.f[i,j] += dx[ind]
        end
    end
end

@views function SparsityPattern!(K, num, pattern, nc) 
    ############ Fields Vx ############
    shift  = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vx --- Vx
        Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][1][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Vy
        Local = num.Vy[i-1:i+2,j-2:j+1] .* pattern[1][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][2][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pt
        Local = num.Pt[i-1:i,j-2:j] .* pattern[1][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][3][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pf
        Local = num.Pf[i-1:i,j-2:j] .* pattern[1][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][4][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Vy ############
    shift  = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vy --- Vx
        Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][1][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Vy
        Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][2][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pt
        Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][3][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pf
        Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][4][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pt ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pt
        Local = num.Pt[i,j] .* pattern[3][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][3][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pf
        Local = num.Pf[i,j] .* pattern[3][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][4][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pf ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pf --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pt
        Local = num.Pt[i,j] .* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ End ############
end

function LineSearch!(rvec, α, dx, R, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, old, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    τ0, P0, Φ0, ρ0 = old
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pi.t .= P.t
    Pi.f .= P.f

    for i in eachindex(α)
        V.x .= Vi.x 
        V.y .= Vi.y
        P.t .= Pi.t
        P.f .= Pi.f
        UpdateSolution!(V, P, α[i].*dx, number, type, nc)
        TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
        ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualContinuity2D!(R, V, P, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 
        ResidualFluidContinuity2D!(R, V, P, ΔP, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 
        rvec[i] = @views norm(R.x[inx_Vx,iny_Vx])/length(R.x[inx_Vx,iny_Vx]) + norm(R.y[inx_Vy,iny_Vy])/length(R.y[inx_Vy,iny_Vy]) + norm(R.pt[inx_c,iny_c])/length(R.pt[inx_c,iny_c]) + norm(R.pf[inx_c,iny_c])/length(R.pf[inx_c,iny_c])  
    end
    imin = argmin(rvec)
    V.x .= Vi.x 
    V.y .= Vi.y
    P.t .= Pi.t
    P.f .= Pi.f
    return imin
end

function GlobalResidual!(α, dx, R, V, P, ε̇, τ, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    UpdateSolution!(V, P, α.*dx, number, type, nc)
    TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
    ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
    ResidualContinuity2D!(R, V, P, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 
    ResidualFluidContinuity2D!(R, V, P, ΔP, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 
end

@inline fnorm(R, inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c) = @views (norm(R.x[inx_Vx,iny_Vx])/sqrt(length(R.x[inx_Vx,iny_Vx])))^2 + (norm(R.y[inx_Vy,iny_Vy])/sqrt(length(R.y[inx_Vy,iny_Vy])))^2 + 1*(norm(R.pt[inx_c,iny_c])/length(R.pt[inx_c,iny_c]))^2 + 1*(norm(R.pf[inx_c,iny_c])/length(R.pf[inx_c,iny_c]))^2

function BackTrackingLineSearch!(rvec, α, dx, R0, R, V, P, ε̇, τ, Vi, Pi, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ; α_init=1.0, β=0.5, c=1e-4)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pi.t .= P.t
    Pi.f .= P.f

    α = α_init
    GlobalResidual!(0.0, dx, R0, V, P, ε̇, τ, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    f0_norm_sq = fnorm(R, inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c) 

    k = 0
    max_iters = 5

    for iter in 1:max_iters
    # # while f_norm_sq >= (1 - c * α * slope) * f0_norm_sq

        k    += 1

        V.x .= Vi.x 
        V.y .= Vi.y
        P.t .= Pi.t
        P.f .= Pi.f

        GlobalResidual!(  α, dx, R, V, P, ε̇, τ, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
        
        f_norm_sq = fnorm(R, inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c) 

        slope = -2 * ( sum(R0.x[inx_Vx,iny_Vx].*R.x[inx_Vx,iny_Vx]) + sum(R0.y[inx_Vy,iny_Vy].*R.y[inx_Vy,iny_Vy]) + 1*sum(R0.pt[inx_c,iny_c].*R.pt[inx_c,iny_c]) + 1*sum(R0.pf[inx_c,iny_c].*R.pf[inx_c,iny_c]) )
    
         if f_norm_sq <= (1 - c * α * slope) * f0_norm_sq
            break        
        end

        # @show α, f_norm_sq, f0_norm_sq, (1 - c * α * slope) * f0_norm_sq


        @show α, f_norm_sq, f0_norm_sq, f_norm_sq/f0_norm_sq

        α *= β

    end

    V.x .= Vi.x 
    V.y .= Vi.y
    P.t .= Pi.t
    P.f .= Pi.f

    @info k, α

    return α
end

    
# function backtracking_line_search(f, x, δx; α_init=1.0, β=0.5, c=1e-4)
#     α = α_init
#     fx = f(x)
#     f_norm_sq = norm(fx)^2
#     slope = -2 * real(dot(fx, f(x + α * δx)))  # approximation to directional derivative

    # while norm(f(x + α * δx))^2 > f_norm_sq - c * α * slope
    #     α *= β
    # end

#     return α
# end

# function Continuity(Vx, Vy, Pt, Pf, old, phase, materials, type, bcv, Δ)
#     Pt0, Pf0, Φ0, ρs0, ρf0 = old
#     invΔx   = 1 / Δ.x
#     invΔy   = 1 / Δ.y
#     Δt      = Δ.t
#     ηΦ      = materials.ξ0[phase]
#     m       = materials.m[phase]
#     KΦ      = materials.KΦ[phase]

#     Φ       = SMatrix{3, 3, Float64}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ηΦ[ii], m[ii], 0., 0., Δt)[1] for ii in eachindex(Φ0) )
#     dΦdt    = SMatrix{3, 3, Float64}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ηΦ[ii], m[ii], 0., 0., Δt)[2] for ii in eachindex(Φ0) ) 

#     divVs   = (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy 
    
#     if type.pt[2,2] == :p_eff
#         fp      = Pt[2,2] - Pf[2,2]
#     else
#         fp      = - dΦdt[2,2]/(1-Φ[2,2]) + divVs
#     end

#     # @show Φ[2,2]
#     # @show (Pf .- Pt) ./ ηΦ
#     # @show dΦdt[2,2]
#     # @show divVs
#     # @show fp
#     # error()

#     # fp    *= max(invΔx, invΔy)
#     return fp
# end

# function FluidContinuity(Vx, Vy, Pt_loc, Pf_loc, ΔPf_loc, old, phase, materials, kμ, type, bcv, Δ)
    
#     Pt0, Pf0, Φ0, ρs0, ρf0 = old
#     invΔx   = 1 / Δ.x
#     invΔy   = 1 / Δ.y
#     Δt      = Δ.t
#     ηΦ      = materials.ξ0[phase]
#     m       = materials.m[phase]
#     KΦ      = materials.KΦ[phase] 
#     n       = materials.n_CK[phase] # Carman-Kozeny

#     # Density - currently explicit in time (= using old fluid density)
#     ρ0f  = SMatrix{3,3, Float64}( materials.ρf[phase])
#     ρfg  = SVector{2, Float64}(@. materials.g[2] * 0.5*(ρ0f[2,1:end-1] + ρ0f[2,2:end]) )
#     Pf   = SetBCPf1(Pf_loc, type.pf, bcv.pf, Δ, ρfg)
#     Pt   = SetBCPf1(Pt_loc, type.pt, bcv.pt, Δ, ρfg)

#     Φ       = SMatrix{3, 3, Float64}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ηΦ[ii], m[ii], 0., 0., Δt)[1] for ii in eachindex(Φ0) )
#     dΦdt    = SMatrix{3, 3, Float64}( Porosity(Φ0[ii], Pt[ii], Pf[ii], Pt0[ii], Pf0[ii], KΦ[ii], ηΦ[ii], m[ii], 0., 0., Δt)[2] for ii in eachindex(Φ0) )

#     # Interpolate porosity to velocity nodes
#     Φxⁿ = SVector{2, Float64}(@. (Φ[1:end-1,2].^n[1:end-1,2] + Φ[2:end,2].^n[2:end,2] )/2 )
#     Φyⁿ = SVector{2, Float64}(@. (Φ[2,1:end-1].^n[2,1:end-1] + Φ[2,2:end].^n[2,2:end] )/2 )

#     qx = SVector{2, Float64}(@. -kμ.xx * Φxⁿ * ((Pf[2:end,2] - Pf[1:end-1,2]) * invΔx      )  )
#     qy = SVector{2, Float64}(@. -kμ.yy * Φyⁿ * ((Pf[2,2:end] - Pf[2,1:end-1]) * invΔy - ρfg)  )

#     divqD = ( (  qx[2] -   qx[1]) * invΔx + (  qy[2] -   qy[1]) * invΔy)
#     divVs = ( (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy) 
    
#     fp = ( dΦdt[2,2]       + Φ[2,2]*divVs + divqD)

#     return fp
# end
