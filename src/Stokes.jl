struct Fields{Tx,Ty,Tp}
    Vx::Tx
    Vy::Ty
    Pt::Tp
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 4
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
end

function Ranges(nc)
    return (inx_Vx=2:nc.x+2, iny_Vx=3:nc.y+2, inx_Vy=3:nc.x+2, iny_Vy=2:nc.y+2, inx_c=2:nc.x+1, iny_c=2:nc.y+1, inx_v=2:nc.x+2, iny_v=2:nc.y+2, size_x=(nc.x + 3, nc.y + 4), size_y=(nc.x + 4, nc.y + 3), size_c=(nc.x + 2, nc.y + 2), size_v=(nc.x + 3, nc.y + 3))
end

function set_boundaries_template!(type, config, nc)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    @info "Setting $(string(config))"

    if config == :all_Dirichlet
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[2, iny_Vx] .= :Dirichlet_normal
        type.Vx[end-1, iny_Vx] .= :Dirichlet_normal
        type.Vx[inx_Vx, 2] .= :Dirichlet_tangent
        type.Vx[inx_Vx, end-1] .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[2, iny_Vy] .= :Dirichlet_tangent
        type.Vy[end-1, iny_Vy] .= :Dirichlet_tangent
        type.Vy[inx_Vy, 2] .= :Dirichlet_normal
        type.Vy[inx_Vy, end-1] .= :Dirichlet_normal
        # -------- Pt -------- #
        type.Pt[2:end-1, 2:end-1] .= :in

    elseif config == :EW_periodic # East/West periodic
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[1, iny_Vx] .= :periodic
        type.Vx[end-1:end, iny_Vx] .= :periodic
        type.Vx[inx_Vx, 2] .= :Dirichlet_tangent
        type.Vx[inx_Vx, end-1] .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[1:2, iny_Vy] .= :periodic
        type.Vy[end-1:end, iny_Vy] .= :periodic
        type.Vy[inx_Vy, 2] .= :Dirichlet_normal
        type.Vy[inx_Vy, end-1] .= :Dirichlet_normal
        # -------- Pt -------- #
        type.Pt[2:end-1, 2:end-1] .= :in
        type.Pt[[1 end], 2:end-1] .= :periodic

    elseif config == :NS_periodic  # North/South periodic
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[2, iny_Vx] .= :Dirichlet_normal
        type.Vx[end-1, iny_Vx] .= :Dirichlet_normal
        type.Vx[inx_Vx, 1:2] .= :periodic
        type.Vx[inx_Vx, end-1:end] .= :periodic
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[2, iny_Vy] .= :Dirichlet_tangent
        type.Vy[end-1, iny_Vy] .= :Dirichlet_tangent
        type.Vy[inx_Vy, 1] .= :periodic
        type.Vy[inx_Vy, end-1:end] .= :periodic
        # -------- Pt -------- #
        type.Pt[2:end-1, 2:end-1] .= :in
        type.Pt[2:end-1, [1 end]] .= :periodic

    elseif config == :NS_Neumann
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[2, iny_Vx] .= :Dirichlet_normal
        type.Vx[end-1, iny_Vx] .= :Dirichlet_normal
        type.Vx[inx_Vx, 2] .= :Dirichlet_tangent
        type.Vx[inx_Vx, end-1] .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[2, iny_Vy] .= :Dirichlet_tangent
        type.Vy[end-1, iny_Vy] .= :Dirichlet_tangent
        type.Vy[inx_Vy, 1] .= :Neumann_normal
        type.Vy[inx_Vy, end] .= :Neumann_normal
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
    elseif config == :NS_stress
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
        type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
        type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
        type.Vy[inx_Vy,1]       .= :normal_stress
        type.Vy[inx_Vy,end]     .= :normal_stress
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
    elseif config == :N_StressFree
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
        type.Vx[inx_Vx,2]       .= :Neumann_tangent
        type.Vx[inx_Vx,end-1]   .= :Neumann_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Neumann_tangent
        type.Vy[end-1,iny_Vy]   .= :Neumann_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
        type.Vy[inx_Vy,end]     .= :Neumann_normal
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in

    elseif config == :EW_Neumann
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[1, iny_Vx] .= :Neumann_normal
        type.Vx[end-0, iny_Vx] .= :Neumann_normal
        type.Vx[inx_Vx, 2] .= :Dirichlet_tangent
        type.Vx[inx_Vx, end-1] .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[2, iny_Vy] .= :Dirichlet_tangent
        type.Vy[end-1, iny_Vy] .= :Dirichlet_tangent
        type.Vy[inx_Vy, 2] .= :Dirichlet_normal
        type.Vy[inx_Vy, end-1] .= :Dirichlet_normal
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
        # type.Pt[[1,end],2:end-1] .= :Neumann_normal
elseif config == :EW_stress
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in      
        type.Vx[1,iny_Vx]       .= :normal_stress
        type.Vx[end,iny_Vx]     .= :normal_stress
        type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
        type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
        type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
        # type.Pt[[1,end],2:end-1] .= :Neumann_normal
    elseif config == :free_slip
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[2, iny_Vx] .= :Dirichlet_normal
        type.Vx[end-1, iny_Vx] .= :Dirichlet_normal
        type.Vx[inx_Vx, 2] .= :Neumann_tangent
        type.Vx[inx_Vx, end-1] .= :Neumann_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[2, iny_Vy] .= :Neumann_tangent
        type.Vy[end-1, iny_Vy] .= :Neumann_tangent
        type.Vy[inx_Vy, 2] .= :Dirichlet_normal
        type.Vy[inx_Vy, end-1] .= :Dirichlet_normal
        # -------- Pt -------- #
        type.Pt[2:end-1, 2:end-1] .= :in

    elseif config == :no_slip
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[2, iny_Vx] .= :Dirichlet_normal
        type.Vx[end-1, iny_Vx] .= :Dirichlet_normal
        type.Vx[inx_Vx, 2] .= :Dirichlet_tangent
        type.Vx[inx_Vx, end-1] .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[2, iny_Vy] .= :Dirichlet_tangent
        type.Vy[end-1, iny_Vy] .= :Dirichlet_tangent
        type.Vy[inx_Vy, 2] .= :Dirichlet_normal
        type.Vy[inx_Vy, end-1] .= :Dirichlet_normal
        # -------- Pt -------- #
        type.Pt[2:end-1, 2:end-1] .= :in

    elseif config == :N_StressFree
        # -------- Vx -------- #
        type.Vx[inx_Vx, iny_Vx] .= :in
        type.Vx[2, iny_Vx] .= :Dirichlet_normal
        type.Vx[end-1, iny_Vx] .= :Dirichlet_normal
        type.Vx[inx_Vx, 2] .= :Neumann_tangent
        type.Vx[inx_Vx, end-1] .= :Neumann_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy, iny_Vy] .= :in
        type.Vy[2, iny_Vy] .= :Neumann_tangent
        type.Vy[end-1, iny_Vy] .= :Neumann_tangent
        type.Vy[inx_Vy, 2] .= :Dirichlet_normal
        type.Vy[inx_Vy, end] .= :Neumann_normal
        # -------- Pt -------- #
        type.Pt[2:end-1, 2:end-1] .= :in

    end
end

function SMomentum_x_Generic(Vx_loc, Vy_loc, Pt, ΔP, τ0, G_loc, 𝐷, materials, type, bcv, Δ)
    
    invΔx, invΔy, BC_sym = 1 / Δ.x, 1 / Δ.y, 1.0

    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    # Interp Vy -> Vx, Vx - > Vy
    V̄y = av2D(Vy)
    V̄x = av2D(Vx)

    # More averages
    Pt_v = SVector(av(Pt))
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
    Ptc = SVector{2}(Pt[i, 2] + comp * ΔP[i] for i = 1:2)

    # Stress
    σxx = SVector{2}(
        (𝐷.c[i][1, 1] - 𝐷.c[i][4, 1]) * ϵ̇xx_c[i] + (𝐷.c[i][1, 2] - 𝐷.c[i][4, 2]) * ϵ̇yy_c[i] + (𝐷.c[i][1, 3] - 𝐷.c[i][4, 3]) * ϵ̇xy_c[i] + (𝐷.c[i][1, 4] - (𝐷.c[i][4, 4] - 1)) * Pt[i, 2] - Ptc[i] for i = 1:2
    )
    τxy = SVector{2}(
        𝐷.v[i][3, 1] * ϵ̇xx_v[i] + 𝐷.v[i][3, 2] * ϵ̇yy_v[i] + 𝐷.v[i][3, 3] * ϵ̇xy_v[i] + 𝐷.v[i][3, 4] * Pt_v[i] for i = 1:2
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
    fx = (σxx[2] - σxx[1]) * invΔx
    fx += (τxy[2] - τxy[1]) * invΔy
    fx *= -Δ.x * Δ.y
    fx *= BC_sym

    return fx
end

function SMomentum_y_Generic(Vx_loc, Vy_loc, Pt, ΔP, τ0, G_loc, ρ_loc, 𝐷, materials, type, bcv, Δ)

    invΔx, invΔy, BC_sym = 1 / Δ.x, 1 / Δ.y, 1.0 
 
    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    # Interp Vy -> Vx, Vx - > Vy
    V̄y = av2D(Vy)   # 2, 2
    V̄x = av2D(Vx)   # 3, 3

    # # More averages
    Pt_v = SVector(av(Pt))
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
    Ptc = SVector{2}(Pt[2, i] + comp * ΔP[i] for i = 1:2)

    # Stress
    σyy = SVector{2}(
        (𝐷.c[i][2, 1] - 𝐷.c[i][4, 1]) * ϵ̇xx_c[i] + (𝐷.c[i][2, 2] - 𝐷.c[i][4, 2]) * ϵ̇yy_c[i] + (𝐷.c[i][2, 3] - 𝐷.c[i][4, 3]) * ϵ̇xy_c[i] + (𝐷.c[i][2, 4] - (𝐷.c[i][4, 4] - 1.)) * Pt[2, i] - Ptc[i] for i = 1:2
    )
    τxy = SVector{2}(
        𝐷.v[i][3, 1] * ϵ̇xx_v[i] + 𝐷.v[i][3, 2] * ϵ̇yy_v[i] + 𝐷.v[i][3, 3] * ϵ̇xy_v[i] + 𝐷.v[i][3, 4] * Pt_v[i] for i = 1:2
    )

    # Gravity
    ρ = SVector{2}(ρ_loc[1, i] for i = 1:2)
    ρg = materials.g[2] * 0.5 * (ρ[1] + ρ[2])

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
    fy = (σyy[2] - σyy[1]) * invΔy
    fy += (τxy[2] - τxy[1]) * invΔx
    fy += ρg
    fy *= -Δ.x * Δ.y
    fy *= BC_sym
    
    return fy
end


function Continuity(Vx, Vy, Pt, Pt0, D, β, ξ, materials, type_loc, bcv_loc, Δ)
    invΔx = 1 / Δ.x
    invΔy = 1 / Δ.y
    invΔt = 1 / Δ.t
    comp = materials.compressible
    f = ((Vx[2, 2] - Vx[1, 2]) * invΔx + (Vy[2, 2] - Vy[2, 1]) * invΔy) + comp * β * (Pt[1] - Pt0) * invΔt + comp * Pt[1] / ξ
    return f
end

function ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)

    shift = (x=1, y=2)
    Threads.@threads for j in 1+shift.y:nc.y+shift.y
        for i in 1+shift.x:nc.x+shift.x+1
            if type.Vx[i, j] == :in
                Vx_loc    = SMatrix{3,3}(V.x[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                Vy_loc    = SMatrix{4,4}(V.y[ii, jj] for ii in i-1:i+2, jj in j-2:j+1)
                bcx_loc   = SMatrix{3,3}(BC.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                bcy_loc   = SMatrix{4,4}(BC.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1)
                typex_loc = SMatrix{3,3}(type.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                typey_loc = SMatrix{4,4}(type.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1)
                typep_loc = SMatrix{2,1}(type.Pt[ii, jj] for ii in i-1:i-0, jj in j-1:j-1)
                P_loc     = SMatrix{2,3}(P[ii, jj] for ii in i-1:i, jj in j-2:j)
                ΔP_loc    = SMatrix{2,1}(ΔP.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
                τxx0      = SMatrix{2,3}(τ0.xx[ii, jj] for ii in i-1:i, jj in j-2:j)
                τyy0      = SMatrix{2,3}(τ0.yy[ii, jj] for ii in i-1:i, jj in j-2:j)
                τxy0      = SMatrix{3,2}(τ0.xy[ii, jj] for ii in i-1:i+1, jj in j-1:j)
                Gc_loc    = SMatrix{2,1}(G.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
                Gv_loc    = SMatrix{1,2}(G.v[ii, jj] for ii in i-0:i-0, jj in j-1:j-0)
                Dc        = SMatrix{2,1}(𝐷.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
                Dv        = SMatrix{1,2}(𝐷.v[ii, jj] for ii in i-0:i-0, jj in j-1:j-0)
                bcv_loc = (x=bcx_loc, y=bcy_loc)
                type_loc = (x=typex_loc, y=typey_loc, p=typep_loc)
                D = (c=Dc, v=Dv)
                τ0_loc = (xx=τxx0, yy=τyy0, xy=τxy0)
                G_loc = (c=Gc_loc, v=Gv_loc)

                R.x[i, j] = SMomentum_x_Generic(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ)
            end
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, P0, ΔP, τ0, 𝐷, G, materials, num, pattern, type, BC, nc, Δ)

    shift = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y
        for i in 1+shift.x:nc.x+shift.x+1

            if type.Vx[i, j] == :in

                bcx_loc = SMatrix{3,3}(BC.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                bcy_loc = SMatrix{4,4}(BC.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1)
                typex_loc = SMatrix{3,3}(type.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                typey_loc = SMatrix{4,4}(type.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1)
                typep_loc = SMatrix{2,1}(type.Pt[ii, jj] for ii in i-1:i-0, jj in j-1:j-1)

                Vx_loc = SMatrix{3,3}(V.x[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                Vy_loc = SMatrix{4,4}(V.y[ii, jj] for ii in i-1:i+2, jj in j-2:j+1)
                P_loc = SMatrix{2,3}(P[ii, jj] for ii in i-1:i, jj in j-2:j)
                ΔP_loc = SMatrix{2,1}(ΔP.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
                Gc_loc = SMatrix{2,1}(G.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
                Gv_loc = SMatrix{1,2}(G.v[ii, jj] for ii in i-0:i-0, jj in j-1:j-0)

                τxx0 = SMatrix{2,3}(τ0.xx[ii, jj] for ii in i-1:i, jj in j-2:j)
                τyy0 = SMatrix{2,3}(τ0.yy[ii, jj] for ii in i-1:i, jj in j-2:j)
                τxy0 = SMatrix{3,2}(τ0.xy[ii, jj] for ii in i-1:i+1, jj in j-1:j)

                Dc = SMatrix{2,1}(𝐷.c[ii, jj] for ii in i-1:i, jj in j-1:j-1)
                Dv = SMatrix{1,2}(𝐷.v[ii, jj] for ii in i-0:i-0, jj in j-1:j-0)
                bcv_loc = (x=bcx_loc, y=bcy_loc)
                type_loc = (x=typex_loc, y=typey_loc, p=typep_loc)
                # ph_loc     = (c=phc_loc, v=phv_loc)
                D = (c=Dc, v=Dv)
                τ0_loc = (xx=τxx0, yy=τyy0, xy=τxy0)
                G_loc = (c=Gc_loc, v=Gv_loc)

                ∂R∂Vx = gradient(x -> SMomentum_x_Generic(x, Vy_loc, P_loc, ΔP_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ), AUTO_DIFF_BACKEND, Vx_loc)
                ∂R∂Vy = gradient(x -> SMomentum_x_Generic(Vx_loc, x, P_loc, ΔP_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ), AUTO_DIFF_BACKEND, Vy_loc)
                ∂R∂Pt = gradient(x -> SMomentum_x_Generic(Vx_loc, Vy_loc, x, ΔP_loc, τ0_loc, G_loc, D, materials, type_loc, bcv_loc, Δ), AUTO_DIFF_BACKEND, P_loc)
                # Vx --- Vx
                Local = SMatrix{3,3}(num.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern[1][1]
                for jj in axes(Local, 2), ii in axes(Local, 1)
                    if (Local[ii, jj] > 0) && num.Vx[i, j] > 0
                        K[1][1][num.Vx[i, j], Local[ii, jj]] = ∂R∂Vx[ii, jj]
                    end
                end
                # Vx --- Vy
                Local = SMatrix{4,4}(num.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1) .* pattern[1][2]
                for jj in axes(Local, 2), ii in axes(Local, 1)
                    if (Local[ii, jj] > 0) && num.Vx[i, j] > 0
                        K[1][2][num.Vx[i, j], Local[ii, jj]] = ∂R∂Vy[ii, jj]
                    end
                end
                # Vx --- Pt
                Local = SMatrix{2,3}(num.Pt[ii, jj] for ii in i-1:i, jj in j-2:j) .* pattern[1][3]
                for jj in axes(Local, 2), ii in axes(Local, 1)
                    if (Local[ii, jj] > 0) && num.Vx[i, j] > 0
                        K[1][3][num.Vx[i, j], Local[ii, jj]] = ∂R∂Pt[ii, jj]
                    end
                end
            end
        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
    shift = (x=2, y=1)
    Threads.@threads for j in 1+shift.y:nc.y+shift.y+1
        for i in 1+shift.x:nc.x+shift.x
            if type.Vy[i, j] == :in
                Vx_loc    = SMatrix{4,4}(V.x[ii, jj] for ii in i-2:i+1, jj in j-1:j+2)
                Vy_loc    = SMatrix{3,3}(V.y[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                bcx_loc   = SMatrix{4,4}(BC.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2)
                bcy_loc   = SMatrix{3,3}(BC.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                typex_loc = SMatrix{4,4}(type.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2)
                typey_loc = SMatrix{3,3}(type.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                P_loc     = SMatrix{3,2}(P[ii, jj] for ii in i-2:i, jj in j-1:j)
                ΔP_loc    = SMatrix{1,2}(ΔP.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                Gc_loc    = SMatrix{1,2}(G.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                Gv_loc    = SMatrix{2,1}(G.v[ii, jj] for ii in i-1:i-0, jj in j-0:j-0)
                ρ_loc     = SMatrix{1,2}(ρ.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                τxx0      = SMatrix{3,2}(τ0.xx[ii, jj] for ii in i-2:i, jj in j-1:j)
                τyy0      = SMatrix{3,2}(τ0.yy[ii, jj] for ii in i-2:i, jj in j-1:j)
                τxy0      = SMatrix{2,3}(τ0.xy[ii, jj] for ii in i-1:i, jj in j-1:j+1)
                Dc        = SMatrix{1,2}(𝐷.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                Dv        = SMatrix{2,1}(𝐷.v[ii, jj] for ii in i-1:i-0, jj in j-0:j-0)
                bcv_loc = (x=bcx_loc, y=bcy_loc)
                type_loc = (x=typex_loc, y=typey_loc)
                G_loc = (c=Gc_loc, v=Gv_loc)
                D = (c=Dc, v=Dv)
                τ0_loc = (xx=τxx0, yy=τyy0, xy=τxy0)

                R.y[i, j] = SMomentum_y_Generic(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, G_loc, ρ_loc, D, materials, type_loc, bcv_loc, Δ)
            end
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, P0, ΔP, τ0, 𝐷, G, ρ, materials, num, pattern, type, BC, nc, Δ)

    shift = (x=2, y=1)
    K21 = K[2][1]
    K22 = K[2][2]
    K23 = K[2][3]

    for j in 1+shift.y:nc.y+shift.y+1
        for i in 1+shift.x:nc.x+shift.x

            if type.Vy[i, j] === :in

                Vx_loc = @inline SMatrix{4,4}(@inbounds V.x[ii, jj] for ii in i-2:i+1, jj in j-1:j+2)
                Vy_loc = @inline SMatrix{3,3}(@inbounds V.y[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                bcx_loc = @inline SMatrix{4,4}(@inbounds BC.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2)
                bcy_loc = @inline SMatrix{3,3}(@inbounds BC.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                typex_loc = @inline SMatrix{4,4}(@inbounds type.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2)
                typey_loc = @inline SMatrix{3,3}(@inbounds type.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1)
                Gc_loc = @inline SMatrix{1,2}(@inbounds G.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                Gv_loc = @inline SMatrix{2,1}(@inbounds G.v[ii, jj] for ii in i-1:i-0, jj in j-0:j-0)
                P_loc = @inline SMatrix{3,2}(@inbounds P[ii, jj] for ii in i-2:i, jj in j-1:j)
                ΔP_loc = @inline SMatrix{1,2}(@inbounds ΔP.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                ρ_loc = @inline SMatrix{1,2}(@inbounds ρ.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                τxx0 = @inline SMatrix{3,2}(@inbounds τ0.xx[ii, jj] for ii in i-2:i, jj in j-1:j)
                τyy0 = @inline SMatrix{3,2}(@inbounds τ0.yy[ii, jj] for ii in i-2:i, jj in j-1:j)
                τxy0 = @inline SMatrix{2,3}(@inbounds τ0.xy[ii, jj] for ii in i-1:i, jj in j-1:j+1)
                Dc = @inline SMatrix{1,2}(@inbounds 𝐷.c[ii, jj] for ii in i-1:i-1, jj in j-1:j)
                Dv = @inline SMatrix{2,1}(@inbounds 𝐷.v[ii, jj] for ii in i-1:i-0, jj in j-0:j-0)
                bcv_loc = (x=bcx_loc, y=bcy_loc)
                type_loc = (x=typex_loc, y=typey_loc)
                G_loc = (c=Gc_loc, v=Gv_loc)
                D = (c=Dc, v=Dv)
                τ0_loc = (xx=τxx0, yy=τyy0, xy=τxy0)

                ∂R∂Vx = gradient(x -> SMomentum_y_Generic(x, Vy_loc, P_loc, ΔP_loc, τ0_loc, G_loc, ρ_loc, D, materials, type_loc, bcv_loc, Δ), AUTO_DIFF_BACKEND, Vx_loc)
                ∂R∂Vy = gradient(x -> SMomentum_y_Generic(Vx_loc, x, P_loc, ΔP_loc, τ0_loc, G_loc, ρ_loc, D, materials, type_loc, bcv_loc, Δ), AUTO_DIFF_BACKEND, Vy_loc)
                ∂R∂Pt = gradient(x -> SMomentum_y_Generic(Vx_loc, Vy_loc, x, ΔP_loc, τ0_loc, G_loc, ρ_loc, D, materials, type_loc, bcv_loc, Δ), AUTO_DIFF_BACKEND, P_loc)


                num_Vy = @inbounds num.Vy[i, j]
                bounds_Vy = num_Vy > 0
                # Vy --- Vx
                Local1 = SMatrix{4,4}(num.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2) .* pattern[2][1]
                # for jj in axes(Local1,2), ii in axes(Local1,1)
                #     if (Local1[ii,jj]>0) && bounds_Vy
                #         @inbounds K21[num_Vy, Local1[ii,jj]] = ∂R∂Vx[ii,jj] 
                #     end
                # end
                # Vy --- Vy
                Local2 = SMatrix{3,3}(num.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern[2][2]
                # for jj in axes(Local2,2), ii in axes(Local2,1)
                #     if (Local2[ii,jj]>0) && bounds_Vy
                #         @inbounds K22[num_Vy, Local2[ii,jj]] = ∂R∂Vy[ii,jj]  
                #     end
                # end
                # Vy --- Pt
                Local3 = SMatrix{3,2}(num.Pt[ii, jj] for ii in i-2:i, jj in j-1:j) .* pattern[2][3]
                # for jj in axes(Local3,2), ii in axes(Local3,1)
                #     if (Local3[ii,jj]>0) && bounds_Vy
                #         @inbounds K23[num_Vy, Local3[ii,jj]] = ∂R∂Pt[ii,jj]  
                #     end
                # end 

                Base.@nexprs 4 jj -> begin
                    Base.@nexprs 4 ii -> begin
                        bounds_Vy && (Local1[ii, jj] > 0) &&
                            (@inbounds K21[num_Vy, Local1[ii, jj]] = ∂R∂Vx[ii, jj])

                        bounds_Vy && ii < 4 && jj < 4 && (Local2[ii, jj] > 0) &&
                            (@inbounds K22[num_Vy, Local2[ii, jj]] = ∂R∂Vy[ii, jj])

                        bounds_Vy && ii < 4 && jj < 3 && (Local3[ii, jj] > 0) &&
                            (@inbounds K23[num_Vy, Local3[ii, jj]] = ∂R∂Pt[ii, jj])
                    end
                end
            end
        end
    end
    return nothing
end

function ResidualContinuity2D!(R, V, P, P0, ΔP, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ)

    Threads.@threads for j in 2:size(R.p, 2)-1
        for i in 2:size(R.p, 1)-1
            if type.Pt[i, j] !== :constant
                Vx_loc = SMatrix{3,2}(V.x[ii, jj] for ii in i:i+2, jj in j:j+1)
                Vy_loc = SMatrix{2,3}(V.y[ii, jj] for ii in i:i+1, jj in j:j+2)
                bcv_loc = (;)
                type_loc = (;)
                D = (;)
                R.p[i, j] = Continuity(Vx_loc, Vy_loc, P[i, j], P0[i, j], D, β.c[i, j], ξ.c[i, j], materials, type_loc, bcv_loc, Δ)
            end
        end
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, Pt0, ΔP, τ0, 𝐷, β, ξ, materials, num, pattern, type, BC, nc, Δ)

    for j in 2:size(P, 2)-1
        for i in 2:size(P, 1)-1
            Vx_loc = SMatrix{2,3}(V.x[ii, jj] for ii in i:i+1, jj in j:j+2)
            Vy_loc = SMatrix{3,2}(V.y[ii, jj] for ii in i:i+2, jj in j:j+1)
            P_loc = SMatrix{1,1}(P[ii, jj] for ii in i:i, jj in j:j)
            bcv_loc = (;)
            type_loc = (;)
            D = (;)

            ∂R∂Vx, ∂R∂Vy, ∂R∂P = ad_partial_gradients(Continuity, (Vx_loc, Vy_loc, P_loc), Pt0[i, j], D, β.c[i, j], ξ.c[i, j], materials, type_loc, bcv_loc, Δ)
            # Pt --- Vx
            Local = SMatrix{2,3}(num.Vx[ii, jj] for ii in i:i+1, jj in j:j+2)# .* pattern[3][1]        
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if Local[ii, jj] > 0 && num.Pt[i, j] > 0
                    K[3][1][num.Pt[i, j], Local[ii, jj]] = ∂R∂Vx[ii, jj]
                end
            end
            # Pt --- Vy
            Local = SMatrix{3,2}(num.Vy[ii, jj] for ii in i:i+2, jj in j:j+1) #.* pattern[3][2]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if Local[ii, jj] > 0 && num.Pt[i, j] > 0
                    K[3][2][num.Pt[i, j], Local[ii, jj]] = ∂R∂Vy[ii, jj]
                end
            end

            # Pt --- Pt
            if num.Pt[i, j] > 0
                K[3][3][num.Pt[i, j], num.Pt[i, j]] = ∂R∂P[1, 1]
            end
        end
    end
    return nothing
end

@views function SparsityPattern!(K, num, pattern, nc)
    ############ Fields Vx ############
    shift = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y
        for i in 1+shift.x:nc.x+shift.x
            # Vx --- Vx
            Local = num.Vx[i-1:i+1, j-1:j+1] .* pattern[1][1]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Vx[i, j] > 0
                    K[1][1][num.Vx[i, j], Local[ii, jj]] = 1
                end
            end
            # Vx --- Vy
            Local = num.Vy[i-1:i+2, j-2:j+1] .* pattern[1][2]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Vx[i, j] > 0
                    K[1][2][num.Vx[i, j], Local[ii, jj]] = 1
                end
            end
            # Vx --- Pt
            Local = num.Pt[i-1:i, j-2:j] .* pattern[1][3]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Vx[i, j] > 0
                    K[1][3][num.Vx[i, j], Local[ii, jj]] = 1
                end
            end
        end
    end
    ############ Fields Vy ############
    shift = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y
        for i in 1+shift.x:nc.x+shift.x
            # Vy --- Vx
            Local = num.Vx[i-2:i+1, j-1:j+2] .* pattern[2][1]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Vy[i, j] > 0
                    K[2][1][num.Vy[i, j], Local[ii, jj]] = 1
                end
            end
            # Vy --- Vy
            Local = num.Vy[i-1:i+1, j-1:j+1] .* pattern[2][2]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Vy[i, j] > 0
                    K[2][2][num.Vy[i, j], Local[ii, jj]] = 1
                end
            end
            # Vy --- Pt
            Local = num.Pt[i-2:i, j-1:j] .* pattern[2][3]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Vy[i, j] > 0
                    K[2][3][num.Vy[i, j], Local[ii, jj]] = 1
                end
            end
        end
    end
    # ############ Fields Pt ############
    shift = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y
        for i in 1+shift.x:nc.x+shift.x
            # Pt --- Vx
            Local = num.Vx[i:i+1, j:j+2] .* pattern[3][1]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Pt[i, j] > 0
                    K[3][1][num.Pt[i, j], Local[ii, jj]] = 1
                end
            end
            # Pt --- Vy
            Local = num.Vy[i:i+2, j:j+1] .* pattern[3][2]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Pt[i, j] > 0
                    K[3][2][num.Pt[i, j], Local[ii, jj]] = 1
                end
            end
            # Pt --- Pt
            Local = num.Pt[i, j] .* pattern[3][3]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                if (Local[ii, jj] > 0) && num.Pt[i, j] > 0
                    K[3][3][num.Pt[i, j], Local[ii, jj]] = 1
                end
            end
        end
    end
    ############ End ############
end

function SetRHS!(r, R, number, type, nc)

    nVx, nVy = maximum(number.Vx), maximum(number.Vy)

    Threads.@threads for j = 2:nc.y+3-1
        for i = 2:nc.x+3-1
            if type.Vx[i, j] == :in
                ind = number.Vx[i, j]
                r[ind] = R.x[i, j]
            end
        end
    end
    Threads.@threads for j = 2:nc.y+3-1
        for i = 2:nc.x+3-1
            if type.Vy[i, j] == :in
                ind = number.Vy[i, j] + nVx
                r[ind] = R.y[i, j]
            end
        end
    end
    Threads.@threads for j = 2:nc.y+1
        for i = 2:nc.x+1
            if type.Pt[i, j] == :in
                ind = number.Pt[i, j] + nVx + nVy
                r[ind] = R.p[i, j]
            end
        end
    end
end

function UpdateSolution!(V, Pt, dx, number, type, nc)

    nVx, nVy = maximum(number.Vx), maximum(number.Vy)

    Threads.@threads for j = 1:size(V.x, 2)
        for i = 1:size(V.x, 1)
            if type.Vx[i, j] == :in
                ind = number.Vx[i, j]
                V.x[i, j] += dx[ind]
            end
        end
    end

    Threads.@threads for j = 1:size(V.y, 2)
        for i = 1:size(V.y, 1)
            if type.Vy[i, j] == :in
                ind = number.Vy[i, j] + nVx
                V.y[i, j] += dx[ind]
            end
        end
    end

    Threads.@threads for I in eachindex(Pt)
        if type.Pt[I] == :in
            ind = number.Pt[I] + nVx + nVy
            Pt[I] += dx[ind]
        end
    end

    # Set E/W periodicity
    Threads.@threads for j = 2:nc.y+3-1
        if type.Vx[nc.x+3-1, j] == :periodic
            V.x[nc.x+3-1, j] = V.x[2, j]
            V.x[nc.x+3-0, j] = V.x[3, j]
            V.x[1, j] = V.x[nc.x+3-2, j]
        end
        if type.Vy[nc.x+3, j] == :periodic
            V.y[nc.x+3-0, j] = V.y[3, j]
            V.y[nc.x+3+1, j] = V.y[4, j]
            V.y[1, j] = V.y[nc.x+3-2, j]
            V.y[2, j] = V.y[nc.x+3-1, j]
        end
        if j <= nc.y + 2
            if type.Pt[nc.x+2, j] == :periodic
                Pt[nc.x+2, j] = Pt[2, j]
                Pt[1, j] = Pt[nc.x+1, j]
            end
        end
    end

    # Set S/N periodicity
    Threads.@threads for i = 2:nc.x+3-1
        if type.Vx[i, nc.y+3] == :periodic
            V.x[i, nc.y+3-0] = V.x[i, 3]
            V.x[i, nc.y+3+1] = V.x[i, 4]
            V.x[i, 1] = V.x[i, nc.y+3-2]
            V.x[i, 2] = V.x[i, nc.y+3-1]
        end
        if type.Vy[i, nc.y+3-1] == :periodic
            V.y[i, nc.y+3-1] = V.y[i, 2]
            V.y[i, nc.y+3-0] = V.y[i, 3]
            V.y[i, 1] = V.y[i, nc.y+3-2]
        end
        if i <= nc.x + 2
            if type.Pt[i, nc.y+2] == :periodic
                Pt[i, nc.y+2] = Pt[i, 2]
                Pt[i, 1] = Pt[i, nc.y+1]
            end
        end
    end

end

function Numbering!(N, type, nc)

    ndof = 0
    neq = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west = sum(any(i -> i == :periodic, type.Vx[1, 3:end-2], dims=2)) > 0
    periodic_south = sum(any(i -> i == :periodic, type.Vx[3:end-2, 2], dims=1)) > 0

    shift = (periodic_west) ? 1 : 0
    # Loop through inner nodes of the mesh
    for j = 3:nc.y+4-2, i = 2:nc.x+3-1
        if type.Vx[i, j] == :Dirichlet_normal || (type.Vx[i, j] == :periodic && i == nc.x + 3 - 1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof += 1
            N.Vx[i, j] = ndof
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1, :] .= N.Vx[end-2, :]
        N.Vx[end-1, :] .= N.Vx[2, :]
        N.Vx[end, :] .= N.Vx[3, :]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        # South
        N.Vx[:, 1] .= N.Vx[:, end-3]
        N.Vx[:, 2] .= N.Vx[:, end-2]
        # North
        N.Vx[:, end] .= N.Vx[:, 4]
        N.Vx[:, end-1] .= N.Vx[:, 3]
    end
    noisy ? printxy(N.Vx) : nothing

    neq = maximum(N.Vx)

    ############ Numbering Vy ############
    ndof = 0
    periodic_west = sum(any(i -> i == :periodic, type.Vy[2, 3:end-2], dims=2)) > 0
    periodic_south = sum(any(i -> i == :periodic, type.Vy[3:end-2, 1], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    # Loop through inner nodes of the mesh
    for j = 2:nc.y+3-1, i = 3:nc.x+4-2
        if type.Vy[i, j] == :Dirichlet_normal || (type.Vy[i, j] == :periodic && j == nc.y + 3 - 1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof += 1
            N.Vy[i, j] = ndof
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vy[:, 1] .= N.Vy[:, end-2]
        N.Vy[:, end-1] .= N.Vy[:, 2]
        N.Vy[:, end] .= N.Vy[:, 3]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        # West
        N.Vy[1, :] .= N.Vy[end-3, :]
        N.Vy[2, :] .= N.Vy[end-2, :]
        # East
        N.Vy[end, :] .= N.Vy[4, :]
        N.Vy[end-1, :] .= N.Vy[3, :]
    end
    noisy ? printxy(N.Vy) : nothing

    neq = maximum(N.Vy)

    ############ Numbering Pt ############
    # neq_Pt                     = nc.x * nc.y
    # N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)
    ii = 0
    for j = 1:nc.y, i = 1:nc.x
        if type.Pt[i+1, j+1] != :constant
            ii += 1
            N.Pt[i+1, j+1] = ii
        end
    end

    if periodic_west
        N.Pt[1, :] .= N.Pt[end-1, :]
        N.Pt[end, :] .= N.Pt[2, :]
    end

    if periodic_south
        N.Pt[:, 1] .= N.Pt[:, end-1]
        N.Pt[:, end] .= N.Pt[:, 2]
    end
    noisy ? printxy(N.Pt) : nothing

    neq = maximum(N.Pt)
end


function LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x
    Vi.y .= V.y
    Pti .= Pt
    for i in eachindex(α)
        V.x .= Vi.x
        V.y .= Vi.y
        Pt .= Pti
        UpdateSolution!(V, Pt, α[i] .* dx, number, type, nc)
        TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
        ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
        rvec[i] = @views norm(R.x[inx_Vx, iny_Vx]) / length(R.x[inx_Vx, iny_Vx]) + norm(R.y[inx_Vy, iny_Vy]) / length(R.y[inx_Vy, iny_Vy]) + 0 * norm(R.p[inx_c, iny_c]) / length(R.p[inx_c, iny_c])
    end
    imin = argmin(rvec)
    V.x .= Vi.x
    V.y .= Vi.y
    Pt .= Pti
    return imin
end

@views function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)

    _ones = @SVector ones(4)
    s = 1
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # Detect periodicity E/W
    periodic_west = false
    @inbounds @views for i in 3:size(type.Vx, 2)-2
        if type.Vx[1, i] === :periodic
            periodic_west = true
            break
        end
    end

    # Detect periodicity N/S
    periodic_south = false
    @inbounds @views for i in 3:size(type.Vx, 1)-2
        if type.Vx[i, 2] === :periodic
            periodic_south = true
            break
        end
    end

    # Loop over centroids
    Threads.@threads for j = 1+s:size(ε̇.xx, 2)-s
        for i = 1+s:size(ε̇.xx, 1)-s
            if (i == 1 && j == 1) || (i == size(ε̇.xx, 1) && j == 1) || (i == 1 && j == size(ε̇.xx, 2)) || (i == size(ε̇.xx, 1) && j == size(ε̇.xx, 2))
                # Avoid the outer corners - nothing is well defined there ;)
            else
                Vx    = SMatrix{2,3}(    V.x[ii, jj] for ii in i:i+1, jj in j:j+2)
                Vy    = SMatrix{3,2}(    V.y[ii, jj] for ii in i:i+2, jj in j:j+1)
                bcx   = SMatrix{2,3}(  BC.Vx[ii, jj] for ii in i:i+1, jj in j:j+2)
                bcy   = SMatrix{3,2}(  BC.Vy[ii, jj] for ii in i:i+2, jj in j:j+1)
                typex = SMatrix{2,3}(type.Vx[ii, jj] for ii in i:i+1, jj in j:j+2)
                typey = SMatrix{3,2}(type.Vy[ii, jj] for ii in i:i+2, jj in j:j+1)
                τxy0  = SMatrix{2,2}(  τ0.xy[ii, jj] for ii in i:i+1, jj in j:j+1)

                # Apply BC's
                Vx = SetBCVx1(Vx, typex, bcx, Δ)
                Vy = SetBCVy1(Vy, typey, bcy, Δ)

                # Interp Vy -> Vx, Vx - > Vy
                V̄y = SMatrix{2,1}(av2D(Vy))
                V̄x = SMatrix{1,2}(av2D(Vx))

                # More averages
                τ0xx = τ0.xx[i, j]
                τ0yy = τ0.yy[i, j]
                τ0xy = av(τxy0)[1]

                # Velocity gradient - centroids
                Dxx = (∂x(Vx)*invΔx)[:, 2:end-1][1]
                Dxy = (∂y(V̄x)*invΔy)[1]
                Dyy = (∂y(Vy)*invΔy)[2:end-1, :][1]
                Dyx = (∂x(V̄y)*invΔx)[1]

                # Deviatoric strain rate
                ε̇xx, ε̇yy, ε̇xy, ε̇kk = deviatoric_strain_rate(Dxx, Dxy, Dyx, Dyy)

                # Effective visco-elastic strain rate
                _2GΔt = inv(2 * G.c[i, j] * Δ.t)
                ϵ̇xx, ϵ̇yy, ϵ̇xy = effective_strain_rate(ε̇xx, ε̇yy, ε̇xy, τ0xx, τ0yy, τ0xy, _2GΔt)
                ε̇vec = SVector{4}(ϵ̇xx, ϵ̇yy, ϵ̇xy, Pt[i, j])

                # Tangent operator used for Newton Linearisation
                stress_state = StressVector!(ε̇vec, ε̇kk, Pt0[i, j], materials, phase_ratios.c[i, j], Δ)
                τ_vec, jac = ad_jacobian_first(StressVector!, ε̇vec, ε̇kk, Pt0[i, j], materials, phase_ratios.c[i, j], Δ)
                _, η_local, λ̇_local, τII_local = stress_state

                @views 𝐷_ctl.c[i, j] .= jac

                # Tangent operator used for Picard Linearisation
                𝐷.c[i, j] .= diagm(2 * η_local * _ones)
                𝐷.c[i, j][4, 4] = 1

                # Update stress
                τ.xx[i, j]  = τ_vec[1]
                τ.yy[i, j]  = τ_vec[2]
                τ.II[i, j]  = τII_local
                τ.θ[i, j]   = 1/2 * atan( 2*τ_vec[3], τ_vec[1] - τ_vec[2] ) + π/2
                ε̇.xx[i, j]  = ε̇xx
                ε̇.yy[i, j]  = ε̇yy
                ε̇.II[i, j]  = sqrt(1 / 2 * (ε̇xx^2 + ε̇yy^2) + ε̇xy^2)
                ε̇.θ[i, j]   = 1/2 * atan( 2*ε̇xy, ε̇xx - ε̇yy ) + π/2
                λ̇.c[i, j]   = λ̇_local
                η.c[i, j]   = η_local
                ΔPt.c[i, j] = (τ_vec[4] - Pt[i, j])
            end
        end
    end

    # For periodic cases
    if periodic_west
        for j = 2:size(ε̇.xx, 2)-1
            i = 1
            @views 𝐷_ctl.c[i, j] .= 𝐷_ctl.c[end-1, j]
            @views 𝐷.c[i, j] .= 𝐷.c[end-1, j]
            i = size(ε̇.xx, 1)
            @views 𝐷_ctl.c[i, j] .= 𝐷_ctl.c[2, j]
            @views 𝐷.c[i, j] .= 𝐷.c[2, j]
        end
    end
    if periodic_south
        for i = 2:size(ε̇.xx, 1)-1
            j = 1
            @views 𝐷_ctl.c[i, j] .= 𝐷_ctl.c[i, end-1]
            @views 𝐷.c[i, j] .= 𝐷.c[i, end-1]
            j = size(ε̇.xx, 2)
            @views 𝐷_ctl.c[i, j] .= 𝐷_ctl.c[i, 2]
            @views 𝐷.c[i, j] .= 𝐷.c[i, 2]
        end
    end

    # Loop over vertices
    Threads.@threads for j = 1+s:size(ε̇.xy, 2)-s
        for i = 1+s:size(ε̇.xy, 1)-s
            Vx    = SMatrix{3,2}(    V.x[ii, jj] for ii in i-1:i+1, jj in j:j+1)
            Vy    = SMatrix{2,3}(    V.y[ii, jj] for ii in i:i+1, jj in j-1:j+1)
            bcx   = SMatrix{3,2}(  BC.Vx[ii, jj] for ii in i-1:i+1, jj in j:j+1)
            bcy   = SMatrix{2,3}(  BC.Vy[ii, jj] for ii in i:i+1, jj in j-1:j+1)
            typex = SMatrix{3,2}(type.Vx[ii, jj] for ii in i-1:i+1, jj in j:j+1)
            typey = SMatrix{2,3}(type.Vy[ii, jj] for ii in i:i+1, jj in j-1:j+1)
            τxx0  = SMatrix{2,2}(  τ0.xx[ii, jj] for ii in i-1:i, jj in j-1:j)
            τyy0  = SMatrix{2,2}(  τ0.yy[ii, jj] for ii in i-1:i, jj in j-1:j)
            P     = SMatrix{2,2}(     Pt[ii, jj] for ii in i-1:i, jj in j-1:j)
            P0    = SMatrix{2,2}(    Pt0[ii, jj] for ii in i-1:i, jj in j-1:j)

            # Apply BC's
            Vx = SetBCVx1(Vx, typex, bcx, Δ)
            Vy = SetBCVy1(Vy, typey, bcy, Δ)

            # Interp Vy -> Vx, Vx - > Vy
            V̄y = SMatrix{1,2}(av2D(Vy))
            V̄x = SMatrix{2,1}(av2D(Vx))

            # # More averages
            τ0xx = av(τxx0)[1]
            τ0yy = av(τyy0)[1]
            τ0xy = τ0.xy[i, j]
            P̄    = av(P)[1]
            P̄0   = av(P0)[1]

            # Velocity gradient - centroids
            Dxx = (∂x(V̄x)*invΔx)[1]
            Dxy = (∂y(Vx)*invΔy)[2:end-1, :][1]
            Dyy = (∂y(V̄y)*invΔy)[1]
            Dyx = (∂x(Vy)*invΔx)[:, 2:end-1][1]

            # Deviatoric strain rate
            ε̇xx, ε̇yy, ε̇xy, ε̇kk = deviatoric_strain_rate(Dxx, Dxy, Dyx, Dyy)

            # Effective visco-elastic strain rate
            _2GΔt = inv(2 * G.v[i, j] * Δ.t)
            ϵ̇xx, ϵ̇yy, ϵ̇xy = effective_strain_rate(ε̇xx, ε̇yy, ε̇xy, τ0xx, τ0yy, τ0xy, _2GΔt)
            ε̇vec = SVector{4}(ϵ̇xx, ϵ̇yy, ϵ̇xy, P̄)

            # Tangent operator used for Newton Linearisation
            stress_state = StressVector!(ε̇vec, ε̇kk, P̄0, materials, phase_ratios.v[i, j], Δ)
            τ_vec, jac = ad_jacobian_first(StressVector!, ε̇vec, ε̇kk, P̄0, materials, phase_ratios.v[i, j], Δ)
            _, η_local, λ̇_local, = stress_state

            @views 𝐷_ctl.v[i, j] .= jac

            # Tangent operator used for Picard Linearisation
            𝐷.v[i, j] .= diagm(2 * η_local * _ones)
            𝐷.v[i, j][4, 4] = 1

            # Update stress
            τ.xy[i, j] = τ_vec[3]
            ε̇.xy[i, j] = ε̇xy
            λ̇.v[i, j] = λ̇_local
            η.v[i, j] = η_local
        end
    end
end