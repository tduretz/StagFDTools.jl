# Initialisation
using CairoMakie, Printf, Statistics, LinearAlgebra, JLD2
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 

kyr = 1e3 * 365.25 * 24 * 3600

# can be replaced by AD
function Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ, Δx, Δy, ncx  ,ncy)
        
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    ebW   = γ[1:end-1,:] 
    ebE   = γ[2:end-0,:] 
    Cxx   = ones(ncx-1, ncy)
    Cxy   = ones(ncx-1, ncy)
    @. Cxx = abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    @. Cxy = abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))

    DVx  = ones(ncx-1, ncy)
    @. DVx .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    ebS  = γ[:,1:end-1] 
    ebN  = γ[:,2:end-0] 
    Cyy  = ones(ncx, ncy-1)
    Cyx  = ones(ncx, ncy-1)
    @. Cyy = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(ebN ./ Δy .^ 2 + (4 // 3) * ηN ./ Δy .^ 2) + abs.(ebS ./ Δy .^ 2 + (4 // 3) * ηS ./ Δy .^ 2) + abs.((ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    @. Cyx = abs.(ebN ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ebN ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy) + ηW ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy) + ηW ./ (Δx .* Δy))

    DVy  = ones(ncx, ncy-1)
    @. DVy .= (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx = 1.0./DVx .* (Cxx .+ Cxy)
    λmaxVy = 1.0./DVy .* (Cyx .+ Cyy)

    return DVx, DVy, λmaxVx, λmaxVy
end

# 2D Stokes routine
@views function Stokes2D(n)

    sc = (
        L = 10e3,
        σ = 1e8,
        t = 1e10,
    )

    Ωη = 10.0
    
    # Physics
    Lx, Ly   = 4e3/sc.L, 2e3/sc.L     # domain size
    radi     = 1e2/sc.L               # inclusion radius
    ηs0      = 1e32/(sc.σ*sc.t)       # viscous viscosity 
    ηi       = 1e32/(sc.σ*sc.t)       # min/max inclusion viscosity
    εbg      = -5e-15*sc.t            # background strain-rate
    G0       = 1e10/(sc.σ)            # shear modulus of the solid
    Gi       = 0.25e10/(sc.σ)         # shear modulus of the solid inclusion
    Ks       = 1e11/(sc.σ)            # bulk modulus of the solid    
    KΦ       = 1e10/(sc.σ)            # bulk modulus of the porous matrix
    Kf       = 1e9/(sc.σ)             # bulk modulus of the fluid
    Φi       = 5e-2   
    k_ηf0    = 1e-15 / (sc.L^2/(sc.σ*sc.t))
    C        = 30e6/(sc.σ)
    ϕ        = 35.0 
    ηvp      = 5e18/(sc.σ*sc.t)             # viscous viscosity of the solid
    sinϕ     = sind(ϕ)
    cosϕ     = cosd(ϕ)
    Pt_ini   = 50e6/sc.σ
    Pf_ini   =  4e6/sc.σ
    # Numerics
    ncx, ncy = 2*n*32, n*32 # numerical grid resolution
    ϵ        = 1e-4         # tolerance
    iterMax  = 1e4          # max number of iters
    nout     = 200          # check frequency
    c_factV  = 0.99         # damping factor
    c_factPf = 0.99
    dτ_local = false        # helps a little bit sometimes, sometimes not! 
    γfact    = 1            # penalty: multiplier to the arithmetic mean of η
    rel_drop = 1e-3         # relative drop of velocity residual per PH iteration
    Δt       = 1e10/(sc.t)  # time step
    # Preprocessing
    Δx, Δy   = Lx/ncx, Ly/ncy
    # Array initialisation
    Φ        = Φi.*ones(ncx  ,ncy  )
    Φ0       = Φi.*ones(ncx  ,ncy  )
    dPtdt    = zeros(ncx  ,ncy  )
    dPfdt    = zeros(ncx  ,ncy  )
    dPsdt    = zeros(ncx  ,ncy  )
    dlnρfdt  = zeros(ncx  ,ncy  )
    dlnρsdt  = zeros(ncx  ,ncy  )
    dΦdt     = zeros(ncx  ,ncy  )
    ∇qD      = zeros(ncx  ,ncy  )
    qDx      = zeros(ncx+1,ncy  )
    qDy      = zeros(ncx  ,ncy+1)
    Pt       = zeros(ncx  ,ncy  )
    Pf       = zeros(ncx+2,ncy+2)
    Pt0      = zeros(ncx  ,ncy  )
    Pf0      = zeros(ncx+2,ncy+2)
    P̄f       = zeros(ncx+1,ncy+1)
    P̄t       = zeros(ncx+1,ncy+1)
    ∇Vc      = zeros(ncx  ,ncy  )
    ∇Vv      = zeros(ncx+1,ncy+1)
    Vx       = zeros(ncx+1,ncy+2) 
    V̄y       = zeros(ncx+1,ncy+2)
    V̄x       = zeros(ncx+2,ncy+1) 
    Vy       = zeros(ncx+2,ncy+1)
    Exxc     = zeros(ncx  ,ncy  )
    Eyyc     = zeros(ncx  ,ncy  )
    Exyc     = zeros(ncx  ,ncy  )
    Exxv     = zeros(ncx+1,ncy+1)
    Eyyv     = zeros(ncx+1,ncy+1)
    Exyv     = zeros(ncx+1,ncy+1)
    Txxc     = zeros(ncx  ,ncy  )
    Tyyc     = zeros(ncx  ,ncy  )
    Txyc     = zeros(ncx  ,ncy  )
    Txxc_PH  = zeros(ncx  ,ncy  )
    Tyyc_PH  = zeros(ncx  ,ncy  )
    Txxv     = zeros(ncx+1,ncy+1)
    Tyyv     = zeros(ncx+1,ncy+1)
    Txyv     = zeros(ncx+1,ncy+1)
    Txx0c    = zeros(ncx  ,ncy  )
    Tyy0c    = zeros(ncx  ,ncy  )
    Txy0c    = zeros(ncx  ,ncy  )
    Txx0v    = zeros(ncx+1,ncy+1)
    Tyy0v    = zeros(ncx+1,ncy+1)
    Txy0v    = zeros(ncx+1,ncy+1)
    τIIc     = zeros(ncx  ,ncy  )
    τIIv     = zeros(ncx+1,ncy+1)
    RVx      = zeros(ncx-1,ncy  )
    RVy      = zeros(ncx  ,ncy-1)
    RVx_PH   = zeros(ncx-1,ncy  )
    RVy_PH   = zeros(ncx  ,ncy-1)
    RPt      = zeros(ncx  ,ncy  )
    RPf      = zeros(ncx  ,ncy  )
    RVx_PH0  = zeros(ncx-1,ncy  )
    RVy_PH0  = zeros(ncx  ,ncy-1)
    RPf0     = zeros(ncx  ,ncy  )
    dVxdτ    = zeros(ncx-1,ncy  )
    dVydτ    = zeros(ncx  ,ncy-1)
    dPfdτ    = zeros(ncx  ,ncy  )
    βVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    βVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    βPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    cVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    cVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    cPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    αVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    αVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    αPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    dVx      = zeros(ncx-1,ncy  )  # Could be computed on the fly
    dVy      = zeros(ncx  ,ncy-1)  # Could be computed on the fly
    dPf      = zeros(ncx  ,ncy  )  # Could be computed on the fly
    KΦ       = zeros(ncx  ,ncy  )
    ηΦ       = zeros(ncx  ,ncy  )
    ηc       = zeros(ncx  ,ncy  )
    ηv       = zeros(ncx+1,ncy+1)
    ηvc      = zeros(ncx  ,ncy  )
    ηvv      = zeros(ncx+1,ncy+1)
    ηec      = G0*Δt .* ones(ncx  ,ncy  )
    ηev      = G0*Δt .* ones(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)
    fc       = zeros(ncx  ,ncy  )
    fv       = zeros(ncx+1,ncy+1)
    λ̇c       = zeros(ncx  ,ncy  )
    λ̇v       = zeros(ncx+1,ncy+1)
    EIIc     = zeros(ncx  ,ncy  )
    EIIv     = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    # Multiple circles with various viscosities
    ηi    = (w=1/ηi, s=ηi) 
    Gi    = (w=1/Gi, s=Gi) 
    x_inc = [0.0   0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1 ] 
    y_inc = [0.0   0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4 ]
    r_inc = [radi  0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07] 
    η_inc = [ηi.s  ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    G_inc = [Gi.s  Gi.w  Gi.w Gi.s Gi.w Gi.s Gi.w Gi.s Gi.s Gi.w]
    # Sharp inclusions
    ηv_sharp   .= ηs0
    for inc in 1:1#eachindex(η_inc)
        ηv_sharp[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηc_sharp   .= ηs0
    for inc in 1:1#eachindex(η_inc)
        ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end  
    # Harmonic averaging mimicking PIC interpolation
    ηvc    .= av4_harm(ηv_sharp)
    ηvv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    ηvv[[1,end],:] .= ηvv[[2,end-1],:]
    ηvv[:,[1,end]] .= ηvv[:,[2,end-1]]
    # Sharp inclusions
    ηv_sharp   .= G0
    for inc in 1:1#eachindex(η_inc)
        ηv_sharp[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= G_inc[inc]
    end
    ηc_sharp   .= G0
    for inc in 1:1#eachindex(η_inc)
        ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= G_inc[inc]
    end  
    # Harmonic averaging mimicking PIC interpolation
    ηec    .= av4_harm(ηv_sharp)*Δt
    ηev[2:end-1,2:end-1] .= av4_harm(ηc_sharp)*Δt
    ηev[[1,end],:] .= ηev[[2,end-1],:]
    ηev[:,[1,end]] .= ηev[:,[2,end-1]]
    # Visco-elasticity
    ηc .= 1 ./ (1 ./ηec .+ 1 ./ηvc)
    ηv .= 1 ./ (1 ./ηev .+ 1 ./ηvv)
    # Bulk viscosity
    ηΦ    .= ηvc.*Ωη
    # Select γ
    γi   = γfact*mean(ηc).*ones(size(ηc))
    # (Pseudo-)compressibility
    γ_eff = zeros(size(ηΦ)) 
    γ_num = γi.*ones(size(ηΦ)) * 1
    γ_phy = ηΦ.*(1 .- Φ)
    γ_eff = ((γ_phy.*γ_num)./(γ_phy.+γ_num))
    # Optimal pseudo-time steps - can be replaced by AD
    DVx, DVy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ_eff, Δx, Δy, ncx ,ncy)
    ηΦ    .= ηs0 ./ Φ
    KΦ    .= G0  ./ Φ
    ηeffΦ  =  1 ./ (1 ./ηΦ .+ 1 ./(KΦ*Δt))
    DPf    =  1 ./ηeffΦ./(1 .- Φ) .+ 2*k_ηf0*(1/Δx^2 + 1/Δy^2) 
    λmaxPf = (1 ./ηeffΦ./(1 .- Φ) .+ 4*k_ηf0*(1/Δx^2 + 1/Δy^2)) ./ DPf 
    # Select dτ
    if dτ_local
        dτVx = 2.0./sqrt.(λmaxVx)*0.99
        dτVy = 2.0./sqrt.(λmaxVy)*0.99
        dτPf = 2.0./sqrt.(λmaxPf)*0.99 
    else
        dτVx = 2.0./sqrt.(maximum(λmaxVx))*0.99 
        dτVy = 2.0./sqrt.(maximum(λmaxVy))*0.99
        dτPf = 2.0./sqrt.(maximum(λmaxPf))*0.99 
    end
    βVx   .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
    βVy   .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
    βPf   .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
    αVx   .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
    αVy   .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
    αPf   .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
    # Initial condition
    Vx     .=   εbg.*xv .+    0*yce'
    Vy     .=     0*xce .- εbg.*yv'
    Vx[2:end-1,:]       .= 0   # ensure non zero initial pressure residual
    Vy[:,2:end-1]       .= 0   # ensure non zero initial pressure residual
    Pf[2:end-1,2:end-1] .= Pf_ini
    Pt                  .= Pt_ini
    # Probes
    nt     =  40
    probes = (τc=zeros(nt), τv=zeros(nt), peffc=zeros(nt), peffv=zeros(nt), Φ=zeros(nt), Vx=zeros(nt), Vy=zeros(nt))
    # Time loop
    for it=1:nt
        Pt0 .= Pt
        Pf0 .= Pf
        Φ0  .= Φ
        Txx0c .= Txxc; Txx0v .= Txxv;
        Tyy0c .= Tyyc; Tyy0v .= Tyyv;
        Txy0v .= Txyv; Txy0c .= Txyc;
        # Iteration loop
        errVx0 = 1.0;  errVy0 = 1.0;  errPf0 = 1.0;   errPt0 = 1.0 
        errVx00= 1.0;  errVy00= 1.0;  errPf00= 1.0 
        iter=1; err_PT=2*ϵ; err_evo_V=[]; err_evo_Pt=[]; err_evo_Pf=[]; err_evo_it=[]; err_glob = 1.0
        @time for itPH = 1:50
            if (err_glob<ϵ) break end
            err_PT = 1.0
            # Set tolerance of velocity solve proportional to residual
            ϵ_vel = err_glob*rel_drop
            # ϵ_vel = 1e-3
            itPT = 0.
            while (err_PT>ϵ_vel && itPT<=iterMax)
                itPT     += 1
                itg      = iter
                # Pseudo-old dudes 
                RVx_PH0 .= RVx_PH
                RVy_PH0 .= RVy_PH
                RPf0    .= RPf
                # Always reset your plastic mulitpliers
                λ̇c      .= 0.0
                λ̇v      .= 0.0
                # Boundaries for primitive fields
                Vx[:,[1,end]] .= Vx[:,[2,end-1]]
                Vy[[1,end],:] .= Vy[[2,end-1],:]
                Pf[[1,end],:] .= Pf[[2,end-1],:]
                Pf[:,[1,end]] .= Pf[:,[2,end-1]]
                # Interpolate primitive fields
                V̄x[2:end-1,:]       .= av(Vx)
                V̄y[:,2:end-1]       .= av(Vy)
                P̄t[2:end-1,2:end-1] .= av(Pt)
                P̄f                  .= av(Pf)
                # Boundaries for interpolated fields
                V̄x[1,:]   .= 2*εbg*xv[1]   .- V̄x[2,:] 
                V̄x[end,:] .= 2*εbg*xv[end] .- V̄x[end-1,:] 
                V̄y[:,1]   .=-2*εbg*yv[1]   .- V̄y[:,2] 
                V̄y[:,end] .=-2*εbg*yv[end] .- V̄y[:,end-1] 
                P̄t[[1,end],:] .=  P̄t[[2,end-1],:]
                P̄t[:,[1,end]] .=  P̄t[:,[2,end-1]]
                # Darcy flux divergence
                qDx     .= -k_ηf0 .* diff(Pf[:,2:end-1], dims=1)/Δx
                qDy     .= -k_ηf0 .* diff(Pf[2:end-1,:], dims=2)/Δy
                ∇qD     .= diff(qDx, dims=1)/Δx .+ diff(qDy, dims=2)/Δy
                # Divergence 
                ∇Vc     .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
                ∇Vv     .= (V̄x[2:end,:]       .- V̄x[1:end-1,:]      )./Δx .+ (V̄y[:,2:end]       .- V̄y[:,1:end-1]      )./Δy
                # Deviatoric strain rate
                Exxc    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇Vc
                Eyyc    .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇Vc
                Exyc    .= 0.5.*((V̄x[2:end-1,2:end] .- V̄x[2:end-1,1:end-1])./Δy .+ (V̄y[2:end,2:end-1] .- V̄y[1:end-1,2:end-1])./Δx)
                Exxv    .= (V̄x[2:end,:] .- V̄x[1:end-1,:])./Δx .- 1.0/3.0.*∇Vv
                Eyyv    .= (V̄y[:,2:end] .- V̄y[:,1:end-1])./Δy .- 1.0/3.0.*∇Vv
                Exyv    .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
                # Two-phase stuff
                ηΦ      .= ηs0 ./ Φ
                KΦ      .= G0  ./ Φ
                dPtdt   .= (Pt .- Pt0) ./ Δt
                dPfdt   .= (Pf[2:end-1,2:end-1] .- Pf0[2:end-1,2:end-1]) ./ Δt
                dΦdt    .= (Pf[2:end-1,2:end-1] .- Pt)./ηΦ  + (dPfdt .- dPtdt)./KΦ 
                Φ       .= Φ0 .+ dΦdt.*Δt
                dPsdt   .= dΦdt .* (Pt .- Pf[2:end-1,2:end-1].*Φ)./(1 .- Φ).^2 .+ (dPtdt .- Φ.*dPfdt .- Pf[2:end-1,2:end-1].*dΦdt) ./ (1 .- Φ)
                dlnρsdt .= dPsdt ./ Ks
                dlnρfdt .= dPfdt ./ Kf
                RPt     .= .-(∇Vc .+ dlnρsdt - 1 ./(1 .- Φ) .* dΦdt )
                # Visco-elasticty
                ηc      .= 1 ./ (1 ./ηec .+ 1 ./ηvc)
                ηv      .= 1 ./ (1 ./ηev .+ 1 ./ηvv)
                # Deviatoric stress predictor
                Txxc    .= 2.0.*ηc.*(Exxc + Txx0c./(2.0.*ηec))  
                Tyyc    .= 2.0.*ηc.*(Eyyc + Tyy0c./(2.0.*ηec))  
                Txyc    .= 2.0.*ηc.*(Exyc + Txy0c./(2.0.*ηec))  
                Txxv    .= 2.0.*ηv.*(Exxv + Txx0v./(2.0.*ηev))  
                Tyyv    .= 2.0.*ηv.*(Eyyv + Tyy0v./(2.0.*ηev))  
                Txyv    .= 2.0.*ηv.*(Exyv + Txy0v./(2.0.*ηev)) 
                # Invariants    
                τIIc    .= sqrt.(0.5 .* (Txxc.^2 .+ Tyyc.^2 .+ 2.0 .* Txyc.^2))
                τIIv    .= sqrt.(0.5 .* (Txxv.^2 .+ Tyyv.^2 .+ 2.0 .* Txyv.^2))
                EIIc    .= sqrt.(0.5 .* ((Exxc + Txx0c./(2.0.*ηec)).^2 .+ (Eyyc + Tyy0c./(2.0.*ηec)) .^2 .+ 2.0 .* (Exyc + Txy0c./(2.0.*ηec)).^2))
                EIIv    .= sqrt.(0.5 .* ((Exxv + Txx0v./(2.0.*ηev)).^2 .+ (Eyyv + Tyy0v./(2.0.*ηev)) .^2 .+ 2.0 .* (Exyv + Txy0v./(2.0.*ηev)).^2))
                # Plasticity
                fc            .= τIIc .- C*cosϕ .- (Pt .- Pf[2:end-1,2:end-1])*sinϕ
                fv            .= τIIv .- C*cosϕ .- (P̄t .- P̄f                 )*sinϕ
                λ̇c[fc.>0.0]   .= fc[fc.>0.0] ./ (ηc[fc.>0.0] .+ ηvp)
                λ̇v[fv.>0.0]   .= fv[fv.>0.0] ./ (ηv[fv.>0.0] .+ ηvp)
                τIIc[fc.>0.0] .= τIIc[fc.>0.0] - λ̇c[fc.>0.0] .* ηc[fc.>0.0]
                τIIv[fv.>0.0] .= τIIv[fv.>0.0] - λ̇v[fv.>0.0] .* ηv[fv.>0.0]
                ηc[fc.>0.0]   .= τIIc[fc.>0.0] ./ (2.0 .* EIIc[fc.>0.0])
                ηv[fv.>0.0]   .= τIIv[fv.>0.0] ./ (2.0 .* EIIv[fv.>0.0])
                # Deviatoric stress corrector
                Txxc    .= 2.0.*ηc.*(Exxc + Txx0c./(2.0.*ηec))  
                Tyyc    .= 2.0.*ηc.*(Eyyc + Tyy0c./(2.0.*ηec))  
                Txyc    .= 2.0.*ηc.*(Exyc + Txy0c./(2.0.*ηec))  
                Txxv    .= 2.0.*ηv.*(Exxv + Txx0v./(2.0.*ηev))  
                Tyyv    .= 2.0.*ηv.*(Eyyv + Tyy0v./(2.0.*ηev))  
                Txyv    .= 2.0.*ηv.*(Exyv + Txy0v./(2.0.*ηev)) 
                # Numerics: Powell-Hestenes stress
                Txxc_PH .= Txxc .- γ_eff .* RPt # Includes numerical bulk viscosity  
                Tyyc_PH .= Tyyc .- γ_eff .* RPt # Includes numerical bulk viscosity 
                # Residuals
                RVx     .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txxc[2:end,:] .- Txxc[1:end-1,:])./Δx .+ (Txyv[2:end-1,2:end] .- Txyv[2:end-1,1:end-1])./Δy)
                RVy     .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyyc[:,2:end] .- Tyyc[:,1:end-1])./Δy .+ (Txyv[2:end,2:end-1] .- Txyv[1:end-1,2:end-1])./Δx)
                RVx_PH  .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txxc_PH[2:end,:] .- Txxc_PH[1:end-1,:])./Δx .+ (Txyv[2:end-1,2:end] .- Txyv[2:end-1,1:end-1])./Δy)
                RVy_PH  .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyyc_PH[:,2:end] .- Tyyc_PH[:,1:end-1])./Δy .+ (Txyv[2:end,2:end-1] .- Txyv[1:end-1,2:end-1])./Δx)
                RPf     .= .-(∇qD .+ Φ.*∇Vc .+ dΦdt .+ Φ.*dlnρfdt)
                # Damping-pongs + preconditioning
                dVxdτ   .= αVx.*dVxdτ .+ RVx_PH ./ DVx
                dVydτ   .= αVy.*dVydτ .+ RVy_PH ./ DVy
                dPfdτ   .= αPf.*dPfdτ .+ RPf    ./ DPf
                # PT updates
                Vx[2:end-1,2:end-1] .+= dVxdτ.*βVx.*dτVx 
                Vy[2:end-1,2:end-1] .+= dVydτ.*βVy.*dτVy 
                Pf[2:end-1,2:end-1] .+= dPfdτ.*βPf.*dτPf 
                # Residual check
                if mod(iter, nout)==0 || iter==1
                    # Error norms
                    errVx    = norm(RVx);    errVy     = norm(RVy);    errPf  =norm(RPf);    errPt  = norm(RPt)
                    errVx_PH = norm(RVx_PH); errVy_PH  = norm(RVy_PH); 
                    # For relative error metrics
                    if (itPH==1 && iter==1) errVx0  = errVx;    errVy0  = errVy;    errPf0  = errPf; errPt0=errPt; end
                    if iter==1              errVx00 = errVx_PH; errVy00 = errVy_PH; errPf00 = errPf;               end
                    # PT error
                    err_PT = maximum([errVx_PH./errVx00, errVy_PH./errVy00]) 
                    # Global error
                    eVx = min(errVx/sqrt(length(Vx)), errVx/errVx0)
                    eVy = min(errVy/sqrt(length(Vy)), errVy/errVy0)
                    ePt = min(errPt/sqrt(length(Pt)), errPt/errPt0)
                    ePf = min(errPf/sqrt(length(Pf)), errPf/errPf0)
                    err_glob = maximum([eVx, eVy, ePt, ePf]) 
                    # Invariants    
                    τIIc    .= sqrt.(0.5 .* (Txxc.^2 .+ Tyyc.^2 .+ 2.0 .* Txyc.^2))
                    τIIv    .= sqrt.(0.5 .* (Txxv.^2 .+ Tyyv.^2 .+ 2.0 .* Txyv.^2))
                    # Plasticity
                    fc            .= τIIc .- C*cosϕ .- (Pt .- Pf[2:end-1,2:end-1])*sinϕ .- ηvp.*λ̇c
                    fv            .= τIIv .- C*cosϕ .- (P̄t .- P̄f)*sinϕ                  .- ηvp.*λ̇v
                    @printf("---> itPH = %02d iter = %06d iter/nx = %03d, err PT = %1.3e, err glob = %1.3e norm[RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e] --- Fc=%1.3e, Fv=%1.3e\n", itPH, iter, iter/ncx, err_PT, err_glob, eVx, eVy, ePt, ePf, maximum(fc), maximum(fv))
                    # Store global errors
                    push!(err_evo_V, errVx/errVx00); push!(err_evo_Pt, errPt/errPt0); push!(err_evo_Pf, errPf/errPf0); push!(err_evo_it, itg)
                    dVx .= dVxdτ.*βVx.*dτVx
                    dVy .= dVydτ.*βVy.*dτVy
                    dPf .= dPfdτ.*βPf.*dτPf 
                    # Compute auto-PT parameters
                    λminV  = abs( sum(dVx.*(RVx_PH .- RVx_PH0)) + sum(dVy.*(RVy_PH .- RVy_PH0))  ) / (sum(dVx.*DVx.*dVx) .+ sum(dVy.*DVy.*dVy))
                    λminPf = abs( sum(dPf.*(RPf .- RPf0))) / sum(dPf.*DPf.*dPf)
                    cVx .= 2*sqrt.(λminV )*c_factV
                    cVy .= 2*sqrt.(λminV )*c_factV
                    cPf .= 2*sqrt.(λminPf)*c_factPf
                    βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
                    βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
                    βPf .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
                    αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
                    αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
                    αPf .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
                end
                iter += 1 
            end
            Pt                  .+= γ_eff.*RPt
        end
        @info "Post"
        probes.τc[it] = mean(τIIc)
        probes.τv[it] = mean(τIIv)
        probes.peffc[it] = mean(Pt.-Pf[2:end-1,2:end-1])
        probes.peffv[it] = mean(P̄t.-P̄f)
        EIIc    .= sqrt.(0.5 .* ((Exxc).^2 .+ (Eyyc) .^2 .+ 2.0 .* (Exyc).^2))

        # Plotting
        fig = Figure(size=(1200, 1000))
        ax  = Axis(fig[1, 1], title="λ̇", xlabel="x", ylabel="y", aspect=DataAspect()) 
        heatmap!(ax, xc*sc.L, yc*sc.L, log10.((λ̇c .+ 1e-13)/sc.t), colormap=:vik)
        ax  = Axis(fig[1, 2], title="Φ", xlabel="x", ylabel="y", aspect=DataAspect()) 
        heatmap!(ax, xc*sc.L, yc*sc.L, Φ, colormap=:vik)
         ax  = Axis(fig[2, 1], title="Pt", xlabel="x", ylabel="y", aspect=DataAspect()) 
        heatmap!(ax, xc*sc.L, yc*sc.L, Pt*sc.σ, colormap=:vik)
        ax  = Axis(fig[2, 2], title="Pf", xlabel="x", ylabel="y", aspect=DataAspect())        
        heatmap!(ax, xc*sc.L, yc*sc.L, Pf[2:end-1,2:end-1]*sc.σ, colormap=:vik)
        ax  = Axis(fig[3, 1], title="Convergence", xlabel="DR iterations / nx", ylabel="errors") 
        lines!(ax, err_evo_it/ncx, log10.(err_evo_V ), label="V" )
        lines!(ax, err_evo_it/ncx, log10.(err_evo_Pt), label="Pt")
        lines!(ax, err_evo_it/ncx, log10.(err_evo_Pf), label="Pf")
        lines!(ax, err_evo_it/ncx, log10.(ϵ.*ones(size(err_evo_it))), label="tol")  
        ax  = Axis(fig[3, 2], title="stress", xlabel="t", ylabel="τ",) 
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.τc[1:it]*sc.σ/1e6, label="τIIc")
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.τv[1:it]*sc.σ/1e6, label="τIIv")
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.peffc[1:it]*sc.σ/1e6, label="τIIc")
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.peffv[1:it]*sc.σ/1e6, label="τIIv")
        display(fig)
        @show iter/ncx
        n   = length(ηc)
        @show η_h = 1.0 / sum(1.0/n ./ηc)
        @show η_g = exp( sum( 1.0/n*log.(ηc)))
        @show η_a = mean(ηc)
    end

    # save("./results/twophases_n$(ncx).jld2", "err_evo_it", err_evo_it, "ncx", ncx, "err_evo_V", err_evo_V, "err_evo_Pt", err_evo_Pt, "err_evo_Pf", err_evo_Pf, "xc", xc, "yc", yc, "∇V", ∇V, "Pt", Pt, "Pf", Pf, "Vx", Vx, "Vy", Vy )

    return
end

n = 2
Stokes2D(n)
