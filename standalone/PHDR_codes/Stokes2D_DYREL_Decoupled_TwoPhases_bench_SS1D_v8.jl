# Initialisation
using CairoMakie, Printf, Statistics, LinearAlgebra, JLD2
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 

kyr = 1e3 * 365.25 * 24 * 3600

# Pad centroid field with one ghost layer: periodic or zero-gradient in x, zero-gradient in y
@views function pad_centroids!(Aext, A, periodic_x)
    Aext[2:end-1,2:end-1] .= A
    if periodic_x
        Aext[1  ,2:end-1] .= A[end,:]
        Aext[end,2:end-1] .= A[1  ,:]
    else
        Aext[1  ,2:end-1] .= A[1  ,:]
        Aext[end,2:end-1] .= A[end,:]
    end
    Aext[:,1  ] .= Aext[:,2    ]
    Aext[:,end] .= Aext[:,end-1]
    return Aext
end

# can be replaced by AD
function Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ, Δx, Δy, ncx  ,ncy; periodic_x=false, Vx_Dirichlet_NS=false)
    
    # Number of Vx unknowns in x: periodic -> the west column is an unknown too
    nVx   = periodic_x ? ncx : ncx-1
    ivx   = periodic_x ? (1:ncx) : (2:ncx)   # vertex indices of Vx unknowns
    ηN    = ones(nVx, ncy)
    ηS    = ones(nVx, ncy)
    ηN[:,1:end-1] .= ηv[ivx,2:end-1]
    ηS[:,2:end-0] .= ηv[ivx,2:end-1]
    if Vx_Dirichlet_NS # ghost Vx = 2*Vbc - Vx: doubles the boundary contribution
        ηN[:,end] .= 2 .* ηv[ivx,end]
        ηS[:,1  ] .= 2 .* ηv[ivx,1  ]
    end
    if periodic_x
        ηW    = [ηc[end:end,:]; ηc[1:end-1,:]]
        ηE    = ηc
        ebW   = [γ[end:end,:]; γ[1:end-1,:]]
        ebE   = γ
    else
        ηW    = ηc[1:end-1,:]
        ηE    = ηc[2:end-0,:]
        ebW   = γ[1:end-1,:] 
        ebE   = γ[2:end-0,:] 
    end
    Cxx   = ones(nVx, ncy)
    Cxy   = ones(nVx, ncy)
    @. Cxx = abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    @. Cxy = abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))

    DVx  = ones(nVx, ncy)
    @. DVx .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    if periodic_x
        ηE[end,:] .= ηv[end,2:end-1]
        ηW[1  ,:] .= ηv[1  ,2:end-1]
    end
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
@views function Stokes2D(n; config=:pure_shear, nt=40, visualization=true, m_anderson=0)

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
    # Kinematic boundary conditions
    if config == :pure_shear          # Vx = εbg*x, Vy = -εbg*y, free slip
        periodic = false
    elseif config == :simple_shear    # Vx = γ̇*y on N/S, periodic E/W
        periodic = true
    else
        error("Unknown config: $(config) (use :pure_shear or :simple_shear)")
    end
    γ̇        = 2*εbg                  # simple shear rate: same ε̇II as pure shear
    G0       = 1e10/(sc.σ)            # shear modulus of the solid
    Gi       = 1e10/(sc.σ)            # shear modulus of the solid inclusion (= G0: only cohesion differs)
    Ks       = 1e11/(sc.σ)            # bulk modulus of the solid    
    KΦ       = 1e10/(sc.σ)            # bulk modulus of the porous matrix
    Kf       = 1e9/(sc.σ)             # bulk modulus of the fluid
    Φi       = 5e-2   
    k_ηf0    = 1e-15 / (sc.L^2/(sc.σ*sc.t))
    C        = 30e6/(sc.σ)
    ϕ        = 0*35.0 
    ηvp      = 5e18/(sc.σ*sc.t)             # viscous viscosity of the solid
    sinϕ     = sind(ϕ)
    cosϕ     = cosd(ϕ)
    C0       = 30e6/(sc.σ)
    Ci       = 10e6/(sc.σ)
    Pt_ini   = 0.0
    Pf_ini   = 0.0
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
    nVxx     = periodic ? ncx : ncx-1 # number of Vx unknowns in x
    ivx      = periodic ? 1 : 2       # first Vx unknown in x
    irx      = periodic ? 2 : 1       # first row of RVx holding an interior Vx node
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
    Pt_ext   = zeros(ncx+2,ncy+2)
    # Anderson acceleration of the Powell-Hestenes pressure update (m_anderson = 0: plain PH)
    nP       = length(Pt)
    ΔP_AA    = zeros(nP, max(m_anderson, 1))
    ΔF_AA    = zeros(nP, max(m_anderson, 1))
    f_AA     = zeros(nP)
    P_prev   = zeros(nP)
    f_prev   = zeros(nP)
    nDR_tot, nPH_tot = 0, 0
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
    RVx      = zeros(nVxx ,ncy  )
    RVy      = zeros(ncx  ,ncy-1)
    RVx_PH   = zeros(nVxx ,ncy  )
    RVy_PH   = zeros(ncx  ,ncy-1)
    RPt      = zeros(ncx  ,ncy  )
    RPf      = zeros(ncx  ,ncy  )
    RVx_PH0  = zeros(nVxx ,ncy  )
    RVy_PH0  = zeros(ncx  ,ncy-1)
    RPf0     = zeros(ncx  ,ncy  )
    dVxdτ    = zeros(nVxx ,ncy  )
    dVydτ    = zeros(ncx  ,ncy-1)
    dPfdτ    = zeros(ncx  ,ncy  )
    βVx      = zeros(nVxx ,ncy  )  # this disappears is dτ is not local
    βVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    βPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    cVx      = zeros(nVxx ,ncy  )  # this disappears is dτ is not local
    cVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    cPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    αVx      = zeros(nVxx ,ncy  )  # this disappears is dτ is not local
    αVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    αPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    dVx      = zeros(nVxx ,ncy  )  # Could be computed on the fly
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
    Cc       = zeros(ncx  ,ncy  )
    Cv       = zeros(ncx+1,ncy+1)
    EIIc     = zeros(ncx  ,ncy  )
    EIIv     = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    # Multiple circles with various viscosities
    ηi    = (w=1/ηi, s=ηi) 
    Gi    = (w=1/Gi, s=Gi) 
    Ci    = (w=1/Ci, s=Ci)
    x_inc = [0.0   0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1 ] 
    y_inc = [0.0   0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4 ]
    r_inc = [radi  0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07] 
    η_inc = [ηi.s  ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    G_inc = [Gi.s  Gi.w  Gi.w Gi.s Gi.w Gi.s Gi.w Gi.s Gi.s Gi.w]
    C_inc = [Ci.s  Ci.w  Ci.w Ci.s Ci.w Ci.s Ci.w Ci.s Ci.s Ci.w]
    # Sharp inclusions
    ηv_sharp   .= ηs0
    for inc in 1:1#eachindex(η_inc)
        ηv_sharp[0*(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηc_sharp   .= ηs0
    for inc in 1:1#eachindex(η_inc)
        ηc_sharp[0*(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end  
    # Harmonic averaging mimicking PIC interpolation
    ηvc    .= av4_harm(ηv_sharp)
    if periodic
        ηvv .= av4_harm(pad_centroids!(zeros(ncx+2,ncy+2), ηc_sharp, true))
    else
        ηvv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
        ηvv[[1,end],:] .= ηvv[[2,end-1],:]
        ηvv[:,[1,end]] .= ηvv[:,[2,end-1]]
    end
    # Sharp inclusions
    ηv_sharp   .= G0
    for inc in 1:1#eachindex(η_inc)
        ηv_sharp[0*(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= G_inc[inc]
    end
    ηc_sharp   .= G0
    for inc in 1:1#eachindex(η_inc)
        ηc_sharp[0*(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= G_inc[inc]
    end  
    # Harmonic averaging mimicking PIC interpolation
    ηec    .= av4_harm(ηv_sharp)*Δt
    if periodic
        ηev .= av4_harm(pad_centroids!(zeros(ncx+2,ncy+2), ηc_sharp, true))*Δt
    else
        ηev[2:end-1,2:end-1] .= av4_harm(ηc_sharp)*Δt
        ηev[[1,end],:] .= ηev[[2,end-1],:]
        ηev[:,[1,end]] .= ηev[:,[2,end-1]]
    end
    # Sharp inclusions
    ηv_sharp   .= C0
    for inc in 1:1#eachindex(η_inc)
        ηv_sharp[0*(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= C_inc[inc]
    end
    ηc_sharp   .= C0
    for inc in 1:1#eachindex(η_inc)
        ηc_sharp[0*(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= C_inc[inc]
    end  
    # Harmonic averaging mimicking PIC interpolation
    Cc    .= av4_harm(ηv_sharp)
    if periodic
        Cv .= av4_harm(pad_centroids!(zeros(ncx+2,ncy+2), ηc_sharp, true))
    else
        Cv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
        Cv[[1,end],:] .= Cv[[2,end-1],:]
        Cv[:,[1,end]] .= Cv[:,[2,end-1]]
    end
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
    DVx, DVy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ_eff, Δx, Δy, ncx ,ncy; periodic_x=periodic, Vx_Dirichlet_NS=periodic)
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
    if periodic
        Vx .= 0*xv .+ γ̇.*yce'
        Vy .= 0.0               # Vy = 0 on N/S
    else
        Vx     .=   εbg.*xv .+    0*yce'
        Vy     .=     0*xce .- εbg.*yv'
        Vx[2:end-1,:]       .= 0   # ensure non zero initial pressure residual
        Vy[:,2:end-1]       .= 0   # ensure non zero initial pressure residual
    end
    Pf[2:end-1,2:end-1] .= Pf_ini
    Pf                  .= Pt_ini
    # Probes
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
        k_AA = 0; err_prev = Inf   # fresh Anderson history for each time step
        @time for itPH = 1:50
            if (err_glob<ϵ) break end
            nPH_tot += 1
            err_PT = 1.0
            # Set tolerance of velocity solve proportional to residual
            ϵ_vel = err_glob*rel_drop
            # ϵ_vel = 1e-3
            itPT = 0.
            # also leave once globally converged: a round-off initial residual makes the relative err_PT unreachable
            while (err_PT>ϵ_vel && err_glob>=ϵ && itPT<=iterMax)
                itPT     += 1
                itg      = iter
                # Pseudo-old dudes 
                RVx_PH0 .= RVx_PH
                RVy_PH0 .= RVy_PH
                RPf0    .= RPf
                # Always reset your plastic multipliers
                λ̇c      .= 0.0
                λ̇v      .= 0.0
                # Boundaries for primitive fields
                if periodic
                    Vx[end,:]     .= Vx[1,:]                          # E = W
                    Vx[:,1  ]     .= 2*γ̇*yv[1  ] .- Vx[:,2    ]      # Dirichlet tangent S
                    Vx[:,end]     .= 2*γ̇*yv[end] .- Vx[:,end-1]      # Dirichlet tangent N
                    Vy[1  ,:]     .= Vy[end-1,:]                      # periodic ghosts
                    Vy[end,:]     .= Vy[2    ,:]
                    Pf[1  ,:]     .= Pf[end-1,:]
                    Pf[end,:]     .= Pf[2    ,:]
                    Pf[:,[1,end]] .= Pf[:,[2,end-1]]
                else
                    Vx[:,[1,end]] .= Vx[:,[2,end-1]]
                    Vy[[1,end],:] .= Vy[[2,end-1],:]
                    Pf[[1,end],:] .= Pf[[2,end-1],:]
                    Pf[:,[1,end]] .= Pf[:,[2,end-1]]
                end
                # Interpolate primitive fields
                V̄x[2:end-1,:]       .= av(Vx)
                V̄y[:,2:end-1]       .= av(Vy)
                P̄f                  .= av(Pf)
                # Boundaries for interpolated fields
                if periodic
                    V̄x[1  ,:] .= V̄x[end-1,:]
                    V̄x[end,:] .= V̄x[2    ,:]
                    V̄y[:,1  ] .= .-V̄y[:,2    ]   # Vy = 0 on N/S
                    V̄y[:,end] .= .-V̄y[:,end-1]
                    P̄t        .= av(pad_centroids!(Pt_ext, Pt, true))
                else
                    V̄x[1,:]   .= 2*εbg*xv[1]   .- V̄x[2,:]
                    V̄x[end,:] .= 2*εbg*xv[end] .- V̄x[end-1,:]
                    V̄y[:,1]   .=-2*εbg*yv[1]   .- V̄y[:,2]
                    V̄y[:,end] .=-2*εbg*yv[end] .- V̄y[:,end-1]
                    P̄t[2:end-1,2:end-1] .= av(Pt)
                    P̄t[[1,end],:] .=  P̄t[[2,end-1],:]
                    P̄t[:,[1,end]] .=  P̄t[:,[2,end-1]]
                end
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
                fc            .= τIIc .- Cc .* cosϕ .- (Pt .- Pf[2:end-1,2:end-1])*sinϕ
                fv            .= τIIv .- Cv .* cosϕ .- (P̄t .- P̄f                 )*sinϕ
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
                RVx[irx:end,:] .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txxc[2:end,:] .- Txxc[1:end-1,:])./Δx .+ (Txyv[2:end-1,2:end] .- Txyv[2:end-1,1:end-1])./Δy)
                RVy     .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyyc[:,2:end] .- Tyyc[:,1:end-1])./Δy .+ (Txyv[2:end,2:end-1] .- Txyv[1:end-1,2:end-1])./Δx)
                RVx_PH[irx:end,:] .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txxc_PH[2:end,:] .- Txxc_PH[1:end-1,:])./Δx .+ (Txyv[2:end-1,2:end] .- Txyv[2:end-1,1:end-1])./Δy)
                if periodic # W/E Vx node: neighbours wrap around
                    RVx[1,:]    .= (.-(Pt[1,:] .- Pt[end,:])./Δx .+ (Txxc[1,:]    .- Txxc[end,:]   )./Δx .+ (Txyv[1,2:end] .- Txyv[1,1:end-1])./Δy)
                    RVx_PH[1,:] .= (.-(Pt[1,:] .- Pt[end,:])./Δx .+ (Txxc_PH[1,:] .- Txxc_PH[end,:])./Δx .+ (Txyv[1,2:end] .- Txyv[1,1:end-1])./Δy)
                end
                RVy_PH  .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyyc_PH[:,2:end] .- Tyyc_PH[:,1:end-1])./Δy .+ (Txyv[2:end,2:end-1] .- Txyv[1:end-1,2:end-1])./Δx)
                RPf     .= .-(∇qD .+ Φ.*∇Vc .+ dΦdt .+ Φ.*dlnρfdt)
                # Damping-pongs + preconditioning
                dVxdτ   .= αVx.*dVxdτ .+ RVx_PH ./ DVx
                dVydτ   .= αVy.*dVydτ .+ RVy_PH ./ DVy
                dPfdτ   .= αPf.*dPfdτ .+ RPf    ./ DPf
                # PT updates
                Vx[ivx:end-1,2:end-1] .+= dVxdτ.*βVx.*dτVx
                Vy[2:end-1,2:end-1] .+= dVydτ.*βVy.*dτVy 
                Pf[2:end-1,2:end-1] .+= dPfdτ.*βPf.*dτPf 
                # Residual check
                if mod(iter, nout)==0 || iter==1
                    # Error norms
                    errVx    = norm(RVx);    errVy     = norm(RVy);    errPf  =norm(RPf);    errPt  = norm(RPt)
                    errVx_PH = norm(RVx_PH); errVy_PH  = norm(RVy_PH); 
                    # For relative error metrics
                    # (a zero initial residual, e.g. divergence-free initial flow, falls back to absolute errors)
                    nz(e) = e > 0 ? e : 1.0
                    if (itPH==1 && iter==1) errVx0  = nz(errVx);    errVy0  = nz(errVy);    errPf0  = nz(errPf); errPt0=nz(errPt); end
                    if iter==1              errVx00 = nz(errVx_PH); errVy00 = nz(errVy_PH); errPf00 = nz(errPf);                   end
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
                    fc            .= τIIc .- Cc .* cosϕ .- (Pt .- Pf[2:end-1,2:end-1])*sinϕ .- ηvp.*λ̇c
                    fv            .= τIIv .- Cv .* cosϕ .- (P̄t .- P̄f)*sinϕ                  .- ηvp.*λ̇v
                    @printf("---> itPH = %02d iter = %06d iter/nx = %03d, err PT = %1.3e, err glob = %1.3e norm[RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e] --- Fc=%1.3e, Fv=%1.3e\n", itPH, iter, iter/ncx, err_PT, err_glob, eVx, eVy, ePt, ePf, maximum(fc), maximum(fv))
                    # Store global errors
                    push!(err_evo_V, errVx/errVx00); push!(err_evo_Pt, errPt/errPt0); push!(err_evo_Pf, errPf/errPf0); push!(err_evo_it, itg)
                    dVx .= dVxdτ.*βVx.*dτVx
                    dVy .= dVydτ.*βVy.*dτVy
                    dPf .= dPfdτ.*βPf.*dτPf 
                    # Compute auto-PT parameters
                    # (skipped while a field has not moved yet: 0/0)
                    denV   = sum(dVx.*DVx.*dVx) + sum(dVy.*DVy.*dVy)
                    denPf  = sum(dPf.*DPf.*dPf)
                    if denV > 0
                        λminV  = abs( sum(dVx.*(RVx_PH .- RVx_PH0)) + sum(dVy.*(RVy_PH .- RVy_PH0))  ) / denV
                        cVx .= 2*sqrt.(λminV )*c_factV
                        cVy .= 2*sqrt.(λminV )*c_factV
                    end
                    if denPf > 0
                        λminPf = abs( sum(dPf.*(RPf .- RPf0))) / denPf
                        cPf .= 2*sqrt.(λminPf)*c_factPf
                    end
                    βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
                    βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
                    βPf .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
                    αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
                    αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
                    αPf .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
                end
                iter += 1 
            end
            # Powell-Hestenes update as a fixed-point map: Pt <- Pt + f, f = γ_eff*RPt
            f_AA .= vec(γ_eff.*RPt)
            # if the error grew since the last PH step, reset the history (avoid extrapolating in a wrong direction)
            k_AA = err_glob > err_prev ? 1 : k_AA + 1
            err_prev = err_glob
            if k_AA > 1 && m_anderson > 0
                slot = (k_AA - 2) % m_anderson + 1 # oldest slot gets replaced
                ΔP_AA[:, slot] .= vec(Pt) .- P_prev
                ΔF_AA[:, slot] .= f_AA    .- f_prev
            end
            P_prev .= vec(Pt)
            f_prev .= f_AA
            # plain PH step
            Pt                  .+= γ_eff.*RPt
            # Anderson correction
            n_AA = min(k_AA - 1, m_anderson)
            if n_AA > 0
                θ = ΔF_AA[:, 1:n_AA] \ f_AA # least squares: θ = argmin ‖f − ΔF⋅θ‖
                if all(isfinite, θ)
                    Pt .-= reshape((ΔP_AA[:, 1:n_AA] .+ ΔF_AA[:, 1:n_AA]) * θ, size(Pt))
                else
                    k_AA = 1 # degenerate history: keep the plain PH step and restart
                end
            end
        end
        nDR_tot += iter - 1
        @info "Post"
        probes.τc[it] = mean(τIIc)
        probes.τv[it] = mean(τIIv)
        probes.peffc[it] = mean(Pt.-Pf[2:end-1,2:end-1])
        probes.peffv[it] = mean(P̄t.-P̄f)
        EIIc    .= sqrt.(0.5 .* ((Exxc).^2 .+ (Eyyc) .^2 .+ 2.0 .* (Exyc).^2))

        # Plotting
        if visualization
        # Vertical profiles averaged along x (periodic: last column duplicates the first)
        ix     = periodic ? (1:ncx) : (1:ncx+1)
        cmyr   = sc.L/sc.t * 100 * 365.25*24*3600            # velocity scale -> cm/yr
        Vx_y   = vec(mean(Vx[ix, 2:end-1], dims=1))          # at yc
        Txy_y  = vec(mean(Txyv[ix, :],     dims=1))          # at yv
        fig = Figure(size=(1200, 1000))
        ax  = Axis(fig[1, 1], title="⟨Vx⟩ₓ", xlabel="Vx [cm/yr]", ylabel="y [m]")
        lines!(ax, Vx_y*cmyr, yc*sc.L, label="numerics")
        periodic && lines!(ax, γ̇.*yc*cmyr, yc*sc.L, linestyle=:dash, color=:black, label="γ̇y")
        axislegend(ax, position=:lt)
        ax  = Axis(fig[1, 2], title="⟨τxy⟩ₓ", xlabel="τxy [MPa]", ylabel="y [m]")
        lines!(ax, Txy_y*sc.σ/1e6, yv*sc.L)
        xlims!(ax, -11, -10)
        ax  = Axis(fig[2, 1], title="Convergence", xlabel="DR iterations / nx", ylabel="errors")
        lines!(ax, err_evo_it/ncx, log10.(err_evo_V ), label="V" )
        lines!(ax, err_evo_it/ncx, log10.(err_evo_Pt), label="Pt")
        lines!(ax, err_evo_it/ncx, log10.(err_evo_Pf), label="Pf")
        lines!(ax, err_evo_it/ncx, log10.(ϵ.*ones(size(err_evo_it))), label="tol")
        axislegend(ax, position=:rt)
        ax  = Axis(fig[2, 2], title="stress", xlabel="t [kyr]", ylabel="[MPa]",)
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.τc[1:it]*sc.σ/1e6, label="τII c")
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.τv[1:it]*sc.σ/1e6, label="τII v")
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.peffc[1:it]*sc.σ/1e6, label="Pe c")
        lines!(ax, (1:it).*Δt*sc.t/kyr, probes.peffv[1:it]*sc.σ/1e6, label="Pe v")
        axislegend(ax, position=:lt)
        display(fig)
        end
        @show iter/ncx
        n   = length(ηc)
        @show η_h = 1.0 / sum(1.0/n ./ηc)
        @show η_g = exp( sum( 1.0/n*log.(ηc)))
        @show η_a = mean(ηc)
    end

    # save("./results/twophases_n$(ncx).jld2", "err_evo_it", err_evo_it, "ncx", ncx, "err_evo_V", err_evo_V, "err_evo_Pt", err_evo_Pt, "err_evo_Pf", err_evo_Pf, "xc", xc, "yc", yc, "∇V", ∇V, "Pt", Pt, "Pf", Pf, "Vx", Vx, "Vy", Vy )

    @printf("Total: %d DR iterations (%.1f per nx), %d PH iterations, m_anderson = %d\n", nDR_tot, nDR_tot/ncx, nPH_tot, m_anderson)
    return (nDR=nDR_tot, nPH=nPH_tot)
end

n = 2
# Stokes2D(n; config=:pure_shear,   m_anderson=3)
Stokes2D(n; config=:simple_shear, m_anderson=3)   # m_anderson=0: plain Powell-Hestenes
