# Two-phase (HM) visco-elasto-viscoplastic DYREL solver, periodic simple shear or pure shear.
# v11 = Julia port of C_TD.m, which itself is based on _bench_SS_v8.jl. Changes w.r.t. SS_v8:
#   - Porosity from the solid mass balance in conservative form (Schmalholz et al. 2023, eq. A2),
#     updated in pseudo-time with dτ_Φ = θΦ*Δt
#   - Pf from the total mass balance in conservative form, preconditioned by subtracting (1-Φ)*RPt
#   - Compressible densities ρf(Pf), ρs(Ps); solid pressure Ps = (Pt - Φ Pf)/(1-Φ)
#   - Porosity-dependent permeability (Kozeny-Carman, n_KC) with log-space relaxation Φ_rel,
#     reference permeability from a compaction length
#   - Bulk viscosity/modulus: ηΦ = Ωη ηv/Φ, KΦ = Ωη G/Φ
#   - Upwind advection (material derivatives) of Pt, Pf, Ps and stresses; advection of C and ηv after each step
#   - Out-of-plane components (Ezz, Tzz) in the invariants
#   - Pf preconditioner/pseudo-time step and Gershgorin velocity preconditioner recomputed every iteration
#   - Elliptical (rotatable) inclusion, accumulated strain, saving/restart
# Advection: periodic in x for :simple_shear, one-sided differences at W/E for :pure_shear (both: one-sided at S/N)
# Performance: all work arrays are preallocated, the PT loop does not allocate (except at residual checks)
using CairoMakie, Printf, Statistics, LinearAlgebra, JLD2
# Macros
# av, av_xa, av_ya are lazy (Broadcasted objects, no array is created): `B .= av(A)` fuses into B without allocating
@views    av(A) = Broadcast.broadcasted((a, b, c, d) -> 0.25*(a + b + c + d), A[1:end-1,1:end-1], A[2:end,1:end-1], A[1:end-1,2:end], A[2:end,2:end])
@views av_xa(A) = Broadcast.broadcasted((a, b) -> 0.5*(a + b), A[1:end-1,:], A[2:end,:])
@views av_ya(A) = Broadcast.broadcasted((a, b) -> 0.5*(a + b), A[:,1:end-1], A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) )

kyr = 1e3 * 365.25 * 24 * 3600

# Weighted squared norm Σ D a², non-allocating
wnorm2(a, D) = sum(i -> D[i]*a[i]^2, eachindex(a, D))

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

# Arithmetic average of a centroid field (ncx,ncy) to x-faces (ncx+1,ncy) and y-faces (ncx,ncy+1)
# Periodic or zero-gradient in x, zero-gradient in y. Aext: work array (ncx+2,ncy+2)
@views function cell_to_faces!(Ax, Ay, Aext, A, periodic_x)
    pad_centroids!(Aext, A, periodic_x)
    Ax .= av_xa(Aext[:,2:end-1])
    Ay .= av_ya(Aext[2:end-1,:])
    return nothing
end

# Centroids -> x-faces, periodic or zero-gradient in x
@views function centroid_to_xface!(B, A, periodic_x)
    B[2:end-1,:] .= av_xa(A)
    if periodic_x
        B[1  ,:] .= 0.5 .* (A[end,:] .+ A[1,:])
        B[end,:] .= B[1,:]
    else
        B[1  ,:] .= A[1  ,:]
        B[end,:] .= A[end,:]
    end
    return B
end

# Centroids -> y-faces, linear extrapolation at the y boundaries
@views function centroid_to_yface!(B, A)
    B[:,2:end-1] .= av_ya(A)
    B[:,1  ]     .= 1.5 .* A[:,1  ] .- 0.5 .* A[:,2    ]
    B[:,end]     .= 1.5 .* A[:,end] .- 0.5 .* A[:,end-1]
    return B
end

# Upwind scalar transport out = U*dA/dx + W*dA/dy (all arrays collocated, same size)
# x periodic (periodic_x = true) or non-periodic; y non-periodic.
# Non-periodic ends use the one available one-sided derivative for either velocity sign
# (outflow: proper upwind; inflow: zero-order extrapolation of the inflowing value).
# Periodic vertex fields carry a duplicate column (A[end,:] = A[1,:]), which is excluded and restored.
function adv_upwind!(out, U, W, A, Δx, Δy, isvertex, periodic_x)
    nx  = (isvertex && periodic_x) ? size(A,1)-1 : size(A,1)
    ny  = size(A,2)
    _Δx, _Δy = 1/Δx, 1/Δy
    @inbounds for j in 1:ny, i in 1:nx
        if periodic_x
            iW = i == 1  ? nx : i-1
            iE = i == nx ? 1  : i+1
            dxb = (A[i ,j] - A[iW,j])*_Δx
            dxf = (A[iE,j] - A[i ,j])*_Δx
        else
            dxb = i > 1  ? (A[i  ,j] - A[i-1,j])*_Δx : (A[2 ,j] - A[1   ,j])*_Δx
            dxf = i < nx ? (A[i+1,j] - A[i  ,j])*_Δx : (A[nx,j] - A[nx-1,j])*_Δx
        end
        dyb = j > 1  ? (A[i,j  ] - A[i,j-1])*_Δy : (A[i,2 ] - A[i,1   ])*_Δy
        dyf = j < ny ? (A[i,j+1] - A[i,j  ])*_Δy : (A[i,ny] - A[i,ny-1])*_Δy
        u, w = U[i,j], W[i,j]
        out[i,j] = max(u,0.0)*dxb + min(u,0.0)*dxf + max(w,0.0)*dyb + min(w,0.0)*dyf
    end
    if isvertex && periodic_x
        @inbounds for j in 1:ny
            out[end,j] = out[1,j]
        end
    end
    return out
end

# Gershgorin estimates (diagonal preconditioner + max eigenvalue) of the Stokes operator with penalty γ, in-place
# can be replaced by AD
function Gershgorin_Stokes2D_SchurComplement!(DVx, DVy, λmaxVx, λmaxVy, ηc, ηv, γ, Δx, Δy; periodic_x=false, Vx_Dirichlet_NS=false)
    ncx, ncy = size(ηc)
    nVx      = size(DVx, 1)   # periodic: ncx (west column is an unknown too), else ncx-1
    _Δx2, _Δy2, _ΔxΔy = 1/Δx^2, 1/Δy^2, 1/(Δx*Δy)
    @inbounds for j in 1:ncy, i in 1:nVx
        iv  = periodic_x ? i : i+1                     # vertex index of the Vx unknown
        # ghost Vx = 2*Vbc - Vx: doubles the boundary contribution
        ηN  = j < ncy ? ηv[iv,j+1] : (Vx_Dirichlet_NS ? 2*ηv[iv,end] : 1.0)
        ηS  = j > 1   ? ηv[iv,j  ] : (Vx_Dirichlet_NS ? 2*ηv[iv,1  ] : 1.0)
        iW, iE = periodic_x ? (i == 1 ? ncx : i-1, i) : (i, i+1)
        ηW, ηE, ebW, ebE = ηc[iW,j], ηc[iE,j], γ[iW,j], γ[iE,j]
        D   = (ηN + ηS)*_Δy2 + (ebE + ebW + 4/3*(ηE + ηW))*_Δx2
        Cxx = abs(ηN*_Δy2) + abs(ηS*_Δy2) + abs((ebE + 4/3*ηE)*_Δx2) + abs((ebW + 4/3*ηW)*_Δx2) + abs(D)
        Cxy = (abs(ebE - 2/3*ηE + ηN) + abs(ebE - 2/3*ηE + ηS) + abs(ebW + ηN - 2/3*ηW) + abs(ebW + ηS - 2/3*ηW))*_ΔxΔy
        DVx[i,j]    = D
        λmaxVx[i,j] = (Cxx + Cxy)/D
    end
    @inbounds for j in 1:ncy-1, i in 1:ncx
        ηE  = (periodic_x || i < ncx) ? ηv[i+1,j+1] : 1.0
        ηW  = (periodic_x || i > 1  ) ? ηv[i  ,j+1] : 1.0
        ηS, ηN, ebS, ebN = ηc[i,j], ηc[i,j+1], γ[i,j], γ[i,j+1]
        D   = (ηE + ηW)*_Δx2 + (ebN + ebS + 4/3*(ηN + ηS))*_Δy2
        Cyy = abs(ηE*_Δx2) + abs(ηW*_Δx2) + abs((ebN + 4/3*ηN)*_Δy2) + abs((ebS + 4/3*ηS)*_Δy2) + abs(D)
        Cyx = (abs(ebN + ηE - 2/3*ηN) + abs(ebN - 2/3*ηN + ηW) + abs(ebS + ηE - 2/3*ηS) + abs(ebS - 2/3*ηS + ηW))*_ΔxΔy
        DVy[i,j]    = D
        λmaxVy[i,j] = (Cyx + Cyy)/D
    end
    return nothing
end

# Rotated ellipse: ((x'/a)^2 + y'^2 < r^2), rotation angle θ (degrees)
in_ellipse(x, y, x0, y0, r, θ, a) = ((x-x0)*cosd(θ) + (y-y0)*sind(θ))^2/a^2 + (-(x-x0)*sind(θ) + (y-y0)*cosd(θ))^2 < r^2

# Sharp phase field on vertices (Av) and centroids (Ac)
function set_phase!(Av, Ac, val0, val_inc, xv, yv, xc, yc, inc_geo; ninc=1)
    Av .= val0
    Ac .= val0
    for inc in 1:ninc
        (; x, y, r, θ, a) = inc_geo
        Av[in_ellipse.(xv, yv', x[inc], y[inc], r[inc], θ, a)] .= val_inc[inc]
        Ac[in_ellipse.(xc, yc', x[inc], y[inc], r[inc], θ, a)] .= val_inc[inc]
    end
end

# Harmonic averaging mimicking PIC interpolation: vertex values -> centroids, centroid values -> vertices
function sharp_to_grid!(Bc, Bv, Av, Ac, periodic)
    ncx, ncy = size(Ac)
    Bc .= av4_harm(Av)
    if periodic
        Bv .= av4_harm(pad_centroids!(zeros(ncx+2,ncy+2), Ac, true))
    else
        Bv[2:end-1,2:end-1] .= av4_harm(Ac)
        Bv[[1,end],:] .= Bv[[2,end-1],:]
        Bv[:,[1,end]] .= Bv[:,[2,end-1]]
    end
end

# 2D Stokes routine
@views function Stokes2D(n; config=:simple_shear, nt=5000, visualization=true, m_anderson=3,
                            save_data=true, nsave=5, out_dir=joinpath(@__DIR__, "Output_NEW_v01"), restart_file=nothing)

    sc = (
        L = 1e3,      # characteristic length
        σ = 1e8,      # characteristic stress
        t = 1e10,     # characteristic time
        ρ = 1e3,      # characteristic density
    )

    Ωη = 1.0

    # Physics
    Lx, Ly   = 2e2/sc.L, 1e2/sc.L     # domain size
    radi     = 1e1/sc.L               # inclusion radius
    l_c      = 2.0*radi               # compaction length (sets the reference permeability)
    ηs0      = 1e21/(sc.σ*sc.t)       # viscous viscosity
    ηi       = 0.5e21/(sc.σ*sc.t)     # inclusion viscosity
    εbg      = 2.5e-14*sc.t           # background strain-rate
    # Kinematic boundary conditions
    if config == :pure_shear          # Vx = εbg*x, Vy = -εbg*y, free slip
        periodic = false
    elseif config == :simple_shear    # Vx = γ̇*y on N/S, periodic E/W
        periodic = true
    else
        error("Unknown config: $(config) (use :pure_shear or :simple_shear)")
    end
    advection = true                  # upwind advection of pressures, stresses and material properties
    γ̇        = 2*εbg                  # simple shear rate: same ε̇II as pure shear
    G0       = 1e10/(sc.σ)            # shear modulus of the solid
    Gi       = 1e10/(sc.σ)            # shear modulus of the solid inclusion
    Ks       = 1e11/(sc.σ)            # bulk modulus of the solid
    Kf       = 1e9/(sc.σ)             # bulk modulus of the fluid
    Φi       = 2e-2                   # initial porosity
    n_KC     = 3                      # Kozeny-Carman exponent
    ρf0      = 1000/sc.ρ              # reference fluid density
    ρs0      = 3000/sc.ρ              # reference solid density
    Cs0      = 0.9                    # concentration of the immobile solid component
    ϕ        = 0*35.0
    ηvp      = 5e17/(sc.σ*sc.t)       # viscoplastic regularisation viscosity
    sinϕ     = sind(ϕ)
    cosϕ     = cosd(ϕ)
    C0       = 40e6/(sc.σ)
    Ci       = 20e6/(sc.σ)
    Pt_ini   = 0.0
    Pf_ini   = 0.0
    # Inclusion geometry: ellipse ((x'/aspect)^2 + y'^2 < r^2), rotated by incl_angle
    incl_angle  = 0.0
    incl_aspect = 3.0
    # Numerics
    ncx, ncy = 10*n*32, 5*n*32 # numerical grid resolution
    ϵ        = 1e-3         # tolerance
    iterMax  = 1e5          # max number of iters
    nout     = 500          # check frequency
    c_factV  = 0.99         # damping factor
    c_factPf = 0.99
    dτ_local = false        # helps a little bit sometimes, sometimes not!
    γfact    = 1            # penalty: multiplier to the arithmetic mean of η
    rel_drop = 1e-3         # relative drop of velocity residual per PH iteration
    Φ_rel    = 0.1          # relaxation of the permeability update (in log space)
    θΦ       = 1.0          # porosity PT step dτ_Φ = θΦ*Δt (Schmalholz et al. 2023, eq. A3)
    Δt       = 0.75e10/(sc.t)  # time step
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
    RΦ       = zeros(ncx  ,ncy  )
    qDx      = zeros(ncx+1,ncy  )
    qDy      = zeros(ncx  ,ncy+1)
    Pt       = zeros(ncx  ,ncy  )
    Pt_ext   = zeros(ncx+2,ncy+2)
    Aext     = zeros(ncx+2,ncy+2)   # work array for face averages
    Ps       = zeros(ncx  ,ncy  )
    Ps0      = zeros(ncx  ,ncy  )
    # Reference pressures of the density laws
    Pf_ref   = zeros(ncx+2,ncy+2)
    Ps_ref   = zeros(ncx  ,ncy  )
    # Densities, solid concentration and mass balance work arrays
    Cs       = Cs0 .* ones(ncx,ncy)
    ρs       = ρs0 .* ones(ncx,ncy)
    ρf       = ρf0 .* ones(ncx+2,ncy+2)
    ρt       = zeros(ncx  ,ncy  )
    ρt_old   = zeros(ncx  ,ncy  )
    Csρs     = zeros(ncx  ,ncy  )
    CsρsΦ_old= zeros(ncx  ,ncy  )
    ρf_x     = zeros(ncx+1,ncy  )
    ρf_y     = zeros(ncx  ,ncy+1)
    ρt_x     = zeros(ncx+1,ncy  )
    ρt_y     = zeros(ncx  ,ncy+1)
    Csρs_x   = zeros(ncx+1,ncy  )
    Csρs_y   = zeros(ncx  ,ncy+1)
    Φ_fx     = zeros(ncx+1,ncy  )
    Φ_fy     = zeros(ncx  ,ncy+1)
    FΦx      = zeros(ncx+1,ncy  )   # solid mass fluxes
    FΦy      = zeros(ncx  ,ncy+1)
    Qfx      = zeros(ncx+1,ncy  )   # total mass fluxes
    Qfy      = zeros(ncx  ,ncy+1)
    # Permeability
    Φ_kx     = zeros(ncx+1,ncy  )
    Φ_ky     = zeros(ncx  ,ncy+1)
    Φ_kx_ini = zeros(ncx+1,ncy  )
    Φ_ky_ini = zeros(ncx  ,ncy+1)
    kηf_x    = zeros(ncx+1,ncy  )
    kηf_y    = zeros(ncx  ,ncy+1)
    # Advection work arrays
    Vxc      = zeros(ncx  ,ncy  )
    Vyc      = zeros(ncx  ,ncy  )
    Vxv      = zeros(ncx+1,ncy+1)
    Vyv      = zeros(ncx+1,ncy+1)
    adv_Pt   = zeros(ncx  ,ncy  )
    adv_Pf   = zeros(ncx  ,ncy  )
    adv_Ps   = zeros(ncx  ,ncy  )
    advc     = zeros(ncx  ,ncy  )
    advv     = zeros(ncx+1,ncy+1)
    TxxHistC = zeros(ncx  ,ncy  )
    TyyHistC = zeros(ncx  ,ncy  )
    TzzHistC = zeros(ncx  ,ncy  )
    TxyHistC = zeros(ncx  ,ncy  )
    TxxHistV = zeros(ncx+1,ncy+1)
    TyyHistV = zeros(ncx+1,ncy+1)
    TzzHistV = zeros(ncx+1,ncy+1)
    TxyHistV = zeros(ncx+1,ncy+1)
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
    Ezzc     = zeros(ncx  ,ncy  )
    Exyc     = zeros(ncx  ,ncy  )
    Exxv     = zeros(ncx+1,ncy+1)
    Eyyv     = zeros(ncx+1,ncy+1)
    Ezzv     = zeros(ncx+1,ncy+1)
    Exyv     = zeros(ncx+1,ncy+1)
    EIIv_str = zeros(ncx+1,ncy+1)
    Txxc     = zeros(ncx  ,ncy  )
    Tyyc     = zeros(ncx  ,ncy  )
    Tzzc     = zeros(ncx  ,ncy  )
    Txyc     = zeros(ncx  ,ncy  )
    Txxc_PH  = zeros(ncx  ,ncy  )
    Tyyc_PH  = zeros(ncx  ,ncy  )
    Txxv     = zeros(ncx+1,ncy+1)
    Tyyv     = zeros(ncx+1,ncy+1)
    Tzzv     = zeros(ncx+1,ncy+1)
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
    RPf_cons = zeros(ncx  ,ncy  )
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
    dτVx     = zeros(nVxx ,ncy  )  # always arrays (uniform if !dτ_local): keeps the code type-stable
    dτVy     = zeros(ncx  ,ncy-1)
    dτPf     = zeros(ncx  ,ncy  )
    dVx      = zeros(nVxx ,ncy  )  # Could be computed on the fly
    dVy      = zeros(ncx  ,ncy-1)  # Could be computed on the fly
    dPf      = zeros(ncx  ,ncy  )  # Could be computed on the fly
    # Preconditioners
    DVx      = zeros(nVxx ,ncy  )
    DVy      = zeros(ncx  ,ncy-1)
    λmaxVx   = zeros(nVxx ,ncy  )
    λmaxVy   = zeros(ncx  ,ncy-1)
    DPf      = zeros(ncx  ,ncy  )
    λmaxPf   = zeros(ncx  ,ncy  )
    aPf      = zeros(ncx  ,ncy  )
    kxy      = zeros(ncx  ,ncy  )
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
    strain_acc = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    # Multiple inclusions (only the first one is active)
    ηi_ws = (w=1/ηi, s=ηi)
    Gi_ws = (w=1/Gi, s=Gi)
    Ci_ws = (w=1/Ci, s=Ci)
    inc_geo = (
        x = [0.0   0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1 ],
        y = [0.0   0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4 ],
        r = [radi  0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07],
        θ = incl_angle,
        a = incl_aspect,
    )
    η_inc = [ηi_ws.s  ηi_ws.w  ηi_ws.w ηi_ws.s ηi_ws.w ηi_ws.s ηi_ws.w ηi_ws.s ηi_ws.s ηi_ws.w]
    G_inc = [Gi_ws.s  Gi_ws.w  Gi_ws.w Gi_ws.s Gi_ws.w Gi_ws.s Gi_ws.w Gi_ws.s Gi_ws.s Gi_ws.w]
    C_inc = [Ci_ws.s  Ci_ws.w  Ci_ws.w Ci_ws.s Ci_ws.w Ci_ws.s Ci_ws.w Ci_ws.s Ci_ws.s Ci_ws.w]
    # Viscosity
    set_phase!(ηv_sharp, ηc_sharp, ηs0, η_inc, xv, yv, xc, yc, inc_geo)
    sharp_to_grid!(ηvc, ηvv, ηv_sharp, ηc_sharp, periodic)
    # Shear modulus -> elastic viscosity G*Δt
    set_phase!(ηv_sharp, ηc_sharp, G0, G_inc, xv, yv, xc, yc, inc_geo)
    sharp_to_grid!(ηec, ηev, ηv_sharp, ηc_sharp, periodic)
    ηec .*= Δt
    ηev .*= Δt
    # Cohesion
    set_phase!(ηv_sharp, ηc_sharp, C0, C_inc, xv, yv, xc, yc, inc_geo)
    sharp_to_grid!(Cc, Cv, ηv_sharp, ηc_sharp, periodic)
    # Visco-elasticity
    ηc .= 1 ./ (1 ./ηec .+ 1 ./ηvc)
    ηv .= 1 ./ (1 ./ηev .+ 1 ./ηvv)
    # Bulk viscosity and modulus of the porous matrix
    Gc     = ηec ./ Δt
    ηΦ    .= Ωη .* ηvc ./ Φ
    KΦ    .= Ωη .* Gc  ./ Φ
    # Select γ
    γi   = γfact*mean(ηc).*ones(size(ηc))
    # (Pseudo-)compressibility
    γ_num = γi.*ones(size(ηΦ)) * 1
    γ_phy = ηΦ.*(1 .- Φ)
    γ_eff = ((γ_phy.*γ_num)./(γ_phy.+γ_num))
    # Optimal pseudo-time steps - can be replaced by AD
    Gershgorin_Stokes2D_SchurComplement!(DVx, DVy, λmaxVx, λmaxVy, ηc, ηv, γ_eff, Δx, Δy; periodic_x=periodic, Vx_Dirichlet_NS=periodic)
    # Permeability from compaction length, porosity dependent (Kozeny-Carman)
    k_ηf0    = l_c^2 / (maximum(ηΦ) + 4/3*maximum(ηvc))
    centroid_to_xface!(Φ_kx_ini, Φ, periodic)
    centroid_to_yface!(Φ_ky_ini, Φ)
    kηf_x   .= k_ηf0 .* (Φ_kx_ini./Φ_kx_ini).^n_KC
    kηf_y   .= k_ηf0 .* (Φ_ky_ini./Φ_ky_ini).^n_KC
    aPf     .= 1 ./ (1 ./ (1 ./ηΦ .+ 1 ./(KΦ*Δt))) ./ (1 .- Φ)
    DPf     .= aPf .+ 2*k_ηf0*(1/Δx^2 + 1/Δy^2)
    λmaxPf  .= (aPf .+ 4*k_ηf0*(1/Δx^2 + 1/Δy^2)) ./ DPf
    # Select dτ
    if dτ_local
        dτVx .= 2.0./sqrt.(λmaxVx)*0.99
        dτVy .= 2.0./sqrt.(λmaxVy)*0.99
        dτPf .= 2.0./sqrt.(λmaxPf)*0.99
    else
        dτVx .= 2.0./sqrt.(maximum(λmaxVx))*0.99
        dτVy .= 2.0./sqrt.(maximum(λmaxVy))*0.99
        dτPf .= 2.0./sqrt.(maximum(λmaxPf))*0.99
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
    Pt                  .= Pt_ini
    Ps                  .= (Pt .- Φ.*Pf[2:end-1,2:end-1]) ./ (1 .- Φ)
    # Probes
    probes   = (τc=Float64[], τv=Float64[], peffc=Float64[], peffv=Float64[])
    time_evo = Float64[]
    it0      = 0
    # Restart: restore the state saved at the end of a time step
    if restart_file !== nothing
        d = load(restart_file)
        for (A, key) in ((Vx,"Vx"), (Vy,"Vy"), (Pt,"Pt"), (Pf,"Pf"), (Φ,"Φ"), (Ps,"Ps"),
                         (Txxc,"Txxc"), (Tyyc,"Tyyc"), (Txyc,"Txyc"), (Txxv,"Txxv"), (Tyyv,"Tyyv"), (Txyv,"Txyv"),
                         (Cc,"Cc"), (Cv,"Cv"), (ηvc,"ηvc"), (ηvv,"ηvv"), (kηf_x,"kηf_x"), (kηf_y,"kηf_y"),
                         (strain_acc,"strain_acc"))
            A .= d[key]
        end
        append!(time_evo, d["time_evo"])
        for k in keys(probes) append!(probes[k], d["probes"][k]) end
        it0 = d["it"]
        @info "Restarting from $(restart_file) at step $(it0)"
    end
    save_data && mkpath(out_dir)
    # Time loop
    for it=it0+1:it0+nt
        Pt0 .= Pt
        Pf0 .= Pf
        Φ0  .= Φ
        Ps0 .= Ps
        Txx0c .= Txxc; Txx0v .= Txxv;
        Tyy0c .= Tyyc; Tyy0v .= Tyyv;
        Txy0v .= Txyv; Txy0c .= Txyc;
        # Old-time-step masses for the conservative solid and total mass balances
        CsρsΦ_old .= Cs .* (1 .- Φ0) .* (ρs0 .+ ρs0./Ks.*(Ps0 .- Ps_ref))                     # solid (immobile) mass
        ρt_old    .= (1 .- Φ0).*(ρs0 .+ ρs0./Ks.*(Ps0 .- Ps_ref)) .+
                     Φ0.*(ρf0 .+ ρf0./Kf.*(Pf0[2:end-1,2:end-1] .- Pf_ref[2:end-1,2:end-1]))  # total mass
        # Iteration loop
        errVx0 = 1.0;  errVy0 = 1.0;  errPf0 = 1.0;   errPt0 = 1.0;  errΦ0 = 1.0
        errVx00= 1.0;  errVy00= 1.0;  errPf00= 1.0
        iter=1; err_PT=2*ϵ; err_evo_V=Float64[]; err_evo_Pt=Float64[]; err_evo_Pf=Float64[]; err_evo_it=Float64[]; err_glob = 1.0
        k_AA = 0; err_prev = Inf   # fresh Anderson history for each time step
        @time for itPH = 1:50
            if (err_glob<ϵ) break end
            nPH_tot += 1
            err_PT = 1.0
            # Set tolerance of velocity solve proportional to residual
            ϵ_vel = err_glob*rel_drop
            itPT = 0
            # also leave once globally converged: a round-off initial residual makes the relative err_PT unreachable
            while (err_PT>ϵ_vel && err_glob>=ϵ && itPT<=iterMax)
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
                if periodic
                    Vx[end,:]  .= Vx[1,:]                          # E = W
                    Vx[:,1  ]  .= 2*γ̇*yv[1  ] .- Vx[:,2    ]      # Dirichlet tangent S
                    Vx[:,end]  .= 2*γ̇*yv[end] .- Vx[:,end-1]      # Dirichlet tangent N
                    Vy[1  ,:]  .= Vy[end-1,:]                      # periodic ghosts
                    Vy[end,:]  .= Vy[2    ,:]
                    Pf[1  ,:]  .= Pf[end-1,:]
                    Pf[end,:]  .= Pf[2    ,:]
                else
                    Vx[:,1  ]  .= Vx[:,2    ]
                    Vx[:,end]  .= Vx[:,end-1]
                    Vy[1  ,:]  .= Vy[2    ,:]
                    Vy[end,:]  .= Vy[end-1,:]
                    Pf[1  ,:]  .= Pf[2    ,:]
                    Pf[end,:]  .= Pf[end-1,:]
                end
                Pf[:,1  ] .= Pf[:,2    ]
                Pf[:,end] .= Pf[:,end-1]
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
                    P̄t[1  ,:] .= P̄t[2    ,:]
                    P̄t[end,:] .= P̄t[end-1,:]
                    P̄t[:,1  ] .= P̄t[:,2    ]
                    P̄t[:,end] .= P̄t[:,end-1]
                end
                # Porosity-dependent permeability, relaxed in log space
                centroid_to_xface!(Φ_kx, Φ, periodic)
                centroid_to_yface!(Φ_ky, Φ)
                Φ_ky[:,1  ] .= Φ[:,1  ]          # zero-gradient at N/S
                Φ_ky[:,end] .= Φ[:,end]
                kηf_x  .= exp.((1 - Φ_rel).*log.(kηf_x) .+ Φ_rel.*log.(k_ηf0.*(Φ_kx./Φ_kx_ini).^n_KC))
                kηf_y  .= exp.((1 - Φ_rel).*log.(kηf_y) .+ Φ_rel.*log.(k_ηf0.*(Φ_ky./Φ_ky_ini).^n_KC))
                # Darcy flux
                qDx     .= .-kηf_x .* (Pf[2:end,2:end-1] .- Pf[1:end-1,2:end-1])./Δx
                qDy     .= .-kηf_y .* (Pf[2:end-1,2:end] .- Pf[2:end-1,1:end-1])./Δy
                # Divergence
                ∇Vc     .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
                ∇Vv     .= (V̄x[2:end,:]       .- V̄x[1:end-1,:]      )./Δx .+ (V̄y[:,2:end]       .- V̄y[:,1:end-1]      )./Δy
                # Deviatoric strain rate
                Exxc    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇Vc
                Eyyc    .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇Vc
                Ezzc    .=                                                .- 1.0/3.0.*∇Vc
                Exyc    .= 0.5.*((V̄x[2:end-1,2:end] .- V̄x[2:end-1,1:end-1])./Δy .+ (V̄y[2:end,2:end-1] .- V̄y[1:end-1,2:end-1])./Δx)
                Exxv    .= (V̄x[2:end,:] .- V̄x[1:end-1,:])./Δx .- 1.0/3.0.*∇Vv
                Eyyv    .= (V̄y[:,2:end] .- V̄y[:,1:end-1])./Δy .- 1.0/3.0.*∇Vv
                Ezzv    .=                                      .- 1.0/3.0.*∇Vv
                Exyv    .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
                EIIv_str .= sqrt.(0.5 .* (Exxv.^2 .+ Eyyv.^2 .+ Ezzv.^2 .+ 2.0 .* Exyv.^2))
                # Two-phase stuff
                ηΦ      .= Ωη .* ηvc ./ Φ
                KΦ      .= Ωη .* Gc  ./ Φ
                # Advection (upwind) of pressures for the material time derivatives
                Pfc      = Pf[2:end-1,2:end-1]
                if advection
                    Vxc .= av_xa(Vx[:,2:end-1])
                    Vyc .= av_ya(Vy[2:end-1,:])
                    adv_upwind!(adv_Pt, Vxc, Vyc, Pt,  Δx, Δy, false, periodic)
                    adv_upwind!(adv_Pf, Vxc, Vyc, Pfc, Δx, Δy, false, periodic)
                end
                dPtdt   .= (Pt  .- Pt0) ./ Δt                   .+ adv_Pt
                dPfdt   .= (Pfc .- Pf0[2:end-1,2:end-1]) ./ Δt  .+ adv_Pf
                # Compaction law (used in the total pressure equation RPt)
                dΦdt    .= (Pfc .- Pt)./ηΦ  .+ (dPfdt .- dPtdt)./KΦ
                # Porosity from the solid mass balance in conservative form (Schmalholz et al. 2023, eq. A2)
                #   d/dt[Cs ρs (1-Φ)] + ∇⋅[Cs ρs (1-Φ) Vs] = 0, residual scaled by Cs ρs (units 1/time)
                ρs      .= ρs0 .+ ρs0./Ks.*(Ps .- Ps_ref)
                Csρs    .= Cs .* ρs
                cell_to_faces!(Csρs_x, Csρs_y, Aext, Csρs, periodic)
                cell_to_faces!(Φ_fx,   Φ_fy,   Aext, Φ,    periodic)
                FΦx     .= Csρs_x.*(1 .- Φ_fx).*Vx[:,2:end-1]
                FΦy     .= Csρs_y.*(1 .- Φ_fy).*Vy[2:end-1,:]
                RΦ      .= ( (Csρs.*(1 .- Φ) .- CsρsΦ_old)./Δt
                             .+ (FΦx[2:end,:] .- FΦx[1:end-1,:])./Δx
                             .+ (FΦy[:,2:end] .- FΦy[:,1:end-1])./Δy ) ./ Csρs
                # PT update: ∂RΦ/∂Φ ≈ -1/Δt (local diagonal) -> dτ_Φ = θΦ*Δt
                Φ       .+= θΦ*Δt .* RΦ
                # Solid pressure and its material derivative
                Ps      .= (Pt .- Φ.*Pfc) ./ (1 .- Φ)
                advection && adv_upwind!(adv_Ps, Vxc, Vyc, Ps, Δx, Δy, false, periodic)
                dPsdt   .= (Ps .- Ps0) ./ Δt .+ adv_Ps
                dlnρsdt .= dPsdt ./ Ks
                dlnρfdt .= dPfdt ./ Kf
                RPt     .= .-(∇Vc .+ dlnρsdt .- dΦdt ./ (1 .- Φ))
                # Stress transport (advection only, no objective-rate terms)
                if advection
                    Vxv .= av_ya(Vx)
                    Vyv .= av_xa(Vy)
                    adv_upwind!(advc, Vxc, Vyc, Txxc, Δx, Δy, false, periodic); TxxHistC .= Txx0c .- Δt .* advc
                    adv_upwind!(advc, Vxc, Vyc, Tyyc, Δx, Δy, false, periodic); TyyHistC .= Tyy0c .- Δt .* advc
                    adv_upwind!(advc, Vxc, Vyc, Txyc, Δx, Δy, false, periodic); TxyHistC .= Txy0c .- Δt .* advc
                    adv_upwind!(advv, Vxv, Vyv, Txxv, Δx, Δy, true , periodic); TxxHistV .= Txx0v .- Δt .* advv
                    adv_upwind!(advv, Vxv, Vyv, Tyyv, Δx, Δy, true , periodic); TyyHistV .= Tyy0v .- Δt .* advv
                    adv_upwind!(advv, Vxv, Vyv, Txyv, Δx, Δy, true , periodic); TxyHistV .= Txy0v .- Δt .* advv
                else
                    TxxHistC .= Txx0c; TyyHistC .= Tyy0c; TxyHistC .= Txy0c
                    TxxHistV .= Txx0v; TyyHistV .= Tyy0v; TxyHistV .= Txy0v
                end
                TzzHistC .= .-(TxxHistC .+ TyyHistC)
                TzzHistV .= .-(TxxHistV .+ TyyHistV)
                # Visco-elasticty
                ηc      .= 1 ./ (1 ./ηec .+ 1 ./ηvc)
                ηv      .= 1 ./ (1 ./ηev .+ 1 ./ηvv)
                # Deviatoric stress predictor
                Txxc    .= 2.0.*ηc.*(Exxc .+ TxxHistC./(2.0.*ηec))
                Tyyc    .= 2.0.*ηc.*(Eyyc .+ TyyHistC./(2.0.*ηec))
                Txyc    .= 2.0.*ηc.*(Exyc .+ TxyHistC./(2.0.*ηec))
                Tzzc    .= .-(Txxc .+ Tyyc)
                Txxv    .= 2.0.*ηv.*(Exxv .+ TxxHistV./(2.0.*ηev))
                Tyyv    .= 2.0.*ηv.*(Eyyv .+ TyyHistV./(2.0.*ηev))
                Txyv    .= 2.0.*ηv.*(Exyv .+ TxyHistV./(2.0.*ηev))
                Tzzv    .= .-(Txxv .+ Tyyv)
                # Invariants (including out-of-plane components)
                τIIc    .= sqrt.(0.5 .* (Txxc.^2 .+ Tyyc.^2 .+ Tzzc.^2 .+ 2.0 .* Txyc.^2))
                τIIv    .= sqrt.(0.5 .* (Txxv.^2 .+ Tyyv.^2 .+ Tzzv.^2 .+ 2.0 .* Txyv.^2))
                EIIc    .= sqrt.(0.5 .* ((Exxc .+ TxxHistC./(2.0.*ηec)).^2 .+ (Eyyc .+ TyyHistC./(2.0.*ηec)).^2 .+ (Ezzc .+ TzzHistC./(2.0.*ηec)).^2 .+ 2.0 .* (Exyc .+ TxyHistC./(2.0.*ηec)).^2))
                EIIv    .= sqrt.(0.5 .* ((Exxv .+ TxxHistV./(2.0.*ηev)).^2 .+ (Eyyv .+ TyyHistV./(2.0.*ηev)).^2 .+ (Ezzv .+ TzzHistV./(2.0.*ηev)).^2 .+ 2.0 .* (Exyv .+ TxyHistV./(2.0.*ηev)).^2))
                # Plasticity
                fc            .= τIIc .- Cc .* cosϕ .- (Pt .- Pfc).*sinϕ
                fv            .= τIIv .- Cv .* cosϕ .- (P̄t .- P̄f ).*sinϕ
                λ̇c      .= ifelse.(fc .> 0.0, fc ./ (ηc .+ ηvp), 0.0)
                λ̇v      .= ifelse.(fv .> 0.0, fv ./ (ηv .+ ηvp), 0.0)
                τIIc    .-= λ̇c .* ηc
                τIIv    .-= λ̇v .* ηv
                ηc      .= ifelse.(fc .> 0.0, τIIc ./ (2.0 .* EIIc), ηc)
                ηv      .= ifelse.(fv .> 0.0, τIIv ./ (2.0 .* EIIv), ηv)
                # Deviatoric stress corrector
                Txxc    .= 2.0.*ηc.*(Exxc .+ TxxHistC./(2.0.*ηec))
                Tyyc    .= 2.0.*ηc.*(Eyyc .+ TyyHistC./(2.0.*ηec))
                Txyc    .= 2.0.*ηc.*(Exyc .+ TxyHistC./(2.0.*ηec))
                Tzzc    .= .-(Txxc .+ Tyyc)
                Txxv    .= 2.0.*ηv.*(Exxv .+ TxxHistV./(2.0.*ηev))
                Tyyv    .= 2.0.*ηv.*(Eyyv .+ TyyHistV./(2.0.*ηev))
                Txyv    .= 2.0.*ηv.*(Exyv .+ TxyHistV./(2.0.*ηev))
                Tzzv    .= .-(Txxv .+ Tyyv)
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
                # Total mass balance in conservative form (Schmalholz et al. 2023, eq. A2)
                #   dρt/dt + ∇⋅[ρf qD + ρt Vs] = 0,  ρt = (1-Φ)ρs + Φρf, residual scaled by ρf (units 1/time)
                ρf      .= ρf0 .+ ρf0./Kf.*(Pf .- Pf_ref)      # uses Pf ghost cells
                ρs      .= ρs0 .+ ρs0./Ks.*(Ps .- Ps_ref)
                ρfc      = ρf[2:end-1,2:end-1]
                ρt      .= (1 .- Φ).*ρs .+ Φ.*ρfc
                ρf_x    .= av_xa(ρf[:,2:end-1])
                ρf_y    .= av_ya(ρf[2:end-1,:])
                cell_to_faces!(ρt_x, ρt_y, Aext, ρt, periodic)
                Qfx     .= ρf_x.*qDx .+ ρt_x.*Vx[:,2:end-1]
                Qfy     .= ρf_y.*qDy .+ ρt_y.*Vy[2:end-1,:]
                RPf_cons .= .-( (ρt .- ρt_old)./Δt .+ (Qfx[2:end,:] .- Qfx[1:end-1,:])./Δx .+ (Qfy[:,2:end] .- Qfy[:,1:end-1])./Δy ) ./ ρfc
                # PT residual for Pf: subtract (1-Φ)*RPt (row combination/preconditioning only).
                # RPt -> 0 at PH convergence, so the converged Pf satisfies RPf_cons = 0.
                RPf     .= RPf_cons .- (1 .- Φ).*RPt
                # Velocity preconditioner from the current (plastic) viscosity
                Gershgorin_Stokes2D_SchurComplement!(DVx, DVy, λmaxVx, λmaxVy, ηc, ηv, γ_eff, Δx, Δy; periodic_x=periodic, Vx_Dirichlet_NS=periodic)
                # Pf preconditioner for variable permeability, faces weighted by ρf_face/ρf_cell
                kxy     .= ( (kηf_x[1:end-1,:].*ρf_x[1:end-1,:] .+ kηf_x[2:end,:].*ρf_x[2:end,:])./Δx^2
                          .+ (kηf_y[:,1:end-1].*ρf_y[:,1:end-1] .+ kηf_y[:,2:end].*ρf_y[:,2:end])./Δy^2 ) ./ ρfc
                # Diagonal of RPf w.r.t. Pf: compaction term (1/ηeffΦ = 1/ηΦ + 1/(KΦ Δt)) + fluid storage
                aPf     .= (1 ./ηΦ .+ 1 ./(KΦ.*Δt))./(1 .- Φ) .+ Φ.*ρf0./(Kf*Δt.*ρfc)
                DPf     .= aPf .+ kxy
                λmaxPf  .= (aPf .+ 2 .* kxy) ./ DPf
                if dτ_local
                    dτPf .= 2.0./sqrt.(λmaxPf).*0.99
                else
                    dτPf .= 2.0/sqrt(maximum(λmaxPf))*0.99
                end
                βPf    .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
                αPf    .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
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
                    # Error norms (Pf: conservative total mass balance)
                    errVx    = norm(RVx);    errVy     = norm(RVy);    errPf  = norm(RPf_cons);    errPt  = norm(RPt);  errΦ = norm(RΦ)
                    errVx_PH = norm(RVx_PH); errVy_PH  = norm(RVy_PH);
                    # For relative error metrics
                    # (a zero initial residual, e.g. divergence-free initial flow, falls back to absolute errors)
                    nz(e) = e > 0 ? e : 1.0
                    if (itPH==1 && iter==1) errVx0  = nz(errVx);    errVy0  = nz(errVy);    errPf0  = nz(errPf); errPt0=nz(errPt); errΦ0 = nz(errΦ); end
                    if iter==1              errVx00 = nz(errVx_PH); errVy00 = nz(errVy_PH); errPf00 = nz(errPf);                   end
                    # PT error
                    err_PT = max(errVx_PH/errVx00, errVy_PH/errVy00)
                    # Global error
                    eVx = min(errVx/sqrt(length(Vx)), errVx/errVx0)
                    eVy = min(errVy/sqrt(length(Vy)), errVy/errVy0)
                    ePt = min(errPt/sqrt(length(Pt)), errPt/errPt0)
                    ePf = min(errPf/sqrt(length(Pf)), errPf/errPf0)
                    eΦ  = min(errΦ /sqrt(length(Φ )), errΦ /errΦ0 )
                    err_glob = max(eVx, eVy, ePt, ePf, eΦ)
                    # Invariants
                    τIIc    .= sqrt.(0.5 .* (Txxc.^2 .+ Tyyc.^2 .+ Tzzc.^2 .+ 2.0 .* Txyc.^2))
                    τIIv    .= sqrt.(0.5 .* (Txxv.^2 .+ Tyyv.^2 .+ Tzzv.^2 .+ 2.0 .* Txyv.^2))
                    # Plasticity
                    fc            .= τIIc .- Cc .* cosϕ .- (Pt .- Pfc).*sinϕ .- ηvp.*λ̇c
                    fv            .= τIIv .- Cv .* cosϕ .- (P̄t .- P̄f ).*sinϕ .- ηvp.*λ̇v
                    @printf("---> itPH = %02d iter = %06d iter/nx = %7.2f, norm[RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e, RΦ=%1.3e] --- Fc=%1.3e, Fv=%1.3e\n", itPH, iter, iter/ncx, eVx, eVy, ePt, ePf, eΦ, maximum(fc), maximum(fv))
                    # Store global errors
                    push!(err_evo_V, errVx/errVx00); push!(err_evo_Pt, errPt/errPt0); push!(err_evo_Pf, errPf/errPf0); push!(err_evo_it, itg)
                    dVx .= dVxdτ.*βVx.*dτVx
                    dVy .= dVydτ.*βVy.*dτVy
                    dPf .= dPfdτ.*βPf.*dτPf
                    # Compute auto-PT parameters (skipped while a field has not moved yet: 0/0)
                    denV   = wnorm2(dVx, DVx) + wnorm2(dVy, DVy)
                    denPf  = wnorm2(dPf, DPf)
                    if denV > 0
                        λminV  = abs( dot(dVx, RVx_PH) - dot(dVx, RVx_PH0) + dot(dVy, RVy_PH) - dot(dVy, RVy_PH0) ) / denV
                        cVx .= 2*sqrt(λminV)*c_factV
                        cVy .= 2*sqrt(λminV)*c_factV
                    end
                    if denPf > 0
                        λminPf = abs( dot(dPf, RPf) - dot(dPf, RPf0) ) / denPf
                        cPf .= 2*sqrt(λminPf)*c_factPf
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
            f_AA .= vec(γ_eff) .* vec(RPt)
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
        # Post-processing
        strain_acc .+= EIIv_str .* Δt
        push!(time_evo, (isempty(time_evo) ? 0.0 : time_evo[end]) + Δt)
        push!(probes.τc,    mean(τIIc))
        push!(probes.τv,    mean(τIIv))
        push!(probes.peffc, mean(Pt.-Pf[2:end-1,2:end-1]))
        push!(probes.peffv, mean(P̄t.-P̄f))

        # Plotting
        if visualization
        fig = Figure(size=(1200, 1000))
        ax  = Axis(fig[1, 1], title="τII [MPa]", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xv*sc.L, yv*sc.L, τIIv*sc.σ/1e6, colormap=:jet)
        Colorbar(fig[1, 1][1, 2], hm)
        ax  = Axis(fig[1, 2], title="Φ", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc*sc.L, yc*sc.L, Φ, colormap=:jet)
        Colorbar(fig[1, 2][1, 2], hm)
        ax  = Axis(fig[2, 1], title="Pt [MPa]", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc*sc.L, yc*sc.L, Pt*sc.σ/1e6, colormap=:jet)
        Colorbar(fig[2, 1][1, 2], hm)
        ax  = Axis(fig[2, 2], title="Vx profile (x = W)", xlabel="Vx", ylabel="y")
        scatterlines!(ax, Vx[1,:], yce*sc.L, color=:black)
        ax  = Axis(fig[3, 1], title="Convergence", xlabel="DR iterations / nx", ylabel="log10 errors")
        lines!(ax, err_evo_it/ncx, log10.(err_evo_V ), label="V" )
        lines!(ax, err_evo_it/ncx, log10.(err_evo_Pt), label="Pt")
        lines!(ax, err_evo_it/ncx, log10.(err_evo_Pf), label="Pf")
        hlines!(ax, log10(ϵ), color=:black, linestyle=:dash, label="tol")
        axislegend(ax, position=:rt)
        ax  = Axis(fig[3, 2], title="Stress and effective pressure [MPa]", xlabel="t [kyr]")
        tplot = time_evo*sc.t/kyr
        scatterlines!(ax, tplot, probes.τc*sc.σ/1e6, color=:black, label="τIIc")
        lines!(ax, tplot, probes.τv*sc.σ/1e6,    label="τIIv")
        lines!(ax, tplot, probes.peffc*sc.σ/1e6, label="Peff c")
        lines!(ax, tplot, probes.peffv*sc.σ/1e6, label="Peff v")
        display(fig)
        end
        # Advect intrinsic material properties (cohesion, viscosity) after the physical time step
        if advection
            Vxc .= av_xa(Vx[:,2:end-1]); Vyc .= av_ya(Vy[2:end-1,:])
            Vxv .= av_ya(Vx);            Vyv .= av_xa(Vy)
            for (Ac, Av) in ((Cc, Cv), (ηvc, ηvv))
                periodic && (Av[end,:] .= Av[1,:])
                adv_upwind!(advc, Vxc, Vyc, Ac, Δx, Δy, false, periodic); Ac .-= Δt .* advc
                adv_upwind!(advv, Vxv, Vyv, Av, Δx, Δy, true , periodic); Av .-= Δt .* advv
            end
        end
        n_c = length(ηc)
        @printf("Step %d: iter/ncx = %g, η_h = %g, η_g = %g, η_a = %g\n", it, iter/ncx, 1.0/sum(1.0/n_c ./ηc), exp(sum(1.0/n_c*log.(ηc))), mean(ηc))
        # Saving data
        if save_data && (mod(it, nsave) == 0 || it == 1)
            fname = joinpath(out_dir, @sprintf("SAVE_HM_%04d.jld2", div(it, nsave)))
            jldsave(fname; it, Δt, sc, xc=collect(xc), yc=collect(yc), xv=collect(xv), yv=collect(yv), xce=collect(xce), yce=collect(yce),
                    Vx, Vy, Pt, Pf, Φ, Ps, Txxc, Tyyc, Txyc, Txxv, Tyyv, Txyv, τIIc, τIIv, λ̇c, λ̇v,
                    Cc, Cv, ηvc, ηvv, ηc, ηv, kηf_x, kηf_y, qDx, qDy, strain_acc, time_evo,
                    probes=Dict(pairs(probes)))
        end
    end

    @printf("Total: %d DR iterations (%.1f per nx), %d PH iterations, m_anderson = %d\n", nDR_tot, nDR_tot/ncx, nPH_tot, m_anderson)
    return (nDR=nDR_tot, nPH=nPH_tot)
end

n = 1
Stokes2D(n; config=:simple_shear, nt=5000, m_anderson=3)
# Restart example:
# Stokes2D(n; config=:simple_shear, nt=100, restart_file=joinpath(@__DIR__, "Output_NEW_v01", "SAVE_HM_0058.jld2"))
