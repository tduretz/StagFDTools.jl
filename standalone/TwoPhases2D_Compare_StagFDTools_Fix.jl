# Initialisation
using Revise, Plots, Printf, Statistics, LinearAlgebra 
Dat = Float64  # Precision (double=Float64 or single=Float32)
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_xi(A) =  0.5*(A[1:end-1,2:end-1].+A[2:end,2:end-1])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av_yi(A) =  0.5*(A[2:end-1,1:end-1].+A[2:end-1,2:end])

# 2D Stokes routine
@views function TwoPhasesPressures()
    viscoelastic = false
    Ωl      = 10^(-1.7)*10
    Ωη      = 10^(2)
    if viscoelastic
        dt      = 1e4
        dtred   = 5*dt*10              # If NaN increase
    else
        dt      = 1e-2
        dtred   = 5              # If NaN increase
    end
    # Adimensionnal numbers
    Ωr     = 0.1             # Ratio inclusion radius / len
    Ωηi    = 1e-1            # Ratio (inclusion viscosity) / (matrix viscosity)
    Ωp     = 1e0             # Ratio (ε̇bg * ηs) / P0
    # Independant
    η0    = 1.              # Shear viscosity
    r      = 0.1             # Box size
    τi     = 1.              # Initial ambiant pressure
    ϕi     = 1e-2
    # Dependant
    ηb0    = Ωη * η0       # Bulk viscosity
    k_ηf0  = (r.^2 * Ωl^2) / (ηb0 + 4/3 * η0) # Permeability / fluid viscosity
    len    = r / Ωr          # Inclusion radius
    ηs_inc = 1 ./ Ωηi * η0  # Inclusion shear viscosity
    εbg    = Ωp * τi / η0   #
    # Physics
    Lx, Ly = len, len          # domain size
    radi   = 0.1               # inclusion radius
    η_ϕ    = ηb0
    ηinc   = ηs_inc 
    if viscoelastic
        G       = 1e-7              # elastic shear modulus
        Ks      = 1e-6
        KΦ      = 1e-6
        Kf      = 1e-5
    else
        G       = 1e10              # elastic shear modulus
        Ks      = 1e10
        KΦ      = 1e10
        Kf      = 1e10
    end
    # Numerics
    nt      = 1                  # number of time steps
    nx, ny  = 101, 101           # numerical grid resolution
    Vdmp    = 5.0                # convergence acceleration (damping)
    Vsc     = 2.0                # iterative time step limiter
    Ptsc    = 3.0                # iterative time step limiter
    ε       = 1e-7               # nonlinear tolerence
    iterMax = 20000#3e4          # max number of iters
    nout    = 200                # check frequency
    # Preprocessing
    dx, dy  = Lx/nx, Ly/ny
    # Array initialisation
    ∇qD     = zeros(Dat, nx  ,ny  )
    qDx     = zeros(Dat, nx+1,ny  )
    qDy     = zeros(Dat, nx  ,ny+1)
    kfx     = zeros(Dat, nx+1,ny  ) # k on Vx points
    kfy     = zeros(Dat, nx  ,ny+1) # k on Vy points
    kfc     = zeros(Dat, nx  ,ny  )
    ln1mϕ   = log.(1.0.-ϕi.*ones(Dat, nx  ,ny  ))
    ln1mϕ0  = zeros(Dat, nx  ,ny  ) .= ln1mϕ
    ϕ       = ϕi.*ones(Dat, nx  ,ny  )
    ϕ0      = ϕi.*ones(Dat, nx  ,ny  )
    ϕ_viz   = ϕi.*ones(Dat, nx  ,ny  )
    ϕex     = zeros(Dat, nx+2,ny+2)
    Pt      = zeros(Dat, nx  ,ny  )
    Pf      = zeros(Dat, nx  ,ny  )
    Pfex    = zeros(Dat, nx+2,ny+2)
    Pt0     = zeros(Dat, nx  ,ny  )
    Pf0     = zeros(Dat, nx  ,ny  )
    RPf     = zeros(Dat, nx  ,ny  )
    RPt     = zeros(Dat, nx  ,ny  )
    Rϕ      = zeros(Dat, nx  ,ny  )
    ∇V      = zeros(Dat, nx  ,ny  )
    Vx      = zeros(Dat, nx+1,ny  )
    Vy      = zeros(Dat, nx  ,ny+1)
    Exx     = zeros(Dat, nx  ,ny  )
    Eyy     = zeros(Dat, nx  ,ny  )
    Exyv    = zeros(Dat, nx+1,ny+1)
    Exx1    = zeros(Dat, nx  ,ny  )
    Eyy1    = zeros(Dat, nx  ,ny  )
    Exy1    = zeros(Dat, nx  ,ny  )
    Exyv1   = zeros(Dat, nx+1,ny+1)
    τxx     = zeros(Dat, nx  ,ny  )
    ρf      =  ones(Dat, nx  ,ny  )
    ρs      =  ones(Dat, nx  ,ny  )
    ρf0     =  ones(Dat, nx  ,ny  )
    ρs0     =  ones(Dat, nx  ,ny  )
    dρfdt   = zeros(Dat, nx  ,ny  )
    dρsdt   = zeros(Dat, nx  ,ny  )
    τyy     = zeros(Dat, nx  ,ny  )
    τxy     = zeros(Dat, nx  ,ny  )
    τxyv    = zeros(Dat, nx+1,ny+1)
    τxx0    = zeros(Dat, nx  ,ny  )
    τyy0    = zeros(Dat, nx  ,ny  )
    τxy0    = zeros(Dat, nx  ,ny  )
    τxyv0   = zeros(Dat, nx+1,ny+1)
    τII     = zeros(Dat, nx  ,ny  )
    Eii     = zeros(Dat, nx  ,ny  )
    Rx      = zeros(Dat, nx-1,ny  )
    Ry      = zeros(Dat, nx  ,ny-1)
    dPtdt   = zeros(Dat, nx  ,ny  )
    dPfdt   = zeros(Dat, nx  ,ny  )
    dVxdt   = zeros(Dat, nx-1,ny  )
    dVydt   = zeros(Dat, nx  ,ny-1)
    dtPt    = zeros(Dat, nx  ,ny  )
    dtPf    = zeros(Dat, nx  ,ny  )
    dtVx    = zeros(Dat, nx-1,ny  )
    dtVy    = zeros(Dat, nx  ,ny-1)
    ηc_v    =   η0*ones(Dat, nx, ny)
    ηv_v    =   η0*ones(Dat, nx+1, ny+1)
    ηc_ve   =   η0*ones(Dat, nx, ny)
    ηv_ve   =   η0*ones(Dat, nx+1, ny+1)
    # Initial condition
    xc, yc    = LinRange(-Lx/2+dx/2, Lx/2-dx/2, nx), LinRange(-Ly/2+dy/2, Ly/2-dy/2, ny)
    xv, yv    = LinRange(-Lx/2, Lx/2, nx+1), LinRange(-Ly/2, Ly/2, ny+1)
    Xc = xc .+ 0*yc'
    Yc = 0*xc .+ yc'
    Xv = xv .+ 0*yv'
    Yv = 0*xv .+ yv'
    θ  = 30.
    ax = 2radi
    ay = radi/2
    (Xvx,Yvx) = ([x for x=xv,y=yc], [y for x=xv,y=yc])
    (Xvy,Yvy) = ([x for x=xc,y=yv], [y for x=xc,y=yv])
    X_tilt = cosd(θ).*Xc .- sind(θ).*Yc
    Y_tilt = sind(θ).*Xc .+ cosd(θ).*Yc
    radc      = ((X_tilt).^2 ./ax.^2 .+ (Y_tilt).^2 ./ay.^2)
    X_tilt = cosd(θ).*Xv .- sind(θ).*Yv
    Y_tilt = sind(θ).*Xv .+ cosd(θ).*Yv
    radv      = ((X_tilt).^2 ./ax.^2 .+ (Y_tilt).^2 ./ay.^2)
    ηc_v[radc.<1] .= ηinc
    ηv_v[radv.<1].= ηinc
    if viscoelastic
        ηc_ve  .= (1.0./(G*dt) .+ 1.0./ηc_v).^-1
        ηv_ve  .= (1.0./(G*dt) .+ 1.0./ηv_v).^-1
    else
        ηc_ve  .= ηc_v
        ηv_ve  .= ηv_v
    end
    Vx     .=   εbg.*Xvx
    Vy     .= .-εbg.*Yvy
    # Time loop
    t=0.0; evo_t=[]; evo_τII=[]; evo_Pt=[]; evo_Pf=[] 
    for it = 1:nt
        iter=1; err=2*ε;
        # Previous time step
        τxx0 .= τxx; τyy0 .= τyy; τxy0 .= τxy; τxyv0 .= τxyv; ϕ0 .= ϕ; ρs0 .= ρs; ρf0 .= ρf 
        ln1mϕ0   .= ln1mϕ;   Pf0  .= Pf;  Pt0  .= Pt
        @printf("it = %d\n", it) 
        while (err>ε && iter<=iterMax)
            # BCs
            ϕex[2:end-1,2:end-1] .= ϕ
            ϕex[1,:] .= 2ϕi.-ϕex[2,:]; ϕex[end,:] .= 2ϕi.-ϕex[end-1,:]; ϕex[:,1] .= 2ϕi.-ϕex[:,2]; ϕex[:,end] .= 2ϕi.-ϕex[:,end-1] 
            Pfex[2:end-1,2:end-1] .= Pf
            Pfex[1,:].= Pfex[2,:]; Pfex[end,:].= Pfex[end-1,:]; Pfex[:,1].= Pfex[:,2]; Pfex[:,end].= Pfex[:,end-1];
            # Darcy flux divergence
            qDx    .= -k_ηf0 .* (diff(Pfex[:,2:end-1], dims=1)/dx )
            qDy    .= -k_ηf0 .* (diff(Pfex[2:end-1,:], dims=2)/dy )
            ∇qD    .= diff(qDx, dims=1)/dx .+ diff(qDy, dims=2)/dy
            # Solid velocity divergence - pressure
            ∇V     .= diff(Vx, dims=1)./dx .+ diff(Vy, dims=2)./dy
            # Strain rates
            Exx    .= diff(Vx, dims=1)./dx .- 1.0/3.0*∇V
            Eyy    .= diff(Vy, dims=2)./dy .- 1.0/3.0*∇V
            Exyv[2:end-1,2:end-1] .= 0.5.*(diff(Vx[2:end-1,:], dims=2)./dy .+ diff(Vy[:,2:end-1], dims=1)./dx)
            # Visco-elastic strain rates
            Exx1   .=    Exx   .+ τxx0 ./2.0./(G*dt)
            Eyy1   .=    Eyy   .+ τyy0 ./2.0./(G*dt)
            Exyv1  .=    Exyv  .+ τxyv0./2.0./(G*dt)
            Exy1   .= av(Exyv) .+ τxy0 ./2.0./(G*dt)
            Eii    .= sqrt.(0.5*(Exx1.^2 .+ Eyy1.^2) .+ Exy1.^2)
            # Trial stress
            τxx    .= 2.0.*ηc_ve.*Exx1
            τyy    .= 2.0.*ηc_ve.*Eyy1
            τxy    .= 2.0.*ηc_ve.*Exy1
            τxyv   .= 2.0.*ηv_ve.*Exyv1
            τII    .= sqrt.(0.5*(τxx.^2 .+ τyy.^2) .+ τxy.^2)
            # Porosity
            ϕ      .= ϕ0 .+ dt*((Pf.-Pt)./η_ϕ .+ 1 ./KΦ .* ((Pf.-Pf0)./dt .- (Pt.-Pt0)./dt ))
            # Density
            dρsdt  .= ρs ./ Ks .*  1 ./ (1 .- ϕ) .* ((Pt.-Pt0)./dt .- ϕ .* (Pf.-Pf0)./dt) 
            dρfdt  .= ρf ./ Kf .* (Pf.-Pf0)./dt 
            ρs     .= ρs0 .+ dt*dρsdt
            ρf     .= ρf0 .+ dt*dρfdt
            # PT timestep
            dtVx   .= min(dx,dy)^2.0./av_xa(ηc_ve)./4.1./Vsc / dtred
            dtVy   .= min(dx,dy)^2.0./av_ya(ηc_ve)./4.1./Vsc / dtred
            dtPt   .= 4.1.*ηc_ve./max(nx,ny)./Ptsc           / dtred
            dtPf   .= min(dx,dy).^2.0./k_ηf0./4.1            / dtred
            # Residuals
            # RPt    .= .-( ∇V  .+ (Pt.-Pf)./η_ϕ./(1.0.-ϕ) .+ 1.0 ./Kd .* (  (Pf.-Pf0)./dt .- α       .* (Pt.-Pt0)./dt ) )
            # RPf    .= .-( ∇qD .- (Pt.-Pf)./η_ϕ./(1.0.-ϕ) .- α   ./Kd .* (  (Pt.-Pt0)./dt .- 1.0 ./B .* (Pf.-Pf0)./dt ) )
            RPt    .= .-(∇V .- 1 ./ (1 .- ϕ) .* (ϕ .- ϕ0)/dt .+ 1 ./ ρs .* dρsdt)
            RPf    .= .-(∇qD .+ ϕ.*∇V .+ (ϕ .- ϕ0)/dt .+ ϕ ./ ρf .* dρfdt)  
            Rx     .= .-diff(Pt, dims=1)./dx .+ diff(τxx, dims=1)./dx .+ diff(τxyv[2:end-1,:], dims=2)./dy
            Ry     .= .-diff(Pt, dims=2)./dy .+ diff(τyy, dims=2)./dy .+ diff(τxyv[:,2:end-1], dims=1)./dx 
            # Updates rates
            dVxdt  .= dVxdt.*(1-Vdmp/nx) .+ Rx 
            dVydt  .= dVydt.*(1-Vdmp/ny) .+ Ry 
            dPfdt  .= dPfdt.*(1-Vdmp/ny) .+ RPf
            dPtdt  .= RPt   # no damping-pong on Pt
            # Updates solutions
            Vx[2:end-1,:] .= Vx[2:end-1,:] .+ dtVx.*dVxdt
            Vy[:,2:end-1] .= Vy[:,2:end-1] .+ dtVy.*dVydt
            Pt            .= Pt            .+ dtPt.*dPtdt
            Pf            .= Pf            .+ dtPf.*dPfdt
            # convergence check
            if mod(iter, nout)==0 || iter==1
                @show extrema(ϕ)
                norm_Rx = norm(Rx)/sqrt(length(Rx)); norm_Ry = norm(Ry)/sqrt(length(Ry)); norm_RPt = norm(RPt)/sqrt(length(RPt)); norm_RPf = norm(RPf)/sqrt(length(RPf)); norm_Rϕ = norm(Rϕ)/sqrt(length(Rϕ))
                err = maximum([norm_Rx, norm_Ry, norm_RPt, norm_RPf, norm_Rϕ])
                # push!(err_evo1, err); push!(err_evo2, itg)
                @printf("iter = %05d, err = %1.2e norm[Rx=%1.2e, Ry=%1.2e, RPt=%1.2e, RPf=%1.2e, Rϕ=%1.2e] \n", iter, err, norm_Rx, norm_Ry, norm_RPt, norm_RPf, norm_Rϕ)
                isnan(err) && error("NaNs!!!")
            end
            iter+=1; #itg=iter
        end

        @show norm( ∇V  .+ (Pt.-Pf)./η_ϕ./(1.0.-ϕ)     ) / sqrt(length(Pt))
        @show norm( ∇qD .- (Pt.-Pf)./η_ϕ./(1.0.-ϕ)     ) / sqrt(length(Pf))
        @show norm( ∇V .- 1 ./ (1 .- ϕ) .* (ϕ .- ϕ0)/dt) / sqrt(length(Pt))
        @show norm( ∇qD .+ ϕ.*∇V .+ (ϕ .- ϕ0)/dt       ) / sqrt(length(Pf))

        t  += dt
        push!(evo_t, t); push!(evo_τII, maximum(τII)-minimum(τII)); push!(evo_Pt, maximum(Pt)-minimum(Pt)); push!(evo_Pf, maximum(Pf)-minimum(Pf))
        # Plotting
        p1 = heatmap(xc, yc, ϕ' , aspect_ratio=1, xlim=extrema(xv), ylim=extrema(yv), c=:inferno, title="ϕ - visu only")
        p3 = heatmap(xc, yc, Pt' , aspect_ratio=1, xlim=extrema(xv), ylim=extrema(yv), c=:inferno, title="Pt")
        p4 = heatmap(xc, yc, Pf' , aspect_ratio=1, xlim=extrema(xv), ylim=extrema(yv), c=:inferno, title="Pf")
        p2 = plot( evo_t./dt, evo_τII, label="ΔTii", xlabel="time", ylabel="Tii, Pt, Pf" )
        p2 = plot!(evo_t./dt, evo_Pt, label="ΔPt")
        p2 = plot!(evo_t./dt, evo_Pf, label="ΔPf")
        display(plot(p1, p2, p3, p4))
    end
    return
end

TwoPhasesPressures()
