# Initialisation
using CairoMakie, Printf, Statistics, LinearAlgebra
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) )
@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

function Gershgorin_Stokes2D_SC(ηc, ηv, ηc_ml, ηv_ml, γ, Δx, Δy, ncx  ,ncy)

    ebE, ebW, ebN, ebS = γ, γ, γ, γ
    Δx, Δy = Δx, Δy

    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    Cxx   = ones(ncx-1, ncy)
    Cxy   = ones(ncx-1, ncy)
    @. Cxx = abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    @. Cxy = abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))

    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    cVxC  = ones(ncx-1, ncy)
    @. cVxC .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    Cyy  = ones(ncx, ncy-1)
    Cyx  = ones(ncx, ncy-1)
    @. Cyy = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(ebN ./ Δy .^ 2 + (4 // 3) * ηN ./ Δy .^ 2) + abs.(ebS ./ Δy .^ 2 + (4 // 3) * ηS ./ Δy .^ 2) + abs.((ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    @. Cyx = abs.(ebN ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ebN ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy) + ηW ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy) + ηW ./ (Δx .* Δy))

    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    cVyC  = ones(ncx, ncy-1)
    @. cVyC .= (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx = 1.0./cVxC .* (Cxx .+ Cxy)
    λmaxVy = 1.0./cVyC .* (Cyx .+ Cyy)

    dτVx =  2.0./sqrt.(λmaxVx)*0.98
    dτVy =  2.0./sqrt.(λmaxVy)*0.98

    return dτVx, dτVy, cVxC, cVyC, λmaxVx, λmaxVy
end

# 2D Stokes routine
@views function Stokes2D_ve(n; m_anderson = 3, γ_fac = 10.0, ϵ_vel_fac = 1e-3, nout = 100, do_plot = true)
    # Physics
    Lx, Ly  = 1.0, 1.0       # domain size
    radi    = 0.2            # inclusion radius
    ξ       = 2.0            # Maxwell relaxation time
    μ0      = 1.0            # viscous viscosity
    G0      = 1.0            # elastic shear modulus
    Musi    = 1e2           # largest viscosity
    εbg     = 1.0            # background strain-rate
    # Numerics
    ncx, ncy = n*63, n*63  # numerical grid resolution
    nt       = 1           # number of time steps
    ϵ        = 1e-7        # tolerance
    iterMax  = 1e4         # max number of iters
    # Preprocessing
    Δx, Δy  = Lx/ncx, Ly/ncy
    dt      = μ0/(G0*ξ + 1e-15)
    # Array initialisation
    Pt      = zeros(ncx  ,ncy  )
    ∇V      = zeros(ncx  ,ncy  )
    Vx      = zeros(ncx+1,ncy+2)
    Vy      = zeros(ncx+2,ncy+1)
    dVx     = zeros(ncx-1,ncy  )
    dVy     = zeros(ncx  ,ncy-1)
    Exx     = zeros(ncx  ,ncy  )
    Eyy     = zeros(ncx  ,ncy  )
    Exy     = zeros(ncx+1,ncy+1)
    Txx     = zeros(ncx  ,ncy  )
    Tyy     = zeros(ncx  ,ncy  )
    Txy     = zeros(ncx+1,ncy+1)
    Txx0    = zeros(ncx  ,ncy  )
    Tyy0    = zeros(ncx  ,ncy  )
    Txy0    = zeros(ncx+1,ncy+1)
    Rx      = zeros(ncx-1,ncy  )
    Ry      = zeros(ncx  ,ncy-1)
    Rp      = zeros(ncx  ,ncy  )
    Rx0     = zeros(ncx-1,ncy  )
    Ry0     = zeros(ncx  ,ncy-1)
    dVxdt   = zeros(ncx-1,ncy  )
    dVydt   = zeros(ncx  ,ncy-1)
    βVx     = zeros(ncx-1,ncy  )
    βVy     = zeros(ncx  ,ncy-1)
    cVx     = zeros(ncx-1,ncy  )
    cVy     = zeros(ncx  ,ncy-1)
    αVx     = zeros(ncx-1,ncy  )
    αVy     = zeros(ncx  ,ncy-1)
    ηc      = zeros(ncx  ,ncy  )
    ηv      = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)

    # Anderson acceleration
    nP      = length(Pt        )
    ΔP      = zeros(nP, m_anderson)
    ΔF      = zeros(nP, m_anderson)
    f       = zeros(nP         )
    P_prev  = zeros(nP         )
    f_prev  = zeros(nP         )
    k_AA    = 0
    err_prev= Inf
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    (Xvx,Yvx) = ([x for x=xv,y=yce], [y for x=xv,y=yce])
    (Xvy,Yvy) = ([x for x=xce,y=yv], [y for x=xce,y=yv])
    # Multiple circles with various viscosities
    ηi    = (w=1/Musi, s=Musi)
    x_inc = [0.0   ]#    0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1]
    y_inc = [0.0   ]#    0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4]
    r_inc = [radi  ]#    0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07]
    η_inc = [ηi.s  ]#    ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    ηv_sharp   .= 1.0
    for inc in eachindex(η_inc)
        ηv_sharp[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηc_sharp   .= 1.0
    for inc in eachindex(η_inc)
        ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    # Harmonic averaging mimicking PIC interpolation
    ηc    .= av4_harm(ηv_sharp)
    ηv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    # Maxloc - not useful apparently
    ηc_maxloc = copy(ηc)
    ηv_maxloc = copy(ηv)
    maxloc!(ηc_maxloc, ηc_sharp)
    maxloc!(ηv_maxloc, ηv_sharp)
    # Select γ
    n   = length(ηc)
    @show γ = γ_fac*mean(ηc)
    # @show γ = 1.0 / sum(1.0/n ./ηc) * 1e5
    # @show γ = exp( sum( 1.0/n*log.(ηc)))*1e3
    # Optimal pseudo-time steps
    dtVx, dtVy, cVxC, cVyC, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SC(ηc, ηv, ηc_maxloc, ηv_maxloc, γ, Δx, Δy, ncx ,ncy)
    βVx .= 2 .* dtVx ./ (2 .+ cVx.*dtVx)
    βVy .= 2 .* dtVy ./ (2 .+ cVy.*dtVy)
    αVx .= (2 .- cVx.*dtVx) ./ (2 .+ cVx.*dtVx)
    αVy .= (2 .- cVy.*dtVy) ./ (2 .+ cVy.*dtVy)
    @show minimum(dtVx), maximum(dtVx)
    @show minimum(dtVy), maximum(dtVy)
    # Initial condition
    Vx     .=   εbg.*Xvx
    Vy     .= .-εbg.*Yvy
    Vx[2:end-1,:] .= 0 # ensure non zero initial pressure residual
    Vy[:,2:end-1] .= 0 # ensure non zero initial pressure residual
    # Time loop
    t=0.0; evo_t=[]; evo_Txx=[]
    nPH = 0; nDR = 0   # this is new: PH passes and DR iterations over all time steps
    @views for it = 1:nt
        errVx0 = 1.0;  errVy0 = 1.0;  errPt0 = 1.0
        errVx00= 1.0;  errVy00= 1.0;
        iter=1; err=2*ϵ; err_evo_V=[]; err_evo_P=[]; err_evo_it=[]
        Txx0.=Txx; Tyy0.=Tyy; Txy0.=Txy
        @time for itPH = 1:50
            nPH += 1   # this is new
            # Boundaries
            Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
            Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
            # Divergence
            ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
            # Deviatoric strain rate
            Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
            Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
            Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
            # Deviatoric stress
            Txx   .= 2.0.*ηc.*Exx
            Tyy   .= 2.0.*ηc.*Eyy
            Txy   .= 2.0.*ηv.*Exy
            # Residuals
            Rx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
            Ry    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
            Rp    .= .-∇V
            # Residual check
            errVx = norm(Rx); errVy = norm(Ry); errPt = norm(Rp)
            if itPH==1 errVx0=errVx; errVy0=errVy; errPt0=errPt; end
            err = maximum([errVx/errVx0, errVy/errVy0, errPt/errPt0])
            @printf("it = %04d, itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, ∇V=%1.3e] \n", it, itPH, iter, iter/ncx, err, errVx/errVx0, errVy/errVy0, errPt/errPt0)
            if (err<ϵ) break end
            # if PH grew since the last step, reset the anderson history to prevent extrapolation in the wrong direction
            k_AA = err > err_prev ? 1 : k_AA + 1
            err_prev = err
            # Set tolerance of velocity solve proportional to residual
            ϵ_vel = err*ϵ_vel_fac
            # ϵ_vel =  ϵ/10
            # ϵ_vel = errPt/errPt0/50000
            # @show ϵ_vel, errVx, errPt, err
            itPT = 0.
            while (err>ϵ_vel && itPT<=iterMax)
                itPT   += 1
                itg    = iter
                # Pseudo-old dudes
                Rx0   .= Rx
                Ry0   .= Ry
                # Boundaries
                Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
                Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
                # Divergence
                ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
                # Deviatoric strain rate
                Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
                Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
                Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
                # Deviatoric stress
                Txx   .= 2.0.*ηc.*Exx .+ γ.*∇V
                Tyy   .= 2.0.*ηc.*Eyy .+ γ.*∇V
                Txy   .= 2.0.*ηv.*Exy
                # Residuals
                Rx    .= (1.0./cVxC).*(.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
                Ry    .= (1.0./cVyC).*(.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
                # Damping-pong
                dVxdt .= αVx.*dVxdt .+ βVx.*Rx
                dVydt .= αVy.*dVydt .+ βVy.*Ry
                # PT updates
                Vx[2:end-1,2:end-1] .+= dVxdt.*dtVx
                Vy[2:end-1,2:end-1] .+= dVydt.*dtVy
                # Residual check
                if mod(iter, nout)==0
                    errVx = norm(cVxC.*Rx); errVy = norm(cVyC.*Ry)
                    if iter==nout errVx00=errVx; errVy00=errVy; end
                    # err = maximum([errVx./errVx00, errVy./errVy00])
    #--->           @ Thibault shouldnt it be errVx0 rather than 00?
                    err = maximum([errVx./errVx0, errVy./errVy0])
                    # push!(err_evo_V, errVx/errVx00); push!(err_evo_P, errPt/errPt0); push!(err_evo_it, itg)
    #--->           Same here
                    push!(err_evo_V, errVx/errVx0); push!(err_evo_P, errPt/errPt0); push!(err_evo_it, itg)
                    dVx .= dVxdt.*dtVx
                    dVy .= dVydt.*dtVy
                    # @printf("it = %d, iter = %d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry)
                    λminV  = (abs.((sum(dVx.*(Rx .- Rx0))) + abs.((sum(dVy.*(Ry .- Ry0))) )/ ( (sum(dVx.*dVx))) + (sum(dVy.*dVy))) )
                    cVx .= 2*sqrt.(λminV)*0.55
                    cVy .= 2*sqrt.(λminV)*0.55
                    βVx .= 2 .* dtVx ./ (2 .+ cVx.*dtVx)
                    βVy .= 2 .* dtVy ./ (2 .+ cVy.*dtVy)
                    αVx .= (2 .- cVx.*dtVx) ./ (2 .+ cVx.*dtVx)
                    αVy .= (2 .- cVy.*dtVy) ./ (2 .+ cVy.*dtVy)
                end
                iter += 1
            end
            # Compute divergence again from the current velocity
            ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
            # plain PH step as a flat vector
            f .= vec(γ.*(.-∇V))
            if k_AA > 1 && m_anderson > 0
                slot = (k_AA - 2) % m_anderson + 1 # old slots get replaced
                ΔP[:, slot] .= vec(Pt) .- P_prev
                ΔF[:, slot] .= f .- f_prev
            end
            # Save for the next differences
            P_prev .= vec(Pt)
            f_prev .= f
            # plain PH step
            Pt .+= γ.*(.-∇V)
            n_AA = min(k_AA - 1, m_anderson)
            if n_AA > 0
                θ = ΔF[:, 1:n_AA] \ f # least-squares problem θ = argmin ‖f − ΔF·θ‖
                all(isfinite, θ) || error("Anderson acceleration is non-finite: θ = $θ")
                Pt .-= reshape((ΔP[:, 1:n_AA] .+ ΔF[:, 1:n_AA]) * θ, size(Pt))
            end
        end
        nDR += iter - 1   # this is new
        t += dt
        push!(evo_t, t); push!(evo_Txx, maximum(Txx))
        # Plotting
        if do_plot
        fig = Figure(size=(1000, 800))
        ax  = Axis(fig[1, 1], title="∇V", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yc, ∇V, colormap=:inferno)
        xlims!(ax, -Lx/2, Lx/2); Colorbar(fig[1, 2], hm)
        ax  = Axis(fig[1, 3], title="Pt", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yc, Pt, colormap=:inferno)
        xlims!(ax, -Lx/2, Lx/2); Colorbar(fig[1, 4], hm)
        ax  = Axis(fig[2, 1], title="Convergence", xlabel="iterations", ylabel="log10 error")
        lines!(ax, err_evo_it, log10.(err_evo_V), label="V")
        lines!(ax, err_evo_it, log10.(err_evo_P), label="P")
        lines!(ax, err_evo_it, log10.(ϵ.*ones(size(err_evo_it))), label="tol")
        axislegend(ax)
        ax  = Axis(fig[2, 3], title="ηc", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yc, log10.(ηc), colormap=:inferno)
        xlims!(ax, -Lx/2, Lx/2); Colorbar(fig[2, 4], hm)
        display(fig)
        save(joinpath(@__DIR__, "Stokes2D_DYREL_Decoupled_ViscousInclusion_m$(m_anderson).png"), fig)
        end
       @show iter/ncx
       n   = length(ηc)
       @show η_h = 1.0 / sum(1.0/n ./ηc)
       @show η_g = exp( sum( 1.0/n*log.(ηc)))
       @show η_a = mean(ηc)
    end
    return (; nPH, nDR)   # this is new
end

# ncx      = n*63
# Stokes2D_ve(8; γ_fac=3.0, ϵ_vel_fac=1e-2, nout=round(Int, 0.4*ncx), do_plot=false)

# Anderson: original settings vs settings retuned for Anderson (smaller γ, looser inner tolerance, nout ∝ ncx)
results = []
for n in (1, 2, 4, 8, 16)
    ncx      = n*63
    original = Stokes2D_ve(n; do_plot=false)
    retuned  = Stokes2D_ve(n; γ_fac=3.0, ϵ_vel_fac=1e-2, nout=round(Int, 0.4*ncx), do_plot=false)
    push!(results, (; n, original, retuned))
end
@printf("\n%3s | %22s | %22s | %7s\n", "n", "original nDR (nPH) /nx", "retuned nDR (nPH) /nx", "speedup")
for r in results
    ncx = r.n*63
    @printf("%3d | %7d (%2d) %6.1f | %7d (%2d) %6.1f | %6.2fx\n", r.n,
        r.original.nDR, r.original.nPH, r.original.nDR/ncx, r.retuned.nDR, r.retuned.nPH, r.retuned.nDR/ncx, r.original.nDR/r.retuned.nDR)
end
