# Initialisation
using CairoMakie, Printf, Statistics, LinearAlgebra
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end
@views av4_harm(A)  = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 

function smooth!(A2, A, fact)
    @. A2[2:end-1,2:end-1] = A[2:end-1,2:end-1] + 1.0/4.1/fact*((-2*A[2:end-1,2:end-1].+A[3:end,2:end-1].+A[1:end-2,2:end-1]) + (-2*A[2:end-1,2:end-1].+A[2:end-1,3:end].+A[2:end-1,1:end-2]))
    return
end

function Gershgorin_Stokes2D(ηc, ηv, ηc_ml, ηv_ml, Δx, Δy, ncx  ,ncy)
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    Cxx = (4 // 3) * abs.(ηE ./ Δx .^ 2) + (4 // 3) * abs.(ηW ./ Δx .^ 2) + abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    Cxy = abs.((2 // 3) * ηE ./ (Δx .* Δy) - ηN ./ (Δx .* Δy)) + abs.((2 // 3) * ηE ./ (Δx .* Δy) - ηS ./ (Δx .* Δy)) + abs.(ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))
    
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    cVxC  = ones(ncx-1, ncy)
    cVxC .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    Cyy = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + (4 // 3) * abs.(ηN ./ Δy .^ 2) + (4 // 3) * abs.(ηS ./ Δy .^ 2) + abs.(((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    Cyx = abs.(ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.((2 // 3) * ηN ./ (Δx .* Δy) - ηW ./ (Δx .* Δy)) + abs.((2 // 3) * ηS ./ (Δx .* Δy) - ηW ./ (Δx .* Δy))
    
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    cVyC  = ones(ncx, ncy-1)
    cVyC .= ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx = 1.0./cVxC .* (Cxx .+ Cxy .+ 2/Δx)
    λmaxVy = 1.0./cVyC .* (Cyx .+ Cyy .+ 2/Δy)
    λmaxPt = (2/Δx + 2/Δy) .* ηc
    # λmaxPt = 1 

    dτVx =  2.0./sqrt.(λmaxVx)*0.98 
    dτVy =  2.0./sqrt.(λmaxVy)*0.98  
    # dτPt =  2.0 * (ηc) ./ λmaxPt *3
    dτPt =  2 * maximum(ηc) ./ maximum(λmaxPt) 
    # dτPt =  4*ηc ./ maximum(λmaxPt)
    # dτPt =  2.0./sqrt.(λmaxPt)*0.98 /10


    Ptsc    = 5.0
    β_n     = 2.0
    max_nxy = 255
    # dτPt = 4.1*(1.0+β_n)/Ptsc/max_nxy*ηc

    return dτVx, dτVy, dτPt, cVxC, cVyC, λmaxVx, λmaxVy, λmaxPt
end

# 2D Stokes routine
@views function Stokes2D_ve(n)
    # Physics
    Lx, Ly  = 10., 10.       # domain size
    radi    = 0.01           # inclusion radius
    ξ       = 2.0            # Maxwell relaxation time
    μ0      = 1.0            # viscous viscosity
    G0      = 1.0            # elastic shear modulus
    Musi    = 1e-3           # elastic shear modulus perturbation
    εbg     = 1.0            # background strain-rate
    # Numerics
    nsm      = 10
    nt       = 1           # number of time steps
    ncx, ncy = 255, 255    # numerical grid resolution
    ϵ        = 1e-5        # nonlinear tolerence
    iterMax  = 5e5         # max number of iters
    nout     = 1000        # check frequency
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
    dPt     = zeros(ncx  ,ncy  )
    Exx     = zeros(ncx  ,ncy  )
    Eyy     = zeros(ncx  ,ncy  )
    Exy     = zeros(ncx+1,ncy+1)
    Txx     = zeros(ncx  ,ncy  )
    Tyy     = zeros(ncx  ,ncy  )
    Txy     = zeros(ncx+1,ncy+1)
    Txx0    = zeros(ncx  ,ncy  )
    Tyy0    = zeros(ncx  ,ncy  )
    Txy0    = zeros(ncx+1,ncy+1)
    Tii     = zeros(ncx  ,ncy  )
    Rx      = zeros(ncx-1,ncy  )
    Ry      = zeros(ncx  ,ncy-1)
    Rp      = zeros(ncx  ,ncy  )
    Rx0     = zeros(ncx-1,ncy  )
    Ry0     = zeros(ncx  ,ncy-1)
    Rp0     = zeros(ncx  ,ncy  )
    dVxdt   = zeros(ncx-1,ncy  )
    dVydt   = zeros(ncx  ,ncy-1)
    dPtdt   = zeros(ncx  ,ncy  )
    βVx     = zeros(ncx-1,ncy  )
    βVy     = zeros(ncx  ,ncy-1)
    βPt     = zeros(ncx  ,ncy  )
    αVx     = zeros(ncx-1,ncy  )
    αVy     = zeros(ncx  ,ncy-1)
    αPt     = zeros(ncx  ,ncy  )
    cVx     = zeros(ncx-1,ncy  )
    cVy     = zeros(ncx  ,ncy-1)
    cPt     = zeros(ncx  ,ncy  )
    Rog     = zeros(ncx  ,ncy  )
    ηc      = zeros(ncx  ,ncy  )
    ηv      = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    (Xvx,Yvx) = ([x for x=xv,y=yce], [y for x=xv,y=yce])
    (Xvy,Yvy) = ([x for x=xce,y=yv], [y for x=xce,y=yv])
    # Multiple circles with various viscosities
    ηi    = (w=1/Musi, s=Musi) 
    x_inc = [0.0  ]
    y_inc = [0.0  ]
    r_inc = [1.0  ] 
    η_inc = [ηi.s ]
    # x_inc = [0.0      0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1]*10 
    # y_inc = [0.0      0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4]*10
    # r_inc = [0.2      0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07]*10 
    # η_inc = [ηi.s     ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    ηc_sharp   .= 1.0
    for inc in eachindex(η_inc)
        ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    
    ηc     .= ηc_sharp
    for ism=1:nsm # *********** nsm --> 1
        smooth!(ηc, ηc_sharp, 1.0)
        ηc_sharp, ηc = ηc, ηc_sharp
    end

    ηv[2:end-1,2:end-1] = av(ηc)
    # ηv_sharp   .= 1.0
    # for inc in eachindex(η_inc)
    #     ηv_sharp[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    # end
    # ηc_sharp   .= 1.0
    # for inc in eachindex(η_inc)
    #     ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    # end  
    # # Harmonic averaging mimicking PIC interpolation
    # ηc    .= av4_harm(ηv_sharp)
    # ηv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    # Maxloc - not useful apparently
    ηc_maxloc = copy(ηc)
    ηv_maxloc = copy(ηv)
    maxloc!(ηc_maxloc, ηc_sharp)
    maxloc!(ηv_maxloc, ηv_sharp)
    # Optimal pseudo-time steps
    dtVx, dtVy, dtPt, cVxC, cVyC, λmaxVx, λmaxVy, λmaxPt = Gershgorin_Stokes2D(ηc, ηv, ηc_maxloc, ηv_maxloc, Δx, Δy, ncx ,ncy)
    @show minimum(dtVx), maximum(dtVx)
    @show minimum(dtVy), maximum(dtVy)
    @show minimum(dtPt), maximum(dtPt)
    # Initial condition
    Vx     .=   εbg.*Xvx
    Vy     .= .-εbg.*Yvy
    # Time loop
    t=0.0; evo_t=[]; evo_Txx=[]
    @views for it = 1:nt
        iter=1; err=2*ϵ; err_evo_V=[]; err_evo_P=[]; err_evo_it=[]
        Txx0 .= Txx; Tyy0 .= Tyy; Txy0 .= Txy
        errVx0 = 1.0; errVy0 = 1.0; errPt0 = 1.0
        local itg
        @time @views while (err>ϵ && iter<=iterMax)
            iter += 1
            itg    = iter
            # Pseudo-old dudes 
            Rx0   .= Rx
            Ry0   .= Ry
            Rp0   .= Rp
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
            Rx    .= (1.0./cVxC).*(.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
            Ry    .= (1.0./cVyC).*(.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
            Rp    .= .-∇V.* ηc
            # Rp    .= .-∇V .+ Pt./ηc

            # Damping-pong
            βVx   .= 2 .* dtVx ./ (2 .+ cVx.*dtVx)
            βVy   .= 2 .* dtVy ./ (2 .+ cVy.*dtVy)
            βPt   .= 2 .* dtPt ./ (2 .+ cPt.*dtPt)
            αVx   .= (2 .- cVx.*dtVx) ./ (2 .+ cVx.*dtVx)
            αVy   .= (2 .- cVy.*dtVy) ./ (2 .+ cVy.*dtVy)
            αPt   .= (2 .- cPt.*dtPt) ./ (2 .+ cPt.*dtPt)
            dVxdt .= αVx.*dVxdt .+ βVx.*Rx
            dVydt .= αVy.*dVydt .+ βVy.*Ry
            # dPtdt .= αPt.*dPtdt .+ βPt.*Rp
            dPtdt .= Rp
            # PT updates
            Vx[2:end-1,2:end-1] .+= dVxdt.*dtVx
            Vy[2:end-1,2:end-1] .+= dVydt.*dtVy
            Pt                  .+= dPtdt.*dtPt
            # convergence check
            if mod(iter, nout)==0 
                errVx = norm(cVxC.*Rx)/length(Rx); errVy = norm(cVyC.*Ry)/length(Ry); errPt = norm(Rp./ηc)/length(Rp)
                if iter==nout errVx0 = errVx; errVy0 = errVy; errPt0 = errPt; end
                err = maximum([errVx./errVx0, errVy./errVy0, errPt./errPt0])
                push!(err_evo_V, errVx./errVx0); push!(err_evo_P, errPt./errPt0); push!(err_evo_it, itg)
                @printf("it = %02d, iter = %06d, iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, ∇V=%1.3e] \n", it, itg, iter/ncx, err, errVx./errVx0, errVy/errVy0, errPt./errPt0)
                dVx   .= dVxdt.*dtVx
                dVy   .= dVydt.*dtVy
                dPt   .= dPtdt.*dtPt
                λminV  = (abs.((sum(dVx.*(Rx .- Rx0))) + abs.((sum(dVy.*(Ry .- Ry0))) )/ ( (sum(dVx.*dVx))) + (sum(dVy.*dVy))) )
                λminVx = abs.((sum(dVx.*(Rx .- Rx0))) / (sum(dVx.*dVx)))
                λminVy = abs.((sum(dVy.*(Ry .- Ry0))) / (sum(dVy.*dVy)))
                λminPt = abs.((sum(dPt.*(Rp .- Rp0))) / (sum(dPt.*dPt)))
                # λminPt = 1.0#abs((sum(dPt.*(Rp .- Rp0))) / (sum(dPt.*dPt)))
                cVx .= 2*sqrt.(λminVx)*1.0
                cVy .= 2*sqrt.(λminVy)*1.0
                cPt .= 2*sqrt.(λminPt)*1.0
            end
        end
        t += dt
        push!(evo_t, t); push!(evo_Txx, maximum(Txx))
        # Plotting
        fig = Figure(size=(1000, 800))
        ax  = Axis(fig[1, 1], title="Pt", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yc, Pt, colormap=:inferno)
        xlims!(ax, -Lx/2, Lx/2); Colorbar(fig[1, 2], hm)
        ax  = Axis(fig[1, 3], title="Vy", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yv, Vy[2:end-1,:], colormap=:inferno)
        xlims!(ax, -Lx/2, Lx/2); Colorbar(fig[1, 4], hm)
        ax  = Axis(fig[2, 1], title="log10 ηc", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yc, log10.(ηc), colormap=:inferno)
        xlims!(ax, -Lx/2, Lx/2); Colorbar(fig[2, 2], hm)
        ax  = Axis(fig[2, 3], title="Convergence", xlabel="iterations", ylabel="log10 error")
        lines!(ax, err_evo_it, log10.(err_evo_V), label="V")
        lines!(ax, err_evo_it, log10.(err_evo_P), label="P")
        lines!(ax, err_evo_it, log10.(ϵ.*ones(size(err_evo_it))), label="tol")
        axislegend(ax)
        display(fig)
       @show iter/ncx
       @show η_h = 1.0 / sum(1.0/n ./ηc)
       @show η_g = exp( sum( 1.0/n*log.(ηc)))
       @show η_a = mean(ηc)
    end
    return
end

n = 2
Stokes2D_ve(n)
