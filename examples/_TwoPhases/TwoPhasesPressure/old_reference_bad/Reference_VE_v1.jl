using StagFDTools, StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
let 

    Ωl = 0.1       # ---> δ/r
    Ωr = 0.1       # ---> r/L
    Ωη = 10^(2)    # ---> ηΦ/ηs

    L  = 1.0       # box size
    ηs = 1.        # Shear viscosity
    Φi = 1e-2      # Reference

    # Compaction length 
    δ      = Ωl * Ωr * L     # δ = δ/r * r/L where L = 1
    ηΦ     = Ωη * ηs  
    n_CK   = 3.0
    k_ηΦ   = δ^2 / (ηΦ + 4/3 * ηs) # Permeability / fluid viscosity

    # Reference conductivity
    k_ηf0  = k_ηΦ/Φi^n_CK 

    # Double check compaction length
    δ1 = sqrt((k_ηf0 * Φi^n_CK) * (ηΦ + 4/3*ηs)) 

    @show k_ηf0, δ, δ1

end

@views function main(nc, Ωl, Ωη)

    M2Di_solver = false

    # Adimensionnal numbers
    Ωr     = 0.1             # Ratio inclusion radius / L
    Ωηi    = 1e-1            # Ratio (inclusion viscosity) / (matrix viscosity)
    Ωp     = 1.              # Ratio (ε̇bg * ηs) / P0
    # Independant
    ηsi    = 1.              # Shear viscosity
    L      = 1.              # Box size
    Pi     = 1.              # Initial ambiant pressure
    Φi     = 1e-2            # Reference
    n_CK   = 3.0
    # Dependant
    @show Ωl, Ωr, L
    δ      = Ωl * Ωr * L     # δ = δ/r * r/L where L = 1
    ηbi    = Ωη * ηsi        # Bulk viscosity
    k_ηΦ   = δ^2 / (ηbi + 4/3 * ηsi) # Permeability / fluid viscosity
    r      = Ωr * L          # Inclusion radius
    ηs_inc = Ωηi * ηsi       # Inclusion shear viscosity
    ε̇      = Ωp * Pi / ηsi   # Background strain rate
    # Time integration
    nt     = 100
    Δt0    = 2.5e-4 #1 / ε̇ / nc.x / 2  * 4

    # @show Δt0
    # error()
    
    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )
   
    # Material parameters
    materials = ( 
        g     = [0. 0.],
        oneway       = false,
        compressible = true,
        plasticity   = :off,
        linearizeΦ   = false, 
        single_phase = false,
        conservative = false,
        n     = [1.0  1.0],
        n_CK  = [n_CK n_CK],
        η0   = [ηsi  ηs_inc], 
        ηΦ    = [ηbi  ηbi],
        G     = [1e0 1e0], 
        ρs    = [1.0  1.0 ],
        ρf    = [1.0  1.0 ],
        Kd    = [1e0 1e0]*1,
        Ks    = [1e0 1e0]*8,
        KΦ    = [1e0 1e0]*5,
        Kf    = [1e0 1e0]*2,
        k_ηf0 = [k_ηΦ/Φi^n_CK k_ηΦ/Φi^n_CK],
        ψ     = [10.    10.  ],
        ϕ     = [35.    35.  ],
        C     = [1e70   1e70],
        ηvp   = [0.0    0.0  ],
        cosϕ  = [0.0    0.0  ],
        sinϕ  = [0.0    0.0  ],
        sinψ  = [0.0    0.0  ],
    )

    k_ηf0 = materials.k_ηf0[1]
    lc = sqrt((k_ηf0) * (materials.ξ0[1] + 4/3*materials.η0[1])) 

    # @show k_ηf0, lc

    # error()

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)
    
    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
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
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Neumann 
    type.Pf[end,:]           .= :Neumann 
    type.Pf[:,1]             .= :Neumann
    type.Pf[:,end]           .= :Neumann
    
    # Equation Fields
    number = Fields(
        fill(0, (nc.x+3, nc.y+4)),
        fill(0, (nc.x+4, nc.y+3)),
        fill(0, (nc.x+2, nc.y+2)),
        fill(0, (nc.x+2, nc.y+2)),
    )
    Numbering!(number, type, nc)

    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0;  0 1 0]),        @SMatrix([0 1 0;  0 1 0])), 
        Fields(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0]),        @SMatrix([0 0; 1 1; 0 0])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1 1 1; 1 1 1; 1 1 1]),  @SMatrix([1 1 1; 1 1 1; 1 1 1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1 1 1; 1 1 1; 1 1 1]),  @SMatrix([1 1 1; 1 1 1; 1 1 1])),
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    nPf   = maximum(number.Pf)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt), ExtendableSparseMatrix(nPt, nPf)),
        Fields(ExtendableSparseMatrix(nPf, nVx), ExtendableSparseMatrix(nPf, nVy), ExtendableSparseMatrix(nPf, nPt), ExtendableSparseMatrix(nPf, nPf)),
    )

    #--------------------------------------------#
    # Intialise field 
    L   = (x=L, y=L)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...), Φ=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ   = (c=Φi.*ones(size_c...), v=Φi.*ones(size_v...) )
    Φ0  = (c=Φi.*ones(size_c...), v=Φi.*ones(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), f = zeros(size_c...)  )
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t=zeros(size_c...), f=zeros(size_c...))
    P0      = (t=zeros(size_c...), f=zeros(size_c...))
    ΔP      = (t=zeros(size_c...), f=zeros(size_c...))
    ρ       = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))
    ρ0      = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))

    P   = (t=zeros(size_c...), f=zeros(size_c...))
    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc  = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
  
    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'

    Xc = xc .+ 0*yc'
    Yc = 0*xc .+ yc'
    Xv = xv .+ 0*yv'
    Yv = 0*xv .+ yv'
    α  = 30.
    # ax = 2
    # ay = 1/2
    ax = 1
    ay = 1
    X_tilt = cosd(α).*Xc .- sind(α).*Yc
    Y_tilt = sind(α).*Xc .+ cosd(α).*Yc
    phases.c[inx_c, iny_c][(X_tilt.^2 ./ax.^2 .+ (Y_tilt).^2 ./ay^2) .< r^2 ] .= 2
    X_tilt = cosd(α).*Xv .- sind(α).*Yv
    Y_tilt = sind(α).*Xv .+ cosd(α).*Yv
    phases.v[inx_v, iny_v][(X_tilt.^2 ./ax.^2 .+ (Y_tilt).^2 ./ay^2) .< r^2 ] .= 2

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)
    
    #--------------------------------------------#

    probes = (
        maxPt = zeros(nt),
        maxPf = zeros(nt),
        maxτ  = zeros(nt),
        Pti = zeros(nt),
        Pfi = zeros(nt),
        Pei = zeros(nt),
        ΔPt = zeros(nt),
        ΔPf = zeros(nt),
        ΔPe = zeros(nt),
        Pe  = zeros(nt),
        Pt  = zeros(nt),
        Pf  = zeros(nt),
        t   = zeros(nt),
    )

    r = zeros(nVx + nVy + nPt + nPf)
    
    for it=1:nt

        @printf("\nStep %04d\n", it)
        @info "Displacement $(2*mean(V.x[2,3:3:end-2])*Δ.t*it)"
        P0.t  .= P.t
        P0.f  .= P.f
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Φ0.c  .= Φ.c
        ρ0.s  .= ρ.s
        ρ0.f  .= ρ.f

        for iter=1:4

            @printf("     Step %04d --- Iteration %04d\n", it, iter)

            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
            ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(R, V, P, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 
            ResidualFluidContinuity2D!(R, V, P, ΔP, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 

            @info "Residuals"
            @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
            AssembleMomentum2D_x!(M, V, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleMomentum2D_y!(M, V, P, P0, ΔP, τ0, Φ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleContinuity2D!(M, V, P, (P0, Φ0, ρ0), phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleFluidContinuity2D!(M, V, P, ΔP, (P0, Φ0, ρ0), phases, materials, number, pattern, type, BC, nc, Δ)

            # Two-phases operator as block matrix
            𝑀 = [
                M.Vx.Vx M.Vx.Vy M.Vx.Pt M.Vx.Pf;
                M.Vy.Vx M.Vy.Vy M.Vy.Pt M.Vy.Pf;
                M.Pt.Vx M.Pt.Vy M.Pt.Pt M.Pt.Pf;
                M.Pf.Vx M.Pf.Vy M.Pf.Pt M.Pf.Pf;
            ]

            @info "System symmetry"
            𝑀diff = 𝑀 - 𝑀'
            dropzeros!(𝑀diff)
            @show norm(𝑀diff)

            #--------------------------------------------#

            if M2Di_solver == false 
                # Direct solver 
                @time dx = - 𝑀 \ r
            else
                # M2Di solver
                fv    = -r[1:(nVx+nVy)]
                fpt   = -r[(nVx+nVy+1):(nVx+nVy+nPt)]
                fpf   = -r[(nVx+nVy+nPt+1):end]
                dv    = zeros(nVx+nVy)
                dpt   = zeros(nPt)
                dpf   = zeros(nPf)
                rvs   = zeros(nVx+nVy)
                rpt   = zeros(nPt)
                rpf   = zeros(nPf)
                rvs_t  = zeros(nVx+nVy)
                rpt_t = zeros(nPt)
                s     = zeros(nPf)
                ddv   = zeros(nVx+nVy)
                ddpt  = zeros(nPt)
                ddpf  = zeros(nPf)

                Jvv  = [M.Vx.Vx M.Vx.Vy;
                        M.Vy.Vx M.Vy.Vy]
                Jvp  = [M.Vx.Pt;
                        M.Vy.Pt]
                Jpv  = [M.Pt.Vx M.Pt.Vy]
                Jpp  = M.Pt.Pt
                Jppf = M.Pt.Pf
                Jpfv = [M.Pf.Vx M.Pf.Vy]
                Jpfp = M.Pf.Pt
                Jpf  = M.Pf.Pf
                Kvv  = Jvv

                @time begin 
                    # γ = 1e-8
                    # Γ = spdiagm(γ*ones(nPt))
                    # Pre-conditionning (~Jacobi)
                    Jpv_t  = Jpv  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfv  
                    Jpp_t  = Jpp  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfp  #.+ Γ
                    Jvv_t  = Kvv  - Jvp *spdiagm(1 ./ diag(Jpp_t))*Jpv 
                    Jpf_h  = cholesky(Hermitian(SparseMatrixCSC(Jpf)), check = false  )        # Cholesky factors
                    Jvv_th = cholesky(Hermitian(SparseMatrixCSC(Jvv_t)), check = false)        # Cholesky factors
                    Jpp_th = spdiagm(1 ./diag(Jpp_t));             # trivial inverse
                    nrvs0, nrpt0, nrpf0 = 1.0, 1.0, 1.0
                    @views for itPH=1:30
                        rvs   .= -( Jvv*dv  + Jvp*dpt             - fv  )
                        rpt   .= -( Jpv*dv  + Jpp*dpt  + Jppf*dpf - fpt )
                        rpf   .= -( Jpfv*dv + Jpfp*dpt + Jpf*dpf  - fpf )
                        nrvs = norm(rvs)/length(rvs); nrpt = norm(rpt)/length(rpt);  nrpf = norm(rpf)/length(rpf)
                        if (itPH == 1) nrvs0, nrpt0, nrpf0 = nrvs, nrpt, nrpf end
                        @printf("  --- iteration %d --- \n",itPH);
                        @printf("  abs. rvs = %2.2e --- rel. rvs = %2.2e\n", nrvs, nrvs/nrvs0)
                        @printf("  abs. rpt = %2.2e --- rel. rpt = %2.2e\n", nrpt, nrpt/nrpt0)
                        @printf("  abs. rpf = %2.2e --- rel. rpf = %2.2e\n", nrpf, nrpf/nrpf0)
                        s     .= Jpf_h \ rpf
                        rpt_t .= -( Jppf*s - rpt)
                        s     .=    Jpp_th*rpt_t
                        rvs_t .= -( Jvp*s  - rvs )
                        ddv   .= Jvv_th \ rvs_t
                        s     .= -( Jpv_t*ddv - rpt_t )
                        ddpt  .=    Jpp_th*s
                        s     .= -( Jpfp*ddpt + Jpfv*ddv - rpf )
                        ddpf  .= Jpf_h \ s
                        dv   .+= ddv
                        dpt  .+= ddpt
                        dpf  .+= ddpf
                        # if ((norm(rvs)/length(rvs)) < tol_linv) && ((norm(rpt)/length(rpt)) < tol_linpt) && ((norm(rpf)/length(rpf)) < tol_linpf), break; end
                        # if ((norm(rvs)/length(rvs)) > (norm(rv0)/length(rv0)) && norm(rvs)/length(rvs) < tol_glob && (norm(rpt)/length(rpt)) > (norm(rpt0)/length(rpt0)) && norm(rpt)/length(rpt) < tol_glob && (norm(rpf)/length(rpf)) > (norm(rpf0)/length(rpf0)) && norm(rpf)/length(rpf) < tol_glob),
                        #     if noisy>=1, fprintf(' > Linear residuals do no converge further:\n'); break; end
                        # end
                        # rv0=rvs; rpt0=rpt; rpf0=rpf; if (itPH==nPH), nfail=nfail+1; end
                    end
                end
                
                dx = zeros(nVx + nVy + nPt + nPf)
                dx[1:(nVx+nVy)] .= dv
                dx[(nVx+nVy+1):(nVx+nVy+nPt)] .= dpt
                dx[(nVx+nVy+nPt+1):end] .= dpf
            end

            #--------------------------------------------#
            UpdateSolution!(V, P, dx, number, type, nc)

        end

        #--------------------------------------------#



        k_ηΦ_x = materials.k_ηf0[1] .* ((Φ.c[2:end,:] .+ Φ.c[1:end-1,:]) / 2).^ materials.n_CK[1]
        k_ηΦ_y = materials.k_ηf0[1] .* ((Φ.c[:,2:end] .+ Φ.c[:,1:end-1]) / 2).^ materials.n_CK[1]

        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
        Vs   = (x=Vxsc, y=Vysc )
        Vs_mag   = sqrt.( Vxsc.^2 .+ Vysc.^2)
        Vxf  = -k_ηΦ_x .* diff(P.f, dims=1)/Δ.x
        Vyf  = -k_ηΦ_y .* diff(P.f, dims=2)/Δ.y
        Vxfc = 0.5*(Vxf[1:end-1,2:end-1] .+ Vxf[2:end,2:end-1])
        Vyfc = 0.5*(Vyf[2:end-1,1:end-1] .+ Vyf[2:end-1,2:end])
        Vf   = (x=Vxfc, y=Vyfc )
        Vf_mag   = sqrt.( Vxfc.^2 .+ Vyfc.^2)

        dΦdt = (Φ.c .- Φ0.c) / Δ.t

        # P.t .-= mean(P.t[inx_c,iny_c]) 
        # P.f .-= mean(P.f[inx_c,iny_c])

        # # p1 = heatmap(xc, yc, Vs[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Vs")
        # p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vf")
        # p2 = heatmap(xc, yc, Φ.c[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Φ")
        # # p3 = heatmap(xc, yc, τII[inx_c,iny_c]',   aspect_ratio=1, xlim=extrema(xc), title="Pt", clims=(-3,3))
        # st = 20
        # p3 = quiver(Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end], quiver=(Vxsc[1:st:end,1:st:end],Vysc[1:st:end,1:st:end]), c=:black,  aspect_ratio=1, xlim=extrema(xc), title="Pt", clims=(-3,3))
        # # divV = diff(V.x[2:end-1,3:end-2], dims=1)/Δ.x  + diff(V.y[3:end-2,2:end-1], dims=2)/Δ.y
        # # p3 = heatmap(xc, yc, divV',   aspect_ratio=1, xlim=extrema(xc), title="Pt")
        
        cmap = (CairoMakie.Reverse(:matter), 1)
        # cmap = :jet1
        st  = 15
        ind = st:st:size(xc,1)-st

        fig = Figure(fontsize = 14, size = (675, 600) )  
    
        ax1 = Axis(fig[3,1],  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$V^\text{s}$"
        hmVs = heatmap!(ax1, xc, yc, Vs_mag, colormap=cmap, colorrange=(0,0.75)) 
        arrows2d!(ax1, xc[ind], yc[ind], Vs.x[ind,ind], Vs.y[ind,ind], lengthscale = 1e-1, color = :white)

        ax2 = Axis(fig[3,2], xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$V^\text{f} \times 1000$"
        hmVf = heatmap!(ax2, xc, yc, Vf_mag*1000, colormap=cmap, colorrange=(0,0.2)) 
        arrows2d!(ax2, xc[ind], yc[ind], Vf.x[ind,ind], Vf.y[ind,ind], lengthscale = 500, color = :white)
        # arrowsize = V.arrow, lengthscale = V.scale)

        ax2 = Axis(fig[3,3], xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$V^\text{f} \times 1000$"
        hmτ = heatmap!(ax2, xc, yc, τ.II[inx_c,iny_c], colormap=cmap, colorrange=(0,3)) 
        # arrows2d!(ax2, xc[ind], yc[ind], σ1.x[ind,ind], σ1.y[ind,ind], lengthscale = 7e-2, color = :white, tipwidth = 0)

        ax1 = Axis(fig[2,1],  xlabel=L"$x$ [-]",  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) #, title=L"$P^\text{t}$"
        hm1=heatmap!(ax1, xc, yc, P.t[inx_c,iny_c], colormap=cmap, colorrange=(-3,3)) 
        # hm1=heatmap!(ax1, xc, yc, Vs.x, colormap=cmap) 

        ax2 = Axis(fig[2,2],  xlabel=L"$x$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) # , title=L"$P^\text{f}$"
        hm2=heatmap!(ax2, xc, yc, P.f[inx_c,iny_c], colormap=cmap, colorrange=(-3,3)) 
        
        ax3 = Axis(fig[2,3],  xlabel=L"$x$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect()) # , title=L"$\dot{\phi}$"
        hm3=heatmap!(ax3, xc, yc, dΦdt[inx_c,iny_c]*100, colormap=cmap, colorrange=(-10.e-1, 10.e-1)) 

        # contour!( ax3, xc, yc, Pe[inx_c,iny_c], levels=[0.1], color=:white)
        
        Colorbar(fig[4,   1], hmVs, label = L"D) $|V^\text{s}|$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = false )
        Colorbar(fig[4,   2], hmVf, label = L"E) $|Q^\text{f}| \times 1000$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = false )
        Colorbar(fig[4,   3], hmτ,  label = L"F) $\tau_{II}$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = false )

        Colorbar(fig[1, 1], hm1, label = L"A) $P^\text{t}$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = true )
        Colorbar(fig[1, 2], hm2, label = L"B) $P^\text{f}$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = true )
        Colorbar(fig[1, 3], hm3, label = L"C) $\dot{\phi} \times 100$ [-]", height=10, width = 150, labelsize = 16, ticklabelsize = 12, vertical=false, valign=true, flipaxis = true )

        display(fig)

        probes.Pti[it]   = mean(P.t[phases.c.==2])
        probes.Pfi[it]   = mean(P.f[phases.c.==2])
        probes.Pei[it]   = mean(P.t[phases.c.==2] .- P.f[phases.c.==2])
        probes.ΔPt[it]   = maximum(P.t) - minimum(P.t)
        probes.ΔPf[it]   = maximum(P.f) - minimum(P.f)
        probes.ΔPe[it]   = maximum(P.t .- P.f) - minimum(P.t .- P.f) 
        probes.Pe[it]    = norm(P.t .- P.f)
        probes.Pt[it]    = norm(P.t)
        probes.Pf[it]    = norm(P.f)
        probes.t[it]     = it*Δ.t
        probes.maxPt[it] = maximum(P.t)
        probes.maxPf[it] = maximum(P.f)
        probes.maxτ[it]  = maximum(τ.II)

        @show mean(P.t[phases.c.==2])
        @show mean(P.f[phases.c.==2])

        fig = Figure(fontsize = 14, size = (600, 600) )  
        ax = Axis(fig[1,1], xlabelsize=20, ylabelsize=20, aspect=DataAspect(), title=L"$\text{max} P^t, P^f, \tau_\text{II}$", xlabel = L"$t$ [-]", ylabel = L"$P, \tau$ [-]")
        lines!(ax,  probes.t[1:it], probes.maxPt[1:it], label=L"$$P^t")
        lines!(ax,  probes.t[1:it], probes.maxPf[1:it], label=L"$$P^f")
        lines!(ax,  probes.t[1:it], probes.maxτ[1:it],  label=L"$$\tau_\text{II}")
        axislegend(framevisible = false, position=:lt)
        display(fig)

    end

    #--------------------------------------------#

    @show Δt0

    save("./examples/_TwoPhases/TwoPhasesPressure/PoroviscousReference.jld2", "Ωl", Ωl, "Ωη", Ωη,"x", (c=xc, v=xv), "y", (c=yc, v=yv), "P", P, "dΦdt", dΦdt, "Φ", Φ, "τ", τ, "Vs", (x=Vxsc, y=Vysc), "Vf", (x=Vxfc, y=Vyfc))

    return P, Δ, (c=xc, v=xv), (c=yc, v=yv)
end

function Run()

    nc = (x=300, y=300)

    # Mode 0   
    # Ωl = 10^(-1.7) # ---> δ/r
    # Ωl = 10^(-1.0)
    Ωη = 10^(2)
    Ωl = 0.2
    main(nc,  Ωl, Ωη);
    
end

Run()
