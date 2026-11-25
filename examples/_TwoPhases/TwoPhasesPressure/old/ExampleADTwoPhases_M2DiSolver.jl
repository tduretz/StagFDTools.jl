using StagFDTools.TwoPhases_v1, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
function main(nc)
    
    # Resolution

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c = Ranges(nc)
    
    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    BC = Fields(
        fill(0., (nc.x+3, nc.y+4)),
        fill(0., (nc.x+4, nc.y+3)),
        fill(0., (nc.x+2, nc.y+2)),
        fill(0., (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx] .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Neumann
    type.Vx[inx_Vx,end-1]   .= :Neumann
    BC.Vx[2,iny_Vx]         .= 0.0
    BC.Vx[end-1,iny_Vx]     .= 0.0
    BC.Vx[inx_Vx,2]         .= 0.0
    BC.Vx[inx_Vx,end-1]     .= 0.0
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy] .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann
    type.Vy[end-1,iny_Vy]   .= :Neumann
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    BC.Vy[2,iny_Vy]         .= 0.0
    BC.Vy[end-1,iny_Vy]     .= 0.0
    BC.Vy[inx_Vy,2]         .= 0.0
    BC.Vy[inx_Vy,end-1]     .= 0.0
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Dirichlet 
    type.Pf[end,:]           .= :Dirichlet 
    type.Pf[:,1]             .= :Dirichlet
    type.Pf[:,end]           .= :Dirichlet
    
    # Equation numbering
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
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1])),
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1 1 1; 1 1 1; 1 1 1])),
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
    L   = (x=10.0, y=10.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (x= ones(size_x...), y= ones(size_y...), p=ones(size_c...) )
    ηΦ  = ones(size_c...) 
    ϕ   = ones(size_c...) 
    kμf = (x= ones(size_x...), y= ones(size_y...))
    P   = (t=zeros(size_c...), f=zeros(size_c...))
    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc  = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)
    xvy = LinRange(-L.x/2-3Δ.x/2, L.x/2+3Δ.x/2, nc.x+4)
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    yvx = LinRange(-L.y/2-3Δ.y/2, L.y/2+3Δ.y/2, nc.y+4)

    # Initial configuration
    ε̇  = -1.0
    V.x[inx_Vx,iny_Vx] .=  ε̇*xv .+ 0*yc' 
    V.y[inx_Vy,iny_Vy] .= 0*xc .-  ε̇*yv' 
    η.y[(xvy.^2 .+ (yvy').^2) .<= 1^2] .= 0.1
    η.x[(xvx.^2 .+ (yvx').^2) .<= 1^2] .= 0.1 
    η.p .= 0.25.*(η.x[1:end-1,2:end-1].+η.x[2:end-0,2:end-1].+η.y[2:end-1,1:end-1].+η.y[2:end-1,2:end-0])
    ηΦ  .= η.p
    ϕ   .= 1e-3
    rheo = (η=η, ηΦ=ηΦ, kμf=kμf, ϕ=ϕ)

    #--------------------------------------------#
    # Residual check
    ResidualContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualFluidContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 

    # Set global residual vector
    r = zeros(nVx + nVy + nPt + nPf)
    SetRHS!(r, R, number, type, nc)

    #--------------------------------------------#
    # Assembly
    @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
    AssembleContinuity2D!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_x!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_y!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    AssembleFluidContinuity2D!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)

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
    # # Direct solver 
    # @time dx = - 𝑀 \ r

    # M2Di solver
    fv    = -r[1:(nVx+nVy)]
    fpt   = -r[(nVx+nVy+1):(nVx+nVy+nPt)]
    fpf   = -r[(nVx+nVy+nPt+1):end]
    dv    = zeros(nVx+nVy)
    dpt   = zeros(nPt)
    dpf   = zeros(nPf)
    rv    = zeros(nVx+nVy)
    rpt   = zeros(nPt)
    rpf   = zeros(nPf)
    rv_t  = zeros(nVx+nVy)
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
        # Pre-conditionning (~Jacobi)
        Jpv_t  = Jpv  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfv
        Jpp_t  = Jpp  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfp
        Jvv_t  = Kvv  - Jvp *spdiagm(1 ./ diag(Jpp_t))*Jpv 
        @show typeof(SparseMatrixCSC(Jpf))
        Jpf_h  = cholesky(Hermitian(SparseMatrixCSC(Jpf)), check = false  )        # Cholesky factors
        Jvv_th = cholesky(Hermitian(SparseMatrixCSC(Jvv_t)), check = false)        # Cholesky factors
        Jpp_th = spdiagm(1 ./diag(Jpp_t));             # trivial inverse
        @views for itPH=1:15
            rv    .= -( Jvv*dv  + Jvp*dpt             - fv  )
            rpt   .= -( Jpv*dv  + Jpp*dpt  + Jppf*dpf - fpt )
            rpf   .= -( Jpfv*dv + Jpfp*dpt + Jpf*dpf  - fpf )
            s     .= Jpf_h \ rpf
            rpt_t .= -( Jppf*s - rpt)
            s     .=    Jpp_th*rpt_t
            rv_t  .= -( Jvp*s  - rv )
            ddv   .= Jvv_th \ rv_t
            s     .= -( Jpv_t*ddv - rpt_t )
            ddpt  .=    Jpp_th*s
            s     .= -( Jpfp*ddpt + Jpfv*ddv - rpf )
            ddpf  .= Jpf_h \ s
            dv   .+= ddv
            dpt  .+= ddpt
            dpf  .+= ddpf
            @printf("  --- iteration %d --- \n",itPH);
            @printf("  ||res.v ||=%2.2e\n", norm(rv)/ 1)
            @printf("  ||res.pt||=%2.2e\n", norm(rpt)/1)
            @printf("  ||res.pf||=%2.2e\n", norm(rpf)/1)
        #     if ((norm(rv)/length(rv)) < tol_linv) && ((norm(rpt)/length(rpt)) < tol_linpt) && ((norm(rpf)/length(rpf)) < tol_linpf), break; end
        #     if ((norm(rv)/length(rv)) > (norm(rv0)/length(rv0)) && norm(rv)/length(rv) < tol_glob && (norm(rpt)/length(rpt)) > (norm(rpt0)/length(rpt0)) && norm(rpt)/length(rpt) < tol_glob && (norm(rpf)/length(rpf)) > (norm(rpf0)/length(rpf0)) && norm(rpf)/length(rpf) < tol_glob),
        #         if noisy>=1, fprintf(' > Linear residuals do no converge further:\n'); break; end
        #     end
        #     rv0=rv; rpt0=rpt; rpf0=rpf; if (itPH==nPH), nfail=nfail+1; end
        end
    end
    
    dx = zeros(nVx + nVy + nPt + nPf)
    dx[1:(nVx+nVy)] .= dv
    dx[(nVx+nVy+1):(nVx+nVy+nPt)] .= dpt
    dx[(nVx+nVy+nPt+1):end] .= dpf

    #--------------------------------------------#
    UpdateSolution!(V, P, dx, number, type, nc)

    #--------------------------------------------#
    # Residual check
    ResidualContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualFluidContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
    
    @info "Residuals"
    @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
    @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
    @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
    @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

    #--------------------------------------------#

    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy")
    p3 = heatmap(xc, yc, P.t[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Pt")
    p4 = heatmap(xc, yc, P.f[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Pf")
    display(plot(p1, p2, p3, p4))

    #--------------------------------------------#
end

main( (x=300, y=300) )