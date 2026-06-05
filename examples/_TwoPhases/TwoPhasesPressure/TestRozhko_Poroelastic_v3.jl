using StagFDTools, StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, ExactFieldSolutions, GridGeometryUtils
import Statistics:mean
using DifferentiationInterface
@views function main(nc)

    # Characteristic scales
    sc  = (σ=1e0, t=1e0, L=1e0)

    # Time steps
    nt     = 1
    Δt0    = 1/sc.t 

    # Newton solver
    niter = 1
    ϵ_nl  = 1e-8
    α     = LinRange(0.05, 1.0, 5)

    # Dependant
    r_in     = 1.0        # Inclusion radius 
    r_out    = 10*r_in
    ε̇        = 0.0    # Background strain rate
    Pf_bot   = 10.0 /sc.σ

    # Set Rozhko values for fluid pressure
    G_anal = 1.0
    ν_anal = 0.25
    K      = 2/3*G_anal*(1+ν_anal)/(1-2ν_anal) 

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )

    # Geometries
    L    = (x=25/sc.L, y=25/sc.L)
    x    = (min=-L.x/2, max=L.x/2)
    y    = (min=-L.y/2, max=L.y/2)
    r1   = Ellipse((0.0, 0.0), r_in, r_in; θ = 0.0)
    r2   = Ellipse((0.0, 0.0), r_out, r_out; θ = 0.0)

    # Material parameters
    kill_elasticity = 1.0 # set to 1 to activate elasticity, set to large value to kill it
    kill_plasticity = 1e20

    materials = ( 
        g     = [0.0 0.0] / (sc.L/sc.t^2),
        oneway       = true, # !!!!!!!!!!! For Rozhko test !!!!!!!!!!!
        linearizeΦ   = true, # !!!!!!!!!!! For Rozhko test !!!!!!!!!!!
        compressible = true,
        plasticity   = :off,
        single_phase = false,
        conservative = false,
        #        mat    inc  
        Φ0    = [1e-6   1e-6  1e-6],
        n     = [1.0    1.0   1.0 ],
        m     = [0.0    0.0   0.0 ],
        n_CK  = [1.0    1.0   1.0 ],
        η0   = [1e40  1e40*1e-6  1e40*1e-6]./sc.σ/sc.t, 
        ξ0   = [1e40  1e40*1e6   1e40*1e-6]./sc.σ/sc.t,
        G     = [G_anal  1e-10 1e-10 ] .* kill_elasticity ./sc.σ, 
        ρs    = [2900   2900  2900]/(sc.σ*sc.t^2/sc.L^2),
        ρf    = [2600   2600  2600]/(sc.σ*sc.t^2/sc.L^2),
        Ks    = [K  K*1e6 1*K/1e6] .* kill_elasticity ./sc.σ,
        KΦ    = [K  K*1e6 1*K/1e6] .* kill_elasticity ./sc.σ,
        Kf    = [K  K*1e6 1*K/1e6] .* kill_elasticity ./sc.σ, 
        k_ηf0 = [1.0    1.0    1.0] ./(sc.L^2/sc.σ/sc.t),
        ϕ     = [35.    35.    35 ].*1,
        ψ     = [10.    10.    10 ].*1,
        C     = [1e7    1e7    1e7] * kill_plasticity ./sc.σ,
        ηvp   = [0.0    0.0    1.0]./sc.σ/sc.t,
        cosϕ  = [0.0    0.0    1.0],
        sinϕ  = [0.0    0.0    1.0],
        sinψ  = [0.0    0.0    1.0],
    )

    # nondim 
    m      = 0.0   # 0 - circle, 0.5 - ellipse, 1 - cut 
    # dependent scales
    Pf_out = 0.    # Fluid pressure on external boundary, Pa
    dPf    = 1.0   # Fluid pressure on cavity - Po    
    Δt0    = 1e0
    nt     = 1
    params = (r_in=r_in, r_out=r_out, P0=Pf_out, dPf=dPf, m=m, nu=ν_anal, G=G_anal)

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
    type.Pf[1,:]             .= :Dirichlet 
    type.Pf[end,:]           .= :Dirichlet 
    type.Pf[:,1]             .= :Dirichlet
    type.Pf[:,end]           .= :Dirichlet

    # Add a constant pressure within a circular region
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    X = GenerateGrid(x, y, Δ, nc)

    @views type.Pf[inx_c,  iny_c ][(X.c.x.^2 .+ (X.c.y').^2) .<= r_in^2 ] .= :constant
    @views type.Pf[inx_c,  iny_c ][(X.c.x.^2 .+ (X.c.y').^2) .>= r_out^2] .= :constant
    
    @views type.Vx[inx_Vx, iny_Vx][(X.v.x.^2 .+ (X.c.y').^2) .<= r_in^2 ] .= :constant
    @views type.Vx[inx_Vx, iny_Vx][(X.v.x.^2 .+ (X.c.y').^2) .>= r_out^2] .= :constant
    
    @views type.Vy[inx_Vy, iny_Vy][(X.c.x.^2 .+ (X.v.y').^2) .<= r_in^2 ] .= :constant
    @views type.Vy[inx_Vy, iny_Vy][(X.c.x.^2 .+ (X.v.y').^2) .>= r_out^2] .= :constant
    
    @views type.Pt[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y').^2) .<= r_in^2 ] .= :constant
    @views type.Pt[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y').^2) .>= r_out^2] .= :constant
    
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
    # Intialise fields
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...), Φ=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    Vi  = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ   = (c=zeros(size_c...), v=zeros(size_v...) )
    Φ0  = (c=zeros(size_c...), v=zeros(size_v...) )
    εp  = zeros(size_c...)
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = ones(size_c...), yy = ones(size_c...), xy = zeros(size_v...) )
    τ       = (xx = ones(size_c...), yy = ones(size_c...), xy = zeros(size_v...), II = zeros(size_c...), f = zeros(size_c...) )
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t = ones(size_c...), f = ones(size_c...))
    Pi      = (t = ones(size_c...), f = ones(size_c...))
    P0      = (t = zeros(size_c...), f = zeros(size_c...))
    ΔP      = (t = zeros(size_c...), f = zeros(size_c...))
    ρ       = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))
    ρ0      = (s = materials.ρs[1]*ones(size_c...), f = materials.ρf[1]*ones(size_c...), t = zeros(size_c...))

    # Generate grid coordinates 
    X = GenerateGrid(x, y, Δ, nc)

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'

    for i in inx_c, j in iny_c   # loop on inner centroids
        𝐱 = @SVector([X.c.x[i-1], X.c.y[j-1]])
        phases.c[i, j] = 3
        if  inside(𝐱, r2)
            phases.c[i, j] = 1
        end
        if  inside(𝐱, r1)
            phases.c[i, j] = 2
        end
        Φ_ini     = materials.Φ0[phases.c[i, j]]
        Φ.c[i, j] = Φ_ini
        ρ.f[i, j] = materials.ρf[phases.c[i, j]]
        ρ.t[i, j] = Φ_ini * materials.ρf[phases.c[i, j]] + (1-Φ_ini) * materials.ρs[phases.c[i, j]]
    end

    for i in inx_v, j in iny_v   # loop on centroids
        𝐱 = @SVector([X.v.x[i-1], X.v.y[j-1]])
        phases.v[i, j] = 3
        if  inside(𝐱, r2)
            phases.v[i, j] = 1
        end
        if  inside(𝐱, r1)
            phases.v[i, j] = 2
        end
        Φ.v[i, j] = materials.Φ0[phases.v[i, j]]
    end

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[1]   .+ D_BC[2,2]*X.v.y)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[end] .+ D_BC[2,2]*X.v.y)
    BC.Pf[     :,     1 ] .= Pf_bot

    #--------------------------------------------#
    Ur_ana = zero(BC.Pf)
    Ux_ana = zero(BC.Pf)
    Ut_ana = zero(BC.Pf)
    Ux_ana = zero(BC.Vx)
    Uy_ana = zero(BC.Vy)
    Pf_ana = zero(BC.Pf)
    Pt_ana = zero(BC.Pf)
    ϵ_Ur   = zero(BC.Pf)
    ϵ_Pf   = zero(BC.Pf)
    ϵ_Pt   = zero(BC.Pf)
    ϵ_Ux   = zero(BC.Vx)

    for i=1:size(BC.Pf,1), j=1:size(BC.Pf,2)
        # coordinate transform
        sol = Poroelasticity2D_Rozhko2008([X.c_e.x[i]; X.c_e.y[j]] ; params)
        BC.Pf[i,j]  = sol.pf
        # P.f[i,j]    = sol.pf
        Pf_ana[i,j] = sol.pf
        # P.t[i,j]    = sol.pt*3/2
        BC.Pt[i,j]  = sol.pt#*3/2
        Pt_ana[i,j] = sol.pt#*3/2
        Ur_ana[i,j] = sol.u_pol[1]
        Ut_ana[i,j] = sol.u_pol[2]
    end

    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)# nc.x+3, nc.y+4
    yvx = LinRange(-L.y/2-3*Δ.y/2, L.y/2+3*Δ.y/2, nc.y+4)
    for i=1:size(BC.Vx,1), j=1:size(BC.Vx,2)
        # coordinate transform
        sol = Poroelasticity2D_Rozhko2008([xvx[i]; yvx[j]] ; params)
        BC.Vx[i,j]  = sol.u[1]
        V.x[i,j]    = sol.u[1]
        Ux_ana[i,j] = sol.u[1]
    end

    xvy = LinRange(-L.x/2-3*Δ.x/2, L.x/2+3*Δ.x/2, nc.x+4)# nc.x+3, nc.y+4
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    for i=1:size(BC.Vy,1), j=1:size(BC.Vy,2)
        # coordinate transform
        sol = Poroelasticity2D_Rozhko2008([xvy[i]; yvy[j]] ; params)
        BC.Vy[i,j]  = sol.u[2]
        V.y[i,j]    = sol.u[2]
        Uy_ana[i,j] = sol.u[2]
    end

    # This will set all correct values for constant pressure points 
    # Inside inner radius / outside outer radius
    P.f .= Pf_ana
    P.t .= Pt_ana

    #--------------------------------------------#

    rvec   = zeros(length(α))
    probes = (
        Pe  = zeros(nt),
        Pt  = zeros(nt),
        Pf  = zeros(nt),
        τ   = zeros(nt),
        Φ   = zeros(nt),
        λ̇   = zeros(nt),
        t   = zeros(nt),
        τII = zeros(nt),
    )

    err  = (x = zeros(niter), y = zeros(niter), pt = zeros(niter), pf = zeros(niter))
    
    # fig  = Figure(fontsize = 20, size = (900, 600) )    
    # step = 10
    # ftsz = 15 
    # eps  = 1e-15
    # ax    = Axis(fig[1,1], aspect=DataAspect(), title=L"$P^f$ (MPa)", xlabel=L"x", ylabel=L"y")
    # field = (phases.c)[inx_c,iny_c].*sc.σ
    # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
    # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
    # hidexdecorations!(ax)
    # Colorbar(fig[2, 1], hm, label = L"$P^f numerics$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
    # display(fig)  

    #--------------------------------------------#
    for it=1:nt

        @printf("\nStep %04d\n", it)
        fill!(err.x,  0e0)
        fill!(err.y,  0e0)
        fill!(err.pt, 0e0)
        fill!(err.pf, 0e0)

        # Swap old values 
        P0.t  .= P.t
        P0.f  .= P.f
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Φ0.c  .= Φ.c 

        for iter=1:niter

            @printf("     Step %04d --- Iteration %04d\n", it, iter)

            λ̇.c   .= 0.0
            λ̇.v   .= 0.0

            #--------------------------------------------#
            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
            ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(R, V, P, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 
            ResidualFluidContinuity2D!(R, V, P, ΔP, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 

            println("min/max λ̇.c  - ",  extrema(λ̇.c[inx_c,iny_c]))
            println("min/max λ̇.v  - ",  extrema(λ̇.v[3:end-2,3:end-2]))
            println("min/max ΔP.t - ",  extrema(ΔP.t[inx_c,iny_c]))
            println("min/max ΔP.f - ",  extrema(ΔP.f[inx_c,iny_c]))

            @info "Residuals"
            @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

            err.x[iter]  = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter]  = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.pt[iter] = @views norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            err.pf[iter] = @views norm(R.pf[inx_c,iny_c])/sqrt(nPt)
            if max(err.x[iter], err.y[iter], err.pt[iter], err.pf[iter]) < ϵ_nl 
                println("Converged")
                break 
            end

            # Set global residual vector
            r = zeros(nVx + nVy + nPt + nPf)
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

            # @show extrema(diag(M.Vx.Vx))
            # @show extrema(diag(M.Vy.Vy))

            # Dy = diag(M.Vy.Vy)
            # @show findmin(Dy)
            # display(Dy[1])

            # printxy(number.Pf)
            # printxy(type.Pf)

            # M.Vx.Vx \ ones(size(M.Vx.Vx,1))
            # M.Vy.Vy \ ones(size(M.Vy.Vy,1))
            # M.Pt.Pt \ ones(size(M.Pt.Pt,1))
            # M.Pf.Pf \ ones(size(M.Pf.Pf,1))

            @info "System symmetry"
            𝑀diff = 𝑀 - 𝑀'
            dropzeros!(𝑀diff)
            @show norm(𝑀diff)

            #--------------------------------------------#
            # Direct solver 
            @time dx = - 𝑀 \ r

            #--------------------------------------------#
            imin = LineSearch!(rvec, α, dx, R, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, (τ0, P0, Φ0, ρ0), λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, P, α[imin]*dx, number, type, nc)
        end

        #--------------------------------------------#

        # Residual check
        TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
        ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualContinuity2D!(R, V, P, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 
        ResidualFluidContinuity2D!(R, V, P, ΔP, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 

        @info "Residuals - posteriori"
        @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
        @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
        @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
        @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

        #--------------------------------------------#

        # Include plasticity corrections
        P.t .= P.t .+ ΔP.t
        P.f .= P.f .+ ΔP.f
        εp  .+= ε̇.II*Δ.t
        
        τxyc = av2D(τ.xy)
        ε̇xyc = av2D(ε̇.xy)

        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
        Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)[2:end-1,2:end-1]
        Vxf  = -materials.k_ηf0[1]*diff(P.f, dims=1)/Δ.x
        Vyf  = -materials.k_ηf0[1]*diff(P.f, dims=2)/Δ.y
        Vyfc = 0.5*(Vyf[1:end-1,:] .+ Vyf[2:end,:])
        Vxfc = 0.5*(Vxf[:,1:end-1] .+ Vxf[:,2:end])
        Vf   = sqrt.( Vxfc.^2 .+ Vyfc.^2)

        #--------------------------------------------#
        probes.Pe[it]   = mean(P.t[inx_c,iny_c] .- P.f[inx_c,iny_c])*sc.σ
        probes.Pt[it]   = mean(P.t[inx_c,iny_c])*sc.σ
        probes.Pf[it]   = mean(P.f[inx_c,iny_c])*sc.σ
        probes.τ[it]    = mean(τ.II[inx_c,iny_c])*sc.σ
        probes.Φ[it]    = mean(Φ.c[inx_c,iny_c])
        probes.λ̇[it]    = mean(λ̇.c[inx_c,iny_c])/sc.t
        probes.t[it]    = it*Δ.t*sc.t

        #-------------------------------------------# 
        Vr_viz  = zero(Vxsc)
        Vt_viz  = zero(Vxsc)
        Pt_viz = copy(P.t)
        Pf_viz = copy(P.f)

        for i in 1:length(X.c_e.x), j in 1:length(X.c_e.y)

            r = sqrt.(X.c_e.x[i].^2 .+ X.c_e.y[j].^2)
            t = atan.(X.c_e.y[j], X.c_e.x[i])

            J = [cos(t) sin(t);    
                -sin(t) cos(t)]
            V_cart = [Vxsc[i,j]; Vysc[i,j]]
            V_pol  =  J*V_cart

            Vr_viz[i,j] = V_pol[1]
            Vt_viz[i,j] = V_pol[2]

            if (X.c_e.x[i].^2 .+ X.c_e.y[j].^2) <= r_in^2 ||  (X.c_e.x[i].^2 .+ X.c_e.y[j].^2) >= r_out^2
                Vr_viz[i,j] = NaN
                Vt_viz[i,j] = NaN
                Pf_viz[i,j] = NaN
                Pt_viz[i,j] = NaN
                Ur_ana[i,j] = NaN
                Ut_ana[i,j] = NaN
            else
                ϵ_Ur[i,j] = abs(Ur_ana[i,j] - Vr_viz[i,j] )
                ϵ_Pf[i,j] = abs(Pf_ana[i,j] - P.f[i,j])
                ϵ_Pt[i,j] = abs(Pt_ana[i,j]*3/2 - P.t[i,j])
            end
            
        end

        for i=1:size(BC.Vx,1), j=1:size(BC.Vx,2)
            ro  = sqrt(xvx[i]^2 + yvx[j]^2)
            if ro <= r_in || ro >= r_out
                # Vx[i,j]     = NaN
            else
                ϵ_Ux[i,j] = abs(Ux_ana[i,j] - V.x[i,j])
            end
        end

        # @show mean(ϵ_Ur)
        # @show mean(ϵ_Ux)
        # @show mean(ϵ_Pf)
        # @show mean(ϵ_Pt)
   
        ymid = Int64(floor(nc.y/2))
        err = mean(abs.(Pf_ana[:, ymid][3:end-2] .-  P.f[:, ymid][3:end-2]))
        @info "Error Pf: $(err)"

        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 600) )    
            step = 10
            ftsz = 15
            eps  = 1e-10

            ax    = Axis(fig[1,1:2], title=L"Horizontal $P^f$ profile")
            # lines!(ax, X.v.x, Ux_ana[:, ymid][2:end-1], color=:black)
            # scatter!(ax, X.v.x, V.x[:, ymid][2:end-1], color=:black)
            lines!(ax, X.c.x, Pf_ana[:, ymid][2:end-1], color=:black)
            scatter!(ax, X.c.x, P.f[:, ymid][2:end-1], color=:black)
            # lines!(ax, X.c.x, Pt_ana[:, ymid][2:end-1], color=:red)
            # scatter!(ax, X.c.x, P.t[:, ymid][2:end-1], color=:red)

            ax    = Axis(fig[2,1], aspect=DataAspect(), title=L"$P^f$ numerics", xlabel=L"x", ylabel=L"y")
            field = (P.f)[inx_c,iny_c].*sc.σ
            hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(Pf_ana)-eps, maximum(Pf_ana)+eps))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[3, 1], hm, label = L"$P^f$ numerics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax    = Axis(fig[2,2], aspect=DataAspect(), title=L"$P^f$ analytics", xlabel=L"x", ylabel=L"y")
            # field = log10.(abs.(R.pt.+eps)[inx_c,iny_c].*sc.σ)
            # field = Pf_ana
            hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[3, 2], hm, label = L"$P^f$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            display(fig) 
            DataInspector(fig)
        end
        with_theme(figure, theme_latexfonts())

        #-------------------------------------------# 

    end

    #--------------------------------------------#

    return P, Δ, (c=X.c.x, v=X.v.x), (c=X.c.y, v=X.v.y)
end

##################################
function Run()

    nc = (x=200, y=200)

    # Mode 0   
    P, Δ, x, y = main(nc);
    # save("/Users/tduretz/PowerFolders/_manuscripts/TwoPhasePressure/benchmark/SchmidTest.jld2", "x", x, "y", y, "P", P )

end

Run()
