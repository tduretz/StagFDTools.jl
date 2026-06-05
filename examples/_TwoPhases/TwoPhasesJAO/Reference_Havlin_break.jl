using StagFDTools, StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, GridGeometryUtils
import Statistics:mean
@views function main(nc)

    sc  = (σ=1e6, t=1e10, L=1e3)        
    # sc  = (σ=1, t=1, L=1)
    cmy = 100*3600*25*365.25

    # Time steps
    nt     = 1
    Δt0    = 1e6/sc.t 

    # Newton solver
    niter = 25
    ϵ_nl  = 1e-8
    α     = LinRange(0.05, 1.0, 5)

    # Background strain rate
    ε̇       = 1e-30.*sc.t
    Pf_top  = 0e6/sc.σ 
    Pt_top  = 0e6/sc.σ 
    Pf_bot  = 1.2841920000000002e9 /sc.σ
    Pt_bot  = 1.366380288e9 /sc.σ

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )

    # Geometries
    L    = (x=10e3/sc.L, y=43.68e3/sc.L)
    x    = (min=-L.x/2, max=L.x/2)
    y    = (min=-L.y,   max=0.0)

    # Material parameters
    kill_elasticity = 1e30 # set to 1 to activate elasticity, set to large value to kill it
    kill_plasticity = 1e30

    k0 = 4.3103448275862073e-7
    
    materials = ( 
        g     = [0. -9.8] / (sc.L/sc.t^2),
        oneway       = false,
        compressible = true,
        plasticity   = :off,
        linearizeΦ   = false,              # !!!!!!!!!!!
        single_phase = false,
        conservative = false,
        #        mush
        Φ0    = [4e-2  ],
        m     = [-1.0  ]  * 1.0,
        n     = [1.0   ],
        n_CK  = [2.7   ] .* 1.0,
        η0   = [1e16  ]./sc.σ/sc.t, 
        ξ0   = [1e16  ]./sc.σ/sc.t,
        G     = [3e10  ] ./sc.σ .* kill_elasticity, 
        ρs    = [3200  ]/(sc.σ*sc.t^2/sc.L^2),
        ρf    = [3000  ]/(sc.σ*sc.t^2/sc.L^2),
        Ks    = [1e31  ] ./sc.σ .* kill_elasticity,
        KΦ    = [1e30  ] ./sc.σ .* kill_elasticity,
        Kf    = [1e30  ] ./sc.σ .* kill_elasticity, 
        k_ηf0 = [1.0   ] .* k0 ./(sc.L^2/sc.σ/sc.t), # a^2/58
        ϕ     = [35.   ].*1,
        ψ     = [10.   ].*1,
        C     = [1e7   ]./sc.σ * kill_plasticity,
        ηvp   = [0.0   ]./sc.σ/sc.t,
        cosϕ  = [0.0   ],
        sinϕ  = [0.0   ],
        sinψ  = [0.0   ],
    )

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
    type.Vx[inx_Vx,2]       .= :Neumann_tangent
    type.Vx[inx_Vx,end-1]   .= :Neumann_tangent
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy]  .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann_tangent
    type.Vy[end-1,iny_Vy]   .= :Neumann_tangent
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    # type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal
    type.Vy[inx_Vy,end]     .= :Neumann_normal
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1]   .= :in
    type.Pt[2:end-1,end]       .= :Dirichlet
    type.Pt[2:end-1,1]         .= :Dirichlet
    type.Pt[1,2:end-1]         .= :Neumann
    type.Pt[end,2:end-1]       .= :Neumann
    # type.Pt[2:end-1,[end-1]] .= :p_eff
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Neumann 
    type.Pf[end,:]           .= :Neumann 
    type.Pf[:,1]             .= :Dirichlet
    type.Pf[:,end]           .= :Dirichlet#:no_flux
    
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
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1;  1 1 1]),        @SMatrix([1 1 1;  1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1]),        @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1])),
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
    Φ   = (c=materials.Φ0[1]*ones(size_c...), v=materials.Φ0[1]*ones(size_v...) )
    Φ0  = (c=materials.Φ0[1]*ones(size_c...), v=materials.Φ0[1]*ones(size_v...) )
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
    P       = (t = zeros(size_c...), f = zeros(size_c...))
    Pi      = (t = zeros(size_c...), f = zeros(size_c...))
    P0      = (t = zeros(size_c...), f = zeros(size_c...))
    ΔP      = (t = zeros(size_c...), f = zeros(size_c...))
    ρ       = (t = zeros(size_c...), f = zeros(size_c...), s = zeros(size_c...))
    ρ0      = (t = zeros(size_c...), f = zeros(size_c...), s = zeros(size_c...))

    # Generate grid coordinates 
    X = GenerateGrid(x, y, Δ, nc)

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'

    for i in eachindex( Φ.c)  # loop on inner centroids
        phases.c[i] = 1
        Φ.c[i] = Φ_ini = materials.Φ0[phases.c[i]]
        ρ.f[i] = materials.ρf[phases.c[i]]
        ρ.t[i] = Φ_ini * materials.ρf[phases.c[i]] + (1-Φ_ini) * materials.ρs[phases.c[i]]
    end

    for i in inx_v, j in iny_v   # loop on centroids
        phases.v[i, j] = 1
        Φ.v[i, j] = materials.Φ0[phases.v[i, j]]
    end

    # Initial pressure fields
    for i in inx_c, j in (nc.y+2-1):-1:2
        # Interpolate densities at Vy points (midpoint)
        ρ̄f = 1/2 * (ρ.f[i,j+1] + ρ.f[i,j])   
        ρ̄t = 1/2 * (ρ.t[i,j+1] + ρ.t[i,j]) 
        
        # @show ρ̄t * materials.g[2] .* Δ.y * sc.σ
        # ∫ (-ρ̄ g) dz (g < 0)
        P.f[i,j] = P.f[i,j+1] - ρ̄f * materials.g[2] .* Δ.y
        P.t[i,j] = P.t[i,j+1] - ρ̄t * materials.g[2] .* Δ.y
    end

    @load "havlin_Stag1D_debug.jld2" Pt Pf τyy ϕ Vy Pt0 Pf0 τyy0 ϕ0
    @show norm(P.f[5,:]*sc.σ .- Pf0)
    @show norm(P.t[5,:]*sc.σ .- Pt0)

    # @show (P.f[5,1:3]*sc.σ)
    # @show Pf0[1:3]

    # @show P.t[5,end-3:end]*sc.σ
    # @show Pt0[end-3:end]

    # error()

    # @show ( P.t[5,:]*sc.σ .- Pt) 

    # @show P.t[5,end]*sc.σ, Pt[end]
    # @show P.t[5,1]*sc.σ, Pt[1]

    # @show P.t[5,end-1]*sc.σ, Pt[end-1]
    # @show P.t[5,2]*sc.σ, Pt[2]

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
    BC.Pf[     :,   end ] .= Pf_top
    BC.Pt[     :,     1 ] .= Pt_bot
    BC.Pt[     :,   end ] .= Pt_top
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

    # fig   = Figure(size = (400,600))
    # ftsz  = 18
    # eps   = 1e-13
    # ax    = Axis(fig[1,1], aspect=DataAspect(), title=L"$$total pressure", xlabel=L"x", ylabel=L"y")
    # field = Float64.(P.t[inx_c, iny_c]*sc.σ)/1e6
    # hm    = heatmap!(ax, X.c.x.*sc.L/1e3, X.c.y.*sc.L/1e3, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
    # hidexdecorations!(ax)
    # Colorbar(fig[1, 2], hm, label = L"$$total pressure", width=20, height = 200, labelsize = ftsz, ticklabelsize = ftsz )

    # ax    = Axis(fig[2,1], aspect=DataAspect(), title=L"$$fluid pressure", xlabel=L"x", ylabel=L"y")
    # field = Float64.(P.f[inx_c, iny_c]*sc.σ)/1e6
    # hm    = heatmap!(ax, X.c.x.*sc.L/1e3, X.c.y.*sc.L/1e3, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
    # hidexdecorations!(ax)
    # Colorbar(fig[2, 2], hm, label = L"$$fluid pressure", width=20, height = 200, labelsize = ftsz, ticklabelsize = ftsz )
    # display(fig)
    # DataInspector(fig)
    
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
        ρ0.f  .= ρ.f
        ρ0.s  .= ρ.s

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

            @load "havlin_Stag1D_debug.jld2" Pt Pf τyy ϕ Vy Pt0 Pf0 τyy0 ϕ0 rVy0 rPt0 rPf0

            rVy_1D = rVy0 / (sc.σ/sc.L)
            rPt_1D = rPt0 / (1/sc.t)
            rPf_1D = rPf0 / (1/sc.t)

            @show rVy_1D
            @show R.y[5,:]
            @show norm(R.y[5,:] .-  rVy_1D) ./ norm(rVy_1D)

            # @show rPt_1D
            # @show R.pt[5,:]
            @show norm(R.pt[5,:] .-  rPt_1D) ./ norm(rPt_1D)

            @show norm(R.pf[5,:] .-  rPf_1D) ./ norm(rPf_1D)


            @show rPf_1D
            @show R.pf[5,:]

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

        # Include plasticity corrections
        P.t  .= P.t .+ ΔP.t
        P.f  .= P.f .+ ΔP.f
        εp  .+= ε̇.II*Δ.t
        
        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])[2:end-1,2:end-1]
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])[2:end-1,2:end-1]
        Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)
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
      
        @load "havlin_DR_debug.jld2" Pt Pf τyy ϕ Vy Pt0 Pf0 τyy0 ϕ0  

        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 600) )    
            step = 10
            ftsz = 15
            eps  = 1e-10

            # ax   = Axis(fig[1,1], aspect=DataAspect(), title=L"$$Plastic strain rate", xlabel=L"x", ylabel=L"y")
            # field = log10.((λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            # ax   = Axis(fig[1,1], aspect=DataAspect(), title=L"$$von Mises strain", xlabel=L"x", ylabel=L"y")
            # field = P.t .- P.f #log10.(εp[inx_c,iny_c])
            # hm = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[2, 1], hm, label = L"$\dot\lambda$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            i_mid_x = Int64(round(nc.x/2))

            ax1   = Axis(fig[1,1], title=L"$$Deviatoric stress (MPa)", xlabel=L"\tau_{yy}", ylabel=L"y")
            hm = lines!(ax1, τ.yy[i_mid_x,2:end-1]*sc.σ/1e6,  X.c.y )
            hm = scatter!(ax1, τyy[2:end-1]/1e6,  X.c.y, marker=:xcross, markersize=20 )

            ax2   = Axis(fig[1,2], title=L"$\Delta P$ (MPa)", xlabel=L"$\Delta P$", ylabel=L"y")
            hm = lines!(ax2, ((P.f.-P.t)./(1 .- Φ.c))[i_mid_x,2:end-1]*sc.σ/1e6,  X.c.y )
            hm = scatter!(ax2, ((Pf.-Pt)./(1 .- ϕ))[2:end-1]/1e6,  X.c.y, marker=:xcross, markersize=20 )

            # ax2   = Axis(fig[1,2], title=L"$P$ (MPa)", xlabel=L"\Delta P", ylabel=L"y")
            # hm = scatter!(ax2, (P.t.-P.f)[i_mid_x,2:end-1]*sc.σ/1e6,  X.c.y )
            # # hm = lines!(ax2, (Pt.-Pf)[2:end-1]/1e6,  X.c.y )
            # hm = scatter!(ax2, (P.f)[i_mid_x,2:end-1]*sc.σ/1e6,  X.c.y )
            # hm = scatter!(ax2, (P.t)[i_mid_x,2:end-1]*sc.σ/1e6,  X.c.y )

        
            # # Top zoom
            # ylims!(ax2, -5, 1)
            # xlims!(ax2, -0, 100)

            # @show Pt[end-1]
            # @show Pf[end-1]
            # @show Pt[end-1] -  Pf[end-1]

            # # Bottom zoom
            # ylims!(ax2, -45, -40)
            # xlims!(ax2, 1300, 1400)


            ax3   = Axis(fig[2,1], title=L"$$Vertical velocity (cm/y)", xlabel=L"V_y", ylabel=L"y")
            hm = lines!(ax3, V.y[i_mid_x,2:end-1]*sc.L/sc.t*cmy,  X.v.y )
            hm = scatter!(ax3, Vy[2:end-1]*cmy,  X.v.y, marker=:xcross, markersize=20 )

            ax4   = Axis(fig[2,2], title=L"$$Porosity", xlabel=L"x", ylabel=L"ϕ")
            @load "havlin_ac.jld2" por_snapshot z
            # lines!(ax, por_snapshot[2:end-1], -z[2:end-1]./1e3, color=:green, label=L"$\phi$ Paris")
            hm = lines!(ax4, Φ.c[i_mid_x,2:end-1]*100,  X.c.y, label=L"$\phi$ StagFD 2D" )
            hm = scatter!(ax4, ϕ[2:end-1]*100,  X.c.y, label=L"$\phi$ DR code", marker=:xcross, markersize=20 )

            @load "havlin_Stag1D_debug.jld2" Pt Pf τyy ϕ Vy Pt0 Pf0 τyy0 ϕ0 

            hm = scatter!(ax1, τyy[2:end-1]/1e6,  X.c.y )
            hm = scatter!(ax2, ((Pf.-Pt)./(1 .- ϕ))[2:end-1]/1e6,  X.c.y )
            hm = scatter!(ax3, Vy[2:end-1]*cmy,  X.v.y )
            hm = scatter!(ax4, ϕ[2:end-1]*100,  X.c.y, label=L"$\phi$ StagFD 1D" )
            axislegend(position=:lt)

            display(fig) 
            DataInspector(fig)
        end
        with_theme(figure, theme_latexfonts())

        
        @show norm(P0.f[5,:]*sc.σ .- Pf0)
        @show norm(P0.t[5,:]*sc.σ .- Pt0)
        
        @show norm(P.f[5,:]*sc.σ .- Pf)
        @show norm(P.t[5,:]*sc.σ .- Pt)

        Pt_stag2D = (P.t[5,2:end-1].- mean(P.t[5,2:end-1]))*sc.σ 
        Pt_stag1D = Pt[2:end-1] .- mean(Pt[2:end-1]) 
        @show norm(Pt_stag2D .- Pt_stag1D)
               

        #-------------------------------------------# 

    end

    #--------------------------------------------#

    return 
end

function Run()

    nc = (x=11, y=51)

    # Mode 0   
    main(nc);
    
end

Run()
