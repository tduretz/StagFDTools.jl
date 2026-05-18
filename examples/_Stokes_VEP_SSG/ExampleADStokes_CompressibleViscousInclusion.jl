using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology 
using JLD2, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs
using ExactFieldSolutions

@views function main(n)
    #--------------------------------------------#

    # Characteristic scales
    sc  = (σ=1e0, t=1e0, L=1e0)

    # Resolution
    nc = (x = n, y = n)

    # Configuration for Stokes2D_Duretz2026 
    params = (ηm = 1.0, ηi = 1e-2, ξm = 1e0, ξi = 1e0, rc = 0.1, γ̇ = 0.0, ε̇ = -1.0)

    # Boundary velocity gradient matrix
    config = :all_Dirichlet
    D_BC   = @SMatrix( [params.ε̇   0;
                        0  -params.ε̇] )

    # Material parameters
    nphases  = 2
    materials = initialize_materials(nphases; compressible=true)
    materials.η0 .= [params.ηm, params.ηi]
    materials.ξ0 .= [params.ξm, params.ξi]
    materials.G  .= [1e50, 1e50]
    materials.β  .= [0.0, 0.0]
    materials.ρ  .= [0.0, 0.0]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Newton solver
    niter = 2
    ϵ_nl  = 1e-8
    α     = LinRange(0.05, 1.0, 10)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    set_boundaries_template!(type, config, nc)

    #--------------------------------------------#
    # Equation numbering
    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    #--------------------------------------------#
    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏  = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r  = zeros(nVx + nVy + nPt)

    #--------------------------------------------#
    # Intialise field
    L   = (x=1., y=1.)
    x   = (min=-L.x/2, max=L.x/2)
    y   = (min=-L.y/2, max=L.y/2)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    ξ       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    G       = (c  = zeros(size_c...), v  = zeros(size_v...))
    β       = (c  = zeros(size_c...), v  = zeros(size_v...))
    ρ       = (c  = zeros(size_c...), v  = zeros(size_v...))
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    ΔPt     = (c=zeros(size_c...), Vx = zeros(size_x...), Vy = zeros(size_y...))

    Dc      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    # Mesh coordinates
    xv  = LinRange(x.min,         x.max,         nc.x+1)
    yv  = LinRange(y.min,         y.max,         nc.y+1)
    xc  = LinRange(x.min+Δ.x/2,  x.max-Δ.x/2,  nc.x)
    yc  = LinRange(y.min+Δ.y/2,  y.max-Δ.y/2,  nc.y)
    xce = LinRange(x.min-Δ.x/2,  x.max+Δ.x/2,  nc.x+2)
    yce = LinRange(y.min-Δ.y/2,  y.max+Δ.y/2,  nc.y+2)
    xve = LinRange(x.min-Δ.x,    x.max+Δ.x,    nc.x+3)
    yve = LinRange(y.min-Δ.y,    y.max+Δ.y,    nc.y+3)

    # Initial velocity & pressure field
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc'
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    Pt[inx_c, iny_c ]  .= 0.
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)

    # Set material geometry
    phases = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))
    rad = params.rc + 1e-13
    phases.c[(xce.^2 .+ (yce').^2) .<= rad^2] .= 2
    phases.v[(xve.^2 .+ (yve').^2) .<= rad^2] .= 2
    phase_ratios = InitialisePhaseRatios(phases, nphases)

    # Analytics
    V_ana = (
        x = zero(BC.Vx),
        y = zero(BC.Vy),
    )
    Pt_ana = zero(BC.Pt)
    ϵV = (
        x   = zero(BC.Vx),
        y   = zero(BC.Vy),
    )
    ϵP   = zero(BC.Pt)

    # Get P analytics
    for i=1:size(BC.Pf,1), j=1:size(BC.Pf,2)
        sol = Stokes2D_Duretz2026( [xce[i], yce[j]]; params )
        Pt_ana[i,j] = sol.p
    end

    # Get Vx analytics
    for i=1:size(BC.Vx,1), j=2:size(BC.Vx,2)-1
        sol = Stokes2D_Duretz2026( [xve[i], xce[j-1]]; params )
        BC.Vx[i,j]   =  sol.V[1]
        V.x[i,j]     = sol.V[1]
        V_ana.x[i,j] = sol.V[1]
    end

    # Get Vy analytics
    for i=2:size(BC.Vy,1)-1, j=1:size(BC.Vy,2)
        sol = Stokes2D_Duretz2026( [xce[i-1], yve[j]]; params )
        BC.Vy[i,j]   = sol.V[2]
        V.y[i,j]     = sol.V[2]
        V_ana.y[i,j] = sol.V[2]
    end

    #--------------------------------------------#

    # Error monitoring, probing and timing
    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    to   = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        @printf("Step %04d\n", it)
        err.x .= 0.
        err.y .= 0.
        err.p .= 0.
        
        # Swap old values
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, nphases)

        for iter=1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = norm(R.p[inx_c,iny_c])/sqrt(nPt)
            max(err.x[iter], err.y[iter]) < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, ξ, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, ρ, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= [M.Pt.Pt;]   
            
            if iter==1 save("DebugInclusionTest.jld2", Dict("M" => M, "r" => r, "R" => R, "V" => V, "Pt" => Pt, "D" => 𝐷, "D_ctl" => 𝐷_ctl)) end
            
            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = -r[1:size(𝐊,1)]
            fp   = -r[size(𝐊,1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            dx[1:size(𝐊,1)]     .= u
            dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
        end

        # Update pressure
        Pt .+= ΔPt.c 

        # Remove mean 
        Pt[inx_c,iny_c]' .-= mean(Pt[inx_c,iny_c])

        # Compute errors
        ϵP[inx_c,iny_c] .= abs.(Pt_ana[inx_c,iny_c] .- Pt[inx_c,iny_c])
        ϵV.x[inx_Vx,iny_Vx] .= abs.(V_ana.x[inx_Vx,iny_Vx] .- V.x[inx_Vx,iny_Vx])
        ϵV.y[inx_Vy,iny_Vy] .= abs.(V_ana.y[inx_Vy,iny_Vy] .- V.y[inx_Vy,iny_Vy])

        @info mean(abs.(ϵV.x))
        @info mean(abs.(ϵV.y))
        @info mean(abs.(ϵP))

        Pt_viz = copy(Pt)
        Pt_viz[Pt.>maximum(Pt_ana)] .= maximum(Pt_ana)
        Pt_viz[Pt.<minimum(Pt_ana)] .= minimum(Pt_ana)

        Vx_viz = copy(V.x)
        Vx_viz[V.x.>maximum(V_ana.x)] .= maximum(V_ana.x)
        Vx_viz[V.x.<minimum(V_ana.x)] .= minimum(V_ana.x)

        Vy_viz = copy(V.y)
        Vy_viz[V.y.>maximum(V_ana.y)] .= maximum(V_ana.y)
        Vy_viz[V.y.<minimum(V_ana.y)] .= minimum(V_ana.y)
        #--------------------------------------------#

        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 900) )    
            step = 10
            ftsz = 15
            eps  = 1e-10

            ax    = Axis(fig[1,1], aspect=DataAspect(), title=L"$P$ numerics", xlabel=L"x", ylabel=L"y")
            field = (Pt_viz)[inx_c,iny_c].*sc.σ
            hm    = heatmap!(ax, xc, yc, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 1], hm, label = L"$P$ numerics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax    = Axis(fig[1,2], aspect=DataAspect(), title=L"$P$ analytics", xlabel=L"x", ylabel=L"y")
            field = (Pt_ana)[inx_c,iny_c].*sc.σ
            hm    = heatmap!(ax, xc, yc, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 2], hm, label = L"$P$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[1,3], aspect=DataAspect(), title=L"$P$ error", xlabel=L"x", ylabel=L"y")
            field = (ϵP)[inx_c,iny_c].*sc.σ
            hm    = heatmap!(ax, xc, yc, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 3], hm, label = L"$P$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ###########################
            ax    = Axis(fig[3,1], aspect=DataAspect(), title=L"$V_{x}$ numerics", xlabel=L"x", ylabel=L"y")
            field = (Vx_viz)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, xv, yc, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 1], hm, label = L"$V_{x}$ numerics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax    = Axis(fig[3,2], aspect=DataAspect(), title=L"$V_{x}$ analytics", xlabel=L"x", ylabel=L"y")
            field = (V_ana.x)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, xv, yc, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 2], hm, label = L"$V_{x}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[3,3], aspect=DataAspect(), title=L"$V_{x}$ error", xlabel=L"x", ylabel=L"y")
            field = (ϵV.x)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, xv, yc, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 3], hm, label = L"$V_{x}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ###########################
            ax    = Axis(fig[5,1], aspect=DataAspect(), title=L"$V_{x}$ numerics", xlabel=L"x", ylabel=L"y")
            field = (Vy_viz)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, xv, yc, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[6, 1], hm, label = L"$V_{y}$ numerics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax    = Axis(fig[5,2], aspect=DataAspect(), title=L"$V_{x}$ analytics", xlabel=L"x", ylabel=L"y")
            field = (V_ana.y)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, xc, yv, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[6, 2], hm, label = L"$V_{y}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[5,3], aspect=DataAspect(), title=L"$V_{x}$ error", xlabel=L"x", ylabel=L"y")
            field = (ϵV.y)[inx_Vy,iny_Vy].*sc.σ
            hm    = heatmap!(ax, xc, yv, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[6, 3], hm, label = L"$V_{y}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            display(fig) 
            DataInspector(fig)
        end
        with_theme(figure, theme_latexfonts())

        # # Visulisation
        # p3 = heatmap(xv, yc, ϵV.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xv), title="Vx", color=:vik)
        # p4 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xv), title="Vx", color=:vik)
        # # p4 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xv), title="Vy", color=:vik)
        # p2 = heatmap(xc, yc, Pt[inx_c,iny_c], aspect_ratio=1, xlim=extrema(xv), title="Pt", color=:vik)
        # p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright, title="Convergence")
        # p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        # p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        # p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        # display(plot(p1, p2, p3, p4, layout=(2,2)))

    end
    display(to)
end


let
    # Run 
    @time main(101)
end