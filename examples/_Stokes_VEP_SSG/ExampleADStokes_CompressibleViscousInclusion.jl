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

    # Solver parameters
    iter_params = IterParams(niter=2, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    #--------------------------------------------#
    # Intialise field
    L   = (x=1., y=1.)
    x   = (min=-L.x/2, max=L.x/2)
    y   = (min=-L.y/2, max=L.y/2)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    nVx = maximum(a.number.Vx)
    nVy = maximum(a.number.Vy)
    nPt = maximum(a.number.Pt)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c]  .= 0.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal)  .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal)  .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Set material geometry
    rad = params.rc + 1e-13
    @views a.phases.c[(a.X.c_e.x.^2 .+ (a.X.c_e.y').^2) .<= rad^2] .= 2
    @views a.phases.v[(a.X.v_e.x.^2 .+ (a.X.v_e.y').^2) .<= rad^2] .= 2
    FillPhaseRatios!(a)

    # Analytics
    V_ana = (
        x = zero(BC.Vx),
        y = zero(BC.Vy),
    )
    Pt_ana = zero(a.Pt)
    ϵV = (
        x   = zero(BC.Vx),
        y   = zero(BC.Vy),
    )
    ϵP   = zero(a.Pt)

    # Get P analytics
    for i=1:size(a.Pt,1), j=1:size(a.Pt,2)
        sol = Stokes2D_Duretz2026( [a.X.c_e.x[i], a.X.c_e.y[j]]; params )
        Pt_ana[i,j] = sol.p
    end

    # Get Vx analytics
    for i=1:size(BC.Vx,1), j=2:size(BC.Vx,2)-1
        sol = Stokes2D_Duretz2026( [a.X.v_e.x[i], a.X.c_e.y[j-1]]; params )
        BC.Vx[i,j]   =  sol.V[1]
        a.V.x[i,j]   = sol.V[1]
        V_ana.x[i,j] = sol.V[1]
    end

    # Get Vy analytics
    for i=2:size(BC.Vy,1)-1, j=1:size(BC.Vy,2)
        sol = Stokes2D_Duretz2026( [a.X.c_e.x[i-1], a.X.v_e.y[j]]; params )
        BC.Vy[i,j]   = sol.V[2]
        a.V.y[i,j]   = sol.V[2]
        V_ana.y[i,j] = sol.V[2]
    end

    #--------------------------------------------#

    # Error monitoring, probing and timing
    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        # Remove mean
        a.Pt[inx_c,iny_c]' .-= mean(a.Pt[inx_c,iny_c])

        # Compute errors
        ϵP[inx_c,iny_c] .= abs.(Pt_ana[inx_c,iny_c] .- a.Pt[inx_c,iny_c])
        ϵV.x[inx_Vx,iny_Vx] .= abs.(V_ana.x[inx_Vx,iny_Vx] .- a.V.x[inx_Vx,iny_Vx])
        ϵV.y[inx_Vy,iny_Vy] .= abs.(V_ana.y[inx_Vy,iny_Vy] .- a.V.y[inx_Vy,iny_Vy])

        @info mean(abs.(ϵV.x))
        @info mean(abs.(ϵV.y))
        @info mean(abs.(ϵP))

        Pt_viz = copy(a.Pt)
        Pt_viz[a.Pt.>maximum(Pt_ana)] .= maximum(Pt_ana)
        Pt_viz[a.Pt.<minimum(Pt_ana)] .= minimum(Pt_ana)

        Vx_viz = copy(a.V.x)
        Vx_viz[a.V.x.>maximum(V_ana.x)] .= maximum(V_ana.x)
        Vx_viz[a.V.x.<minimum(V_ana.x)] .= minimum(V_ana.x)

        Vy_viz = copy(a.V.y)
        Vy_viz[a.V.y.>maximum(V_ana.y)] .= maximum(V_ana.y)
        Vy_viz[a.V.y.<minimum(V_ana.y)] .= minimum(V_ana.y)
        #--------------------------------------------#

        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 900) )
            step = 10
            ftsz = 15
            eps  = 1e-10

            ax    = Axis(fig[1,1], aspect=DataAspect(), title=L"$P$ numerics", xlabel=L"x", ylabel=L"y")
            field = (Pt_viz)[inx_c,iny_c].*sc.σ
            hm    = heatmap!(ax, a.X.c.x, a.X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 1], hm, label = L"$P$ numerics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[1,2], aspect=DataAspect(), title=L"$P$ analytics", xlabel=L"x", ylabel=L"y")
            field = (Pt_ana)[inx_c,iny_c].*sc.σ
            hm    = heatmap!(ax, a.X.c.x, a.X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 2], hm, label = L"$P$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[1,3], aspect=DataAspect(), title=L"$P$ error", xlabel=L"x", ylabel=L"y")
            field = (ϵP)[inx_c,iny_c].*sc.σ
            hm    = heatmap!(ax, a.X.c.x, a.X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 3], hm, label = L"$P$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ###########################
            ax    = Axis(fig[3,1], aspect=DataAspect(), title=L"$V_{x}$ numerics", xlabel=L"x", ylabel=L"y")
            field = (Vx_viz)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, a.X.v.x, a.X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 1], hm, label = L"$V_{x}$ numerics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[3,2], aspect=DataAspect(), title=L"$V_{x}$ analytics", xlabel=L"x", ylabel=L"y")
            field = (V_ana.x)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, a.X.v.x, a.X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 2], hm, label = L"$V_{x}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[3,3], aspect=DataAspect(), title=L"$V_{x}$ error", xlabel=L"x", ylabel=L"y")
            field = (ϵV.x)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, a.X.v.x, a.X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 3], hm, label = L"$V_{x}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ###########################
            ax    = Axis(fig[5,1], aspect=DataAspect(), title=L"$V_{x}$ numerics", xlabel=L"x", ylabel=L"y")
            field = (Vy_viz)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, a.X.v.x, a.X.c.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[6, 1], hm, label = L"$V_{y}$ numerics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[5,2], aspect=DataAspect(), title=L"$V_{x}$ analytics", xlabel=L"x", ylabel=L"y")
            field = (V_ana.y)[inx_Vx,iny_Vx].*sc.σ
            hm    = heatmap!(ax, a.X.c.x, a.X.v.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[6, 2], hm, label = L"$V_{y}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            ax    = Axis(fig[5,3], aspect=DataAspect(), title=L"$V_{x}$ error", xlabel=L"x", ylabel=L"y")
            field = (ϵV.y)[inx_Vy,iny_Vy].*sc.σ
            hm    = heatmap!(ax, a.X.c.x, a.X.v.y, field, colormap=(Makie.Reverse(:matter), 1), colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[6, 3], hm, label = L"$V_{y}$ analytics", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )

            display(fig)
            DataInspector(fig)
        end
        with_theme(figure, theme_latexfonts())

    end
    display(to)
end


let
    # Run
    @time main(101)
end
