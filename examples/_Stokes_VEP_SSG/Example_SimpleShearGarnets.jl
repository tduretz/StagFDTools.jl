using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using DifferentiationInterface
using TimerOutputs, CairoMakie, Interpolations, GridGeometryUtils

function splot(ax, x, y, u, v)
    intu, intv = linear_interpolation((x, y), u), linear_interpolation((x, y), v)
    f(x) = Point2f(intu(x...), intv(x...))
    return streamplot!(ax, f, x, y, colormap=:magma, arrow_size=0)
end

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x=600, y=600)

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 3
    materials = initialize_materials(nphases; compressible=true)
    materials.g .= [0.0, 0.0]
    materials.ρ .= [1.0, 1.0, 1.0]
    materials.n .= [1.0, 1.0, 1.0]
    materials.η0 .= [1e0, 1e4, 1e-1]
    materials.G .= [1e1, 1e1, 1e1]
    materials.β .= [1e-2, 1e-2, 1e-2]
    preprocess!(materials)

    # Material geometries
    garnets = (
        Hexagon((-0.075, 0.075), 0.100; θ=π / 4),
        Hexagon((0.04, -0.04), 0.075; θ=π / 4),
        Hexagon((0.18, -0.18), 0.120; θ=π / 4),
        Hexagon((-0.2, -0.19), 0.100; θ=π / 4),
        Hexagon((-0.21, -0.05), 0.050; θ=π / 4),
    )

    micas = (
        Rectangle((0.1, -0.1), 0.03, 0.07; θ=-π / 4), #0.1, -0.1, 0.03, 0.07, -45
    )

    # Time steps
    Δt0 = 0.5
    nt = 1

    # Newton solver
    niter = 3
    ϵ_nl = 1e-8
    α = LinRange(0.05, 1.0, 10)

    # Intialise field
    L = (x=1.0, y=1.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y, max=0.0)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)


    # Initial velocity & pressure field
    # Initial velocity & pressure
    @views a.V.x .= D_BC[1, 1] * a.X.vx_e.x .+ D_BC[1, 2] * a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1] * a.X.vy_e.x .+ D_BC[2, 2] * a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * a.X.v.x .+ D_BC[1, 2] * a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * a.X.v.x .+ D_BC[1, 2] * a.X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * a.X.v.x[1] .+ D_BC[2, 2] * a.X.v.y)
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * a.X.v.x[end] .+ D_BC[2, 2] * a.X.v.y)
    end

    # Set material geometry 
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([xc[i-1], yc[j-1]])

        for igeom in eachindex(garnets) # Garnets: phase 2
            if inside(𝐱, garnets[igeom])
                phases.c[i, j] = 2
            end
        end
        for igeom in eachindex(micas) # Micas: phase 3
            if inside(𝐱, micas[igeom])
                phases.c[i, j] = 3
            end
        end
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([xv[i-1], yv[j-1]])

        # Garnets: phase 2
        for igeom in eachindex(garnets) # Garnets: phase 2
            if inside(𝐱, garnets[igeom])
                phases.v[i, j] = 2
            end
        end

        for igeom in eachindex(micas) # Micas: phase 3
            if inside(𝐱, micas[igeom])
                phases.v[i, j] = 3
            end
        end
    end
    phase_ratios = InitialisePhaseRatios(phases, nphases)

    #--------------------------------------------#

    rvec = zeros(length(α))
    err = (x=zeros(niter), y=zeros(niter), p=zeros(niter))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        @printf("Step %04d\n", it)
        err.x .= 0.
        err.y .= 0.
        err.p .= 0.

        # Swap old values
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0 .= Pt

        compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, nphases)

        for iter = 1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
                @show extrema(λ̇.c)
                @show extrema(λ̇.v)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = norm(R.x[inx_Vx, iny_Vx]) / sqrt(nVx)
            err.y[iter] = norm(R.y[inx_Vy, iny_Vy]) / sqrt(nVy)
            err.p[iter] = norm(R.p[inx_c, iny_c]) / sqrt(nPt)
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
            𝐊 .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐 .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏 .= [M.Pt.Pt;]

            #--------------------------------------------#

            # Direct-iterative solver
            fu = -r[1:size(𝐊, 1)]
            fp = -r[size(𝐊, 1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu, ηb=1e3, niter_l=10, ϵ_l=1e-9)
            dx[1:size(𝐊, 1)] .= u
            dx[size(𝐊, 1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)

            UpdateSolution!(V, Pt, α[imin] * dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)

        end

        # Update pressure
        Pt .+= ΔPt.c

        #--------------------------------------------#

        # Principal stress
        σ1 = (x=zeros(size(Pt)), y=zeros(size(Pt)), v=zeros(size(Pt)))

        τxyc = 0.25 * (τ.xy[1:end-1, 1:end-1] .+ τ.xy[2:end-0, 1:end-1] .+ τ.xy[1:end-1, 2:end-0] .+ τ.xy[2:end-0, 2:end-0])

        @show size(τxyc)
        @show size(τ.xx)

        for i in inx_c, j in iny_c
            σ = @SMatrix[-Pt[i, j]+τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -Pt[i, j]+τ.yy[i, j] 0.; 0. 0. -Pt[i, j]+(-τ.xx[i, j]-τ.yy[i, j])]
            v = eigvecs(σ)
            σp = eigvals(σ)
            σ1
            scale = sqrt(v[1, 1]^2 + v[2, 1]^2)
            σ1.x[i, j] = v[1, 1] / scale
            σ1.y[i, j] = v[2, 1] / scale
            # σ3.x[i] = v[1,3]
            # σ3.y[i] = v[2,3]
            σ1.v[i] = σp[1]
            # σ3.v[i] = σp[3]
        end

        fig = Figure()
        ax = Axis(fig[1, 1], aspect=DataAspect())
        heatmap!(ax, xc, yc, Pt[inx_c, iny_c], colormap=:bluesreds)
        # heatmap!(ax, xc, yc,  phases.c[inx_c,iny_c], colormap=:bluesreds)
        st = 10
        # arrows!(ax, xc[1:st:end], yc[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.02, linewidth=1, color=:white)
        splot(ax, xc[1:st:end], yc[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end])
        display(fig)

    end

    display(to)

end


let
    # Boundary condition templates
    BCs = [
        :EW_periodic,
    ]

    # Boundary velocity gradient matrix
    D_BCs = [
        @SMatrix([0 1; 0 0]),
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end