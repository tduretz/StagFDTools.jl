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
    # niter = 3
    # ϵ_nl = 1e-8
    # α = LinRange(0.05, 1.0, 10)
    iter_params = IterParams()


    # Intialise field
    L = (x=1.0, y=1.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y, max=0.0)

    nmpc = (x=4, y=4)  # markers per cell
    noise = false         # noise in marker distribution

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases, nmpc, noise)

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
        𝐱 = @SVector([a.X.c.x[i-1], a.X.c.y[j-1]])

        for igeom in eachindex(garnets) # Garnets: phase 2
            if inside(𝐱, garnets[igeom])
                a.phases.c[i, j] = 2
            end
        end
        for igeom in eachindex(micas) # Micas: phase 3
            if inside(𝐱, micas[igeom])
                a.phases.c[i, j] = 3
            end
        end
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([a.X.v.x[i-1], a.X.v.y[j-1]])

        # Garnets: phase 2
        for igeom in eachindex(garnets) # Garnets: phase 2
            if inside(𝐱, garnets[igeom])
                a.phases.v[i, j] = 2
            end
        end

        for igeom in eachindex(micas) # Micas: phase 3
            if inside(𝐱, micas[igeom])
                a.phases.v[i, j] = 3
            end
        end
    end
    # Set phase ratios on grid
    SetPhaseRatios!(a.phase_ratios, a.m, a.X.c_e.x, a.X.c_e.y, a.X.v_e.x, a.X.v_e.y, Δ, nphases)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Principal stress
        σ1 = (x=zeros(size(a.Pt)), y=zeros(size(a.Pt)), v=zeros(size(a.Pt)))

        τxyc = 0.25 * (a.τ.xy[1:end-1, 1:end-1] .+ a.τ.xy[2:end-0, 1:end-1] .+ a.τ.xy[1:end-1, 2:end-0] .+ a.τ.xy[2:end-0, 2:end-0])

        @show size(τxyc)
        @show size(a.τ.xx)

        for i in inx_c, j in iny_c
            σ = @SMatrix[-a.Pt[i, j]+a.τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -a.Pt[i, j]+a.τ.yy[i, j] 0.; 0. 0. -a.Pt[i, j]+(-a.τ.xx[i, j]-a.τ.yy[i, j])]
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
        heatmap!(ax, a.X.c.x, a.X.c.y, a.Pt[inx_c, iny_c], colormap=:bluesreds)
        # heatmap!(ax, a.X.c.x, a.X.c.y,  phases.c[inx_c,iny_c], colormap=:bluesreds)
        st = 10
        # arrows!(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.02, linewidth=1, color=:white)
        splot(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end])
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