using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2
using GridGeometryUtils
import Statistics: mean
using DifferentiationInterface
using TimerOutputs

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x=128, y=128)

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 3
    materials = initialize_materials(nphases; compressible=false)
    materials.g .= [0.0, 0.0]
    materials.ρ .= [1.0, 1.0, 1.0]
    materials.n .= [1.0, 1.0, 1.0]
    materials.η0 .= [1e0, 1e-3, 1e+3]
    materials.G .= [1e6, 1e6, 1e6]
    materials.β .= [1e-2, 1e-2, 1e-2]
    preprocess!(materials)

    phase = (3, 2, 2, 3, 2, 3, 2, 3, 3, 2)

    L = (x=1.0, y=1.0)
    inclusions = (
        Ellipse((0.0, 0.0), 0.2, 0.2; θ=0.0),
        Ellipse((0.2, 0.4), 0.09, 0.09; θ=0.0),
        Ellipse((-0.3, 0.4), 0.05, 0.05; θ=0.0),
        Ellipse((-0.4, -0.3), 0.08, 0.08; θ=0.0),
        Ellipse((0.0, -0.2), 0.08, 0.08; θ=0.0),
        Ellipse((-0.3, 0.2), 0.1, 0.1; θ=0.0),
        Ellipse((0.4, -0.2), 0.07, 0.07; θ=0.0),
        Ellipse((0.3, -0.4), 0.08, 0.08; θ=0.0),
        Ellipse((0.35, 0.2), 0.07, 0.07; θ=0.0),
        Ellipse((-0.1, -0.4), 0.07, 0.07; θ=0.0),
    )

    # With little shift upward to make inclusion cross boundary
    # inclusions = (
    #     Ellipse((0.0 , 0.0 ), 0.2 , 0.2 ; θ = 0.0),
    #     Ellipse((0.2 , 0.5 ), 0.09, 0.09; θ = 0.0),
    #     Ellipse((-0.3, 0.5 ), 0.05, 0.05; θ = 0.0),
    #     Ellipse((-0.4, -0.3), 0.08, 0.08; θ = 0.0),
    #     Ellipse((0.0 , -0.2), 0.08, 0.08; θ = 0.0),
    #     Ellipse((-0.3, 0.2 ), 0.1 , 0.1 ; θ = 0.0),
    #     Ellipse((0.4 , -0.2), 0.07, 0.07; θ = 0.0),
    #     Ellipse((0.3 , -0.4), 0.08, 0.08; θ = 0.0),
    #     Ellipse((0.35, 0.2 ), 0.07, 0.07; θ = 0.0),
    #     Ellipse((-0.1, -0.4), 0.07, 0.07; θ = 0.0),
    # )

    # Time steps
    Δt0 = 0.5
    nt = 1

    # Solver parameters
    iter_params = IterParams(niter=2, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field
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
        for inc in eachindex(inclusions)
            if inside(𝐱, inclusions[inc])
                a.phases.c[i, j] = phase[inc]
            end
        end
    end

    for i in inx_v, j in iny_v   # loop on vertices
        𝐱 = @SVector([a.X.v.x[i-1], a.X.v.y[j-1]])
        for inc in eachindex(inclusions)
            if inside(𝐱, inclusions[inc])
                a.phases.v[i, j] = phase[inc]
            end
        end
    end

    FillPhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        fig = Figure()
        ax = Axis(fig[1, 1], title="Pt", aspect=DataAspect())
        heatmap!(ax, a.X.c.x, a.X.c.y, a.Pt[inx_c, iny_c] .- mean(a.Pt[inx_c, iny_c]), colormap=:turbo, colorrange=(-5, 5))
        display(fig)

    end

    display(to)

end


let
    # # Boundary condition templates
    BCs = [
        :free_slip,
        # :no_slip,
    ]

    # # Boundary deformation gradient matrix
    # D_BCs = [
    #     @SMatrix( [1 0; 0 -1] ),
    # ]

    # BCs = [
    #     # :EW_periodic,
    #     :all_Dirichlet,
    # ]

    # Boundary velocity gradient matrix
    er = -1
    # ∂𝐕∂𝐱 - velocity gradient tensor
    D_BCs = [
        #  @SMatrix( [0 1; 0  0] ),
        @SMatrix([er 0;        #    ∂Vx∂x ∂Vx∂y
            0 -er]),    #    ∂Vy∂x ∂Vy∂y  div(V) = 0 = ∂Vx∂x + ∂Vy∂y --> ∂Vy∂y = - ∂Vx∂x
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end
