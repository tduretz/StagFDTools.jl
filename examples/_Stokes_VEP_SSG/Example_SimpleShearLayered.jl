using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using DifferentiationInterface
using TimerOutputs, Interpolations, GridGeometryUtils
import CairoMakie as cm

@views function main(nc, layering, BC_template, D_template)
    #--------------------------------------------#

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 3
    materials = initialize_materials(nphases; compressible=false)
    materials.g .= [0., 0.]
    materials.ρ .= [1.0, 1.0, 1.0]
    materials.n .= [1.0, 1.0, 1.0]
    materials.η0 .= [2e0, 2 / 10, 1e-1]
    materials.G .= [1e6, 1e6, 1e6]
    materials.β .= [1e-6, 1e-6, 1e-6]
    materials.B .= [0., 0., 0.]
    preprocess!(materials)

    println(typeof(materials.plasticity))
    println(materials)

    # Time steps
    Δt0 = 0.5
    nt = 1

    # Solver parameters
    iter_params = IterParams(niter=3, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Intialise field
    L = (x=1.0, y=1.0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)
    Grid = a.X

    τII = ones(size_c...)
    ε̇II = ones(size_c...)

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
        isin = inside(𝐱, layering)
        if isin
            a.phases.c[i, j] = 2
        end
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([a.X.v.x[i-1], a.X.v.y[j-1]])
        isin = inside(𝐱, layering)
        if isin
            a.phases.v[i, j] = 2
        end
    end
    # Convert to phase ratios
    FillPhaseRatios!(a)

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

        τxyc = av2D(a.τ.xy)
        ε̇xyc = av2D(a.ε̇.xy)
        τII[inx_c, iny_c] .= sqrt.(0.5 .* (a.τ.xx[inx_c, iny_c] .^ 2 + a.τ.yy[inx_c, iny_c] .^ 2 + 0 * (-a.τ.xx[inx_c, iny_c] - a.τ.yy[inx_c, iny_c]) .^ 2) .+ τxyc[inx_c, iny_c] .^ 2)
        ε̇II[inx_c, iny_c] .= sqrt.(0.5 .* (a.ε̇.xx[inx_c, iny_c] .^ 2 + a.ε̇.yy[inx_c, iny_c] .^ 2 + 0 * (-a.ε̇.xx[inx_c, iny_c] - a.ε̇.yy[inx_c, iny_c]) .^ 2) .+ ε̇xyc[inx_c, iny_c] .^ 2)

        for i in inx_c, j in iny_c
            σ = @SMatrix[-a.Pt[i, j]+a.τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -a.Pt[i, j]+a.τ.yy[i, j] 0.; 0. 0. -a.Pt[i, j]+(-a.τ.xx[i, j]-a.τ.yy[i, j])]
            v = eigvecs(σ)
            σp = eigvals(σ)
            scale = sqrt(v[1, 1]^2 + v[2, 1]^2)
            σ1.x[i, j] = v[1, 1] / scale
            σ1.y[i, j] = v[2, 1] / scale
            σ1.v[i] = σp[1]
        end

        fig = cm.Figure()
        ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect())
        cm.heatmap!(ax, a.X.c.x, a.X.c.y, τII[inx_c, iny_c], colormap=:bluesreds)
        st = 10
        cm.arrows2d!(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
        display(fig)
    end

    # display(to)

    # Only account for the subdomain
    imin_x = argmin(abs.(Grid.c_e.x .+ 0.3))
    imax_x = argmin(abs.(Grid.c_e.x .- 0.3))
    imin_y = argmin(abs.(Grid.c_e.y .+ 0.3))
    imax_y = argmin(abs.(Grid.c_e.y .- 0.3))
    inner_x = imin_x:imax_x
    inner_y = imin_y:imax_y

    return mean(τII[inner_x, inner_y])

end

let
    # Boundary condition templates
    BCs = [
        # :EW_periodic,
        # :all_Dirichlet,
        :free_slip,
    ]

    # Boundary velocity gradient matrix
    D_BCs = [
        #  @SMatrix( [0 1; 0  0] ),
        @SMatrix([1 0; 0 -1]),
    ]

    nc = (x=200, y=200)

    # Discretise angle of layer
    nθ = 30
    θ = LinRange(0, π, nθ)
    τ_cart = zeros(nθ)

    # Run them all
    for iθ in eachindex(θ)

        layering = Layering(
            (0 * 0.25, 0.025),
            0.1,
            0.5;
            θ=θ[iθ],
            perturb_amp=0 * 1.0,
            perturb_width=1.0
        )
        τ_cart[iθ] = main(nc, layering, BCs[1], D_BCs[1])
    end

    ε̇bg = sqrt(sum(1 / 2 .* D_BCs[1][:] .^ 2))

    α1 = 0.5
    α2 = 1 - α1

    η1 = 2 / 10
    η2 = 2
    m = η2 / η1

    # Strongest end-member
    ηeff = α1 * η1 + α2 * η2
    @show τstrong = 2 * ηeff * ε̇bg

    # Strongest end-member
    ηeff = (α1 / η1 + α2 / η2)^(-1)
    @show τweak = 2 * ηeff * ε̇bg

    fig = cm.Figure()
    ax = cm.Axis(fig[1, 1], xlabel="θ", ylabel="τII") #, aspect=DataAspect()
    cm.lines!(ax, θ * 180 / π, τ_cart)
    cm.lines!(ax, θ * 180 / π, τstrong * ones(size(θ)))
    cm.lines!(ax, θ * 180 / π, τweak * ones(size(θ)))
    display(fig)

end
