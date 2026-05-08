using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using DifferentiationInterface
using TimerOutputs, Interpolations, GridGeometryUtils
import CairoMakie as cm

@views function main(nc, layering, BC_template, D_template, factorization)
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

    # Newton solver
    niter = 3
    ϵ_nl = 1e-8
    α = LinRange(0.05, 1.0, 10)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x + 3, nc.y + 4)),
        fill(:out, (nc.x + 4, nc.y + 3)),
        fill(:out, (nc.x + 2, nc.y + 2)),
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
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]), @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])),
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1; 1 1 1]), @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0; 0 1 0]), @SMatrix([0 0; 1 1; 0 0]), @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx = maximum(number.Vx)
    nVy = maximum(number.Vy)
    nPt = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)),
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)),
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊 = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐 = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏 = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r = zeros(nVx + nVy + nPt)

    #--------------------------------------------#
    # Intialise field
    L = (x=1.0, y=1.0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    Grid = GenerateGrid(x, y, Δ, nc)

    # Allocations
    R = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V = (x=zeros(size_x...), y=zeros(size_y...))
    Vi = (x=zeros(size_x...), y=zeros(size_y...))
    η = (c=ones(size_c...), v=ones(size_v...))
    G = (c=zeros(size_c...), v=zeros(size_v...))
    β = (c=zeros(size_c...), v=zeros(size_v...))
    ρ = (c=zeros(size_c...), v=zeros(size_v...))
    λ̇ = (c=zeros(size_c...), v=zeros(size_v...))
    ε̇ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    τ0 = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...))
    τ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    Pt = zeros(size_c...)
    Pti = zeros(size_c...)
    Pt0 = zeros(size_c...)
    ΔPt = (c=zeros(size_c...), Vx=zeros(size_x...), Vy=zeros(size_y...))

    Dc = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    Dv = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷 = (c=Dc, v=Dv)
    D_ctl_c = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    D_ctl_v = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷_ctl = (c=D_ctl_c, v=D_ctl_v)
    τII = ones(size_c...)
    ε̇II = ones(size_c...)

    # Initialize phase_ratios from discrete phases
    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    V.x[inx_Vx, iny_Vx] .= D_BC[1, 1] * Grid.v.x .+ D_BC[1, 2] * Grid.c.y'
    V.y[inx_Vy, iny_Vy] .= D_BC[2, 1] * Grid.c.x .+ D_BC[2, 2] * Grid.v.y'
    Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    BC.Vx[2, iny_Vx] .= (type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
    BC.Vx[end-1, iny_Vx] .= (type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
    BC.Vx[inx_Vx, 2] .= (type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * Grid.v.x .+ D_BC[1, 2] * Grid.v.y[1])
    BC.Vx[inx_Vx, end-1] .= (type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * Grid.v.x .+ D_BC[1, 2] * Grid.v.y[end])
    BC.Vy[inx_Vy, 2] .= (type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
    BC.Vy[inx_Vy, end-1] .= (type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
    BC.Vy[2, iny_Vy] .= (type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * Grid.v.x[1] .+ D_BC[2, 2] * Grid.v.y)
    BC.Vy[end-1, iny_Vy] .= (type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * Grid.v.x[end] .+ D_BC[2, 2] * Grid.v.y)

    # Set material geometry 
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([Grid.c.x[i-1], Grid.c.y[j-1]])
        isin = inside(𝐱, layering)
        if isin
            phases.c[i, j] = 2
        end
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([Grid.v.x[i-1], Grid.v.y[j-1]])
        isin = inside(𝐱, layering)
        if isin
            phases.v[i, j] = 2
        end
    end
    # Convert to phase ratios
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

        # Compute bulk and shear moduli
        compute_grid_fields!(G, β, ρ, materials, phase_ratios, nc, size_c, size_v, nphases)

        for iter = 1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, materials, number, type, BC, nc, Δ)
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
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, ρ, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=factorization, ηb=1e3, niter_l=10, ϵ_l=1e-9)
            dx[1:size(𝐊, 1)] .= u
            dx[size(𝐊, 1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)

            UpdateSolution!(V, Pt, α[imin] * dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)

        end

        # Update pressure
        Pt .+= ΔPt.c

        #--------------------------------------------#

        # Principal stress
        σ1 = (x=zeros(size(Pt)), y=zeros(size(Pt)), v=zeros(size(Pt)))

        τxyc = av2D(τ.xy)
        ε̇xyc = av2D(ε̇.xy)
        τII[inx_c, iny_c] .= sqrt.(0.5 .* (τ.xx[inx_c, iny_c] .^ 2 + τ.yy[inx_c, iny_c] .^ 2 + 0 * (-τ.xx[inx_c, iny_c] - τ.yy[inx_c, iny_c]) .^ 2) .+ τxyc[inx_c, iny_c] .^ 2)
        ε̇II[inx_c, iny_c] .= sqrt.(0.5 .* (ε̇.xx[inx_c, iny_c] .^ 2 + ε̇.yy[inx_c, iny_c] .^ 2 + 0 * (-ε̇.xx[inx_c, iny_c] - ε̇.yy[inx_c, iny_c]) .^ 2) .+ ε̇xyc[inx_c, iny_c] .^ 2)

        for i in inx_c, j in iny_c
            σ = @SMatrix[-Pt[i, j]+τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -Pt[i, j]+τ.yy[i, j] 0.; 0. 0. -Pt[i, j]+(-τ.xx[i, j]-τ.yy[i, j])]
            v = eigvecs(σ)
            σp = eigvals(σ)
            scale = sqrt(v[1, 1]^2 + v[2, 1]^2)
            σ1.x[i, j] = v[1, 1] / scale
            σ1.y[i, j] = v[2, 1] / scale
            σ1.v[i] = σp[1]
        end

        fig = cm.Figure()
        ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect())
        cm.heatmap!(ax, Grid.c.x, Grid.c.y, τII[inx_c, iny_c], colormap=:bluesreds)
        st = 10
        cm.arrows2d!(ax, Grid.c.x[1:st:end], Grid.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
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
        τ_cart[iθ] = main(nc, layering, BCs[1], D_BCs[1], :chol)
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