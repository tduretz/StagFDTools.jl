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

    # Newton solver
    niter = 2
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
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)

    # Allocations
    R = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V = (x=zeros(size_x...), y=zeros(size_y...))
    Vi = (x=zeros(size_x...), y=zeros(size_y...))
    η = (c=ones(size_c...), v=ones(size_v...))
    ξ = (c=ones(size_c...), v=ones(size_v...))
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

    # Mesh coordinates
    xv = LinRange(-L.x / 2, L.x / 2, nc.x + 1)
    yv = LinRange(-L.y / 2, L.y / 2, nc.y + 1)
    xc = LinRange(-L.x / 2 + Δ.x / 2, L.x / 2 - Δ.x / 2, nc.x)
    yc = LinRange(-L.y / 2 + Δ.y / 2, L.y / 2 - Δ.y / 2, nc.y)
    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))

    # Initial velocity & pressure field
    V.x[inx_Vx, iny_Vx] .= D_BC[1, 1] * xv .+ D_BC[1, 2] * yc'
    V.y[inx_Vy, iny_Vy] .= D_BC[2, 1] * xc .+ D_BC[2, 2] * yv'
    Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    BC.Vx[2, iny_Vx] .= (type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
    BC.Vx[end-1, iny_Vx] .= (type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
    BC.Vx[inx_Vx, 2] .= (type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * xv .+ D_BC[1, 2] * yv[1]) * 1
    BC.Vx[inx_Vx, end-1] .= (type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * xv .+ D_BC[1, 2] * yv[end]) * 1
    BC.Vy[inx_Vy, 2] .= (type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
    BC.Vy[inx_Vy, end-1] .= (type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
    BC.Vy[2, iny_Vy] .= (type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * xv[1] .+ D_BC[2, 2] * yv) * 1
    BC.Vy[end-1, iny_Vy] .= (type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * xv[end] .+ D_BC[2, 2] * yv) * 1

    # Set material geometry 
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([xc[i-1], yc[j-1]])
        for inc in eachindex(inclusions)
            if inside(𝐱, inclusions[inc])
                phases.c[i, j] = phase[inc]
            end
        end
    end

    for i in inx_v, j in iny_v   # loop on centroids
        𝐱 = @SVector([xv[i-1], yv[j-1]])
        for inc in eachindex(inclusions)
            if inside(𝐱, inclusions[inc])
                phases.v[i, j] = phase[inc]
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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol, ηb=1e3, niter_l=10, ϵ_l=1e-11)
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

        fig = Figure()
        ax = Axis(fig[1, 1], title="Pt", aspect=DataAspect())
        heatmap!(ax, xc, yc, Pt[inx_c, iny_c] .- mean(Pt[inx_c, iny_c]), colormap=:turbo, colorrange=(-5, 5))
        # heatmap!(ax, xc, yc,  Pt[inx_c,iny_c].-mean( Pt[inx_c,iny_c]), colormap=:bluesreds, colorrange=(-10,10))
        display(fig)

        # Psave = Pt[inx_c,iny_c].-mean( Pt[inx_c,iny_c])
        # @save "MultiInclusions_StagFD.jld2" P=Psave

        # p3 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xv), title="Vx", color=:vik)
        # p4 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy", color=:vik)
        # p2 = heatmap(xc, yc,  Pt[inx_c,iny_c]'.-mean( Pt[inx_c,iny_c]), aspect_ratio=1, xlim=extrema(xc), title="Pt", color=:vik)
        # p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright, title=BC_template)
        # p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        # p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        # p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        # display(plot(p1, p2, p3, p4, layout=(2,2)))

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