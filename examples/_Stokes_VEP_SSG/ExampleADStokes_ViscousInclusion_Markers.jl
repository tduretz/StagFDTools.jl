using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using DifferentiationInterface
using TimerOutputs
using GridGeometryUtils

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x=50, y=50) # number of cells
    nmpc = (x=4, y=4)  # markers per cell
    noise = false         # noise in marker distribution

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; compressible=false)

    # Parameters
    params_bg = (ρ=1.0, n=1.0, η0=1e0, G=1e6, β=0.5)
    params_in = (ρ=1.0, n=1.0, η0=1e5, G=1e6, β=0.5)

    materials.g .= [0., 0.]
    materials.ρ .= [params_bg.ρ, params_in.ρ]
    materials.n .= [params_bg.n, params_in.n]
    materials.η0 .= [params_bg.η0, params_in.η0]
    materials.G .= [params_bg.G, params_in.G]
    materials.β .= [params_bg.β, params_in.β]

    preprocess!(materials)

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
    L = (x=1.0, y=1.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

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
    Grid = GenerateGrid(x, y, Δ, nc)

    # Initial velocity & pressure field
    @views V.x[inx_Vx, iny_Vx] .= D_BC[1, 1] * Grid.v.x .+ D_BC[1, 2] * Grid.c.x'
    @views V.y[inx_Vy, iny_Vy] .= D_BC[2, 1] * Grid.c.x .+ D_BC[2, 2] * Grid.v.y'
    Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    BC.Vx[2, iny_Vx] .= (type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
    BC.Vx[end-1, iny_Vx] .= (type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
    BC.Vx[inx_Vx, 2] .= (type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * Grid.v.x .+ D_BC[1, 2] * Grid.v.y[1])
    BC.Vx[inx_Vx, end-1] .= (type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * Grid.v.y .+ D_BC[1, 2] * Grid.v.y[end])
    BC.Vy[inx_Vy, 2] .= (type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
    BC.Vy[inx_Vy, end-1] .= (type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
    BC.Vy[2, iny_Vy] .= (type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * Grid.v.x[1] .+ D_BC[2, 2] * Grid.v.y)
    BC.Vy[end-1, iny_Vy] .= (type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * Grid.v.x[end] .+ D_BC[2, 2] * Grid.v.y)

    # --------------------------------------------#
    # Initialise marker field
    m = InitialiseParticleField(nc, nmpc, L, Δ, materials, noise)
    phase_ratios, phase_weights = InitialisePhaseRatios(m, ε̇)
    mphase = ones(Int64, m.num...)

    # Set material geometry
    # rad = 0.1 + 1e-13
    # mphase[(m.xm.^2 .+ (m.ym)'.^2) .<= rad^2] .= 2
    incl = Hexagon((0.0, 0.0), 0.2; θ=π / 10)
    for I in CartesianIndices(mphase)
        i, j = I[1], I[2]
        𝐱 = SVector(m.xm[i], m.ym[j])
        isin = inside(𝐱, incl)
        if isin
            mphase[I] = 2
        end
    end

    # Set phase ratios on grid
    SetPhaseRatios!(phase_ratios, phase_weights, m, mphase, Grid.c_e.x, Grid.c_e.y, Grid.v_e.x, Grid.v_e.y, Δ)

    for I in CartesianIndices(phase_ratios.c)
        s = sum(phase_ratios.c[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.center at $I: sum = $s, values = $(phase_ratios.center[I])"
        end
    end

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
        compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, size_c, size_v, m.nphases)


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
        # Plot
        Fig = Figure(size=(1200, 900), fontsize=14)
        ax1 = Axis(Fig[1, 1], aspect=DataAspect(), title="Vx", xlabel="x", ylabel="y")
        ax2 = Axis(Fig[1, 3], aspect=DataAspect(), title="Vy", xlabel="x", ylabel="y")
        ax3 = Axis(Fig[2, 1], aspect=DataAspect(), title="Pt", xlabel="x", ylabel="y")
        # ax4 = Axis(Fig[2, 3], aspect=DataAspect(), title="Markers", xlabel="x", ylabel="y")
        ax4 = Axis(Fig[2, 3], xlabel="Iterations step $(it)", ylabel="log₁₀ error")

        hm1 = heatmap!(ax1, Grid.v.x, Grid.c.y, V.x[inx_Vx, iny_Vx]', colormap=:redsblues)
        hm2 = heatmap!(ax2, Grid.c.x, Grid.c.y, V.y[inx_Vy, iny_Vy]', colormap=:redsblues)
        hm3 = heatmap!(ax3, Grid.c.x, Grid.c.y, Pt[inx_c, iny_c]' .- mean(Pt[inx_c, iny_c]), colormap=:redsblues)
        Colorbar(Fig[1, 2], hm1)
        Colorbar(Fig[1, 4], hm2)
        Colorbar(Fig[2, 2], hm3)

        # hm4 = heatmap!(ax4, m.xm, m.ym, mphase, colormap=:viridis)

        scatter!(ax4, 1:niter, log10.(err.x[1:niter]), markersize=8, label="Vx")
        scatter!(ax4, 1:niter, log10.(err.y[1:niter]), markersize=8, label="Vy")
        scatter!(ax4, 1:niter, log10.(err.p[1:niter]), markersize=8, label="Pt")
        axislegend(ax4, position=:rt)
        display(Fig)
    end

    display(to)

end


let
    # # Boundary condition templates
    BCs = [
        :free_slip,
    ]

    # # Boundary deformation gradient matrix
    # D_BCs = [
    #     @SMatrix( [1 0; 0 -1] ),
    # ]

    # BCs = [
    #     # :EW_periodic,
    #     :all_Dirichlet,
    # ]

    # Boundary deformation gradient matrix
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