using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils, MAT
import Statistics: mean
using DifferentiationInterface
using TimerOutputs, CairoMakie

@views function main(nc)
    #--------------------------------------------#

    # Resolution

    # Load data
    filepath = joinpath(@__DIR__, "DataM2Di_EP_test01.mat")
    data = matread(filepath)
    @show keys(data)

    # Scales
    sc = (σ=3e10, L=1e3, t=1e10)

    # Boundary loading type
    config = :free_slip
    ε̇bg = 5.0e-15 .* sc.t
    D_BC = @SMatrix([-ε̇bg 0.;
        0. ε̇bg])

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; compressible=true, plasticity=DruckerPrager)
    materials.n .= [1.0, 1.0]            # Power law exponent
    materials.η0 .= [1e30, 1e30] ./ sc.σ / sc.t # Reference viscosity 
    materials.ξ0 .= [1e60, 1e60] ./ sc.σ / sc.t
    materials.G .= [1e10, 0.25e10] ./ sc.σ      # Shear modulus
    materials.plasticity.C .= [3e7, 3e7] ./ sc.σ      # Cohesion
    # materials.plasticity.σT .= [5e6, 5.0e6] ./ sc.σ  # Kiss2023 / Tensile / Hyperbolic
    materials.plasticity.ϕ .= [30., 30.]            # Friction angle
    materials.plasticity.ψ .= [10., 10.0]            # Dilation angle
    materials.plasticity.ηvp .= [1e19, 1e19] .* 0.0 ./ sc.σ / sc.t # Viscoplastic regularisation
    materials.β .= [5e-11, 5e-11] .* sc.σ      # Compressibility
    preprocess!(materials)

    # Geometry
    seed = (
        Ellipse((0.0, -1e3 / sc.L), 100 / sc.L, 100 / sc.L; θ=0.0),
    )
    nmpc = (x=4, y=4)
    noise = false

    # Time steps
    Δt0 = 1e10 / sc.t
    nt = 35

    # Newton solver
    niter = 20
    ϵ_nl = 1e-11
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
    L = (x=4e3 / sc.L, y=2e3 / sc.L)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y, max=0.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)

    # Allocations
    R = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V = (x=zeros(size_x...), y=zeros(size_y...))
    Vi = (x=zeros(size_x...), y=zeros(size_y...))
    η = (c=ones(size_c...), v=ones(size_v...))
    ξ = (c=ones(size_c...), v=ones(size_v...))
    G = (c=ones(size_c...), v=ones(size_v...))
    ρ = (c=ones(size_c...), v=ones(size_v...))
    β = (c=ones(size_c...), v=ones(size_v...))
    λ̇ = (c=zeros(size_c...), v=zeros(size_v...))
    λ = (c=zeros(size_c...), v=zeros(size_v...))
    ε̇ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    τ0 = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...))
    τ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    Pt = zeros(size_c...)
    Pti = zeros(size_c...)
    Pt0 = zeros(size_c...)
    ΔPt = (c=zeros(size_c...), Vx=zeros(size_x...), Vy=zeros(size_y...))

    bifurc = (detA=zeros(size_c...), θ=zeros(size_c...))

    Dc = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    Dv = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷 = (c=Dc, v=Dv)
    D_ctl_c = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    D_ctl_v = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷_ctl = (c=D_ctl_c, v=D_ctl_v)

    # Mesh coordinates
    X = GenerateGrid(x, y, Δ, nc)
    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    @show size(V.x), size(X.v_e.x), size(X.c_e.y)
    @views V.x .= D_BC[1, 1] * X.vx_e.x .+ D_BC[1, 2] * X.vx_e.y'
    @views V.y .= D_BC[2, 1] * X.vy_e.x .+ D_BC[2, 2] * X.vy_e.y'
    @views Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * X.v.x .+ D_BC[1, 2] * X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * X.v.x .+ D_BC[1, 2] * X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * X.v.x[1] .+ D_BC[2, 2] * X.v.y)
        BC.Vy[end-1, iny_Vy] .= (type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * X.v.x[end] .+ D_BC[2, 2] * X.v.y)
    end

    # NO MARKERS:
    # Set material geometry 
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([X.c_e.x[i], X.c_e.y[j]])
        for igeom in eachindex(seed) # seed
            if inside(𝐱, seed[igeom])
                phases.c[i, j] = 2
            end
        end
    end

    for i in inx_c, j in iny_c  # loop on vertices
        𝐱 = @SVector([X.v_e.x[i], X.v_e.y[j]])

        for igeom in eachindex(seed) # seed
            if inside(𝐱, seed[igeom])
                phases.v[i, j] = 2
            end
        end
    end
    phase_ratios = InitialisePhaseRatios(phases, nphases)

    # # YES MARKERS:
    # m = InitialiseParticleField(nc, nmpc, L, Δ, x, y, noise) #    |
    # phase_ratios, phase_weights = InitialisePhaseRatios(nphases, ε̇) #        |

    # for I in eachindex(m.phase)
    #     xm = m.Xm[I]
    #     ym = m.Ym[I]
    #     𝐱 = @SVector([xm, ym])
    #     for igeom in eachindex(seed) # seed
    #         if inside(𝐱, seed[igeom])
    #             m.phase[I] = 2
    #         end
    #     end
    # end
    # # Set phase ratios
    # SetPhaseRatios!(phase_ratios, phase_weights, m, X.c_e.x, X.c_e.y, X.v_e.x, X.v_e.y, Δ, nphases)


    Pt .= 0.0
    Pt0 .= Pt
    Pti .= Pt

    #--------------------------------------------#

    rvec = zeros(length(α))
    err = (x=zeros(niter), y=zeros(niter), p=zeros(niter))
    probes = (τII=zeros(nt), fric=zeros(nt), t=zeros(nt), str=zeros(nt), λ=zeros(nt))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        @printf("Step %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)

        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0 .= Pt

        # Compute material properties on grid
        compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, nphases)

        for iter = 1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
                @show extrema(λ̇.c[inx_c, iny_c])
                @show extrema(λ̇.v[inx_v, iny_v])
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
                @show extrema(λ̇.c[inx_c, iny_c])
                @show extrema(λ̇.v[inx_v, iny_v])
            end

            err.x[iter] = @views norm(R.x[inx_Vx, iny_Vx]) / sqrt(nVx)
            err.y[iter] = @views norm(R.y[inx_Vy, iny_Vy]) / sqrt(nVy)
            err.p[iter] = @views norm(R.p[inx_c, iny_c]) / sqrt(nPt)
            max(min(err.x[iter] / err.x[1], err.x[iter]), min(err.y[iter] / err.y[1], err.y[iter])) < ϵ_nl ? break : nothing

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
            𝐏 .= M.Pt.Pt

            #--------------------------------------------#

            # Direct-iterative solver
            fu = @views -r[1:size(𝐊, 1)]
            fp = @views -r[size(𝐊, 1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu, ηb=1e3, niter_l=10, ϵ_l=1e-11)
            @views dx[1:size(𝐊, 1)] .= u
            @views dx[size(𝐊, 1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)
            UpdateSolution!(V, Pt, α[imin] * dx, number, type, nc)

        end

        # Update pressure
        Pt .+= ΔPt.c
        λ.c .= λ̇.c
        λ.v .= λ̇.v

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = av2D(τ.xy)
        τII = sqrt.(0.5 .* (τ.xx[inx_c, iny_c] .^ 2 + τ.yy[inx_c, iny_c] .^ 2 + (-τ.xx[inx_c, iny_c] - τ.yy[inx_c, iny_c]) .^ 2) .+ τxyc[inx_c, iny_c] .^ 2)
        ε̇xyc = av2D(ε̇.xy)
        ε̇II = sqrt.(0.5 .* (ε̇.xx[inx_c, iny_c] .^ 2 + ε̇.yy[inx_c, iny_c] .^ 2 + (-ε̇.xx[inx_c, iny_c] - ε̇.yy[inx_c, iny_c]) .^ 2) .+ ε̇xyc[inx_c, iny_c] .^ 2)

        # Principal stress
        σ1 = (x=zeros(size(Pt)), y=zeros(size(Pt)), v=zeros(size(Pt)))
        τxyc = 0.25 * (τ.xy[1:end-1, 1:end-1] .+ τ.xy[2:end-0, 1:end-1] .+ τ.xy[1:end-1, 2:end-0] .+ τ.xy[2:end-0, 2:end-0])

        for i in inx_c, j in iny_c
            σ = @SMatrix[-Pt[i, j]+τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -Pt[i, j]+τ.yy[i, j] 0.; 0. 0. -Pt[i, j]+(-τ.xx[i, j]-τ.yy[i, j])]
            v = eigvecs(σ)
            σp = eigvals(σ)
            scale = sqrt(v[1, 1]^2 + v[2, 1]^2)
            σ1.x[i, j] = v[1, 1] / scale
            σ1.y[i, j] = v[2, 1] / scale
            σ1.v[i] = σp[1]
        end

        # Store probes data
        probes.t[it] = it * Δ.t
        probes.τII[it] = mean(τII)
        probes.λ[it] = mean(λ.c[inx_c, iny_c])
        probes.str[it] = ε̇bg * it * Δ.t
        i_midx = Int64(floor(nc.x))
        probes.fric[it] = mean(.-τxyc[i_midx, end-3] ./ (-Pt[i_midx, end-3] .+ τ.yy[i_midx, end-3]))

        # Bifurcation analysis
        Te = @SMatrix([2/3 -1/3 0; -1/3 2/3 0; 0 0 1; 1 1 0])
        Ts = @SMatrix([1 0 0 -1; 0 1 0 -1; 0 0 1 0])
        θ = LinRange(0, 90, 180)
        detA = zeros(size(θ))
        for i in inx_c, j in iny_c

            D = SMatrix{1,1}(𝐷_ctl.c[ii, jj] for ii in i:i, jj in j:j)
            phase = phases.c[i, j]
            χe = 1 / materials.β[phase] * Δ.t
            C = @SMatrix([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -χe])
            𝐃ep = Ts * (D[1] * C) * Te

            for i in eachindex(θ)
                n = @SVector([cosd(θ[i]), sind(θ[i])])
                𝐧 = @SVector([n[1], n[2], 2 * n[1] * n[2]])
                # display( 𝐃ep )
                # error()
                detA[i] = det(𝐧' * 𝐃ep * 𝐧)
            end
            bifurc.detA[i, j] = detA[argmin(detA)]
            bifurc.θ[i, j] = abs(θ[argmin(detA)])
        end

        @info minimum(bifurc.detA[inx_c, iny_c])
        @info extrema(bifurc.θ[inx_c, iny_c])
        sleep(0.5)

        if minimum(bifurc.detA[inx_c, iny_c]) < 0
            @show extrema(bifurc.detA[inx_c, iny_c])
            error()
        end

        # Visualise
        function figure()
            fig = Figure()
            ax = Axis(fig[1:1, 1], aspect=DataAspect(), title="Pressure", xlabel="x", ylabel="y")
            # heatmap!(ax, X.c.x, X.c.y,  log10.(λ̇.c[inx_c,iny_c]), colormap=:bluesreds)
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            heatmap!(ax, X.c.x, X.c.y, Pt[inx_c, iny_c] * sc.σ, colormap=:jet, colorrange=(-6e6, 4e6))
            # heatmap!( ax, X.v_e.x, X.v_e.y, λ̇.v )
            # heatmap!(ax, X.c.x, X.c.y, bifurc.detA[inx_c,iny_c], colormap=:jet)

            # st = 10
            # arrows!(ax, X.c.x[1:st:end], X.c.y[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.04, linewidth=2, color=:white)
            ax = Axis(fig[2, 1], xlabel="Iterations @ step $(it) ", ylabel=L"$\log_{10}$ error")
            scatter!(ax, 1:niter, log10.(err.x[1:niter] ./ err.x[1]))
            scatter!(ax, 1:niter, log10.(err.y[1:niter] ./ err.y[1]))
            scatter!(ax, 1:niter, log10.(err.p[1:niter] ./ err.p[1]))
            ylims!(ax, -15, 1)

            ax = Axis(fig[1, 2], xlabel="Strain", ylabel="Mean stress invariant")
            lines!(ax, data["strvec"][1:nt], data["Tiivec"][1:nt])
            scatter!(ax, probes.str[1:2:nt], probes.τII[1:2:nt] * sc.σ)

            ax = Axis(fig[2, 2], xlabel="Strain", ylabel="Mean plastic strain rate")
            lines!(ax, data["strvec"][1:nt], data["dgvec"][1:nt])
            scatter!(ax, probes.str[1:2:nt], probes.λ[1:2:nt])

            display(fig)
        end
        with_theme(figure, theme_latexfonts())
        # @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))
    end

    display(to)

end

let
    main((x=150, y=75))
end