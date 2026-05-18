using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils
import CairoMakie as cm

function Analytical(θ, η, δ, D_BC)
    #= define velocity gradient components and resulting deviatoric strain rate components
    pure shear   ε̇ = [ε̇xx  0 ;  0  -ε̇xx]
    simple shear ε̇ = [ 0  ε̇xy; ε̇xy   0 ] =#
    Dxx = D_BC[1, 1]
    Dyy = -Dxx
    Dxy = D_BC[1, 2]
    Dkk = Dxx + Dyy

    ε̇ = @SVector([Dxx - Dkk / 3, Dyy - Dkk / 3, Dxy])

    # Normal vector of anisotropic direction
    n1 = -cos(θ)
    n2 = sin(θ)

    # compute isotropic and layered components for 𝐷
    Δ0 = 2 * n1^2 * n2^2
    Δ1 = n1 * n2^3 - n2 * n1^3
    Δ = @SMatrix([Δ0 -Δ0 2*Δ1; -Δ0 Δ0 -2*Δ1; Δ1 -Δ1 1-2*Δ0])
    A = @SMatrix([1 0 0; 0 1 0; 0 0 1])

    # compute 𝐷
    𝐷 = 2 * η * A - 2 * (η - η / δ) * Δ

    τ = 𝐷 * ε̇

    τ_II = sqrt(0.5 * (τ[1]^2 + τ[2]^2 + (-τ[1] - τ[2])^2) + τ[3]^2)
    return τ_II
end

@views function main(nc, nt, layering, BC_template, D_template, factorization, η1, η2, G1, G2, C1, C2)
    #--------------------------------------------#   

    # Boundary loading type
    config = BC_template
    D_BC = D_template

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; plasticity=VonMises, compressible=false)
    materials.η0 .= [η1, η2]
    materials.G .= [G1, G2]
    materials.plasticity.C .= [C1, C2]
    preprocess!(materials)

    nmpc = (x=4, y=4)
    noise = false

    # Time steps
    Δt0 = 0.5

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
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocations
    R = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V = (x=zeros(size_x...), y=zeros(size_y...))
    Vi = (x=zeros(size_x...), y=zeros(size_y...))
    η = (c=ones(size_c...), v=ones(size_v...))
    λ̇ = (c=zeros(size_c...), v=zeros(size_v...))
    ε̇ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    G = (c=zeros(size_c...), v=zeros(size_v...))
    β = (c=zeros(size_c...), v=zeros(size_v...))
    ξ = (c=zeros(size_c...), v=zeros(size_v...))
    ρ = (c=zeros(size_c...), v=zeros(size_v...))
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
    τIIev = ones(nt)
    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))

    # Mesh coordinates
    Grid = GenerateGrid(x, y, Δ, nc)

    # Only account for the subdomain
    imin_x = argmin(abs.(Grid.c_e.x .+ 0.3))
    imax_x = argmin(abs.(Grid.c_e.x .- 0.3))
    imin_y = argmin(abs.(Grid.c_e.y .+ 0.3))
    imax_y = argmin(abs.(Grid.c_e.y .- 0.3))
    inner_x = imin_x:imax_x
    inner_y = imin_y:imax_y

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

    # MARKERS ------------------------------------------------------------
    # Initialise markers and derive phase ratios from markers #        |
    m = InitialiseParticleField(nc, nmpc, L, Δ, materials, noise) #    |
    phase_ratios, phase_weights = InitialisePhaseRatios(m, ε̇) #        |
    mphase = ones(Int64, m.num...) #                                   |
    |
    # Assign marker phases from layering geometry (1 or 2) #           |
    for I in CartesianIndices(mphase) #                                |
        xm = m.Xm[I]
        ym = m.Ym[I]
        isin = inside(@SVector([xm, ym]), layering)
        mphase[I] = isin ? 2 : 1
    end

    # Build extended vertex arrays (with ghost vertices) and accumulate marker contributions
    PhaseRatios!(phase_ratios, phase_weights, m, mphase, Grid.c_e.x, Grid.c_e.y, Grid.v_e.x, Grid.v_e.y, Δ)

    # # NO MARKERS: --------------------------------------------------------
    # for i in inx_c, j in iny_c   # loop on centroids                     |
    #     𝐱 = @SVector([Grid.c.x[i-1], Grid.c.y[j-1]]) #                   |
    #     isin = inside(𝐱, layering) #                                     |
    #     if isin #                                                        |
    #         phases.c[i, j] = 2 #                                         |
    #     end
    # end

    # for i in inx_v, j in iny_v  # loop on vertices
    #     𝐱 = @SVector([Grid.v.x[i-1], Grid.v.y[j-1]])
    #     isin = inside(𝐱, layering)
    #     if isin
    #         phases.v[i, j] = 2
    #     end
    # end
    # phase_ratios = InitialisePhaseRatios(phases, nphases)

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

        # Compute material properties on grid
        compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, size_c, size_v, nphases)

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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=factorization, ηb=1e3, niter_l=10, ϵ_l=1e-9)
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

        for i in inx_c, j in iny_c
            τxyc = 1 / 4 * (τ.xy[i, j] + τ.xy[i+1, j] + τ.xy[i, j+1] + τ.xy[i+1, j+1])
            σ = @SMatrix[-Pt[i, j]+τ.xx[i, j] τxyc 0.; τxyc -Pt[i, j]+τ.yy[i, j] 0.; 0. 0. -Pt[i, j]+(-τ.xx[i, j]-τ.yy[i, j])]
            v = eigvecs(σ)
            σp = eigvals(σ)
            scale = sqrt(v[1, 1]^2 + v[2, 1]^2)
            σ1.x[i, j] = v[1, 1] / scale
            σ1.y[i, j] = v[2, 1] / scale
            σ1.v[i] = σp[1]
        end
        τIIev[it] = mean(τ.II[inner_x, inner_y])

        fig = cm.Figure()
        ax = cm.Axis(fig[1, 1], aspect=cm.DataAspect())
        hm = cm.heatmap!(ax, Grid.c.x, Grid.c.y, τ.II[inx_c, iny_c], colormap=:bluesreds)
        cm.poly!(ax, cm.Rect(Grid.c_e.x[imin_x], Grid.c_e.y[imin_y], Grid.c_e.x[imax_x] - Grid.c_e.x[imin_x], Grid.c_e.y[imax_y] - Grid.c_e.y[imin_y]), strokecolor=:white, strokewidth=2, color=:transparent)
        st = 15
        cm.arrows2d!(ax, Grid.c.x[1:st:end], Grid.c.y[1:st:end], σ1.x[inx_c, iny_c][1:st:end, 1:st:end], σ1.y[inx_c, iny_c][1:st:end, 1:st:end], tiplength=0, lengthscale=0.02, tipwidth=1, color=:white)
        cm.Colorbar(fig[1, 2], hm, label="τII")
        ax2 = cm.Axis(fig[1, 3], aspect=cm.DataAspect())
        hm2 = cm.heatmap!(ax2, Grid.c.x, Grid.c.y, η.c[inx_c, iny_c], colormap=:bluesreds)
        cm.Colorbar(fig[1, 4], hm2, label="η")
        ax3 = cm.Axis(fig[2, 1], aspect=cm.DataAspect())
        hm3 = cm.heatmap!(ax3, Grid.c.x, Grid.c.y, V.x[inx_Vx, iny_Vx], colormap=:bluesreds)
        cm.Colorbar(fig[2, 2], hm3, label="Vx")
        ax4 = cm.Axis(fig[2, 3], aspect=cm.DataAspect())
        hm4 = cm.heatmap!(ax4, Grid.c.x, Grid.c.y, V.y[inx_Vx, iny_Vx], colormap=:bluesreds)
        cm.Colorbar(fig[2, 4], hm4, label="Vy")

        ax5 = cm.Axis(fig[3, 1:4])
        cm.xlims!(ax5, 0, nt)
        # cm.ylims!(ax5, 0, 2.5)
        cm.lines!(ax5, 1:it, τIIev[1:it])
        display(fig)
        display(fig)
    end

    # display(to)

    return mean(τ.II[inner_x, inner_y]), τIIev

end

let
    # Boundary condition templates
    BCs = [
        # :EW_periodic,
        # :all_Dirichlet,
        :free_slip,
    ]

    # Boundary deformation gradient matrix
    D_BCs = [
        @SMatrix([1 0; 0 -1]),
    ]

    nc = (x=50, y=50)
    nt = 40

    # Discretise angle of layer 
    nθ = 30
    θ = LinRange(0, π, nθ)
    τ_cart = zeros(nθ)
    τ_cart_lay = zeros(nθ)
    τ_cart_ana = zeros(nθ)
    τ_time = zeros(nθ, nt)

    #  Anisotropy parameters
    η2 = 1e2
    m = 1
    η1 = η2 / m

    α2 = 0.5
    α1 = 1 - α2

    ηn = α1 * η1 + α2 * η2
    δ = (α1 + α2 * m) * (α1 + α2 / m)

    # elasticity
    tmax = 1.0
    G2 = G1 = 1.0
    C2 = C1 = 10.
    C2 = 4.
    C1 = C2 / 4    # @abacaxi-seco HARDCODED factor 2, to remove

    # Run them all
    for iθ in eachindex(θ)

        layering = Layering(
            (0 * 0.25, 0.025),
            0.15,
            α2;
            θ=θ[iθ],
            perturb_amp=0 * 1.0,
            perturb_width=1.0
        )

        # @abacaxi-seco Note that I switched to LU factorisation as the Jacobian is already not symmetric with elasto-palsticity
        τ_cart_lay[iθ], τ_time[iθ, :] = main(nc, nt, layering, BCs[1], D_BCs[1], :lu, η1, η2, G1, G2, C1, C2)
        τ_cart_ana[iθ] = Analytical(θ[iθ], ηn, δ, D_BCs[1])

    end

    ε̇bg = sqrt(sum(1 / 2 .* D_BCs[1][:] .^ 2))

    # Strongest end-member
    ηeff = α1 * η1 + α2 * η2
    @show τstrong = 2 * ηeff * ε̇bg

    # Weakest end-member
    ηeff = (α1 / η1 + α2 / η2)^(-1)
    @show τweak = 2 * ηeff * ε̇bg

    τ_cart .= τstrong * sqrt.(((δ^2 - 1) * cos.(2 .* θ) .^ 2 .+ 1) / (δ^2))

    cm.with_theme(cm.theme_latexfonts()) do
        fig = cm.Figure(fontsize=15)

        ax = cm.Axis(fig[0, 1], xlabel=cm.L"$$step]", ylabel=cm.L"$\tau_{II}$ [-]")
        for iθ in eachindex(θ)
            cm.lines!(ax, 1:nt, τ_time[iθ, :])
        end
        ax = cm.Axis(fig[1, 1], xlabel=cm.L"$\theta$ [$^{\circ}$]", ylabel=cm.L"$\tau_{II}$ [-]")
        cm.lines!(ax, θ * 180 / π, τ_cart_lay, label="Layering")
        # cm.lines!(ax, θ*180/π, τstrong*ones(size(θ)), color=:gray, linestyle=:dash, label="End-Member (Biot et al., 1965)")
        # cm.lines!(ax, θ*180/π, τweak*ones(size(θ)), color=:gray, linestyle=:dash, label="End-Member (Biot et al., 1965)")
        # cm.scatter!(ax, θ[1:6:end]*180/π, τ_cart[1:6:end], label="Expression", markersize=10)
        # cm.scatter!(ax, θ[1:8:end]*180/π, τ_cart_ana[1:8:end], label="Analytical", marker=:utriangle, markersize=10, color=cm.CGrid.c.yled(3))
        cm.Legend(fig[2, 1], ax, framevisible=false, orientation=:horizontal, unique=true, nbanks=3, cm.L"$\tau_{II}$    ($δ \approx$ %$(round(Int,δ)))")
        display(fig)
    end

end