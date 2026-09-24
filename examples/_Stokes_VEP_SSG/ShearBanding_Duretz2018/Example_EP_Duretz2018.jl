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

    # Time steps
    Δt0 = 1e10 / sc.t
    nt = 35

    # Solver parameters
    niter = 20
    ϵ_nl = 1e-11
    α = LinRange(0.05, 1.0, 10)
    iter_params = IterParams(solver_type=:PH, niter=niter, ϵ_nl=ϵ_nl, α=α)

    L = (x=4e3 / sc.L, y=2e3 / sc.L)
    x = (min=(-L.x / 2), max=L.x / 2)
    y = (min=(-L.y), max=0.0)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field
    @show size(a.V.x), size(a.X.v_e.x), size(a.X.c_e.y)
    @views a.V.x .= D_BC[1, 1] * a.X.vx_e.x .+ D_BC[1, 2] * a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1] * a.X.vy_e.x .+ D_BC[2, 2] * a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values 
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[1] .+ D_BC[2, 2]*a.X.v.y)
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[end] .+ D_BC[2, 2]*a.X.v.y)
    end

    # NO MARKERS:
    # Set material geometry 
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([a.X.c_e.x[i], a.X.c_e.y[j]])
        for igeom in eachindex(seed) # seed
            if inside(𝐱, seed[igeom])
                a.phases.c[i, j] = 2
            end
        end
    end

    for i in inx_c, j in iny_c  # loop on vertices
        𝐱 = @SVector([a.X.v_e.x[i], a.X.v_e.y[j]])
        for igeom in eachindex(seed) # seed
            if inside(𝐱, seed[igeom])
                a.phases.v[i, j] = 2
            end
        end
    end

    Set_PhaseRatios!(a)

    # # IF MARKERS:

    # nxcell = (5, 5) # initial number of particles per cell
    # max_xcell = 40 # maximum number of particles per cell
    # min_xcell = 10 # minimum number of particles per cell
    # args = 1 # Fields to be advected (1=phase)
    # adv = Markers(backend, a, nxcell, max_xcell, min_xcell, nc, nphases, args)
    # phases, = adv.particle_args

    # for i in axes(phases, 1)
    #     for ip in cellaxes(phases)
    #         @index(particles.index[ip, i, j]) == 0 && continue
    #         x = @index particles.coords[1][ip, i, j]
    #         y = @index particles.coords[2][ip, i, j]
    #         𝐱 = SVector(x, y)
    #         if inside(𝐱, seed[igeom])
    #             @index phases[ip, i, j] = 2.0 # seed
    #         end
    #     end
    # end
    # # Set phase ratios
    # SetPhaseRatios!(a, adv.phase_ratios, adv.particles, adv.phases)

    a.Pt .= 0.0
    a.Pt0 .= a.Pt
    a.Pti .= a.Pt

    #--------------------------------------------#

    rvec = zeros(length(α))
    err = (x=zeros(niter), y=zeros(niter), p=zeros(niter))
    probes = (τII=zeros(nt), fric=zeros(nt), t=zeros(nt), str=zeros(nt), λ=zeros(nt))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        @time main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = av2D(a.τ.xy)
        τII = sqrt.(0.5 .* (a.τ.xx[inx_c, iny_c] .^ 2 + a.τ.yy[inx_c, iny_c] .^ 2 + (-a.τ.xx[inx_c, iny_c] - a.τ.yy[inx_c, iny_c]) .^ 2) .+ τxyc[inx_c, iny_c] .^ 2)
        ε̇xyc = av2D(a.ε̇.xy)
        ε̇II = sqrt.(0.5 .* (a.ε̇.xx[inx_c, iny_c] .^ 2 + a.ε̇.yy[inx_c, iny_c] .^ 2 + (-a.ε̇.xx[inx_c, iny_c] - a.ε̇.yy[inx_c, iny_c]) .^ 2) .+ ε̇xyc[inx_c, iny_c] .^ 2)

        # Principal stress
        σ1 = (x=zeros(size(a.Pt)), y=zeros(size(a.Pt)), v=zeros(size(a.Pt)))
        τxyc = 0.25 * (a.τ.xy[1:(end-1), 1:(end-1)] .+ a.τ.xy[2:(end-0), 1:(end-1)] .+ a.τ.xy[1:(end-1), 2:(end-0)] .+ a.τ.xy[2:(end-0), 2:(end-0)])

        for i in inx_c, j in iny_c
            σ = @SMatrix[-a.Pt[i, j]+a.τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -a.Pt[i, j]+a.τ.yy[i, j] 0.; 0. 0. -a.Pt[i, j]+(-a.τ.xx[i, j]-a.τ.yy[i, j])]
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
        probes.λ[it] = mean(a.λ̇.c[inx_c, iny_c])
        probes.str[it] = ε̇bg * it * Δ.t
        i_midx = Int64(floor(nc.x))
        probes.fric[it] = mean(.-τxyc[i_midx, end-3] ./ (-a.Pt[i_midx, end-3] .+ a.τ.yy[i_midx, end-3]))

        # Bifurcation analysis
        Te = @SMatrix([2/3 -1/3 0; -1/3 2/3 0; 0 0 1; 1 1 0])
        Ts = @SMatrix([1 0 0 -1; 0 1 0 -1; 0 0 1 0])
        θ = LinRange(0, 90, 180)
        detA = zeros(size(θ))
        bifurc = (detA=zeros(size(a.Pt)), θ=zeros(size(a.Pt)))
        for i in inx_c, j in iny_c

            D = SMatrix{1,1}(a.𝐷_ctl.c[ii, jj] for ii in i:i, jj in j:j)
            phase = a.phases.c[i, j]
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
            # heatmap!(ax, a.X.c.x, a.X.c.y,  log10.(a.λ̇.c[inx_c,iny_c]), colormap=:bluesreds)
            # contour!(ax, a.X.c.x, a.X.c.y,  a.phases.c[inx_c,iny_c], color=:black)
            heatmap!(ax, a.X.c.x, a.X.c.y, a.Pt[inx_c, iny_c] * sc.σ, colormap=:jet, colorrange=(-6e6, 4e6))
            # heatmap!( ax, a.X.v_e.x, a.X.v_e.y, a.λ̇.v )
            # heatmap!(ax, a.X.c.x, a.X.c.y, bifurc.detA[inx_c,iny_c], colormap=:jet)

            # st = 10
            # arrows!(ax, a.X.c.x[1:st:end], a.X.c.y[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.04, linewidth=2, color=:white)
            ax = Axis(fig[2, 1], xlabel="Iterations @ step $(it) ", ylabel=L"$\log_{10}$ error")
            scatter!(ax, 1:iter_params.niter, log10.(err.x[1:iter_params.niter] ./ err.x[1]))
            scatter!(ax, 1:iter_params.niter, log10.(err.y[1:iter_params.niter] ./ err.y[1]))
            scatter!(ax, 1:iter_params.niter, log10.(err.p[1:iter_params.niter] ./ err.p[1]))
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