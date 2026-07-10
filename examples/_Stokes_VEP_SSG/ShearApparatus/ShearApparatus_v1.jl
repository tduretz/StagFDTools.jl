using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics: mean
using DifferentiationInterface
using TimerOutputs, CairoMakie

@views function main(nc, θgouge)
    #--------------------------------------------#

    # Boundary loading type
    config = :free_slip
    D_BC = @SMatrix([1. 0.;
        0 -1])

    # Material parameters
    nphases = 3
    materials = initialize_materials(nphases; plasticity=DruckerPrager, compressible=true)
    materials.g .= [0., 0.]
    #                 rock   gouge  salt
    materials.ρ .= [0.0, 0.0, 0.0]
    materials.n .= [1.0, 1.0, 1.0]      # Power law exponent
    materials.η0 .= [1e3, 1e3, 1e-3]      # Reference viscosity
    materials.G .= [1e1, 2e1, 1e1]      # Shear modulus
    #                            rock   gouge  salt
    materials.plasticity.C .= [150, 100, 150]      # Cohesion
    materials.plasticity.ϕ .= [35., 30., 35.]      # Friction angle
    materials.plasticity.ψ .= [0.0, 5.0, 0.0]      # Dilation angle
    materials.plasticity.ηvp .= [1.0, 1.0, 1.0] .* 0.3 # Viscoplastic regularisation
    materials.β .= [1e-2, 0.5e-2, 1e-2]      # Compressibility
    preprocess!(materials)

    # Geometry
    gouge = (
        Rectangle((0.0, 0.0), 0.2, 2.0; θ=θgouge),
    )
    salt = (
        Rectangle((-0.5, 0.0), 0.5, 2.0; θ=0),
        Rectangle((0.5, 0.0), 0.5, 2.0; θ=0),
    )

    # Time steps
    Δt0 = 0.25
    nt = 200

    # Newton solver
    iter_params = IterParams(niter=15, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Intialise field
    L = (x=1.0, y=1.5)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

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
        𝐱 = @SVector([a.X.c_e.x[i], a.X.c_e.y[j]])
        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                a.phases.c[i, j] = 2
            end
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                a.phases.c[i, j] = 3
            end
        end
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([a.X.v_e.x[i], a.X.v_e.y[j]])
        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                a.phases.v[i, j] = 2
            end
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                a.phases.v[i, j] = 3
            end
        end
    end
    FillPhaseRatios!(a)

    a.Pt .= 200 * rand(size(a.Pt)...)
    a.Pt0 .= a.Pt
    a.Pti .= a.Pt

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    probes = (τII=zeros(nt), fric=zeros(nt), t=zeros(nt))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = 0.25 * (a.τ.xy[1:end-1, 1:end-1] .+ a.τ.xy[2:end-0, 1:end-1] .+ a.τ.xy[1:end-1, 2:end-0] .+ a.τ.xy[2:end-0, 2:end-0])
        τII = sqrt.(0.5 .* (a.τ.xx[inx_c, iny_c] .^ 2 + a.τ.yy[inx_c, iny_c] .^ 2 + (-a.τ.xx[inx_c, iny_c] - a.τ.yy[inx_c, iny_c]) .^ 2) .+ τxyc[inx_c, iny_c] .^ 2)

        # Store probes data
        probes.t[it] = it * Δ.t
        probes.τII[it] = mean(τII)
        i_midx = Int64(floor(nc.x))
        probes.fric[it] = mean(.-τxyc[i_midx, end-3] ./ (-a.Pt[i_midx, end-3] .+ a.τ.yy[i_midx, end-3]))

        # Visualise
        fig = Figure()
        ax = Axis(fig[1:2, 1], aspect=DataAspect(), title="Strain rate", xlabel="x", ylabel="y")
        heatmap!(ax, a.X.c.x, a.X.c.y, log10.(a.λ̇.c[inx_c, iny_c]), colormap=:bluesreds)
        contour!(ax, a.X.c.x, a.X.c.y, a.phases.c[inx_c, iny_c], color=:black)
        ax = Axis(fig[1, 2], xlabel="Time", ylabel="Effective stress (τII)")
        scatter!(ax, probes.t[1:nt], probes.τII[1:nt])
        ax = Axis(fig[2, 2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error")
        scatter!(ax, 1:iter, log10.(err.x[1:iter]))
        scatter!(ax, 1:iter, log10.(err.y[1:iter]))
        scatter!(ax, 1:iter, log10.(err.p[1:iter]))
        ylims!(ax, -10, 5)
        display(fig)

    end

    display(to)

end

let
    main((x=250, y=200), -π / 2.4)
end
