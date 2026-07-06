using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, CairoMakie
import Statistics:mean
using JustPIC, JustPIC._2D
import JustPIC.@index
const backend = JustPIC.CPUBackend
using DifferentiationInterface
using TimerOutputs

function set_phases!(phases, particles)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                # quick escape
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                if (x^2 + (y)^2) <= 0.1^2
                    @index phases[ip, i, j] = 2.0
                else
                    @index phases[ip, i, j] = 1.0
                end
            end
        end
    end
end

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 25, y = 25)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Material parameters
    nphases  = 2
    materials = initialize_materials(nphases; compressible=false)
    materials.g  .= [0.0,   0.0]
    materials.ρ  .= [1.0,   1.0]
    materials.n  .= [1.0,   1.0]
    materials.η0 .= [1e0,   1e5]
    materials.G  .= [1e60,  2e60]
    materials.β  .= [1e-20, 2e-20]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 20
    C     = 0.1

    # Solver parameters
    iter_params = IterParams(niter=2, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c]  .= 0.0
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Initialize particles
    nxcell    = 36 # initial number of particles per cell
    max_xcell = 36*2 # maximum number of particles per cell
    min_xcell = 6 # minimum number of particles per cell
    args      = 1 # Fields to be advected (1=phase)
    adv       = JustPICAdvection(backend, a, nxcell, max_xcell, min_xcell, nc, nphases, args)
    phases,   = adv.particle_args

    # Set material geometry
    set_phases!(phases, adv.particles)
    update_JustPIC!(a, adv.phase_ratios, adv.particles, adv.particle_args[1])

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        # Adaptive time step
        Vmax = max(maximum(abs.(a.V.x)), maximum(abs.(a.V.y)))
        Δ    = (x=Δ.x, y=Δ.y, t = C * min(Δ.x, Δ.y)/Vmax)

        @time main_loop(a, adv, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Visualise
        function visualisation()
            #-----------
            fig = Figure(size=(500,800))
            #-----------
            ax  = Axis(fig[1,1], aspect=DataAspect(), title="Pressure", xlabel="x", ylabel="y")
            heatmap!(ax, a.X.c.x, a.X.c.y,  (a.Pt[inx_c,iny_c]), colormap=:bluesreds)
            Vxc = 0.5.*(a.V.x[inx_Vx,iny_Vx][1:end-1,:] .+ a.V.x[inx_Vx,iny_Vx][2:end,:])
            Vyc = 0.5.*(a.V.y[inx_Vy,iny_Vy][:,1:end-1] .+ a.V.y[inx_Vy,iny_Vy][:,2:end])
            arrows2d!(ax, a.X.c.x, a.X.c.y, Vxc, Vyc, lengthscale = 0.05)
            ax  = Axis(fig[1,2], aspect=DataAspect(), title="Particles", xlabel="x", ylabel="y")
            ppx, ppy = adv.particles.coords
            pxv  = ppx.data[:]
            pyv  = ppy.data[:]
            clr  = phases.data[:]
            idxv = adv.particles.index.data[:]
            scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=5)
            ax  = Axis(fig[2,1], aspect=DataAspect(), title="Txx", xlabel="x", ylabel="y")
            heatmap!(ax, a.X.c.x, a.X.c.y,  a.τ.xx[inx_c,iny_c], colormap=:bluesreds)
            ax  = Axis(fig[2,2], aspect=DataAspect(), title="Tyy", xlabel="x", ylabel="y")
            heatmap!(ax, a.X.c.x, a.X.c.y,  a.τ.yy[inx_c,iny_c], colormap=:bluesreds)
            ax  = Axis(fig[3,1], aspect=DataAspect(), title="Gc", xlabel="x", ylabel="y")
            heatmap!(ax, a.X.c.x, a.X.c.y,  a.G.c[inx_c,iny_c], colormap=:bluesreds)
            ax  = Axis(fig[3,2], aspect=DataAspect(), title="Gv", xlabel="x", ylabel="y")
            heatmap!(ax, a.X.v.x, a.X.v.y,  a.G.v[inx_v,iny_v], colormap=:bluesreds)
            #-----------
            display(fig)
        end
        with_theme(visualisation, theme_latexfonts())
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

    # Boundary velocity gradient matrix
    D_BCs = [
        #  @SMatrix( [0 1; 0  0] ),
        @SMatrix( [1 0; 0 -1] ),
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end
