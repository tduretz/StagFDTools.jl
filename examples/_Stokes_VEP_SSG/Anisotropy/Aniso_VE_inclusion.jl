using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs, Interpolations, GridGeometryUtils, JLD2
import CairoMakie as cm
using DifferentiationInterface

@views function main(BC_template, D_template)
    # Resolution
    nc = (x = 100, y = 100)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; compressible=false, mixing=:voigt, anisotropy=Anisotropy)

    # Parameters
    params_bg = (ρ=1.0, n=1.0, η0=1e1, G=1e1)
    params_in = (ρ=1.0, n=1.0, η0=1e10, G=1e1)

    materials.g .= [0., 0.]
    materials.ρ .= [params_bg.ρ, params_in.ρ]
    materials.n .= [params_bg.n, params_in.n]
    materials.η0 .= [params_bg.η0, params_in.η0]
    materials.G .= [params_bg.G, params_in.G]

    # Anisotropy
    α  = 0.5   # volume fraction of the strong layer
    an = materials.anisotropy
    an.mv .= [4., 1.]      # B in power law
    an.me .= [4., 1.]
    an.α  .= [0.5, α]
    an.dv .= [-sind(30), cosd(30)]
    an.n1 .= [3., 1.]
    an.n2 .= [2., 1.]

    preprocess!(materials)

    # Time steps
    Δt0   = 0.15
    nt    = 250

    # Solver parameters
    iter_params = IterParams(niter=20)

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
    @views a.Pt[inx_c, iny_c]  .= 10.
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

    # Set material geometry
    rad = 0.1 + 1e-13
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2 ) .<= rad^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2 ) .<= rad^2] .= 2
    Set_PhaseRatios!(a)

    # Time loop (grid only)
    rvec = zeros(length(iter_params.α))
    err  = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to   = TimerOutput()
    nout = 1   # plot every nout steps
    for it = 1:nt
        main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
        
        # Plot
        cm.with_theme(cm.theme_latexfonts()) do
            # Layer traces (⟂ director) in the cells of the anisotropic phases, every `step` cells
            # (director field, or materials.anisotropy.dv where not set)
            step   = 5
            xc, yc = a.X.c.x, a.X.c.y
            nd0    = materials.anisotropy.dv
            px, py, tx, ty = Float64[], Float64[], Float64[], Float64[]
            for j in iny_c[1:step:end], i in inx_c[1:step:end]
                is_anisotropic(materials, a.phases.c[i, j]) || continue
                nx, ny = a.τ0.nd.x_c[i, j], a.τ0.nd.y_c[i, j]
                (iszero(nx) && iszero(ny)) && ((nx, ny) = (nd0[1], nd0[2]))
                push!(px, xc[i-1]); push!(py, yc[j-1]); push!(tx, ny); push!(ty, -nx)
            end

            fig = cm.Figure(size=(1000,600))
            ax1 = cm.Axis(fig[1,1], xlabel="x", ylabel="y", aspect=cm.DataAspect(), title="timestep $(it)")
            ax2 = cm.Axis(fig[1,3],  xlabel="x", ylabel="y", aspect=cm.DataAspect(), title="timestep $(it)")
            hm = cm.heatmap!(ax1, a.X.c.x, a.X.c.y, a.τ.II[inx_c, iny_c], colormap=cm.cgrad(:vik))
            cm.Colorbar(fig[1,2], hm, label = cm.L"$\tau_{II}$")
            hm2 = cm.heatmap!(ax2, a.X.c.x, a.X.c.y, a.ε̇.II[inx_c, iny_c], colormap=cm.cgrad(:vik))
            cm.Colorbar(fig[1,4], hm2, label = cm.L"$\dot\varepsilon_{II}$")
            for ax in (ax1, ax2)
                isempty(px) || cm.arrows2d!(ax, px, py, tx, ty, lengthscale=0.03, tiplength=0, shaftwidth=1.5, align=:center, color=:white)
            end
            display(fig)
        end
        
    end

    return a
end

let
    BC = :free_slip
    # Boundary velocity gradient matrix
    er    = -2
    D_BC = @SMatrix( [er 0;
                   0 -er] )
    main(BC, D_BC)
    nothing
end


######################
# 2 ways of defining an anisotropic material
#   a) Initialise two materials and then define a builder function
#      that creates the upscaled composite from individual material parameters
#
#   b) Give bulk values of the material plus the anisotropic coefficient;
#      then the single phase parameters are found if needed within the computation
#      of rheology; here an additional container into materials struct