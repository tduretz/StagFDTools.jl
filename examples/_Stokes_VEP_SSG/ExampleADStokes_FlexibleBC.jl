using CairoMakie
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using StagFDTools: IterParams
using DifferentiationInterface
using TimerOutputs

@views function main(BC_template, D_template, σ_template=@SMatrix(zeros(2, 2)))
    #--------------------------------------------#

    # Resolution
    nc = (x = 20, y = 20)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template
    σ_BC   = σ_template

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; compressible=false)
    materials.g   .= [0.0,  0.0]
    materials.ρ   .= [1.0,  1.0]
    materials.n   .= [1.0,  1.0]
    materials.η0  .= [1e2,  1e-1]
    materials.G   .= [1e1,  1e1]
    materials.β   .= [1e-2, 1e-2]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Solver parameters
    iter_params = IterParams(niter=3, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

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
    @views a.Pt[inx_c, iny_c]  .= 0.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1] .+ (a.type.Vx[     1, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1] .+ (a.type.Vx[   end, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  ) .+ (a.type.Vx[inx_Vx,     2] .== :tangential_stress) .* σ_BC[1,2]
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end]) .+ (a.type.Vx[inx_Vx, end-1] .== :tangential_stress) .* σ_BC[1,2]
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2] .+ (a.type.Vy[inx_Vy,     1 ] .== :normal_stress) .* σ_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2] .+ (a.type.Vy[inx_Vy,   end ] .== :normal_stress) .* σ_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y) .+ (a.type.Vy[    2, iny_Vy] .== :tangential_stress) .* σ_BC[2,1]
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y) .+ (a.type.Vy[end-1, iny_Vy] .== :tangential_stress) .* σ_BC[2,1]
    end

    # Set material geometry
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= 0.1^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= 0.1^2] .= 2
    Set_PhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    iter = 0
    for it=1:nt
        iter, err = main_loop(a, nothing, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
    end

    # Boundary strain rates of the solution (W/E: ∂Vx/∂x, S/N: ∂Vy/∂y), to compare with D_BC
    Vx, Vy = a.V.x[inx_Vx,iny_Vx], a.V.y[inx_Vy,iny_Vy]
    ε̇_bnd = (
        W = mean(Vx[2,:]   .- Vx[1,:])     / Δ.x,
        E = mean(Vx[end,:] .- Vx[end-1,:]) / Δ.x,
        S = mean(Vy[:,2]   .- Vy[:,1])     / Δ.y,
        N = mean(Vy[:,end] .- Vy[:,end-1]) / Δ.y,
    )

    # Normal stresses in the centroids next to each boundary, to compare with σ_BC
    σxx = a.τ.xx[inx_c,iny_c] .- a.Pt[inx_c,iny_c]
    σyy = a.τ.yy[inx_c,iny_c] .- a.Pt[inx_c,iny_c]
    σ_bnd = (W = mean(σxx[1,:]), E = mean(σxx[end,:]), S = mean(σyy[:,1]), N = mean(σyy[:,end]))
    # Normal stresses reached on the boundaries (linear extrapolation of the first two centroids)
    σ_face = (
        W = mean(1.5 .* σxx[1,:]   .- 0.5 .* σxx[2,:]),
        E = mean(1.5 .* σxx[end,:] .- 0.5 .* σxx[end-1,:]),
        S = mean(1.5 .* σyy[:,1]   .- 0.5 .* σyy[:,2]),
        N = mean(1.5 .* σyy[:,end] .- 0.5 .* σyy[:,end-1]),
    )

    # Shear stresses reached on the boundaries (linear extrapolation of the first two vertex rows/columns inside)
    τxy = a.τ.xy[inx_v, iny_v]
    τ_face = (
        W = mean(2 .* τxy[2,:]   .- τxy[3,:]),
        E = mean(2 .* τxy[end-1,:] .- τxy[end-2,:]),
        S = mean(2 .* τxy[:,2]   .- τxy[:,3]),
        N = mean(2 .* τxy[:,end-1] .- τxy[:,end-2]),
    )

    return (; X=a.X, Vx, Vy, Pt=a.Pt[inx_c,iny_c], err, iter, ε̇_bnd, σ_bnd, σ_face, τ_face)
end

let
    # Boundary condition templates
    BCs = [
        :all_Dirichlet,
        :free_slip,
        :NS_stress,
        :EW_stress,
        :NS_periodic,
        :EW_periodic,
        :EW_periodic_shear_stress,
        :NS_periodic_shear_stress,
    ]

    # Boundary velocity gradient matrix
    D_BCs = [
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [1 0; 0 -1] ),
        @SMatrix( [0 0; 1  0] ),
        @SMatrix( [0 1; 0  0] ),
        @SMatrix( [0 1; 0  0] ), # initial guess and S velocity (V = 0 would give a NaN tangent: ε̇II = √(…) at 0)
        @SMatrix( [0 0; 1  0] ), # initial guess and W velocity
    ]

    # Boundary stresses: normal (diagonal, *_stress templates) and shear (off-diagonal, *_shear_stress templates)
    σ_BCs = [
        @SMatrix( [  0.0   0.0;   0.0   0.0] ),
        @SMatrix( [  0.0   0.0;   0.0   0.0] ),
        @SMatrix( [-50.0   0.0;   0.0 -50.0] ),
        @SMatrix( [-50.0   0.0;   0.0 -50.0] ),
        @SMatrix( [  0.0   0.0;   0.0   0.0] ),
        @SMatrix( [  0.0   0.0;   0.0   0.0] ),
        @SMatrix( [  0.0   1.0;   1.0   0.0] ),
        @SMatrix( [  0.0   1.0;   1.0   0.0] ),
    ]

    # Run them all and collect the results on a single figure (one row per BC template)
    fig = Figure(size=(1300, 330*length(BCs)), fontsize=14)
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        r = main(BCs[iBC], D_BCs[iBC], σ_BCs[iBC])
        D, σ = D_BCs[iBC], σ_BCs[iBC]
        e, s, b, t = r.ε̇_bnd, r.σ_bnd, r.σ_face, r.τ_face
        @printf("%-14s boundary σxx W, E = %8.3f, %8.3f   σyy S, N = %8.3f, %8.3f   (σ_BC = %g, %g)   boundary τxy S, N = %8.3f, %8.3f   W, E = %8.3f, %8.3f   (τ_BC = %g)\n", BCs[iBC], b.W, b.E, b.S, b.N, σ[1,1], σ[2,2], t.S, t.N, t.W, t.E, σ[1,2])
        Label(fig[iBC, 0], "$(BCs[iBC])\nD = $(D[1,:]) $(D[2,:])\nσ_BC = $(σ[1,1]), $(σ[2,2]), τ_BC = $(σ[1,2])\n\n∂Vx/∂x W, E: $(round(e.W, digits=3)), $(round(e.E, digits=3))\n∂Vy/∂y S, N: $(round(e.S, digits=3)), $(round(e.N, digits=3))\nσxx W, E: $(round(s.W, digits=2)), $(round(s.E, digits=2))\nσyy S, N: $(round(s.S, digits=2)), $(round(s.N, digits=2))\nat boundary:\nσxx W, E: $(round(b.W, digits=2)), $(round(b.E, digits=2))\nσyy S, N: $(round(b.S, digits=2)), $(round(b.N, digits=2))\nτxy S, N: $(round(t.S, digits=3)), $(round(t.N, digits=3))\nτxy W, E: $(round(t.W, digits=3)), $(round(t.E, digits=3))", tellheight=false)
        ax = Axis(fig[iBC, 1], aspect=DataAspect(), title="Vx")
        hm = heatmap!(ax, r.X.v.x, r.X.c.y, r.Vx); Colorbar(fig[iBC, 2], hm)
        ax = Axis(fig[iBC, 3], aspect=DataAspect(), title="Vy")
        hm = heatmap!(ax, r.X.c.x, r.X.v.y, r.Vy); Colorbar(fig[iBC, 4], hm)
        ax = Axis(fig[iBC, 5], aspect=DataAspect(), title="Pt")
        hm = heatmap!(ax, r.X.c.x, r.X.c.y, r.Pt); Colorbar(fig[iBC, 6], hm)
        ax = Axis(fig[iBC, 7], title="Convergence", xlabel="Iterations", ylabel="log₁₀ error")
        scatter!(ax, 1:r.iter, log10.(r.err.x[1:r.iter]), label="Vx")
        scatter!(ax, 1:r.iter, log10.(r.err.y[1:r.iter]), label="Vy")
        scatter!(ax, 1:r.iter, log10.(r.err.p[1:r.iter]), label="Pt")
        iBC == 1 && axislegend(ax)
    end
    display(fig)
end
