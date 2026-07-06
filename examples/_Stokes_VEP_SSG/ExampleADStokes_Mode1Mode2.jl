using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs

function line(p, K, Δt, η_ve, ψ, p1, t1)
    p2 = p1 + K*Δt*sind(ψ)
    t2 = t1 - η_ve
    a  = (t2-t1)/(p2-p1)
    b  = t2 - a*p2
    return a*p + b
end

@views function main(nc)
    #--------------------------------------------#

    # Resolution

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [ -1. 0.;
                          0  1 ])

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; plasticity=Kiss2023, compressible=false)
    materials.g   .= [0.0,  0.0]
    materials.ρ   .= [1.0,  1.0]
    materials.n   .= [1.0,  1.0]
    materials.η0  .= [1e3,  1e-1]
    materials.G   .= [1e1,  1e1]
    materials.β   .= [1e-2, 1e-2]
    materials.plasticity.C   .= [100.0, 100.0]
    materials.plasticity.σT  .= [50.0,  50.0]
    materials.plasticity.δσT .= [10.0,  10.0]
    materials.plasticity.ϕ   .= [30.0,  30.0]
    materials.plasticity.ηvp .= [1.0,   1.0]
    materials.plasticity.ψ   .= [3.0,   3.0]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 100

    # Solver parameters
    iter_params = IterParams(niter=20, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

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
    @views a.Pt[inx_c, iny_c ]  .= 0.
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
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= 0.1^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= 0.1^2] .= 2
    FillPhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        τxyc = av2D(a.τ.xy)
        τII  = sqrt.( 0.5.*(a.τ.xx[inx_c,iny_c].^2 + a.τ.yy[inx_c,iny_c].^2 + (-a.τ.xx[inx_c,iny_c]-a.τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        ε̇xyc = av2D(a.ε̇.xy)
        ε̇II  = sqrt.( 0.5.*(a.ε̇.xx[inx_c,iny_c].^2 + a.ε̇.yy[inx_c,iny_c].^2 + (-a.ε̇.xx[inx_c,iny_c]-a.ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )

        p_tr1 = LinRange(-100, 0, 100)
        p_tr2 = LinRange(0, 200, 100)
        p_tr3 = LinRange(50, 200, 100)

        K      = 1 / materials.β[1]
        η_ve   = materials.G[1] * Δ.t
        pc1    = materials.plasticity.P1[1]
        pc2    = materials.plasticity.P2[1]
        τc1    = materials.plasticity.τ1[1]
        τc2    = materials.plasticity.τ2[1]
        φ      = materials.plasticity.ϕ[1]
        C      = materials.plasticity.C[1]
        ψ      = materials.plasticity.ψ[1]
        η_vp   = materials.plasticity.ηvp[1]

        l1    = line.(p_tr1, K, Δ.t, η_ve, 90., pc1, τc1)
        l2    = line.(p_tr2, K, Δ.t, η_ve, 90., pc2, τc2)
        l3    = line.(p_tr3, K, Δ.t, η_ve,   ψ, pc2, τc2)

        P_end =  1000

        ########################################
        fig = Figure(size = (1200, 900))

        ax11 = Axis(fig[1, 1],
            xlabel = "Iterations @ step $(it)",
            ylabel = "log₁₀ error")
        s1 = scatter!(ax11, 1:iter, log10.(err.x[1:iter]); color = :blue)
        s2 = scatter!(ax11, 1:iter, log10.(err.y[1:iter]); color = :orange)
        s3 = scatter!(ax11, 1:iter, log10.(err.p[1:iter]); color = :green)
        axislegend(ax11, [s1, s2, s3], ["Vx", "Vy", "Pt"]; position = :rt)

        ax12 = Axis(fig[1, 2], title = "ε̇II", aspect = DataAspect())
        heatmap!(ax12, a.X.c.x, a.X.c.y, log10.(ε̇II)'; colormap = :coolwarm)
        # set x-limits correctly
        xlims!(ax12, extrema(a.X.c.x))

        ax21 = Axis(fig[2, 1], xlabel = "P", ylabel = "τII", aspect = DataAspect())
        lines!(ax21, [pc1, pc1, pc2, P_end], [0.0, τc1, τc2, P_end * sind(φ) + C * cosd(φ)]; color = :black)
        lines!(ax21, p_tr1, l1; color = :red)
        lines!(ax21, p_tr2, l2; color = :blue)
        lines!(ax21, p_tr3, l3; color = :green)
        scatter!(ax21, a.Pt[inx_c, iny_c][:], τII[:] ; color = :black)

        ax22 = Axis(fig[2, 2], title = "τII", aspect = DataAspect())
        heatmap!(ax22, a.X.c.x, a.X.c.y, τII'; colormap = :turbo)
        xlims!(ax22, extrema(a.X.c.x))

        display(fig)
        # or save("my_figure.png", fig)
        ########################################

        @show (3/materials.β[1] - 2*materials.G[1]) / (2*(3/materials.β[1] + 2*materials.G[1]))

    end

    display(to)

end

let
    main((x = 100, y = 100))
end
