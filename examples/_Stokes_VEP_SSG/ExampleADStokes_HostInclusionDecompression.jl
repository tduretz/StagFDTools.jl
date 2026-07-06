using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using DifferentiationInterface
using TimerOutputs

function line(p, K, Δt, η_ve, ψ, p1, t1)
    p2 = p1 + K * Δt * sind(ψ)
    t2 = t1 - η_ve
    a = (t2 - t1) / (p2 - p1)
    b = t2 - a * p2
    return a * p + b
end

@views function main(nc, radius)
    #--------------------------------------------#

    # Scales
    sc = (σ=3e10, L=1e-2, t=1e10)
    L = (x=1e-2 / sc.L, y=1e-2 / sc.L)

    # Boundary loading type
    config = :free_slip
    ε̇kk = 0.5e-14 .* sc.t
    P0 = 1e9 / sc.σ
    D_BC = @SMatrix([ε̇kk 0.0;
        0.0 ε̇kk])

    # Material parameters
    G0 = 3e10
    K0 = 4 * G0

    nphases = 3
    materials = initialize_materials(nphases; plasticity=DruckerHyperbolic, compressible=true)
    materials.g .= [0.0, 0.0]
    materials.ρ .= [0.0, 0.0, 0.0]
    materials.n .= [1.0, 1.0, 1.0]
    materials.η0 .= [1e50, 1e50, 1e50] ./ (sc.σ * sc.t)
    materials.G .= [G0, G0 / 4, 2 * G0] ./ sc.σ
    materials.β .= [1 / K0, 1 / (K0 / 4), 1 / (2 * K0)] .* sc.σ
    materials.plasticity.C .= [50e6, 50e6, 50e6] ./ sc.σ
    materials.plasticity.σT .= [50e6, 50e6, 50e6] ./ sc.σ
    materials.plasticity.ϕ .= [35.0, 35.0, 35.0]
    materials.plasticity.ηvp .= [1e19, 1e19, 1e19] ./ (sc.σ * sc.t)
    materials.plasticity.ψ .= [5.0, 5.0, 5.0]
    preprocess!(materials)

    # Time steps
    Δt0 = 5e9 / sc.t
    nt = 1#145

    # Solver parameters
    iter_params = IterParams(niter=15, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-0.0, max=L.x)
    y = (min=-0.0, max=L.y)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    nVx = maximum(a.number.Vx)
    nVy = maximum(a.number.Vy)
    nPt = maximum(a.number.Pt)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1, 1] * a.X.vx_e.x .+ D_BC[1, 2] * a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1] * a.X.vy_e.x .+ D_BC[2, 2] * a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c] .= P0
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
    a_line, b_line = -1., 1.0
    xc2 = a.X.c.x .+ 0 * a.X.c.y'
    yc2 = 0 * a.X.c.x .+ a.X.c.y'
    xv2 = a.X.v.x .+ 0 * a.X.v.y'
    yv2 = 0 * a.X.v.x .+ a.X.v.y'
    @views @. a.phases.c[inx_c, iny_c][yc2<0.75&&xc2<0.75&&yc2<(xc2*a_line+b_line)] .= 3
    @views @. a.phases.v[inx_v, iny_v][yv2<0.75&&xv2<0.75&&yv2<(xv2*a_line+b_line)] .= 3
    @views @. a.phases.c[inx_c, iny_c][yc2<radius&&xc2<radius] .= 2
    @views @. a.phases.v[inx_v, iny_v][yv2<radius&&xv2<radius] .= 2
    FillPhaseRatios!(a)

    fig_init = Figure()
    ax_p1 = Axis(fig_init[1, 1], title="phases.c", aspect=DataAspect())
    heatmap!(ax_p1, a.X.c.x, a.X.c.y, a.phases.c[inx_c, iny_c]')
    ax_p2 = Axis(fig_init[1, 2], title="phases.v", aspect=DataAspect())
    heatmap!(ax_p2, a.X.v.x, a.X.v.y, a.phases.v[inx_v, iny_v]')
    display(fig_init)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()

    #--------------------------------------------#

    for it = 1:nt

        @printf("Step %04d --- mean(Pt) = %1.2f GPa\n", it, mean(a.Pt) .* sc.σ / 1e9)

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#
        τxyc = av2D(a.τ.xy)
        τII = sqrt.(0.5 .* (a.τ.xx[inx_c, iny_c] .^ 2 + a.τ.yy[inx_c, iny_c] .^ 2 + (-a.τ.xx[inx_c, iny_c] - a.τ.yy[inx_c, iny_c]) .^ 2) .+ τxyc[inx_c, iny_c] .^ 2)
        ε̇xyc = av2D(a.ε̇.xy)
        ε̇II = sqrt.(0.5 .* (a.ε̇.xx[inx_c, iny_c] .^ 2 + a.ε̇.yy[inx_c, iny_c] .^ 2 + (-a.ε̇.xx[inx_c, iny_c] - a.ε̇.yy[inx_c, iny_c]) .^ 2) .+ ε̇xyc[inx_c, iny_c] .^ 2)

        mp = materials.plasticity
        φ = mp.ϕ[1]
        C = mp.C[1]
        σT = mp.σT[1]
        P_end = 0.05

        fig = Figure(size=(1200, 900))

        ax1 = Axis(fig[1, 1], xlabel="Iterations @ step $(it)", ylabel="log₁₀ error")
        scatter!(ax1, 1:iter, log10.(err.x[1:iter]), label="Vx")
        scatter!(ax1, 1:iter, log10.(err.y[1:iter]), label="Vy")
        scatter!(ax1, 1:iter, log10.(err.p[1:iter]), label="Pt")
        axislegend(ax1, position=:rt)

        ax2 = Axis(fig[1, 2], title="log10 ε̇II [1/s]", aspect=DataAspect())
        heatmap!(ax2, a.X.c.x * sc.L * 1e2, a.X.c.y * sc.L * 1e2, log10.(ε̇II ./ sc.t)', colormap=:coolwarm)
        xlims!(ax2, extrema(a.X.c.x * sc.L * 1e2))

        ax3 = Axis(fig[2, 1], xlabel="P [GPa]", ylabel="τII [GPa]")
        function F_hyperbolic(τ, P, φ, C, σT)
            return sqrt.(τ .^ 2 .+ (C * cosd(φ) - σT * sind(φ)) .^ 2) .- (C * cosd(φ) .+ P * sind(φ))
        end
        P_ax = LinRange(-σT, P_end, 100)
        τ_ax = collect(P_ax * sind(φ) .+ C * cosd(φ))
        for _ in 1:10
            τ_ax .-= F_hyperbolic(τ_ax, P_ax, φ, C, σT)
        end
        lines!(ax3, P_ax .* sc.σ / 1e9, τ_ax .* sc.σ / 1e9, color=:black)
        scatter!(ax3, a.Pt[inx_c, iny_c][:] .* sc.σ / 1e9, τII[:] .* sc.σ / 1e9, markersize=3)

        ax4 = Axis(fig[2, 2], title="τII [MPa]", aspect=DataAspect())
        heatmap!(ax4, a.X.c.x * sc.L * 1e2, a.X.c.y * sc.L * 1e2, τII' .* sc.σ ./ 1e6, colormap=:turbo)
        xlims!(ax4, extrema(a.X.c.x * sc.L * 1e2))

        display(fig)

        @show (3 / materials.β[1] - 2 * materials.G[1]) / (2 * (3 / materials.β[1] + 2 * materials.G[1]))

    end

    display(to)

end

let
    # r = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4]
    r = [0.4 0.45]
    for i in eachindex(r)
        main((x=100, y=100), r[i])
    end
end
