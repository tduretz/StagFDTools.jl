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

    # Intialise field
    radius = 0.1

    Δt0 = 0.5
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x = (min= -L.x / 2, max= L.x / 2)
    y = (min= -L.y / 2, max= L.y / 2)

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [ -1e-3    0.;   # Make background shear rate negligible
                          0    1e-3 ])

    # Material parameters
    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; compressible=true, plasticity=Kiss2023)

    materials.ρ    .= [0.0,    0.0  ]
    materials.g     .= [0.0,   0.0]
    materials.η0   .= [1e3,    1e-1 ]
    materials.G    .= [1e1,   1e1  ]
    materials.plasticity.C    .= [100.0,  100.0]
    materials.plasticity.σT   .= [50.0,   50.0 ] # Kiss2023
    materials.plasticity.δσT  .= [10.0,   10.0 ] # Kiss2023
    materials.plasticity.P1   .= [0.0,    0.0  ] # Kiss2023
    materials.plasticity.τ1   .= [0.0,    0.0  ] # Kiss2023
    materials.plasticity.P2   .= [0.0,    0.0  ] # Kiss2023
    materials.plasticity.ϕ    .= [30.0,   30.0 ]
    materials.plasticity.ηvp  .= [0.1,    0.1  ]
    materials.β    .= [1e-2,   1e-2 ]
    materials.plasticity.ψ    .= [3.0,    3.0  ]

    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 245

    # Solver parameters
    iter_params = IterParams(niter=20, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)
    nVx = maximum(a.number.Vx)
    nVy = maximum(a.number.Vy)
    nPt = maximum(a.number.Pt)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c ]  .= 0.
    @views a.Pt[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= 0.1^2] .= 1.0
    @views a.type.Pt[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= radius^2] .= :constant
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
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= radius^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= radius^2] .= 2
    FillPhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()
    time = 0.0

    #--------------------------------------------#

    # Note: this example imposes a growing Dirichlet pressure inside the hole
    # (`a.Pt` at `:constant` cells) at every time step, which must happen
    # strictly between the τ0/Pt0 swap and the nonlinear iterations so that
    # the pressure increase is registered as new visco-elastic loading. This
    # ordering is not exposed by `main_loop`, so the Newton loop is kept
    # explicit here (built from the same blocks `main_loop` uses internally).
    for it=1:nt

        time += Δ.t
        @printf("Step %04d --- time = %1.3f \n", it, time)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)

        # Swap old values
        a.τ0.xx .= a.τ.xx
        a.τ0.yy .= a.τ.yy
        a.τ0.xy .= a.τ.xy
        a.Pt0   .= a.Pt

        # Update pressure in the hole
        @views a.Pt[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= radius^2] .= 1 + 5*time
        compute_grid_fields!(a.G, a.β, a.ρ, a.ξ, materials, a.phase_ratios, nc, nphases)

        for iter=1:iter_params.niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check
            @timeit to "Residual" begin
                TangentOperator!(a.𝐷, a.𝐷_ctl, a.τ, a.τ0, a.ε̇, a.λ̇, a.η, a.G, a.V, a.Pt, a.Pt0, a.ΔPt, a.type, BC, materials, a.phase_ratios, Δ)
                ResidualContinuity2D!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ, materials, a.number, a.type, BC, nc, Δ)
                ResidualMomentum2D_x!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, materials, a.number, a.type, BC, nc, Δ)
                ResidualMomentum2D_y!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ, materials, a.number, a.type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(a.R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(a.R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(a.R.p[inx_c,iny_c])/sqrt(nPt)
            max(err.x[iter], err.y[iter]) < iter_params.ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(a.r, a.R, a.number, a.type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                AssembleContinuity2D!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.β, a.ξ, materials, a.number, a.pattern, a.type, BC, nc, Δ)
                AssembleMomentum2D_x!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, materials, a.number, a.pattern, a.type, BC, nc, Δ)
                AssembleMomentum2D_y!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, a.ρ, materials, a.number, a.pattern, a.type, BC, nc, Δ)
            end

            #--------------------------------------------#
            # Stokes operator as block matrices
            a.𝐊  .= [a.M.Vx.Vx a.M.Vx.Vy; a.M.Vy.Vx a.M.Vy.Vy]
            a.𝐐  .= [a.M.Vx.Pt; a.M.Vy.Pt]
            a.𝐐ᵀ .= [a.M.Pt.Vx a.M.Pt.Vy]
            a.𝐏  .= a.M.Pt.Pt

            #--------------------------------------------#

            # Direct-iterative solver
            fu   = @views -a.r[1:size(a.𝐊,1)]
            fp   = @views -a.r[size(a.𝐊,1)+1:end]
            u, p = DecoupledSolver(a.𝐊, a.𝐐, a.𝐐ᵀ, a.𝐏, fu, fp; fact=:lu,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            @views a.dx[1:size(a.𝐊,1)]     .= u
            @views a.dx[size(a.𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, iter_params.α, a.dx, a.R, a.V, a.Pt, a.ε̇, a.τ, a.Vi, a.Pti, a.ΔPt, a.Pt0, a.τ0, a.λ̇, a.η, a.G, a.β, a.ξ, a.ρ, a.𝐷, a.𝐷_ctl, a.number, a.type, BC, materials, a.phase_ratios, nc, Δ)
            UpdateSolution!(a.V, a.Pt, iter_params.α[imin]*a.dx, a.number, a.type, nc)
            TangentOperator!(a.𝐷, a.𝐷_ctl, a.τ, a.τ0, a.ε̇, a.λ̇, a.η, a.G, a.V, a.Pt, a.Pt0, a.ΔPt, a.type, BC, materials, a.phase_ratios, Δ)

        end

        # Update pressure
        a.Pt .+= a.ΔPt.c

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

        P_end =  600

        fig = Figure(resolution = (1100,800))

        # Top-left: nonlinear solver convergence (iterations)
        ax_conv = Axis(fig[1,1], xlabel = "Iterations @ step $(it)", ylabel = "log₁₀ error")
        lines!(ax_conv, 1:iter_params.niter, log10.(err.x[1:iter_params.niter]), color = :blue)
        scatter!(ax_conv, 1:iter_params.niter, log10.(err.y[1:iter_params.niter]), color = :orange)
        scatter!(ax_conv, 1:iter_params.niter, log10.(err.p[1:iter_params.niter]), color = :green)

        # Top-right: pressure field
        ax_pt = Axis(fig[1,2], title = "Pt")
        heatmap!(ax_pt, a.X.c.x, a.X.c.y, a.Pt[inx_c,iny_c]')

        # Bottom-left: yield/τII diagram
        ax_yield = Axis(fig[2,1], xlabel = "P", ylabel = "τII", aspect = DataAspect())
        lines!(ax_yield, [pc1, pc1, pc2, P_end], [0.0, τc1, τc2, P_end*sind(φ)+C*cosd(φ)], color = :black)
        lines!(ax_yield, p_tr1, l1, color = :blue)
        lines!(ax_yield, p_tr2, l2, color = :blue)
        lines!(ax_yield, p_tr3, l3, color = :blue)
        scatter!(ax_yield, vec(a.Pt[inx_c,iny_c]), vec(τII), color = :red)

        # Bottom-right: strain-rate magnitude
        ax_e = Axis(fig[2,2], title = "ε̇II")
        heatmap!(ax_e, a.X.c.x, a.X.c.y, log10.(ε̇II)')

        display(fig)

    end

    display(to)

end

let
    main((x = 100, y = 100))
end
