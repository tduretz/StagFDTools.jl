
function compute_vorticity!(allocs, Δ)
    @views allocs.ω.xy_v .= 0.5 .* (
        (allocs.V.x[:, 2:end] .- allocs.V.x[:, 1:(end-1)]) ./ Δ.y .-
            (allocs.V.y[2:end, :] .- allocs.V.y[1:(end-1), :]) ./ Δ.x
    )
    allocs.ω.xy_c .= av2D(allocs.ω.xy_v)
    return nothing
end

function _assemble!(a::Allocs, materials, BC, nc, Δ)
    # Jacobian
    AssembleContinuity2D!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.β, a.ξ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_x!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_y!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, a.ρ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    # Picard preconditioner
    AssembleContinuity2D!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_x!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_y!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
end

function update_solution!(a::Allocs, materials, BC, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params)
    a.𝐊 .= [a.M.Vx.Vx a.M.Vx.Vy; a.M.Vy.Vx a.M.Vy.Vy]
    a.𝐐 .= [a.M.Vx.Pt; a.M.Vy.Pt]
    a.𝐐ᵀ .= [a.M.Pt.Vx a.M.Pt.Vy]
    a.𝐏 .= a.M.Pt.Pt
    a.𝐊_PC .= [a.M_PC.Vx.Vx a.M_PC.Vx.Vy; a.M_PC.Vy.Vx a.M_PC.Vy.Vy]
    a.𝐐_PC .= [a.M_PC.Vx.Pt; a.M_PC.Vy.Pt]
    a.𝐐ᵀ_PC .= [a.M_PC.Pt.Vx a.M_PC.Pt.Vy]
    a.𝐏_PC .= a.M_PC.Pt.Pt

    ϵ_l = iter_params.inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : iter_params.ϵ_l
    @printf("Abs. res. = %02e --- Rel. res = %02e  --- ϵ_l = %1.2e\n", ϵ, ϵ / ϵ0, ϵ_l)

    @timeit to "Linear solve" begin
        mechanical_solver!(a.dx, a.M, a.r, a.𝐊, a.𝐐, a.𝐐ᵀ, a.𝐏, a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC; solver=iter_params.solver_type, ηb=iter_params.γ, ϵ_l=ϵ_l, niter_l=10, restart=20)
    end

    @timeit to "Line search" begin
        imin = LineSearch!(rvec, iter_params.α, a.dx, a.R, a.V, a.Pt, a.ε̇, a.τ, a.Vi, a.Pti, a.ΔPt, a.Pt0, a.τ0, a.λ̇, a.η, a.G, a.β, a.ξ, a.ρ, a.𝐷, a.𝐷_ctl, a.number, a.type, BC, materials, a.phase_ratios, nc, Δ)
    end
    UpdateSolution!(a.V, a.Pt, iter_params.α[imin] * a.dx, a.number, a.type, nc)
end

function Solve!(a::Allocs, materials, BC, nc, Δ, to,
    rvec, iter, ϵ0, ϵ, iter_params)
    @timeit to "Assembly" _assemble!(a, materials, BC, nc, Δ)
    update_solution!(a, materials, BC, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params)
end

function main_solver!(a::Allocs, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c,
    inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    nVx = maximum(a.number.Vx)
    nVy = maximum(a.number.Vy)
    nPt = maximum(a.number.Pt)

    compute_grid_fields!(a.G, a.β, a.ρ, a.ξ, materials, a.phase_ratios, nc, nphases)

    @printf("Time step %04d (nthreads = %03d)\n", it, Threads.nthreads())
    iter, ϵ0, ϵ = 0, 0.0, 0.0

    while iter < iter_params.niter
        iter += 1
        @printf("Iteration %04d\n", iter)

        @timeit to "Residual" begin
            TangentOperator!(a.𝐷, a.𝐷_ctl, a.τ, a.τ0, a.ε̇, a.λ̇, a.η, a.G, a.V, a.Pt, a.Pt0, a.ΔPt, a.type, BC, materials, a.phase_ratios, Δ)
            ResidualContinuity2D!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ, materials, a.number, a.type, BC, nc, Δ)
            ResidualMomentum2D_x!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, materials, a.number, a.type, BC, nc, Δ)
            ResidualMomentum2D_y!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ, materials, a.number, a.type, BC, nc, Δ)
        end

        err.x[iter] = @views norm(a.R.x[inx_Vx, iny_Vx]) / sqrt(nVx)
        err.y[iter] = @views norm(a.R.y[inx_Vy, iny_Vy]) / sqrt(nVy)
        err.p[iter] = @views norm(a.R.p[inx_c, iny_c]) / sqrt(nPt)
        ϵ = max(err.x[iter], err.y[iter])
        (iter == 1) && (ϵ0 = ϵ)
        ϵ < iter_params.ϵ_nl && break

        SetRHS!(a.r, a.R, a.number, a.type, nc)
        Solve!(a, materials, BC, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params)
    end

    a.Pt .+= a.ΔPt.c

    return iter, err
end

# No advection
main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err) = main_loop(a, nothing, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

function main_loop(a::Allocs, adv::Nothing, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
    @printf("Step %04d\n", it)
    a.τ0.xx .= a.τ.xx
    a.τ0.yy .= a.τ.yy
    a.τ0.xy .= a.τ.xy
    a.Pt0 .= a.Pt
    return main_solver!(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
end

function main_loop(a::Allocs, adv::Markers, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    # Solve
    main_solver!(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    @views V = (a.V.x[2:(end-1), 2:(end-1)], a.V.y[2:(end-1), 2:(end-1)])

    advection!(adv.particles, RungeKutta2(), V, Δ.t)
    move_particles!(adv.particles, adv.particle_args)
    inject_particles!(adv.particles, adv.particle_args)

    # Update phase_ratios for the solver (includes ghost nodes)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, adv.particle_args[1])
end

function main_loop(a::Allocs, adv::Markers, tm::TensorMarkers, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    # Solve
    main_solver!(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    @views V = (a.V.x[2:(end-1), 2:(end-1)], a.V.y[2:(end-1), 2:(end-1)])

    increment_stress!(a, tm, adv.particles, Δ)
    advection!(adv.particles, RungeKutta2(), V, Δ.t)
    move_particles!(adv.particles, adv.particle_args)
    inject_particles!(adv.particles, adv.particle_args)
    stress_ToGrid!(tm, a, adv.particles)

    # Update phase_ratios for the solver (includes ghost nodes)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, adv.particle_args[1])
end