
function compute_vorticity!(allocs, Δ)
    @views allocs.ω.xy_v .= 0.5 .* (
        (allocs.V.x[:, 2:end] .- allocs.V.x[:, 1:(end-1)]) ./ Δ.y .-
            (allocs.V.y[2:end, :] .- allocs.V.y[1:(end-1), :]) ./ Δ.x
    )
    allocs.ω.xy_c .= av2D(allocs.ω.xy_v)
    return nothing
end

"""
    assemble_jacobian!(M, a, 𝐷, materials, BC, nc, Δ)

Assemble the Stokes block matrix `M` (CSC blocks) with the tangent operator `𝐷`.
The sparsity pattern is recorded on the first call; later calls write the values
in place, in parallel.
"""
function assemble_jacobian!(M, a::Allocs, 𝐷, materials, BC, nc, Δ)
    if nnz(M.Vx.Vx) == 0
        K_pat = pattern_storage(M)
        _assemble_blocks!(K_pat, a, 𝐷, materials, BC, nc, Δ)
        set_sparsity_pattern!(M, K_pat)
    end
    zero_values!(M)
    _assemble_blocks!(fixed_pattern(M), a, 𝐷, materials, BC, nc, Δ)
    return M
end

function _assemble_blocks!(K, a::Allocs, 𝐷, materials, BC, nc, Δ)
    AssembleContinuity2D!(K, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, 𝐷, a.β, a.ξ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_x!(K, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, 𝐷, a.G,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_y!(K, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, 𝐷, a.G, a.ρ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
end

function _assemble!(a::Allocs, materials, BC, nc, Δ; Newton=true, refresh_PC=true)
    Newton && assemble_jacobian!(a.M, a, a.𝐷_ctl, materials, BC, nc, Δ)        # Jacobian (not needed for Picard)
    refresh_PC && assemble_jacobian!(a.M_PC, a, a.𝐷, materials, BC, nc, Δ)    # Picard preconditioner
end

"""
    set_block_matrices!(𝐊, 𝐐, 𝐐ᵀ, 𝐏, M)

Copy the velocity (𝐊), gradient (𝐐), divergence (𝐐ᵀ) and pressure (𝐏) blocks of `M`.
"""
function set_block_matrices!(𝐊, 𝐐, 𝐐ᵀ, 𝐏, M)
    M = dropzeros(M)
    𝐊 .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
    𝐐 .= [M.Vx.Pt; M.Vy.Pt]
    𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
    𝐏 .= M.Pt.Pt
    return nothing
end

# Newton: Jacobian (M, assembled with 𝐷_ctl) as operator; Picard: M_PC (assembled with 𝐷) as operator
# :GCR: the cached solver (a.gcr) works on the blocks of M and M_PC directly; refresh_PC: new Schur complement and Cholesky factor
function update_solution!(a::Allocs, materials, BC, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params; Newton=true, refresh_PC=true)
    if iter_params.solver_type !== :GCR
        Newton && set_block_matrices!(a.𝐊, a.𝐐, a.𝐐ᵀ, a.𝐏, a.M)
        set_block_matrices!(a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC, a.M_PC)
    end

    ϵ_l = iter_params.inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : iter_params.ϵ_l
    # @printf("Abs. res. = %02e --- Rel. res = %02e  --- ϵ_l = %1.2e\n", ϵ, ϵ / ϵ0, ϵ_l)

    its = 0
    @timeit to "Linear solve" begin
        if iter_params.solver_type === :GCR
            a.gcr.initialized || setup_gcr!(a.gcr, a.M, a.M_PC; restart=20)
            (refresh_PC || a.gcr.Kf === nothing) && @timeit to "Preconditioner" update_preconditioner!(a.gcr, a.M_PC; ηb=iter_params.γ)
            update_operator!(a.gcr, Newton ? a.M : a.M_PC)
            its = KSP_GCR_Stokes!(a.dx, a.gcr, a.r; ϵ_l=ϵ_l, maxit=iter_params.niter_KSP, noisy=false)
        elseif Newton
            its = mechanical_solver!(a.dx, a.M, a.r, a.𝐊, a.𝐐, a.𝐐ᵀ, a.𝐏, a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC; solver=iter_params.solver_type, ηb=iter_params.γ, ϵ_l=ϵ_l, niter_l=10, restart=20, maxit=iter_params.niter_KSP, noisy=false)
        else
            its = mechanical_solver!(a.dx, a.M_PC, a.r, a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC, a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC; solver=iter_params.solver_type, ηb=iter_params.γ, ϵ_l=ϵ_l, niter_l=10, restart=20, maxit=iter_params.niter_KSP, noisy=false)
        end
    end

    @timeit to "Line search" begin
        if iter_params.line_search === :grid
            imin = LineSearch!(rvec, iter_params.α, a.dx, a.R, a.V, a.Pt, a.ε̇, a.τ, a.Vi, a.Pti, a.ΔPt, a.Pt0, a.τ0, a.λ̇, a.η, a.G, a.β, a.ξ, a.ρ, a.𝐷, a.𝐷_ctl, a.number, a.type, BC, materials, a.phase_ratios, nc, Δ; εp=a.εp)
            α = iter_params.α[imin]
        else
            α, success = BackTrackingLineSearch!(a.R, a.dx, a.V, a.Pt, a.ε̇, a.τ, a.Vi, a.Pti, a.ΔPt, a.Pt0, a.τ0, a.λ̇, a.η, a.G, a.β, a.ξ, a.ρ, a.𝐷, a.𝐷_ctl, a.number, a.type, BC, materials, a.phase_ratios, nc, Δ;
                α0=Newton ? 1.0 : iter_params.α_Picard, εp=a.εp)
        end
    end
    UpdateSolution!(a.V, a.Pt, α * a.dx, a.number, a.type, nc)
    return Newton, α, its
end

function Solve!(a::Allocs, materials, BC, nc, Δ, to,
    rvec, iter, ϵ0, ϵ, iter_params)
    # Picard iterations until the relative residual drops below Pic2Newt, then Newton
    Newton = ϵ / ϵ0 < iter_params.Pic2Newt
    # Picard preconditioner: at each Picard iteration (it is the operator), else once per step if PC_refresh = :step
    refresh_PC = !Newton || iter == 1 || iter_params.PC_refresh !== :step || iter_params.solver_type !== :GCR
    @timeit to "Assembly" _assemble!(a, materials, BC, nc, Δ; Newton, refresh_PC)
    return update_solution!(a, materials, BC, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params; Newton, refresh_PC)
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
    # Iteration log, printed at the end of the step: residuals, step type, α and KSP iterations
    log_ϵ, log_step, log_α = zeros(iter_params.niter), fill("", iter_params.niter), zeros(iter_params.niter)
    log_KSP = zeros(Int, iter_params.niter)

    while iter < iter_params.niter
        iter += 1
        @printf("Time step %04d -- Iter. %03d\n", it, iter)

        @timeit to "Residual" begin
            TangentOperator!(a.𝐷, a.𝐷_ctl, a.τ, a.τ0, a.ε̇, a.λ̇, a.η, a.G, a.V, a.Pt, a.Pt0, a.ΔPt, a.type, BC, materials, a.phase_ratios, Δ; εp=a.εp)
            ResidualContinuity2D!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ, materials, a.number, a.type, BC, nc, Δ)
            ResidualMomentum2D_x!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, materials, a.number, a.type, BC, nc, Δ)
            ResidualMomentum2D_y!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ, materials, a.number, a.type, BC, nc, Δ)
        end

        err.x[iter] = @views norm(a.R.x[inx_Vx, iny_Vx]) / sqrt(nVx)
        err.y[iter] = @views norm(a.R.y[inx_Vy, iny_Vy]) / sqrt(nVy)
        err.p[iter] = @views norm(a.R.p[inx_c, iny_c]) / sqrt(nPt)
        ϵ = max(err.x[iter], err.y[iter])
        (iter == 1) && (ϵ0 = ϵ)
        ϵ_l = iter_params.inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : iter_params.ϵ_l
        @printf("Abs. res. = %02e --- Rel. res = %02e  --- ϵ_l = %1.2e\n", ϵ, ϵ / ϵ0, ϵ_l)
        log_ϵ[iter] = ϵ
        ϵ < iter_params.ϵ_nl && break

        SetRHS!(a.r, a.R, a.number, a.type, nc)
        Newton, log_α[iter], log_KSP[iter] = Solve!(a, materials, BC, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params)
        log_step[iter] = Newton ? "Newton" : "Picard"
    end

    @printf("Time step %04d: iteration log\n", it)
    for k in 1:iter
        if log_step[k] == ""
            @printf("    Iter. %02d. %-9s abs. res. = %.3e, rel. res. = %.3e\n", k, "converged", log_ϵ[k], log_ϵ[k] / ϵ0)
        else
            @printf("    Iter. %02d. %-9s abs. res. = %.3e, rel. res. = %.3e, α = %.3f, KSP its = %d\n", k, log_step[k] * ".", log_ϵ[k], log_ϵ[k] / ϵ0, log_α[k], log_KSP[k])
        end
    end

    # Stop instead of carrying on with an unconverged (or NaN) solution, unless below the last resort tolerance
    if !(ϵ < iter_params.ϵ_nl)
        ϵ < iter_params.ϵ_nl_last || error(@sprintf("Time step %04d: Newton did not converge in %d iterations (abs. res. = %.2e > ϵ_nl_last = %.2e)", it, iter, ϵ, iter_params.ϵ_nl_last))
        @printf("Time step %04d: WARNING, not converged in %d iterations, accepted with abs. res. = %.2e < ϵ_nl_last = %.2e\n", it, iter, ϵ, iter_params.ϵ_nl_last)
    end

    a.Pt .+= a.ΔPt.c

    return iter, err
end

function update_old_stress!(a::Allocs)
    a.τ0.xx .= a.τ.xx
    a.τ0.yy .= a.τ.yy
    a.τ0.xy .= a.τ.xy
    a.Pt0 .= a.Pt
    return nothing
end

# No advection
main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err) = main_loop(a, nothing, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

function main_loop(a::Allocs, adv::Nothing, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
    update_old_stress!(a)
    return main_solver!(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
end

# Advect (no stress correction)
function main_loop(a::Allocs, adv::Markers, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err; advect::Bool=true)

    # Static markers update
    !advect && return main_loop(a, nothing, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    # Set old stresses
    update_old_stress!(a)

    # Solve
    main_solver!(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    @views V = (a.V.x[2:(end-1), 2:(end-1)], a.V.y[2:(end-1), 2:(end-1)])

    # Advect
    advection!(adv.particles, RungeKutta4(), V, Δ.t)
    move_particles!(adv.particles, adv.particle_args)
    inject_particles!(adv.particles, adv.particle_args)

    # Update phase_ratios for the solver (includes ghost nodes)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, adv.particle_args[1])
end

# Advect and rotate stresses
function main_loop(a::Allocs, adv::Markers, tm::TensorMarkers, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    # Solve
    main_solver!(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

    @views V = (a.V.x[2:(end-1), 2:(end-1)], a.V.y[2:(end-1), 2:(end-1)])

    # Advect
    stress_ToParticles!(a, tm, materials, adv.particles, Δ)
    advection!(adv.particles, RungeKutta4(), V, Δ.t)
    move_particles!(adv.particles, adv.particle_args)
    inject_particles!(adv.particles, adv.particle_args)

    # Update fields with new positions
    stress_ToGrid!(tm, a, adv.particles)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, adv.particle_args[1])
end