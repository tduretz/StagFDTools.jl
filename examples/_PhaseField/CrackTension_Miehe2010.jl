#---------------------------------------------------------------------------------------
# Single edge notched tension test (Miehe et al., 2010) with the PhaseFieldMechanics module.
# Same setup as Mechanics_v11_ElasticDamage_v1.jl of FDMIC_Geodynamics.jl:
#   1 mm × 1 mm plate, horizontal notch from the W side to the centre (phase field pinned to 1),
#   K = 174.9 GPa, G = 80.77 GPa, gc = 2.7 kN/m, l = 15 μm, Δu = 1e-5 mm per step.
# The (Vx, Vy, Pt, d) system is solved monolithically with Newton iterations.
#
# config = :free_slip      : FDMIC BCs (Vx = 0 on W/E, free slip on all sides, Vy prescribed on S/N)
# config = :clamped_tension: Miehe BCs (S clamped, N: Vx = 0 and Vy prescribed, W/E traction free)
#
# solver = :lu  : sparse LU of the full Jacobian
# solver = :gcr : GCR of the TwoPhases module, block preconditioner with Cholesky factors of the d-d block and of
#                 the velocity Schur complement
#---------------------------------------------------------------------------------------
using StagFDTools, StagFDTools.PhaseFieldMechanics, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf

@views function main(; nc=(x=150, y=150), config=:free_slip, amor=false, u_max=7e-3, Δu_coarse=1e-4, u_fine=4e-3, Δu_fine=1e-5,
                       Δu_min=1e-6, ηd_phys=1e3, ε̇_load=2e-6, Δd_relax=1e-3,
                       niter=15, ϵ_nl=1e-10, solver=:lu, ϵ_lin=1e-10, nout=50, noisy=false, debug=false, lag_pc=:none, figname=nothing)

    # Characteristic scales: lengths in mm, stresses in GPa -> forces (per mm thickness) in kN
    sc = (L = 1e-3, S = 1e9, t = 1.0)

    # Material parameters
    materials = (k_res = 1e-6, amor = amor)
    K   = 174.9e9   / sc.S
    G   = 80.77e9   / sc.S
    η   = Inf
    gc  = 2.7e3     / (sc.S * sc.L)
    l   = 15e-6     / sc.L
    ηd  = ηd_phys * ε̇_load / sc.S    # viscous regularisation [Pa⋅s] (FDMIC/Miehe: 1e3), time unit = 1/ε̇_load
    τd  = ηd * l / gc                # relaxation time of the phase field

    # Loading: velocity of the N boundary = 1 (mm per unit time), Δt = displacement increment
    L   = (x = 1.0, y = 1.0)
    Δ   = (x = L.x/nc.x, y = L.y/nc.y, t = Δu_coarse)
    v_N = 1.0

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    X = GenerateGrid((min=-L.x/2, max=L.x/2), (min=0.0, max=L.y), Δ, nc)
    xc, yc = X.c.x, X.c.y

    # Node types, notch: phase field prescribed (d = 1)
    type   = set_BC_template(config, nc)
    notch  = (xc .< 0.0) .& (abs.(yc' .- L.y/2) .< 7.5e-6/sc.L)
    type.d[inx_c, iny_c][notch] .= :constant

    # Numbering
    number = Fields(fill(0, size_x), fill(0, size_y), fill(0, size_c), fill(0, size_c))
    Numbering!(number, type, nc)
    nVx, nVy, nPt, nd = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt), maximum(number.d)
    ndof = nVx + nVy + nPt + nd
    @printf("nc = %d × %d, l/Δx = %1.2f, %d dofs\n", nc.x, nc.y, l/Δ.x, ndof)
    dx, r = zeros(ndof), zeros(ndof)

    # Fields
    V    = (x = zeros(size_x...), y = zeros(size_y...))
    Vi   = (x = zeros(size_x...), y = zeros(size_y...))
    P    = (t = zeros(size_c...),)
    Pi   = (t = zeros(size_c...),)
    d    = zeros(size_c...)
    di   = zeros(size_c...)
    d[inx_c, iny_c][notch] .= 1.0
    τ    = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...))
    ψ    = zeros(size_c...)
    old  = (τ = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...)), P = zeros(size_c...), d = copy(d), H = zeros(size_c...), Wp = zeros(size_c...))
    R    = (x = zeros(size_x...), y = zeros(size_y...), pt = zeros(size_c...), d = zeros(size_c...))
    rheo = (G  = (c = fill(G, size_c...), v = fill(G, size_v...)),
            η  = (c = fill(η, size_c...), v = fill(η, size_v...)),
            K  = (c = fill(K, size_c...),),
            gc = (c = fill(gc, size_c...),),
            l  = (c = fill(l, size_c...),),
            ηd = (c = fill(ηd, size_c...),),
            ph = (c = ones(Int, size_c...), v = ones(Int, size_v...)))

    # Boundary velocities: Vy = v_N y on S/N (Dirichlet_normal values live in V), Vx = 0 elsewhere
    V.y[inx_Vy, iny_Vy] .= v_N .* X.v.y'
    BC = (Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), d = zeros(size_c...))

    # Global Jacobian: minimal sparsity pattern (recorded once), LU symbolic analysis reused
    J = JacobianCache(V, P, d, old, rheo, materials, number, type, BC, nc, Δ)
    @printf("Jacobian: %d nonzeros\n", nnz(J.A))

    # Storage
    u, it = 0.0, 0
    hist  = (u = Float64[], F = Float64[], iters = Int[], dmax = Float64[])
    backup = (V = deepcopy(V), P = deepcopy(P), d = copy(d))
    # Previous step (for the predictor: linear extrapolation of d and P in time)
    prev   = (P = zeros(size_c...), d = copy(d), Δt = Inf, mode = :load)
    is_d   = type.d .=== :in

    # Step control. Loading steps: N boundary moves by Δu. When Δu falls below Δu_min (unstable crack growth,
    # snap-back), relaxation steps are taken: the boundary is held fixed and the crack advances in time through
    # the viscous regularisation ηd ∂d/∂t. Loading resumes once d no longer evolves
    # with Δt ≫ τd (Δt > 1e3 τd and max Δd τd/Δt < Δd_relax).
    mode, Δu, Δt_relax = :load, Δu_coarse, 10τd
    ϵref = nothing
    t_solve, n_newton, n_lin, max_lin, n_lin_fail = 0.0, 0, 0, 0, 0   # linear solver statistics

    u_kick  = 0.0
    n_stuck = 0                      # load failures → relax → arrest cycles without any accepted load step
    while u < u_max - 1e-12
        Δu_target = u < u_fine - 1e-12 ? Δu_coarse : Δu_fine
        Δu        = min(Δu, Δu_target)
        # Relaxation: the first step applies the failed increment u_kick quickly (Δt ~ τd) to push the system
        # past the limit point; then the boundary is held fixed
        v_step    = mode == :load ? v_N : u_kick / Δt_relax
        Δ = merge(Δ, (t = mode == :load ? Δu / v_N : Δt_relax,))
        V.y[inx_Vy, iny_Vy[end]] .= v_step
        backup.V.x .= V.x; backup.V.y .= V.y; backup.P.t .= P.t; backup.d .= d
        # Predictor
        # Only extrapolate within the same mode (the relax → load ratio Δt/prev.Δt is ~1e3+)
        if isfinite(prev.Δt) && prev.mode == mode
            ω = min(Δ.t / prev.Δt, 2.0)
            @. P.t = old.P + ω * (old.P - prev.P)
            @. d   = ifelse(is_d, clamp(old.d + ω * (old.d - prev.d), 0.0, 1.0), d)
        end

        # Newton iterations. Residuals are normalised per field by a fixed reference (running maximum of the
        # first-iteration residuals, i.e. the loading scale), not by the first residual of the current step:
        # the latter can itself be at round-off level (e.g. Rp = ΔP/(KΔt) when P is well predicted).
        converged, iter = false, 0
        ϵhist = zeros(niter)
        nflip = zeros(Int, niter); act_prev = falses(nc.x, nc.y); flips = CartesianIndex{2}[]
        for outer iter = 1:niter
            Residuals!(R, V, P, d, old, rheo, materials, type, BC, nc, Δ)
            if debug   # active set of the history switch H = max(H⁰, ψ)
                UpdateStress!(τ, ψ, V, P, d, old, rheo, materials, type, BC, nc, Δ)
                act = ψ[inx_c, iny_c] .> old.H[inx_c, iny_c]
                iter > 1 && (flips = findall(act .!= act_prev); nflip[iter] = length(flips))
                act_prev .= act
            end
            ϵ = residual_norms(R, number, nc)
            if iter == 1
                ϵref = isnothing(ϵref) ? ϵ : map(max, ϵref, ϵ)
                ϵref = map(e -> max(e, 1e-12 * maximum(ϵref)), ϵref)
            end
            ϵrel = max(ϵ.x/ϵref.x, ϵ.y/ϵref.y, ϵ.pt/ϵref.pt, ϵ.d/ϵref.d)
            noisy && @printf("    iter %02d: |Rx| = %1.2e |Ry| = %1.2e |Rp| = %1.2e |Rd| = %1.2e  rel = %1.2e\n", iter, ϵ.x, ϵ.y, ϵ.pt, ϵ.d, ϵrel)
            if ϵrel < ϵ_nl
                converged = true
                break
            end
            # Stagnation (e.g. cycling of the history switch max(H⁰, ψ) when no equilibrium is nearby):
            # give up early instead of spending niter iterations
            ϵhist[iter] = ϵrel
            iter > 3 && ϵrel > ϵhist[iter-3] / 2 && break
            SetRHS!(r, R, number, type, nc)
            zero_values!(J)
            Assemble!(J.K, V, P, d, old, rheo, materials, number, type, BC, nc, Δ)
            # GCR preconditioner lag: :none (refresh every solve), :step (refresh at the first Newton iteration
            # of each step, then reuse while GCR iterations stay low), :always (reuse across steps too)
            refresh  = lag_pc === :none || (lag_pc === :step && iter == 1) ? true : :auto
            t_solve += @elapsed nlin = solve!(dx, J, r; solver=solver, reltol=ϵ_lin, refresh=refresh)
            n_newton += 1; n_lin += abs(nlin); max_lin = max(max_lin, abs(nlin)); n_lin_fail += nlin < 0
            α, _ = BackTrackingLineSearch!(R, dx, V, P, d, Vi, Pi, di, old, rheo, materials, number, type, BC, nc, Δ, ϵref)
            noisy && α < 1 && @printf("    line search: α = %1.3f\n", α)
        end
        if !converged
            @printf("    failed Newton history (rel): %s\n", join([@sprintf("%1.1e", e) for e in ϵhist[1:iter] if e > 0], " "))
            if debug
                @printf("    active-set flips per iteration: %s\n", join(nflip[2:iter], " "))
                for I in flips[1:min(end, 6)]
                    @printf("      flip at x = %+1.4f, y = %1.4f, d = %1.3f, ψ/H⁰ = %1.4f\n", xc[I[1]], yc[I[2]], d[inx_c, iny_c][I], ψ[inx_c, iny_c][I]/old.H[inx_c, iny_c][I])
                end
                Imax = argmax(abs.(R.d[inx_c, iny_c]))
                @printf("      max |Rd| at x = %+1.4f, y = %1.4f (d = %1.3f); d range [%1.3e, %1.6f]\n", xc[Imax[1]], yc[Imax[2]], d[inx_c, iny_c][Imax], extrema(d[inx_c, iny_c])...)
            end
            # Restart the increment with a halved step
            V.x .= backup.V.x; V.y .= backup.V.y; P.t .= backup.P.t; d .= backup.d
            if mode == :load
                Δu /= 2
                if Δu < Δu_min
                    # Restart relaxation from a short time step: Δt_relax may still hold the (huge) value
                    # reached when the previous episode ended, which would declare arrest immediately
                    mode, u_kick, Δu, Δt_relax = :relax, 2Δu, Δu_fine, 10τd
                    n_stuck += 1
                    n_stuck > 3 && error("no progress at u = $u: loading fails right after crack arrest")
                    @printf("u = %1.4e mm: Newton failed, switching to relaxation steps (Δt = %1.2e τd)\n", u, Δt_relax/τd)
                else
                    @printf("u = %1.4e mm: Newton failed, Δu -> %1.2e mm\n", u, Δu)
                end
            else
                Δt_relax /= 2
                @printf("u = %1.4e mm: Newton failed, Δt -> %1.2e τd\n", u, Δt_relax/τd)
                Δt_relax < 1e-3τd && error("Δt too small")
            end
            continue
        end

        # Accept step: update history
        mode_step = mode
        it += 1
        u  += Δ.t * v_step
        mode == :relax && (u_kick = 0.0)
        Δd  = maximum(abs.(d .- old.d))
        easy = iter - 1 <= 4
        if mode == :load
            n_stuck = 0
            easy && (Δu = min(2Δu, Δu_target))
        else
            easy && (Δt_relax *= 2)
            # Arrested: d no longer evolves although Δt ≫ τd (viscous effects negligible)
            if Δ.t > 1e3τd && Δd * τd / Δ.t < Δd_relax
                mode = :load
                @printf("u = %1.4e mm: crack arrested, loading resumes\n", u)
            end
        end
        UpdateStress!(τ, ψ, V, P, d, old, rheo, materials, type, BC, nc, Δ)
        old.τ.xx .= τ.xx; old.τ.yy .= τ.yy; old.τ.xy .= τ.xy
        prev.P   .= old.P
        prev.d   .= old.d
        prev     = merge(prev, (Δt = Δ.t, mode = mode_step))
        old.P    .= P.t
        old.d    .= d
        old.H    .= max.(old.H, ψ)

        # Reaction force on N boundary (per mm thickness): σyy of the last row of centroids
        σ = total_stress(τ, P.t, d, materials)
        F = sum(σ.yy[inx_c, iny_c[end]]) * Δ.x
        push!(hist.u, u); push!(hist.F, F); push!(hist.iters, iter-1); push!(hist.dmax, maximum(d[inx_c, iny_c][.!notch]))
        @printf("step %04d (%s): u = %1.4e mm, F = %1.4e kN, Newton its = %d, max(Δd) = %1.2e, max(d) = %1.3f\n", it, mode, u, F, iter-1, Δd, hist.dmax[end]); flush(stdout)

        if mod(it, nout) == 0 || u >= u_max - 1e-12
            fig = plot_state(X, d, σ, hist, inx_c, iny_c, u)
            display(fig)
            isnothing(figname) || save(figname, fig)
        end
    end
    @printf("Linear solver %s: %d Newton iterations, %1.1f s in solves (%1.3f s per solve), %d linear its (max %d per solve, %d unconverged)\n",
            solver, n_newton, t_solve, t_solve/n_newton, n_lin, max_lin, n_lin_fail)
    solver === :gcr && @printf("GCR preconditioner factorisations: %d for %d solves\n", J.gcr[].n_refresh[], n_newton)
    return (; X, d = d[inx_c, iny_c], hist)
end

function plot_state(X, d, σ, hist, inx_c, iny_c, u)
    fig = Figure(size=(1500, 450), fontsize=16)
    ax  = Axis(fig[1,1], aspect=DataAspect(), title=@sprintf("d, u = %1.3e mm", u), xlabel="x [mm]", ylabel="y [mm]")
    hm  = heatmap!(ax, X.c.x, X.c.y, d[inx_c, iny_c], colormap=:turbo, colorrange=(0,1)); Colorbar(fig[1,2], hm)
    ax  = Axis(fig[1,3], aspect=DataAspect(), title="σyy [GPa]", xlabel="x [mm]")
    hm  = heatmap!(ax, X.c.x, X.c.y, σ.yy[inx_c, iny_c], colormap=:roma); Colorbar(fig[1,4], hm)
    ax  = Axis(fig[1,5], title="Load-displacement", xlabel="u [mm]", ylabel="F [kN]")
    lines!(ax, hist.u, hist.F)
    return fig
end

r = main(config=:free_slip, nout=50)
