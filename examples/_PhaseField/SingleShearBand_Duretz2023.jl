# Single shear band configuration of Duretz et al. (2023, G-Cubed, "Strain localization in frictional plasticity:
# viscoplastic, gradient and Cosserat regularization"): pre-stressed Drucker-Prager medium under pure shear with a
# weak (G/4) seed in the south-west corner, linear cohesion softening C = C⁰ + h εp (h_phys), regularised through the phase field:
#       gc/l d - ∇⋅(gc l ∇d) - 2(1-d) βp Wp + ηd ∂d/∂t = 0,   Wp = ∫ τII λ̇ dt
#       F = τII - P sinϕ - C cosϕ + Kd ∇²d                  (damage = false: d does not degrade the material)
# ∇²d is taken from the phase-field equation at the point, so λ̇ stays a local variable.
# Kd ∇²d acts as an implicit gradient: Kg k²/(1 + l²k²), Kg ≈ Kd 2βp τy/(gc/l). It stabilises the shortest modes
# only if Kg/l² exceeds the critical (non-associated + softening) modulus: Kd ≳ 3e12 Pa m² for l = 100 m and
# gc/l = 1e8 Pa. Then no viscoplasticity is needed (ηvp = 0); weaker Kd needs ηvp > 0 at the onset of yielding of
# the pre-stressed background. Adaptive sub-stepping (Δt/2 on Newton failure) handles that onset.
# Only Kd βp/(gc/l) matters; gc/l = 1e8 Pa keeps d < ~0.5 at these stress levels.
# Band width (282×208, l = 100 m, h = -2e8 Pa, ηvp = 0): ≈ 616, 894, 1407, 1901 m for Kd = 3e12, 1e13, 3e13, 6e13,
# i.e. W ≈ 230 m + 2.15e-4 √Kd (gradient plasticity scaling, Duretz et al., 2023); Kd = 1e14 gives a conjugate X pattern.
# band_width() extracts the shear band width from a Gaussian fit of the plastic strain profile across the band
# (as in Fig. 7 of Duretz et al., 2023).
using StagFDTools, StagFDTools.PhaseFieldMechanics, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics: mean

# Gaussian fit a exp(-s²/2σ²) + b of a profile: linear least squares in (a, b) for each σ, golden section in σ
function gaussian_fit(s, f)
    misfit(σ) = begin
        A = hcat(exp.(-s.^2 ./ (2σ^2)), ones(length(s)))
        c = A \ f
        sum(abs2, A * c .- f), c
    end
    σs   = range(0.01, 1.0, 400) .* (maximum(s) - minimum(s))
    i    = argmin(first.(misfit.(σs)))
    a, b = σs[max(i-1, 1)], σs[min(i+1, end)]
    φ    = (sqrt(5) - 1) / 2
    for _ in 1:60
        c1, c2 = b - φ*(b - a), a + φ*(b - a)
        first(misfit(c1)) < first(misfit(c2)) ? (b = c2) : (a = c1)
    end
    σ = (a + b) / 2
    return σ, misfit(σ)[2]
end

# Shear band width: centre line from the εp maximum along rows y ∈ yrange (x ∈ xrange), straight-line fit,
# εp binned against the distance s to the line, Gaussian fit. Returns σ, FWHM (same unit as X), angle, profile
function band_width(xc, yc, εp; yrange, xrange = (-Inf, Inf), smax = Inf)
    ys, xs = Float64[], Float64[]
    for j in eachindex(yc)
        yrange[1] <= yc[j] <= yrange[2] || continue
        ii = findall(x -> xrange[1] <= x <= xrange[2], xc)
        k  = ii[argmax(εp[ii, j])]
        δ  = 0.0                                    # sub-cell peak position (parabola)
        if 1 < k < length(xc)
            f0, fm, fp = εp[k, j], εp[k-1, j], εp[k+1, j]
            den = fm - 2f0 + fp
            δ   = den < 0 ? 0.5 * (fm - fp) / den : 0.0
        end
        push!(ys, yc[j]); push!(xs, xc[k] + δ * (xc[2] - xc[1]))
    end
    a, b  = hcat(ones(length(ys)), ys) \ xs        # x = a + b y
    θ     = atan(1, b) * 180 / π                    # angle of the band to the x axis
    Δs    = (xc[2] - xc[1]) / 2
    nb    = round(Int, min(smax, 0.25 * (xc[end] - xc[1])) / Δs)
    sbin  = (-nb:nb) .* Δs
    fsum, cnt = zeros(length(sbin)), zeros(length(sbin))
    for j in eachindex(yc), i in eachindex(xc)
        yrange[1] <= yc[j] <= yrange[2] || continue
        s = (xc[i] - a - b * yc[j]) / sqrt(1 + b^2)
        k = round(Int, s / Δs) + nb + 1
        1 <= k <= length(sbin) || continue
        fsum[k] += εp[i, j]; cnt[k] += 1
    end
    ok    = cnt .> 0
    s, f  = collect(sbin[ok]), fsum[ok] ./ cnt[ok]
    σ, c  = gaussian_fit(s, f)
    return (; σ, fwhm = 2sqrt(2log(2)) * σ, θ, s, f, fit = c[1] .* exp.(-s.^2 ./ (2σ^2)) .+ c[2])
end

@views function main(; nc=(x=282, y=208), nt=40, βp=1.0, gc_l=1e8, l_phys=100.0, ηd_phys=0.0, ηvp_phys=0.0,
                       ψ_dil=0.0, h_phys=-2e8, damage=false, Kd_phys=1e13,
                       niter=30, ϵ_nl=1e-10, nout=10, noisy=false, figname=nothing)

    # Physical parameters (Table 1, single shear band; cohesion softening h is not used)
    Lx, Ly  = 14.1e3, 10.4e3
    pconf   = 250e6
    ε̇bg_phy = 2e-13
    Δt_phy  = 8e8
    rad     = 5e2

    # Scales: stresses G, lengths 1 km, time 1/ε̇bg
    sc  = (σ = 1e10, L = 1e3, t = 1 / ε̇bg_phy)
    ε̇bg = ε̇bg_phy * sc.t
    Δt  = Δt_phy / sc.t

    # Materials: matrix / seed (shear modulus G/4), Drucker-Prager without softening
    G_mat, G_inc = 1e10 / sc.σ, 0.25e10 / sc.σ
    K   = 2e10 / sc.σ
    pl  = DruckerPrager(C = [1.75e7 / sc.σ], ϕ = [30.0], ψ = [ψ_dil], ηvp = [ηvp_phys / sc.σ / sc.t], cosϕ = [0.0], sinϕ = [0.0], sinψ = [0.0], cosψ = [0.0])
    preprocess!(pl)

    # Geometry and grid: x ∈ [0, Lx], y ∈ [0, Ly]
    L   = (x = Lx / sc.L, y = Ly / sc.L)
    Δ   = (x = L.x / nc.x, y = L.y / nc.y, t = Δt)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    X   = GenerateGrid((min = 0.0, max = L.x), (min = 0.0, max = L.y), Δ, nc)

    # Phase field (driven by plastic work only) and gradient regularisation Kd [Pa m²]
    l   = l_phys / sc.L
    gc  = gc_l / sc.σ * l
    ηd  = ηd_phys / sc.σ / sc.t
    Kd  = Kd_phys / sc.σ / sc.L^2
    materials = (k_res = 1e-6, amor = false, plasticity = pl, βp = βp, ψc = Inf, damage = damage, Kd = Kd, h = h_phys / sc.σ, C_min = 0.0)

    # Node types and numbering
    type   = set_BC_template(:free_slip, nc)
    number = Fields(fill(0, size_x), fill(0, size_y), fill(0, size_c), fill(0, size_c))
    Numbering!(number, type, nc)
    ndof   = maximum(number.Vx) + maximum(number.Vy) + maximum(number.Pt) + maximum(number.d)
    @printf("nc = %d × %d, Δx = %1.1f m, l = %1.1f m, l/Δx = %1.2f, Kd = %1.2e Pa m², %d dofs\n", nc.x, nc.y, Δ.x*sc.L, l*sc.L, l/Δ.x, Kd_phys, ndof)
    dx, r  = zeros(ndof), zeros(ndof)

    # Fields, pre-stressed initial state: P = Pconf, τxx = -Pconf/2, τyy = Pconf/2
    V    = (x = zeros(size_x...), y = zeros(size_y...))
    Vi   = (x = zeros(size_x...), y = zeros(size_y...))
    P    = (t = fill(pconf / sc.σ, size_c...),)
    Pi   = (t = zeros(size_c...),)
    d    = zeros(size_c...)
    di   = zeros(size_c...)
    τ    = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...))
    ψ    = zeros(size_c...)
    plastic = (λ̇ = (c = zeros(size_c...), v = zeros(size_v...)), τII = (c = zeros(size_c...),), Pc = (c = zeros(size_c...),))
    εp   = zeros(size_c...)
    old  = (τ = (xx = fill(-pconf/2/sc.σ, size_c...), yy = fill(pconf/2/sc.σ, size_c...), xy = zeros(size_v...)),
            P = fill(pconf / sc.σ, size_c...), d = zeros(size_c...), H = zeros(size_c...), Wp = zeros(size_c...), εp = zeros(size_c...))
    R    = (x = zeros(size_x...), y = zeros(size_y...), pt = zeros(size_c...), d = zeros(size_c...))

    # Weak seed: circle of radius 500 m centred on the south-west corner
    inc(x, y) = x^2 + y^2 < (rad / sc.L)^2
    Gc = fill(G_mat, size_c...); Gv = fill(G_mat, size_v...)
    Gc[inx_c, iny_c] .= ifelse.(inc.(X.c.x, X.c.y'), G_inc, G_mat)
    Gv[inx_v, iny_v] .= ifelse.(inc.(X.v.x, X.v.y'), G_inc, G_mat)
    rheo = (G  = (c = Gc, v = Gv),
            η  = (c = fill(Inf, size_c...), v = fill(Inf, size_v...)),
            K  = (c = fill(K, size_c...),),
            gc = (c = fill(gc, size_c...),),
            l  = (c = fill(l, size_c...),),
            ηd = (c = fill(ηd, size_c...),),
            ph = (c = ones(Int, size_c...), v = ones(Int, size_v...)))

    # Pure shear (x shortening), as in the original script: Vx = -ε̇ (x - Lx/2), Vy = ε̇ (y - Ly); zero shear stress
    V.x[inx_Vx, iny_Vx] .= -ε̇bg .* (X.v.x .- L.x/2)
    V.y[inx_Vy, iny_Vy] .=  ε̇bg .* (X.v.y' .- L.y)
    BC = (Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), d = zeros(size_c...))

    J = JacobianCache(V, P, d, old, rheo, materials, number, type, BC, nc, Δ)

    hist = (t = Float64[], τII = Float64[], dmax = Float64[], iters = Int[])
    ϵref = nothing

    # Newton solve of one (sub)step of length Δs.t; returns (converged, iterations)
    function newton!(Δs)
        for iter in 1:niter
            Residuals!(R, V, P, d, old, rheo, materials, type, BC, nc, Δs)
            ϵ = residual_norms(R, number, nc)
            any(isnan, values(ϵ)) && return false, iter
            if iter == 1
                ϵref = isnothing(ϵref) ? ϵ : map(max, ϵref, ϵ)
                ϵref = map(e -> max(e, 1e-12 * maximum(ϵref)), ϵref)
                # Physical floors: pure shear is isochoric, Rd vanishes before yielding
                ϵref = merge(ϵref, (pt = max(ϵref.pt, ε̇bg * Δ.t), d = max(ϵref.d, gc / l * Δ.x * Δ.y)))
            end
            ϵrel = max(ϵ.x/ϵref.x, ϵ.y/ϵref.y, ϵ.pt/ϵref.pt, ϵ.d/ϵref.d)
            noisy && @printf("    iter %02d: |Rx| = %1.2e |Ry| = %1.2e |Rp| = %1.2e |Rd| = %1.2e  rel = %1.2e\n", iter, ϵ.x, ϵ.y, ϵ.pt, ϵ.d, ϵrel)
            ϵrel < ϵ_nl && return true, iter - 1
            SetRHS!(r, R, number, type, nc)
            zero_values!(J)
            Assemble!(J.K, V, P, d, old, rheo, materials, number, type, BC, nc, Δs)
            solve!(dx, J, r; solver=:lu)
            all(isfinite, dx) || return false, iter
            α, _ = BackTrackingLineSearch!(R, dx, V, P, d, Vi, Pi, di, old, rheo, materials, number, type, BC, nc, Δs, ϵref)
            noisy && α < 1 && @printf("    line search: α = %1.3f\n", α)
        end
        return false, niter
    end

    # History update after a converged (sub)step
    function update_history!(Δs)
        UpdateStress!(τ, ψ, V, P, d, old, rheo, materials, type, BC, nc, Δs; plastic)
        old.τ.xx .= τ.xx; old.τ.yy .= τ.yy; old.τ.xy .= τ.xy
        old.P    .= plastic.Pc.c          # pressure corrected for plastic dilatancy
        P.t      .= old.P
        old.d    .= d
        old.Wp  .+= plastic.τII.c .* plastic.λ̇.c .* Δs.t
        εp      .+= sqrt(2/3) .* plastic.λ̇.c .* Δs.t       # εp = √(2/3) ∫ λ̇ dt, as in Duretz et al. (2023)
        old.εp   .= εp                                       # cohesion C = C⁰ + h εp
        set_ghosts!(old.d); set_ghosts!(old.Wp); set_ghosts!(old.εp)
    end

    # Adaptive sub-stepping: a failed Newton solve is restarted from the last converged state with Δt/2
    # (rate-independent plasticity: the onset of yielding of the pre-stressed background is the hard part)
    bak = (Vx = copy(V.x), Vy = copy(V.y), P = copy(P.t), d = copy(d))
    Δt_sub, nsub_tot = Δt, 0
    for it in 1:nt
        t_left, iters = Δt, 0
        while t_left > 1e-12 * Δt
            Δs = merge(Δ, (t = min(Δt_sub, t_left),))
            bak.Vx .= V.x; bak.Vy .= V.y; bak.P .= P.t; bak.d .= d
            ok, iter = newton!(Δs)
            iters += iter
            if ok
                update_history!(Δs)
                t_left  -= Δs.t
                nsub_tot += 1
                Δt_sub   = min(2Δt_sub, Δt)
            else
                V.x .= bak.Vx; V.y .= bak.Vy; P.t .= bak.P; d .= bak.d
                Δt_sub = Δs.t / 2
                Δt_sub < Δt / 2^8 && error("step $it: no convergence down to Δt/256")
                noisy && @printf("  step %d: Newton failed, Δt → Δt/%d\n", it, round(Int, Δt / Δt_sub))
            end
        end

        g    = damage ? (1 .- d[inx_c, iny_c]).^2 .+ materials.k_res : ones(nc.x, nc.y)
        τxyc = 0.25 .* (τ.xy[1:end-1, 1:end-1] .+ τ.xy[2:end, 1:end-1] .+ τ.xy[1:end-1, 2:end] .+ τ.xy[2:end, 2:end])[inx_c, iny_c]
        txx, tyy = τ.xx[inx_c, iny_c], τ.yy[inx_c, iny_c]
        τII  = g .* sqrt.(0.5 .* (txx.^2 .+ tyy.^2 .+ (txx .+ tyy).^2) .+ τxyc.^2)
        push!(hist.t, it * Δt_phy / 3.15576e10); push!(hist.τII, mean(τII) * sc.σ); push!(hist.dmax, maximum(d)); push!(hist.iters, iters)
        @printf("step %03d: t = %1.3f kyr, mean τII = %1.4e MPa, max εp = %1.3e, max d = %1.3f, Newton its = %d (%d substeps so far)\n",
                it, hist.t[end], hist.τII[end]/1e6, maximum(εp), hist.dmax[end], iters, nsub_tot); flush(stdout)

        if mod(it, nout) == 0 || it == nt
            fig = Figure(size = (1100, 800))
            ax  = Axis(fig[1, 1], aspect = DataAspect(), title = "εp (Kd = $(Kd_phys) Pa m², l = $(l_phys) m)", xlabel = "x [km]", ylabel = "y [km]")
            hm  = heatmap!(ax, X.c.x, X.c.y, εp[inx_c, iny_c], colormap = :inferno)
            Colorbar(fig[1, 2], hm)
            ax  = Axis(fig[1, 3], aspect = DataAspect(), title = "phase field d", xlabel = "x [km]", ylabel = "y [km]")
            hm  = heatmap!(ax, X.c.x, X.c.y, d[inx_c, iny_c], colormap = :turbo)
            Colorbar(fig[1, 4], hm)
            ax  = Axis(fig[2, 1], aspect = DataAspect(), title = "τII [MPa]", xlabel = "x [km]", ylabel = "y [km]")
            hm  = heatmap!(ax, X.c.x, X.c.y, τII .* sc.σ ./ 1e6, colormap = :viridis)
            Colorbar(fig[2, 2], hm)
            ax  = Axis(fig[2, 3], xlabel = "t [kyr]", ylabel = "mean τII [MPa]")
            scatterlines!(ax, hist.t, hist.τII ./ 1e6, color = :red, markersize = 6)
            display(fig)
            isnothing(figname) || save(figname, fig)
        end
    end
    return (; X, d = d[inx_c, iny_c], εp = εp[inx_c, iny_c], hist)
end

# Shear band width vs regularisation parameter (282×208, l = 100 m, ηvp = 0): Kd = 3e12, 1e13, 3e13, 6e13 Pa m²
#   r  = main(Kd_phys = 1e13)
#   bw = band_width(r.X.c.x, r.X.c.y, r.εp; yrange = (2.5, 7.5), smax = 2.5)
#   @printf("band width: Gaussian FWHM = %1.0f m, angle = %1.1f°\n", 1e3bw.fwhm, bw.θ)
r = main(Kd_phys = 1e13)
