# Pure shear shortening of an elasto-plastic (Drucker-Prager, no softening) layer with a weak inclusion,
# setup of Duretz et al. (2018, M2Di EP test 01), with the phase field driven by plastic work (Hageman, 2026):
#       gc/l d - ∇⋅(gc l ∇d) - 2(1-d) (H + βp Wp) + ηd ∂d/∂t = 0,   Wp = ∫ τII λ̇ dt
# Yield is checked on the degraded stress. Elastic driving is disabled (ψc = Inf), so damage only develops
# where plastic work accumulates (shear bands). βp = 0 reproduces the elasto-plastic reference (M2Di).
using StagFDTools, StagFDTools.PhaseFieldMechanics, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, MAT
import Statistics: mean

@views function main(; nc=(x=150, y=75), nt=35, βp=0.0, gc_l=3e5, l_Δx=2.0, l_phys=nothing, ηd_phys=0.0, ψc=Inf, ηvp_phys=0.0,
                       damage=true, Kd_phys=0.0,
                       niter=25, ϵ_nl=1e-10, nout=5, noisy=false, figname=nothing)

    # Scales (Duretz et al., 2018): stresses 3e10 Pa, lengths 1 km, time 1e10 s
    sc = (σ = 3e10, L = 1e3, t = 1e10)

    # Loading and time step
    ε̇bg = 5e-15 * sc.t
    Δt  = 1e10 / sc.t

    # Materials: matrix / inclusion (shear modulus), K = 1/β
    G_mat, G_inc = 1e10 / sc.σ, 0.25e10 / sc.σ
    K   = 1 / (5e-11 * sc.σ)
    pl  = DruckerPrager(C = [3e7 / sc.σ], ϕ = [30.0], ψ = [10.0], ηvp = [ηvp_phys / sc.σ / sc.t], cosϕ = [0.0], sinϕ = [0.0], sinψ = [0.0], cosψ = [0.0])
    preprocess!(pl)

    # Geometry and grid
    L   = (x = 4e3 / sc.L, y = 2e3 / sc.L)
    Δ   = (x = L.x / nc.x, y = L.y / nc.y, t = Δt)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    X   = GenerateGrid((min = -L.x/2, max = L.x/2), (min = -L.y, max = 0.0), Δ, nc)

    # Phase field: length scale l = l_Δx Δx (or l_phys [m]), gc set by gc/l (energy per volume, Pa); driven by
    # plastic work only. damage = false: d does not degrade the material, it only regularises plasticity through
    # Kd ∇²d in the yield function (Kd_phys [Pa m²])
    l   = isnothing(l_phys) ? l_Δx * Δ.x : l_phys / sc.L
    gc  = gc_l / sc.σ * l
    ηd  = ηd_phys / sc.σ / sc.t
    Kd  = Kd_phys / sc.σ / sc.L^2
    materials = (k_res = 1e-6, amor = false, plasticity = pl, βp = βp, ψc = ψc, damage = damage, Kd = Kd)

    # Node types and numbering
    type   = set_BC_template(:free_slip, nc)
    number = Fields(fill(0, size_x), fill(0, size_y), fill(0, size_c), fill(0, size_c))
    Numbering!(number, type, nc)
    ndof   = maximum(number.Vx) + maximum(number.Vy) + maximum(number.Pt) + maximum(number.d)
    @printf("nc = %d × %d, l = %1.1f m, l/Δx = %1.2f, %d dofs\n", nc.x, nc.y, l*sc.L, l/Δ.x, ndof)
    dx, r  = zeros(ndof), zeros(ndof)

    # Fields
    V    = (x = zeros(size_x...), y = zeros(size_y...))
    Vi   = (x = zeros(size_x...), y = zeros(size_y...))
    P    = (t = zeros(size_c...),)
    Pi   = (t = zeros(size_c...),)
    d    = zeros(size_c...)
    di   = zeros(size_c...)
    τ    = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...))
    ψ    = zeros(size_c...)
    plastic = (λ̇ = (c = zeros(size_c...), v = zeros(size_v...)), τII = (c = zeros(size_c...),), Pc = (c = zeros(size_c...),))
    εp   = zeros(size_c...)
    old  = (τ = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...)), P = zeros(size_c...), d = zeros(size_c...),
            H = zeros(size_c...), Wp = zeros(size_c...))
    R    = (x = zeros(size_x...), y = zeros(size_y...), pt = zeros(size_c...), d = zeros(size_c...))

    # Weak inclusion (circle of radius 100 m at (0, -1 km)), centroids and vertices
    inc(x, y) = x^2 + (y + 1e3/sc.L)^2 < (100/sc.L)^2
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

    # Pure shear velocity (x shortening): Dirichlet normal values live in V, zero shear stress on all sides
    V.x[inx_Vx, iny_Vx] .= -ε̇bg .* X.v.x
    V.y[inx_Vy, iny_Vy] .=  ε̇bg .* X.v.y'
    BC = (Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), d = zeros(size_c...))

    J = JacobianCache(V, P, d, old, rheo, materials, number, type, BC, nc, Δ)
    @printf("Jacobian: %d nonzeros\n", nnz(J.A))

    # Reference (M2Di): mean τII vs strain
    data = matread(joinpath(pkgdir(StagFDTools), "examples", "_Stokes_VEP_SSG", "ShearBanding_Duretz2018", "DataM2Di_EP_test01.mat"))
    hist = (str = Float64[], τII = Float64[], dmax = Float64[], iters = Int[])

    ϵref = nothing
    for it in 1:nt
        converged, iter = false, 0
        for outer iter in 1:niter
            Residuals!(R, V, P, d, old, rheo, materials, type, BC, nc, Δ)
            ϵ = residual_norms(R, number, nc)
            if iter == 1
                ϵref = isnothing(ϵref) ? ϵ : map(max, ϵref, ϵ)
                ϵref = map(e -> max(e, 1e-12 * maximum(ϵref)), ϵref)
                # Physical floors: pure shear is isochoric (Rp starts at round-off), Rd vanishes before yielding
                ϵref = merge(ϵref, (pt = max(ϵref.pt, ε̇bg * Δ.t), d = max(ϵref.d, gc / l * Δ.x * Δ.y)))
            end
            ϵrel = max(ϵ.x/ϵref.x, ϵ.y/ϵref.y, ϵ.pt/ϵref.pt, ϵ.d/ϵref.d)
            noisy && @printf("    iter %02d: |Rx| = %1.2e |Ry| = %1.2e |Rp| = %1.2e |Rd| = %1.2e  rel = %1.2e\n", iter, ϵ.x, ϵ.y, ϵ.pt, ϵ.d, ϵrel)
            if ϵrel < ϵ_nl
                converged = true
                break
            end
            SetRHS!(r, R, number, type, nc)
            zero_values!(J)
            Assemble!(J.K, V, P, d, old, rheo, materials, number, type, BC, nc, Δ)
            solve!(dx, J, r; solver=:lu)
            α, _ = BackTrackingLineSearch!(R, dx, V, P, d, Vi, Pi, di, old, rheo, materials, number, type, BC, nc, Δ, ϵref)
            noisy && α < 1 && @printf("    line search: α = %1.3f\n", α)
        end
        converged || @warn "step $it: Newton did not converge in $niter iterations"

        # Update history
        UpdateStress!(τ, ψ, V, P, d, old, rheo, materials, type, BC, nc, Δ; plastic)
        old.τ.xx .= τ.xx; old.τ.yy .= τ.yy; old.τ.xy .= τ.xy
        old.P    .= plastic.Pc.c          # pressure corrected for plastic dilatancy
        P.t      .= old.P                 # initial guess of the next step
        old.d    .= d
        old.H    .= max.(old.H, max.(ψ .- ψc, 0.0))
        old.Wp  .+= plastic.τII.c .* plastic.λ̇.c .* Δ.t
        set_ghosts!(old.d); set_ghosts!(old.H); set_ghosts!(old.Wp)
        εp      .+= plastic.λ̇.c .* Δ.t

        # Mean stress invariant (degraded stresses), as in the reference
        g    = damage ? (1 .- d[inx_c, iny_c]).^2 .+ materials.k_res : ones(nc.x, nc.y)
        τxyc = 0.25 .* (τ.xy[1:end-1, 1:end-1] .+ τ.xy[2:end, 1:end-1] .+ τ.xy[1:end-1, 2:end] .+ τ.xy[2:end, 2:end])[inx_c, iny_c]
        txx, tyy = τ.xx[inx_c, iny_c], τ.yy[inx_c, iny_c]
        τII  = g .* sqrt.(0.5 .* (txx.^2 .+ tyy.^2 .+ (txx .+ tyy).^2) .+ τxyc.^2)
        push!(hist.str, ε̇bg * it * Δ.t); push!(hist.τII, mean(τII) * sc.σ); push!(hist.dmax, maximum(d)); push!(hist.iters, iter - 1)
        @printf("step %03d: strain = %1.3e, mean τII = %1.4e MPa, mean P = %1.3e MPa (corrected %1.3e), max λ̇ = %1.3e, max d = %1.3f, Newton its = %d\n",
                it, hist.str[end], hist.τII[end]/1e6, mean(P.t[inx_c, iny_c])*sc.σ/1e6, mean(plastic.Pc.c[inx_c, iny_c])*sc.σ/1e6,
                maximum(plastic.λ̇.c), hist.dmax[end], iter - 1); flush(stdout)

        if mod(it, nout) == 0 || it == nt
            fig = Figure(size = (900, 700))
            ax  = Axis(fig[1, 1], aspect = DataAspect(), title = "plastic strain εp", xlabel = "x [km]", ylabel = "y [km]")
            hm  = heatmap!(ax, X.c.x, X.c.y, εp[inx_c, iny_c], colormap = :inferno)
            Colorbar(fig[1, 2], hm)
            ax  = Axis(fig[2, 1], aspect = DataAspect(), title = "phase field d (βp = $(βp), damage = $(damage), Kd = $(Kd_phys) Pa m²)", xlabel = "x [km]", ylabel = "y [km]")
            hm  = heatmap!(ax, X.c.x, X.c.y, d[inx_c, iny_c], colormap = :turbo)
            Colorbar(fig[2, 2], hm)
            ax  = Axis(fig[3, 1], xlabel = "strain", ylabel = "mean τII [MPa]")
            n   = min(length(data["strvec"]), length(hist.str))
            lines!(ax, vec(data["strvec"])[1:n], vec(data["Tiivec"])[1:n] ./ 1e6, label = "M2Di (Duretz et al., 2018)")
            scatter!(ax, hist.str, hist.τII ./ 1e6, color = :red, markersize = 6, label = "PhaseFieldMechanics")
            axislegend(ax, position = :rb)
            display(fig)
            isnothing(figname) || save(figname, fig)
        end
    end
    n   = min(length(data["strvec"]), length(hist.str))
    err = maximum(abs.(hist.τII[1:n] .- vec(data["Tiivec"])[1:n]) ./ vec(data["Tiivec"])[1:n])
    @printf("max relative difference of mean τII to M2Di over %d steps: %1.2e\n", n, err)
    return (; X, d = d[inx_c, iny_c], εp = εp[inx_c, iny_c], hist)
end

# Elasto-plastic reference (βp = 0, matches M2Di of Duretz et al., 2018):  r = main(βp = 0.0)
# Damage driven by plastic work in the shear bands, viscoplastic regularisation:   r = main(βp = 1.0, nt = 60, ηvp_phys = 1e19)
# Gradient regularisation through the phase field (no damage, no softening, ηvp = 0): mesh-converged shear bands
r = main(nc = (x=150, y=75), nt = 50, βp = 1.0, gc_l = 3e6, l_phys = 100.0, damage = false, Kd_phys = 3e10)
