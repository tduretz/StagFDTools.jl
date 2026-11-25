# Variational free surface + elasto-visco-plasticity: Buiter et al. (2016, JSG), experiment 1 (stable wedge)
#
# A quartz sand wedge (base 8.24 cm, height 3 cm at the mobile wall, surface slope α = 20°, horizontal base) is
# pushed by the mobile wall at 2.5 cm/h over a frictional base, for 4 cm. The wedge lies in the stable domain of
# critical taper theory: it should translate without internal deformation. Analytical values: surface slope 20°,
# root-mean-square velocity 6.944e-6 m/s, wall force Fa = ½ρgA tan(ϕb) + Cb L = 7.9 N/m (A = 8.24 × 3 cm²).
#
# Sand: ρ = 1560 kg/m³, ϕ = 36°, C = 30 Pa, ψ = 0, background viscosity 1e12 Pa s (Table 1, peak strengths, no
# softening as in the analytical force), compressible elasticity E = 5e6 Pa, ν = 0.25 (as Elfen), Drucker-Prager with
# tension cap, viscoplastic regularisation ηvp
# (strong in the nearly rigid sand, weak in the basal layer where it adds ηvp λ̇ to the basal shear stress).
# Boundary friction: a 1 mm basal layer with ϕb = 16°, Cb = 30 Pa (the "layer" approach of most codes).
# The grid is fixed, so the problem is solved in the frame of the mobile wall: wall at x = 13.24 cm (Vx = 0), base
# and left (inflow) boundary moving at +v. Lab-frame quantities: v_lab = v - (v, 0).
# Nondimensionalisation: L = 1 cm, σ = ρgL, t = L/v.
using StagFDTools, StagFDTools.Stokes, StagFDTools.StokesVariational, StagFDTools.Rheology
using StaticArrays, LinearAlgebra, Printf, CairoMakie
using JustPIC
using CellArraysIndexing: @index
const backend = JustPIC.CPU

const sc = let L = 1e-2, ρ = 1560.0, g = 9.81, v = 2.5e-2 / 3600
    σ = ρ * g * L
    (L=L, ρ=ρ, g=g, σ=σ, v=v, t=L / v)
end

const sand, basal, air = 1, 2, 3
const α_wedge = atand(3.0 / 8.24)   # 20.0°

function set_initial_phases!(phases, particles, h_layer)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                y = @index particles.coords[2][ip, i, j]
                @index phases[ip, i, j] = Float64(y < h_layer ? basal : sand)
            end
        end
    end
end

# Wall force ∫ (P - τxx) dy on the wall (last centroid column, fluid-weighted), lab-frame vrms over the sand (rock fraction
# of the control volumes, basal layer included), surface slope (least-squares line through the chain between toe and wall)
function diagnostics(a, fs, chain, nc, Δ, xv, h_layer)
    i = nc.x + 1
    F = sum(fs.φ.c[i, j] * (a.Pt[i, j] - a.τ.xx[i, j]) * Δ.y for j in 2:(nc.y+1))
    A = sum(fs.φ.c[2:end-1, 2:end-1]) * Δ.x * Δ.y
    vx2 = sum(fs.φ.Vx[i, j] * (a.V.x[i, j] - 1.0)^2 for i in 2:(nc.x+2), j in 3:(nc.y+2)) * Δ.x * Δ.y
    vy2 = sum(fs.φ.Vy[i, j] * a.V.y[i, j]^2 for i in 3:(nc.x+2), j in 2:(nc.y+2)) * Δ.x * Δ.y
    vrms = sqrt((vx2 + vy2) / A)
    h = chain.h_vertices
    sel = findall(k -> h[k] > h_layer + 0.3 && xv[k] < xv[end] - 0.5, eachindex(xv))
    X, Y = xv[sel], h[sel]
    slope = atand(sum((X .- mean(X)) .* (Y .- mean(Y))) / sum((X .- mean(X)) .^ 2))
    return (F=F, vrms=vrms, slope=slope)
end
mean(x) = sum(x) / length(x)

function main(; nc=(x=265, y=70), Δt_s=36.0, shortening_cm=4.0, h_layer=0.1, G=5e6 / 2.5, K=5e6 / 1.5, ηvp=(sand=5e5, basal=1e3), θ_fs=1.0,
                nout=50, niter=20, ϵ_nl=1e-8, nxcell=12, noisy=false, pc_type=:sym_newton, Pic2Newt=Inf, ϵ_rel=1e-2, ηvp_start=5e5, nstart=8)

    # Materials: sand, basal friction layer, air (removed from the phase ratios)
    nphases = 3
    # Compressible Drucker-Prager with tension cap: the pressure is a state variable that the return mapping can update,
    # so tensile (Newton/line-search) iterates near the wall corner are mapped back onto the cap. (Incompressible: P is a
    # Lagrange multiplier, fixed in the return mapping; the cap then admits no state below its tensile end and plain
    # Drucker-Prager returns a negative strength below the apex.) Elasticity as Elfen: E = 5e6 Pa, ν = 0.25.
    materials = initialize_materials(nphases; plasticity=DruckerPragerCap, compressible=true)
    materials.g  .= [0.0, -1.0]
    materials.ρ  .= [1560.0, 1560.0, 0.0] ./ sc.ρ
    materials.n  .= 1.0
    materials.η0 .= [1e12, 1e12, 1e12] ./ (sc.σ * sc.t)
    materials.G  .= [G, G, G] ./ sc.σ
    materials.β  .= [1/K, 1/K, 1/K] .* sc.σ
    pl = materials.plasticity
    pl.C   .= [30.0, 30.0, 30.0] ./ sc.σ
    pl.ϕ   .= [36.0, 16.0, 36.0]
    pl.ψ   .= 0.0
    pl.Pt  .= -0.5 .* pl.C ./ tand.(pl.ϕ)          # tension cap (must be > -C/tanϕ)
    pl.Δϕ  .= 0.0; pl.ΔC .= 0.0; pl.εh .= 1.0      # no softening
    ηvp_ph = [ηvp.sand, ηvp.basal, ηvp.sand] ./ (sc.σ * sc.t)   # sand nearly rigid (λ̇ ≈ 0): strong regularisation costs little
    pl.ηvp .= ηvp_ph
    preprocess!(materials)

    # Grid: 13.24 cm × 3.5 cm (3 cm of sand at the wall + air)
    x  = (min=0.0, max=13.24)
    y  = (min=0.0, max=3.5)
    Δ  = (x=(x.max - x.min) / nc.x, y=(y.max - y.min) / nc.y, t=Δt_s / sc.t)
    a  = Allocs(nc, :free_slip, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    # Wall frame: base moves at +v (no slip), left boundary inflow at +v, mobile wall (right) fixed, free slip along walls
    a.type.Vx[inx_Vx, 2] .= :Dirichlet_tangent
    BC.Vx[inx_Vx, 2]     .= min.(1.0, (x.max .- a.X.v.x) ./ Δ.x)   # base velocity, ramped to 0 over the last cell at the wall (as MILAMIN_VEP)
    a.V.x .= 1.0
    a.V.x[end-1, :]      .= 0.0                    # mobile wall: Vx = 0 (Dirichlet_normal)

    # Free surface: wedge from the toe (x = 5 cm) to the wall, basal layer thickness in front
    xv = collect(a.X.v.x)
    h0 = @. max(h_layer, (xv - 5.0) * 3.0 / 8.24)
    chain = init_markerchain(backend, 20, 10, 40, xv, h0)

    # Material markers
    adv = Markers(backend, a, nxcell, 2nxcell, nxcell ÷ 3, nc, nphases)
    set_initial_phases!(adv.particle_args[1], adv.particles, h_layer)
    set_grid_phases_free_surface!(a, adv, chain; air_phase=air, fallback_phase=sand)

    fs = VariationalFreeSurface(a, nc; θ_fs)
    update_free_surface!(fs, a, chain, nc, Δ)

    # Time loop
    nt = round(Int, shortening_cm / Δ.t)
    hist = (s=zeros(nt), F=zeros(nt), vrms=zeros(nt), slope=zeros(nt), iter=zeros(Int, nt))
    tsolve, fallbacks = 0.0, 0
    for it in 1:nt
        update_old_stress!(a)
        if it == 1
            # Continuation within the first step (sudden gravity + boundary loading of an unstressed wedge, far from
            # equilibrium): solve with a viscoplastic regularisation decreased geometrically from ηvp_start, no advection
            for k in 0:(nstart-1)
                pl.ηvp .= max.(ηvp_ph, ηvp_start / (sc.σ * sc.t) .* (ηvp_ph .* sc.σ .* sc.t ./ ηvp_start) .^ (k / nstart))
                solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter, ϵ_nl, ϵ_rel, noisy, pc_type, Pic2Newt)
            end
            pl.ηvp .= ηvp_ph
        end
        tsolve += @elapsed iter, err = solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter, ϵ_nl, ϵ_rel, noisy, pc_type, Pic2Newt)
        ϵ = max(err.x[iter], err.y[iter], err.p[iter])
        fallbacks += fs.gcr.sym_fallbacks; fs.gcr.sym_fallbacks = 0   # sym_newton PC not positive definite → Picard factor
        ϵ > max(ϵ_nl, ϵ_rel * max(err.x[1], err.y[1], err.p[1])) && @warn @sprintf("step %d: not converged after %d iterations (res. %.2e)", it, iter, ϵ)
        d = diagnostics(a, fs, chain, nc, Δ, xv, h_layer)
        hist.s[it], hist.F[it], hist.vrms[it], hist.slope[it], hist.iter[it] = it * Δ.t, d.F, d.vrms, d.slope, iter
        if it % nout == 0 || it == 1 || it == nt
            @printf("step %04d, shortening = %.2f cm, %2d its, F = %6.3f N/m, vrms = %.4e m/s, slope = %5.2f°, max λ̇ = %.2e (%.2f s per solve)\n",
                    it, hist.s[it], iter, d.F * sc.σ * sc.L, d.vrms * sc.v, d.slope, maximum(a.λ̇.c) / sc.t, tsolve / it)
            fallbacks > 0 && @printf("          (sym_newton PC fell back to Picard in %d solves so far)\n", fallbacks)
        end
        advect_free_surface!(a, adv, chain, BC, Δ; air_phase=air, refill_phase=basal, fallback_phase=sand, conserve_mean=false)
        update_free_surface!(fs, a, chain, nc, Δ)
    end

    # Figures
    fig = Figure(size=(1300, 950), fontsize=14)
    s = hist.s
    ax = Axis(fig[1, 1], xlabel="shortening [cm]", ylabel="surface slope [°]"); lines!(ax, s, hist.slope); hlines!(ax, [α_wedge], color=:black, linestyle=:dash)
    ax = Axis(fig[1, 2], xlabel="shortening [cm]", ylabel="vrms [m/s]");        lines!(ax, s, hist.vrms .* sc.v); hlines!(ax, [sc.v], color=:black, linestyle=:dash)
    ax = Axis(fig[1, 3], xlabel="shortening [cm]", ylabel="wall force [N/m]");  lines!(ax, s, hist.F .* sc.σ .* sc.L); hlines!(ax, [7.9], color=:black, linestyle=:dash)
    mask(A) = (B = copy(A); B[fs.φ.c[2:end-1, 2:end-1] .== 0] .= NaN; B)
    xc, yc = a.X.c.x, a.X.c.y
    ph = [argmax(a.phase_ratios.c[i+1, j+1]) for i in 1:nc.x, j in 1:nc.y]
    fields = (("phases", mask(Float64.(ph)), :Set2_3), ("pressure [Pa]", mask(a.Pt[2:end-1, 2:end-1] .* sc.σ), :viridis),
              ("log₁₀ ε̇II [1/s]", mask(log10.(a.ε̇.II[2:end-1, 2:end-1] ./ sc.t .+ 1e-20)), :inferno))
    for (k, (title, A, cmap)) in enumerate(fields)
        ax = Axis(fig[k+1, 1:3], aspect=DataAspect(), title="$(title) after $(round(s[end], digits=2)) cm of shortening", xlabel="x [cm]", ylabel="y [cm]")
        hm = heatmap!(ax, xc, yc, A, colormap=cmap)
        lines!(ax, xv, chain.h_vertices, color=:black)
        Colorbar(fig[k+1, 4], hm)
    end
    save(joinpath(@__DIR__, "Buiter2016_Exp1.png"), fig)
    display(fig)

    return (hist=hist, a=a, fs=fs, chain=chain)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
