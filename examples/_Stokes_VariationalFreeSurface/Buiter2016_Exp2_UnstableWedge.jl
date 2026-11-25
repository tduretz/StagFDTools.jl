# Variational free surface + elasto-visco-plasticity with strain softening: Buiter et al. (2016, JSG), experiment 2
#
# Unstable (subcritical) wedge: three horizontal layers of quartz - corundum - quartz sand (1 cm each, 35 cm long) are
# shortened by a mobile wall moving at 2.5 cm/h, over 10 cm. A critically tapered thrust wedge builds by forward and
# backward thrusting (Buiter et al., 2016, Figs. 8-11; MILAMIN_VEP, I3VIS, SOPALE, ...).
#
# Materials (Table 1): quartz ρ = 1560, corundum ρ = 1890 kg/m³, ϕ = 36° → 31° (peak → dynamic stable), C = 30 Pa,
# basal friction quartz on foil ϕb = 16° → 14°, Cb = 30 Pa (1 mm basal layer), ψ = 0.
# Rheology: compressible elasticity (E = 5e6 Pa, ν = 0.25, as Elfen), Drucker-Prager with tension cap, moderate
# viscoplastic regularisation. Softening of the cohesion only, C + ΔC tanh(εp/εh) (default 30 → 15 Pa, εh = 0.5), friction
# kept at its peak value (the paper softens the friction 36° → 31°, 16° → 14°; with the implicit solve the friction
# softening gave a negative tangent stiffness at ~20 % of the points and the linear solves failed).
# εp is carried by the markers (second marker field).
# Frame of the mobile wall (fixed grid): wall at x = 35 cm (Vx = 0), base and left boundary moving at +v (the sand flows
# in from the left: the far end behaves as an infinitely long layer instead of the front wall of the sandbox).
# Nondimensionalisation: L = 1 cm, σ = ρgL, t = L/v.
using StagFDTools, StagFDTools.Stokes, StagFDTools.StokesVariational, StagFDTools.Rheology
using StaticArrays, LinearAlgebra, Printf, CairoMakie, JLD2
using JustPIC
using CellArraysIndexing: @index
const backend = JustPIC.CPU

const sc = let L = 1e-2, ρ = 1560.0, g = 9.81, v = 2.5e-2 / 3600
    σ = ρ * g * L
    (L=L, ρ=ρ, g=g, σ=σ, v=v, t=L / v)
end

const quartz, corundum, basal, air = 1, 2, 3, 4

function set_initial_phases!(phases, particles, h_layer)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                y = @index particles.coords[2][ip, i, j]
                ph = y < h_layer ? basal : (1.0 ≤ y < 2.0 ? corundum : quartz)
                @index phases[ip, i, j] = Float64(ph)
            end
        end
    end
end

function main(; nc=(x=350, y=100), Lx=35.0, ytop=8.0, Δt_s=36.0, shortening_cm=10.0, h_layer=0.1,
                G=5e6 / 2.5, K=5e6 / 1.5, ηvp=2e5, ΔC=-15.0, εh=0.5, θ_fs=1.0, snapshots=0:2:10,
                niter=30, ϵ_nl=1e-8, ϵ_rel=1e-2, inexact=true, pc_type=:sym_newton, ηvp_start=1e6, nstart=6, nxcell=12, nout=10, tag="")

    # Materials: quartz, corundum, basal layer (quartz on foil), air (removed from the phase ratios)
    nphases = 4
    materials = initialize_materials(nphases; plasticity=DruckerPragerCap, compressible=true)
    materials.g  .= [0.0, -1.0]
    materials.ρ  .= [1560.0, 1890.0, 1560.0, 0.0] ./ sc.ρ
    materials.n  .= 1.0
    materials.η0 .= 1e12 / (sc.σ * sc.t)
    materials.G  .= G / sc.σ
    materials.β  .= sc.σ / K
    pl = materials.plasticity
    pl.C   .= 30.0 / sc.σ
    pl.ϕ   .= [36.0, 36.0, 16.0, 36.0]
    pl.Δϕ  .= 0.0                                            # no friction softening (implicit solve: negative tangent stiffness)
    pl.ΔC  .= [ΔC, ΔC, ΔC, 0.0] ./ sc.σ                      # cohesion softening C + ΔC tanh(εp/εh)
    pl.εh  .= εh
    pl.ψ   .= 0.0
    pl.Pt  .= -0.5 .* pl.C ./ tand.(pl.ϕ)                     # tension cap (> -C/tanϕ, also after softening)
    pl.ηvp .= ηvp / (sc.σ * sc.t)
    preprocess!(materials)

    # Grid and boundary conditions (frame of the mobile wall)
    x  = (min=0.0, max=Lx)
    y  = (min=0.0, max=ytop)
    Δ  = (x=Lx / nc.x, y=ytop / nc.y, t=Δt_s / sc.t)
    a  = Allocs(nc, :free_slip, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    a.type.Vx[inx_Vx, 2] .= :Dirichlet_tangent
    BC.Vx[inx_Vx, 2]     .= min.(1.0, (x.max .- a.X.v.x) ./ Δ.x)   # base moves at +v, ramped to 0 over the last cell
    a.V.x .= 1.0                                                    # left boundary: inflow at +v
    a.V.x[end-1, :] .= 0.0                                          # mobile wall: Vx = 0

    # Free surface: flat, 3 cm
    xv = collect(a.X.v.x)
    chain = init_markerchain(backend, 20, 10, 40, xv, fill(3.0, length(xv)))

    # Markers: phase and plastic strain
    adv = Markers(backend, a, nxcell, 2nxcell, nxcell ÷ 3, nc, nphases; args=2)
    set_initial_phases!(adv.particle_args[1], adv.particles, h_layer)
    adv.particle_args[2].data .= 0.0
    set_grid_phases_free_surface!(a, adv, chain; air_phase=air, fallback_phase=quartz)

    fs = VariationalFreeSurface(a, nc; θ_fs)
    update_free_surface!(fs, a, chain, nc, Δ)

    # Time loop
    nt   = round(Int, shortening_cm / Δ.t)
    snap = Dict{Float64,Any}()
    take_snapshot!(s) = (snap[s] = (ph=[argmax(a.phase_ratios.c[i+1, j+1]) for i in 1:nc.x, j in 1:nc.y], εp=copy(a.εp.c[inx_c, iny_c]),
                                    ε̇II=copy(a.ε̇.II[inx_c, iny_c]) ./ sc.t, φ=copy(fs.φ.c[inx_c, iny_c]), h=copy(chain.h_vertices)))
    take_snapshot!(0.0)
    tsolve, fallbacks, nstall = 0.0, 0, 0
    for it in 1:nt
        update_old_stress!(a)
        if it == 1   # continuation within the first step (sudden gravity loading of an unstressed layer)
            for k in 0:(nstart-1)
                pl.ηvp .= max(ηvp, ηvp_start * (ηvp / ηvp_start)^(k / nstart)) / (sc.σ * sc.t)
                solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter, ϵ_nl, ϵ_rel, inexact, noisy=false, pc_type)
            end
            pl.ηvp .= ηvp / (sc.σ * sc.t)
        end
        tsolve += @elapsed iter, err = solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter, ϵ_nl, ϵ_rel, inexact, noisy=false, pc_type)
        ϵ = max(err.x[iter], err.y[iter], err.p[iter])
        stalled = ϵ > max(ϵ_nl, ϵ_rel * max(err.x[1], err.y[1], err.p[1]))
        nstall += stalled
        fallbacks += fs.gcr.sym_fallbacks; fs.gcr.sym_fallbacks = 0
        s = it * Δ.t
        if it % nout == 0 || it == 1 || it == nt
            @printf("step %04d, shortening = %5.2f cm, %2d its%s, max topo = %.3f cm, max εp = %.2f, (%.2f s per solve, %d stalled steps, %d sym→Picard PC)\n",
                    it, s, iter, stalled ? " (stalled)" : "", maximum(chain.h_vertices), maximum(a.εp.c[inx_c, iny_c]), tsolve / it, nstall, fallbacks)
        end
        advect_free_surface!(a, adv, chain, BC, Δ; air_phase=air, refill_phase=quartz, fallback_phase=quartz, conserve_mean=false)
        update_free_surface!(fs, a, chain, nc, Δ)
        any(t -> abs(t - s) < Δ.t / 2, snapshots) && take_snapshot!(round(s, digits=2))
    end
    jldsave(joinpath(@__DIR__, "Buiter2016_Exp2$(tag).jld2"); snap, xc=collect(a.X.c.x), yc=collect(a.X.c.y), xv)

    # Figure: material field (masked by the fluid fraction) and plastic strain at the snapshots
    ks  = sort(collect(keys(snap)))
    fig = Figure(size=(1400, 230 * length(ks)), fontsize=13)
    for (r, k) in enumerate(ks)
        S = snap[k]
        ph = Float64.(S.ph); ph[S.φ .== 0] .= NaN
        εp = copy(S.εp); εp[S.φ .== 0] .= NaN
        ax = Axis(fig[r, 1], aspect=DataAspect(), title="$(k) cm: material", ylabel="y [cm]")
        heatmap!(ax, a.X.c.x, a.X.c.y, ph, colormap=cgrad([:wheat, :gray30, :steelblue], 3, categorical=true), colorrange=(1, 3))
        lines!(ax, xv, S.h, color=:black); ylims!(ax, 0, ytop)
        ax = Axis(fig[r, 2], aspect=DataAspect(), title="$(k) cm: plastic strain εp")
        hm = heatmap!(ax, a.X.c.x, a.X.c.y, εp, colormap=:inferno, colorrange=(0, 2))
        lines!(ax, xv, S.h, color=:white); ylims!(ax, 0, ytop)
        r == 1 && Colorbar(fig[r, 3], hm)
    end
    save(joinpath(@__DIR__, "Buiter2016_Exp2$(tag).png"), fig)
    return (a=a, fs=fs, chain=chain, snap=snap, materials=materials, BC=BC, nc=nc, Δ=Δ, nphases=nphases)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
