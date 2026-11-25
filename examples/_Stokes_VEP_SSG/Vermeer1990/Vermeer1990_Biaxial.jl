# Vermeer (1990, Géotechnique 40, 223-236): "The orientation of shear bands in biaxial tests", post-bifurcation analysis
#
# Plane-strain biaxial test on a cohesionless sand: G = 10 MPa, ν = 0, ϕ = 40°, ψ = 10°, c = 0, perfect plasticity
# (no material softening). The sample is loaded to failure at constant cell pressure σh = -100 kPa,
# σv = (1 + sinϕ)/(1 - sinϕ) σh = 4.6 σh. Once a shear band forms, the vertical stress drops although the friction angle is
# constant ("non-associated softening"): the residual ratio for the critical band θ = 45° + α/2 is
# σv/σh = (1 + sinα)/(1 - sinα) = 3.7, tanα = cosψ sinϕ/(1 - sinψ sinϕ) (Eq. 42, Table 2), and for a band at angle θ
# from the minor compressive stress (horizontal) σv/σh = (sin(2θ - α) + sin α)/(sin(2θ - α) - sin α) (Eq. 39).
#
# Model: rectangular sample W × H. Sides: normal stress σh (cell pressure) and τxy = 0 (membrane).
# Top/bottom: vertical shortening imposed (Vy), τxy = 0 (lubricated platens); Vx pinned at one bottom node.
# Initial state: homogeneous preload σxx = σh, σyy = r0 σh (r0 just below the peak ratio), σzz = (σxx + σyy)/2 (τzz = 0),
# then the vertical stress is raised by the imposed shortening up to failure. A small weak seed (lower ϕ) at the centre
# triggers the band (off-centre, to get a single band rather than a conjugate pair).
# Plasticity: plasticity = :mc (default) uses the Mohr-Coulomb model (σ2-independent, as Vermeer; θt = 0: exact, principal
# stresses; θt > 0: Abbo & Sloan corner rounding). plasticity = :dp: Drucker-Prager, which equals Vermeer's criterion only
# while τzz = 0 (ψ = 0, ν → ½); otherwise σzz drifts and the strength depends on σ2. Elasticity: ν = 0 → K = 2G/3.
# Mild viscoplastic regularisation (ηvp).
# Nondimensionalisation: σ = |σh|, L = W, t = 1/ε̇ (ε̇: imposed vertical shortening rate).
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, StaticArrays, LinearAlgebra, Printf, CairoMakie, JLD2
import Statistics: mean
using TimerOutputs

# Vermeer's closed-form ratios
tanα(ϕ, ψ) = cosd(ψ) * sind(ϕ) / (1 - sind(ψ) * sind(ϕ))
ratio_peak(ϕ) = (1 + sind(ϕ)) / (1 - sind(ϕ))
ratio_residual(ϕ, ψ) = (α = atand(tanα(ϕ, ψ)); (1 + sind(α)) / (1 - sind(α)))                          # Eq. 42, critical band
ratio_residual(ϕ, ψ, θ) = (α = atand(tanα(ϕ, ψ)); (sind(2θ - α) + sind(α)) / (sind(2θ - α) - sind(α))) # Eq. 39, band at θ

# Band angle from the horizontal (minor compressive stress), from the structure tensor of the plastic strain rate field:
# the band normal is the dominant direction of ∇λ̇ (doubled-angle average, so that the two faces of a band add up), weighted by
# |∇λ̇|², away from the boundaries and from the seed. Parallel bands count together; bands reflected at the platens (other
# family) are a minor contribution as long as they are short.
function band_angle(xc, yc, λ̇; margin=0.1, x_seed=(Inf, Inf), r_excl=0.0)
    W, H = xc[end] - xc[1], yc[end] - yc[1]
    dx, dy = xc[2] - xc[1], yc[2] - yc[1]
    Jxx = Jyy = Jxy = 0.0
    for j in 2:length(yc)-1, i in 2:length(xc)-1
        (abs(xc[i]) < (0.5 - margin) * W && abs(yc[j]) < (0.5 - margin) * H && hypot(xc[i] - x_seed[1], yc[j] - x_seed[2]) > r_excl) || continue
        gx = (λ̇[i+1, j] - λ̇[i-1, j]) / 2dx
        gy = (λ̇[i, j+1] - λ̇[i, j-1]) / 2dy
        Jxx += gx^2; Jyy += gy^2; Jxy += gx * gy
    end
    Jxx + Jyy == 0 && return NaN
    θn = 0.5 * atand(2Jxy, Jxx - Jyy)  # band normal
    θ  = mod(θn + 90, 180)              # band direction, in [0, 180)
    return θ > 90 ? 180 - θ : θ         # angle with the horizontal, in [0, 90]
end

@views function main(; nc=(x=50, y=100), H=2.0, G=10e6, ν=0.0, ϕ=40.0, ψ=10.0, σh=-100e3, ϕ_seed=36.0, r_seed=0.06, x_seed=(-0.5, -0.4), r0=4.4,
                     Δε=2e-5, εmax=0.01, ηvp=1e-3, nsub_max=4, niter=25, ϵ_nl=1e-9, nout=25, tag="", show=false,
                     plasticity=:mc, θt=0.0)

    # Scales: σ = |σh|, L = W, t = 1/ε̇
    sc = (σ=abs(σh), L=1.0, t=1.0)

    # Materials: sand, seed
    nphases   = 2
    materials = initialize_materials(nphases; compressible=true, plasticity=plasticity === :mc ? MohrCoulomb : DruckerPrager)
    materials.n   .= 1.0
    materials.η0  .= 1e40
    materials.G   .= G / sc.σ
    materials.β   .= 1 / (2G * (1 + ν) / (3 * (1 - 2ν)) / sc.σ)   # K = 2G(1 + ν)/(3(1 - 2ν)) (Vermeer: ν = 0, K = 2G/3)
    pl = materials.plasticity
    pl.C   .= 0.0
    pl.ϕ   .= [ϕ, ϕ_seed]
    pl.ψ   .= ψ
    pl.ηvp .= ηvp
    plasticity === :mc && (pl.θt .= θt)   # 0: exact Mohr-Coulomb (principal stresses), > 0: Abbo & Sloan rounding
    preprocess!(materials)

    # Grid: sample centred on (0, 0)
    W  = 1.0
    x  = (min=-W / 2, max=W / 2)
    y  = (min=-H / 2, max=H / 2)
    Δ  = (x=W / nc.x, y=H / nc.y, t=Δε)            # ε̇ = 1: a step shortens the sample by Δε
    a  = Allocs(nc, :EW_stress, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Boundary conditions: sides σxx = σh, τxy = 0; top/bottom Vy = ∓ε̇H/2, τxy = 0, Vx pinned at the bottom centre
    a.type.Vy[2, iny_Vy]     .= :tangential_stress
    a.type.Vy[end-1, iny_Vy] .= :tangential_stress
    a.type.Vx[inx_Vx, 2]     .= :tangential_stress
    a.type.Vx[inx_Vx, end-1] .= :tangential_stress
    a.type.Vx[inx_Vx[argmin(abs.(a.X.v.x))], 2] = :Dirichlet_tangent
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    BC.Vx[2, iny_Vx]     .= σh / sc.σ
    BC.Vx[end-1, iny_Vx] .= σh / sc.σ
    a.V.x .= 0.0
    a.V.y .= -1.0 .* a.X.vy_e.y'                    # ε̇yy = -1
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Weak seed (default: on the left side, below mid-height, so that a single band crosses the sample)
    for i in axes(a.phases.c, 1), j in axes(a.phases.c, 2)
        a.phases.c[i, j] = hypot(a.X.c_e.x[i] - x_seed[1], a.X.c_e.y[j] - x_seed[2]) < r_seed ? 2 : 1
    end
    for i in axes(a.phases.v, 1), j in axes(a.phases.v, 2)
        a.phases.v[i, j] = hypot(a.X.v_e.x[i] - x_seed[1], a.X.v_e.y[j] - x_seed[2]) < r_seed ? 2 : 1
    end
    Set_PhaseRatios!(a)

    # Preload: σxx = σh, σyy = r0 σh, τzz = 0
    σxx0, σyy0 = σh / sc.σ, r0 * σh / sc.σ
    P0 = -(σxx0 + σyy0) / 2
    a.Pt .= P0; a.Pt0 .= P0; a.Pti .= P0
    a.τ.xx .= σxx0 + P0; a.τ.yy .= σyy0 + P0; a.τ.xy .= 0.0
    a.τ0.xx .= a.τ.xx; a.τ0.yy .= a.τ.yy; a.τ0.xy .= 0.0

    # Solver: GCR + symmetrised-Newton Cholesky preconditioner
    iter_params = IterParams(niter=niter, ϵ_nl=ϵ_nl, ϵ_nl_last=1e-6, solver_type=:GCR, pc_type=:sym_newton, niter_KSP=200)
    rvec = zeros(length(iter_params.α))
    err  = (x=zeros(niter), y=zeros(niter), p=zeros(niter))
    to   = TimerOutput()

    nt = round(Int, εmax / Δε)
    hist = (ε=zeros(nt + 1), σv=zeros(nt + 1), σh=zeros(nt + 1), λ̇max=zeros(nt + 1), θband=fill(NaN, nt + 1))
    σyy = similar(a.Pt); σxx = similar(a.Pt)
    function record!(k, ε)
        σyy .= a.τ.yy .- a.Pt; σxx .= a.τ.xx .- a.Pt
        hist.ε[k]     = ε
        hist.σv[k]    = mean(0.5 .* (σyy[inx_c, iny_c[end]] .+ σyy[inx_c, iny_c[1]])) # top/bottom platens (load cell)
        hist.σh[k]    = mean(0.5 .* (σxx[inx_c[1], iny_c] .+ σxx[inx_c[end], iny_c]))
        hist.λ̇max[k]  = maximum(a.λ̇.c[inx_c, iny_c])
        hist.θband[k] = hist.λ̇max[k] > 0 ? band_angle(a.X.c.x, a.X.c.y, a.λ̇.c[inx_c, iny_c]; x_seed, r_excl=2r_seed) : NaN
    end
    record!(1, 0.0)
    # State at the start of a step, restored when the step fails (then it is done in 2, 4, ... sub-steps)
    state  = (V=a.V, Pt=a.Pt, τ=(a.τ.xx, a.τ.yy, a.τ.xy, a.τ.II), λ̇=(a.λ̇.c, a.λ̇.v))
    backup = deepcopy(state)
    copy_state!(dst, src) = (foreach(k -> dst[k] isa Tuple ? foreach(copyto!, dst[k], src[k]) : (dst[k] isa NamedTuple ? foreach(copyto!, values(dst[k]), values(src[k])) : copyto!(dst[k], src[k])), keys(dst)); nothing)
    nfail = 0
    ttot = @elapsed for it in 1:nt
        copy_state!(backup, state)
        iter = 0
        for isub in 0:nsub_max
            nsub = 2^isub
            try
                for _ in 1:nsub
                    iter, = main_loop(a, it, materials, BC, nc, (x=Δ.x, y=Δ.y, t=Δ.t / nsub), to, nphases, iter_params, rvec, err)
                end
                break
            catch e
                ((e isa ErrorException || e isa TaskFailedException) && isub < nsub_max) || rethrow()
                nfail += 1
                @printf("step %04d failed with %d sub-step(s), retry with %d\n", it, nsub, 2nsub)
                copy_state!(state, backup)
            end
        end
        record!(it + 1, it * Δε)
        if it % nout == 0 || it == nt
            @printf("step %04d, εv = %.3f %%, σv/σh = %.4f, σh = %.4f, max λ̇ = %.2e, band θ = %.1f°, %d its\n",
                    it, 100it * Δε, hist.σv[it+1] / hist.σh[it+1], hist.σh[it+1], hist.λ̇max[it+1], hist.θband[it+1], iter)
        end
    end
    @printf("total %.1f s, %d failed (sub-stepped) attempts\n", ttot, nfail)

    # Reference values
    rp, rr = ratio_peak(ϕ), ratio_residual(ϕ, ψ)
    θC, θR, θA = 45 + ϕ / 2, 45 + ψ / 2, 45 + (ϕ + ψ) / 4
    θend = hist.θband[end]
    @printf("peak: model %.3f, Vermeer %.3f | end: model %.3f, Vermeer critical band %.3f, Eq. 39 at θ = %.1f°: %.3f\n",
            maximum(hist.σv ./ hist.σh), rp, hist.σv[end] / hist.σh[end], rr, θend, ratio_residual(ϕ, ψ, θend))
    @printf("band angle %.1f° (Roscoe %.1f°, Arthur %.1f°, Coulomb %.1f°)\n", θend, θR, θA, θC)

    # Figure
    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size=(1100, 520), fontsize=16)
        ax = Axis(fig[1, 1], xlabel="axial shortening εv [%]", ylabel="σv/σh", title="Vermeer (1990) biaxial test, ϕ = $(ϕ)°, ψ = $(ψ)°")
        hlines!(ax, [rp], color=:gray, linestyle=:dash, label="peak (1+sinϕ)/(1−sinϕ)")
        hlines!(ax, [rr], color=:black, linestyle=:dash, label="residual, critical band (Eq. 42)")
        isnan(θend) || hlines!(ax, [ratio_residual(ϕ, ψ, θend)], color=:red, linestyle=:dot, label=@sprintf("residual, Eq. 39 at θ = %.1f°", θend))
        lines!(ax, 100hist.ε, hist.σv ./ hist.σh, color=:steelblue, linewidth=3, label="model")
        ylims!(ax, 3.0, 5.0)
        axislegend(ax, position=:rb, framevisible=false, labelsize=13)
        ax = Axis(fig[1, 2], aspect=DataAspect(), title=@sprintf("plastic strain rate, θ = %.1f°", θend), xlabel="x/W", ylabel="y/W")
        hm = heatmap!(ax, a.X.c.x, a.X.c.y, a.λ̇.c[inx_c, iny_c], colormap=:inferno)
        Colorbar(fig[1, 3], hm, label="λ̇/ε̇")
        colsize!(fig.layout, 2, Auto(0.45))
        fig
    end
    save(joinpath(@__DIR__, "Vermeer1990_Biaxial$(tag).png"), fig)
    show && display(fig)
    jldsave(joinpath(@__DIR__, "Vermeer1990_Biaxial$(tag).jld2"); hist, λ̇=a.λ̇.c[inx_c, iny_c], xc=collect(a.X.c.x), yc=collect(a.X.c.y),
            params=(; nc, H, G, ν, ϕ, ψ, σh, ϕ_seed, r_seed, x_seed, r0, Δε, εmax, ηvp, plasticity, θt))
    return (; a, hist, materials)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
