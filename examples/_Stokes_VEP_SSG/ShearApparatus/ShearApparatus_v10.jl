using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics: mean
using DifferentiationInterface
using TimerOutputs, CairoMakie, JLD2, Random
using JustPIC
using CellArraysIndexing: @index
const backend = JustPIC.CPU
BLAS.set_num_threads(1) # the sparse Cholesky (CHOLMOD) is faster on 1 BLAS thread, and multithreaded BLAS collapses when the machine is loaded

# comment rand() in initial pressure
# make large VP
# made :free_slip

# x_tip: gouge with |x| > x_tip (ends of the gouge, against the salt) is phase 5: static gouge, no rate-and-state
# ingouge(𝐱): gouge membership (rectangle, possibly with rough walls); isalt(𝐱): gouge marker belongs to the second gouge phase 6 (heterogeneity)
function set_material_geometry!(phases, particles, ingouge, salt, plate, x_tip=Inf, isalt=𝐱->false)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                𝐱 = @SVector([x, y])
                if any(g -> inside(𝐱, g), salt)
                    @index phases[ip, i, j] = 3.
                elseif any(g -> inside(𝐱, g), plate)
                    @index phases[ip, i, j] = 4.
                elseif ingouge(𝐱)
                    @index phases[ip, i, j] = abs(x) > x_tip ? 5. : isalt(𝐱) ? 6. : 2.
                else
                    @index phases[ip, i, j] = 1.
                end
            end
        end
    end
end

# Stationary Gaussian random field with Gaussian covariance of length ℓ (random Fourier features, unit variance)
function random_field_2D(ℓ, seed; N=400)
    rng = Random.Xoshiro(seed)
    k   = [randn(rng, 2) ./ ℓ for _ in 1:N]
    φ   = 2π .* rand(rng, N)
    return 𝐱 -> sqrt(2/N) * sum(cos(k[i][1]*𝐱[1] + k[i][2]*𝐱[2] + φ[i]) for i in 1:N)
end
# Band-limited random profile along the gouge (wavelengths in [λ/2, 2λ]), RMS amplitude 1
function random_profile_1D(λ, seed; N=40)
    rng = Random.Xoshiro(seed)
    k   = 2π ./ (λ .* 2 .^ (2 .* rand(rng, N) .- 1))
    φ   = 2π .* rand(rng, N)
    return η -> sqrt(2/N) * sum(cos(k[i]*η + φ[i]) for i in 1:N)
end

function visualise_markers(particles, phases, sc, x, y, time)
    with_theme(theme_latexfonts()) do
    ppx, ppy = particles.coords
    idxv = particles.index.data[:]
    px   = Array(ppx.data[:][idxv]) .* sc.L
    py   = Array(ppy.data[:][idxv]) .* sc.L
    ph   = Int.(Array(phases.data[:][idxv]))
    phase_colors = [:blue, :red, :lightblue, :gray80, :orange, :darkred] # rock, gouge, salt, plate, gouge ends (static), gouge 2 (heterogeneity)
    Lx, Ly = x.max - x.min, y.max - y.min
    fig = Figure(size=(500, 500*Ly/Lx))
    ax  = Axis(fig[1, 1], aspect=DataAspect(), title=@sprintf("t = %.1f min", time*sc.t/60), xlabel="x [m]", ylabel="y [m]")
    scatter!(ax, px, py, color=phase_colors[ph], markersize=3)
    limits!(ax, x.min*sc.L, x.max*sc.L, y.min*sc.L, y.max*sc.L)
    Legend(fig[1, 2], [MarkerElement(color=c, marker=:circle, markersize=12) for c in phase_colors], ["rock", "gouge", "salt", "plate", "gouge (static)", "gouge 2"], "Phase")
    resize_to_layout!(fig)
    return fig
    end
end

function softening(Ep, Cmax, Cmin, E0, w)
    return Cmin + (Cmax-Cmin)*(1 - tanh((Ep-E0)/w))
end

"""
    plot_shear_apparatus(d)

Figure of one time step from the data `d` built in `main` (also stored in the `step_XXXX.jld2` files):
`plot_shear_apparatus(load_step("ShearApparatus_Tet30_H02/step_0100.jld2"))`.
"""
function plot_shear_apparatus(d)
    it, sc, pl = d.it, d.sc, d.plasticity
    return with_theme(theme_latexfonts()) do
        ftsz = 25
        fig = Figure(size=(1000, 1000)) # (1000, 1150) with the Δt panel
        empty!(fig)
        # Probe colors: shared by the markers on the heatmap and the local metrics they measure
        c_salt, c_plate, c_MC1, c_MC2, c_app, c_sub = :gold, :magenta, :cyan, :limegreen, :darkorange, :gray
        has_plates = get(d, :plates, true) # steps saved by v8 have plates

        # Split heatmap of the apparatus
        # ax  = Axis(fig[1:3,1], aspect=DataAspect(), title="Apparent / True friction", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
        ax = Axis(fig[1:3, 1], aspect=DataAspect(), title="Plastic strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        eps = 1e-1
        # field = (τ.xy)[inx_c,iny_c]  .* sc.σ / 1e6
        # field = app_fric[inx_c,iny_c]
        # field = (τ.yy .- Pt)[inx_c,iny_c]  .* sc.σ / 1e6
        field = log10.((d.λ̇c .+ eps)/sc.t)
        crange = (minimum(field)-eps, maximum(field)+eps)

        # field = C.c*sc.σ./1e6
        # crange = (5, 7)

        # field = phases.c[inx_c,iny_c]
        hm = heatmap!(ax, d.xc .* sc.L, d.yc .* sc.L, field, colormap=:bluesreds, colorrange=crange)
        # hm1 = heatmap!(ax, d.xc[1:ind_mid_x].*sc.L, d.yc.*sc.L, field1[1:ind_mid_x,:], colormap=:bluesreds, colorrange=(minimum(field1)-eps, maximum(field1)+eps))
        # hm2 = heatmap!(ax, d.xc[ind_mid_x:end].*sc.L, d.yc.*sc.L, field2[ind_mid_x:end,:], colormap=:bluesreds, colorrange=(minimum(field2)-eps, maximum(field2)+eps))
        contour!(ax, d.xc .* sc.L, d.yc .* sc.L, d.phc, color=:white)
        # Colorbar(fig[4, 1], hm1, label = L"$\phi^\text{app}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
        # Colorbar(fig[4, 2], hm2, label = L"$\phi^\text{true}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
        step = 10
        # arrows2d!(ax, d.xc[1:step:end].*sc.L, d.yc[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=50000.4, color=:white)
        arrows2d!(ax, d.xc[1:step:end] .* sc.L, d.yc[1:step:end] .* sc.L, cos.(d.θc)[1:step:end, 1:step:end], sin.(d.θc)[1:step:end, 1:step:end], lengthscale=0.001, color=:white, tiplength=0)
        xlims!(ax, d.xv_min .* sc.L, d.xv_max .* sc.L)
        lines!(ax, d.xc[d.ind_mid_x] .* sc.L * ones(size(d.yc)), d.yc .* sc.L, color=:white, linewidth=4)
        # Probes
        scatter!(ax, d.probe_salt.x[it] .* sc.L, d.probe_salt.y[it] .* sc.L, marker=:star5, color=c_salt, markersize=20)
        has_plates && scatter!(ax, d.probe_plate.x[it] .* sc.L, d.probe_plate.y[it] .* sc.L, marker=:star5, color=c_plate, markersize=20)
        # Subdomain
        # outline of the cells i1:i2, j1:j2 (indices include ghost cells, d.xc/d.yc do not)
        Δxc, Δyc = d.xc[2] - d.xc[1], d.yc[2] - d.yc[1]
        bx = (d.xc[d.i1-1] - Δxc/2, d.xc[d.i2-1] + Δxc/2) .* sc.L
        by = (d.yc[d.j1-1] - Δyc/2, d.yc[d.j2-1] + Δyc/2) .* sc.L
        lines!(ax, [bx[1], bx[2], bx[2], bx[1], bx[1]], [by[1], by[1], by[2], by[2], by[1]], color=c_sub, linewidth=1)
        # Mohr-circles
        scatter!(ax, d.probe_MC1.x[it] .* sc.L, d.probe_MC1.y[it] .* sc.L, marker=:xcross, color=c_MC1, markersize=20)
        scatter!(ax, d.probe_MC2.x[it] .* sc.L, d.probe_MC2.y[it] .* sc.L, marker=:xcross, color=c_MC2, markersize=20)
        # Gouge coordinate axes, taken from the rows of Rot (as used in σ′ = Rot * σ * Rot')
        x0, y0, ℓ = 0.0, d.Ly/2 * sc.L, 0.008
        for (k, c, lab) in ((1, :black, "x′"), (2, :white, "y′"))
            e = (d.Rot[k, 1], d.Rot[k, 2])
            lines!(ax, [x0 - ℓ*e[1], x0 + ℓ*e[1]], [y0 - ℓ*e[2], y0 + ℓ*e[2]], color=c, linewidth=3)
            text!(ax, x0 + 1.15ℓ*e[1], y0 + 1.15ℓ*e[2], text=lab, color=c, fontsize=20, align=(:center, :center))
        end

        # # Zoom on the gouge
        # ax  = Axis(fig[2,2], aspect=DataAspect(), title="Plastic Strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
        # eps   = 1e-1
        # # field = (a.τ.xy)[inx_c,iny_c]  .* sc.σ / 1e6
        # # field = app_fric[inx_c,iny_c]
        # # field = (a.τ.yy .- a.Pt)[inx_c,iny_c]  .* sc.σ / 1e6
        # field = log10.((a.λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
        # # field = a.phases.c[inx_c,iny_c]
        # hm = heatmap!(ax, d.xc.*sc.L, d.yc.*sc.L, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
        # contour!(ax, d.xc.*sc.L, d.yc.*sc.L,  a.phases.c[inx_c,iny_c], color=:white)
        # # arrows2d!(ax, d.xc[1:step:end].*sc.L, d.yc[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=50000.4, color=:white)
        # arrows2d!(ax, d.xc[1:step:end].*sc.L, d.yc[1:step:end].*sc.L, cos.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], sin.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], lengthscale=0.001, color=:white, tiplength = 0)
        # xlims!(ax, -0.02, 0.02)
        # ylims!(ax, 0.04, 0.06)

        ax = Axis(fig[2, 2], aspect=DataAspect(), title="Mohr circles", xlabel="σn", ylabel="τ", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        θ = LinRange(0, π, 100)

        # In-plane Mohr circle (compression positive): centre -(σxx+σyy)/2, radius √(((σxx-σyy)/2)² + σxy²)
        # (τII and P include τzz, so they only match this in the incompressible limit)
        Mohr(p) = (c = -(p.σxx[it] + p.σyy[it])/2; R = sqrt(((p.σxx[it] - p.σyy[it])/2)^2 + p.σxy[it]^2); (x=R .* cos.(θ) .+ c, y=R .* sin.(θ)))
        MC1 = Mohr(d.probe_MC1)
        MC2 = Mohr(d.probe_MC2)

        τII_MC3 = -(d.probe_plate.σyy[it] - d.probe_salt.σxx[it])/2
        P_MC3   = -(d.probe_plate.σyy[it] + d.probe_salt.σxx[it])/2
        MC3     = (x=(τII_MC3 .* cos.(θ) .+ P_MC3), y=(τII_MC3 .* sin.(θ)))
        MC3_μ   = (x=(τII_MC3 .* (cos.(2*(π/2-d.θgouge))) .+ P_MC3), y=(τII_MC3 .* sin.(2*(π/2-d.θgouge))))

        lines!(ax, MC1.x*sc.σ ./ 1e6, MC1.y*sc.σ ./ 1e6, color=c_MC1, label="probe 1 (rock)")
        lines!(ax, MC2.x*sc.σ ./ 1e6, MC2.y*sc.σ ./ 1e6, color=c_MC2, label="probe 2 (gouge)")
        lines!(ax, MC3.x*sc.σ ./ 1e6, MC3.y*sc.σ ./ 1e6, color=c_app, label="apparatus")
        σmin = minimum([minimum(MC1.x), minimum(MC2.x), minimum(MC3.x)])
        σmax = maximum([maximum(MC1.x), maximum(MC2.x), maximum(MC3.x)])
        scatter!(ax, MC3_μ.x*sc.σ ./ 1e6, MC3_μ.y*sc.σ ./ 1e6, color=c_app, label="apparatus, gouge plane")
        σn_ax = LinRange(σmin-0.1*(σmin), σmax+0.1*σmin, 100)
        # τ1    = pl.cosϕ[1]/(1+pl.sinϕ[1]) * (pl.C[1] * pl.cosϕ[1] .+ σn_ax.*pl.sinϕ[1])
        # τ2    = pl.cosϕ[2]/(1+pl.sinϕ[2]) * (pl.C[2] * pl.cosϕ[2] .+ σn_ax.*pl.sinϕ[2])
        τ1 = pl.C[1] .+ σn_ax .* pl.sinϕ[1] ./ pl.cosϕ[1]
        τ2 = pl.C[2] .+ σn_ax .* pl.sinϕ[2] ./ pl.cosϕ[2]
        lines!(ax, σn_ax*sc.σ/1e6, τ1*sc.σ/1e6, color=:black, label="yield rock")
        lines!(ax, σn_ax*sc.σ/1e6, τ2*sc.σ/1e6, color=:black, linestyle=:dash, label="yield gouge")
        Legend(fig[2, 3], ax, framevisible=false, labelsize=12, halign=:left)

        # ax  = Axis(fig[1,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        # scatter!(ax, 1:niter, log10.(err.x[1:niter]./err.x[1]) )
        # scatter!(ax, 1:niter, log10.(err.y[1:niter]./err.y[1]) )
        # scatter!(ax, 1:niter, log10.(err.p[1:niter]./err.p[1]) )
        # ylims!(ax, -15, 1)
        ax = Axis(fig[3, 2], title=L"$$Stress space", xlabel=L"$P$", ylabel=L"$\tau_{II}$", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        # Yield surface: Drucker-Prager, or with DruckerPragerCap: Drucker-Prager for P ≥ pd, circle (centre py, radius Ry) down to Pt
        cap = get(pl, :cap, nothing)
        incap(P, τ, ph) = cap !== nothing && τ * (cap.py[ph] - cap.pd[ph]) < cap.τd[ph] * (cap.py[ph] - P)
        τ_yield(P, ph) = cap === nothing || P >= cap.pd[ph] ? pl.C[ph] * pl.cosϕ[ph] + P * pl.sinϕ[ph] :
                         P >= cap.Pt[ph] ? sqrt(max(cap.Ry[ph]^2 - (P - cap.py[ph])^2, 0.0)) : NaN # not drawn below Pt
        # Stress state without the viscoplastic overstress (λ̇ηvp): along τ for Drucker-Prager, radial (by λ̇ηvp/a) on the cap
        function on_yield(P, τ, λ̇, ph)
            incap(P, τ, ph) || return P, τ - λ̇ * pl.ηvp[ph]
            Rf = sqrt(τ^2 + (P - cap.py[ph])^2)
            s  = (Rf - λ̇ * pl.ηvp[ph] / cap.a[ph]) / Rf
            return cap.py[ph] + s * (P - cap.py[ph]), s * τ
        end
        Pmin = cap === nothing ? minimum(d.P_rock) : min(minimum(d.P_rock), cap.Pt[1], cap.Pt[2])
        P_ax = LinRange(Pmin, maximum(d.P_rock), 200)
        lines!(ax, P_ax*sc.σ/1e6, τ_yield.(P_ax, 2)*sc.σ/1e6, color=:black, linestyle=:dash, label="yield gouge")
        lines!(ax, P_ax*sc.σ/1e6, τ_yield.(P_ax, 1)*sc.σ/1e6, color=:black, label="yield rock")
        for (P_ph, τ_ph, λ̇_ph, ph, col, lab) in ((d.P_rock, d.τII_rock, d.λ̇_rock, 1, :blue, "rock"), (d.P_gouge, d.τII_gouge, d.λ̇_gouge, 2, :red, "gouge"))
            pts = on_yield.(P_ph, τ_ph, λ̇_ph, ph)
            scatter!(ax, first.(pts)*sc.σ/1e6, last.(pts)*sc.σ/1e6, color=col, label=lab)
        end
        Legend(fig[3, 3], ax, framevisible=false, labelsize=12, halign=:left)

        # τ_ax_gouge = materials.C[2]*materials.cosϕ[2] .+ P_ax.*materials.sinϕ[2]
        # lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:red)
        # scatter!(ax, d.P_gouge*sc.σ/1e6, d.τII_gouge*sc.σ/1e6, color=:red )

        ax = Axis(fig[0, 1], xlabel="Axial shortening", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        # scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.PW[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20
        # scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.PE[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20
        # scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.σxxW[1:it]*sc.σ./1e6, marker=:star5 ) #, markersize=20
        # scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.σxxE[1:it]*sc.σ./1e6, marker=:star5 ) #, markersize=20
        # scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.σyyN[1:it]*sc.σ./1e6, marker=:circle )
        # scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.σyyS[1:it]*sc.σ./1e6, marker=:circle )
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, -d.probe_salt.σxx[1:it]*sc.σ ./ 1e6, marker=:star5, color=c_salt, label=L"$-\sigma_{xx}$ salt")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, -d.probe_plate.σyy[1:it]*sc.σ ./ 1e6, marker=:star5, color=c_plate, label=has_plates ? L"$-\sigma_{yy}$ plate" : L"$-\sigma_{yy}$ load cell (boundary mean)")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, -d.probes.σxx_salt_grid[1:it]*sc.σ ./ 1e6, marker=:circle, markersize=5, color=:black, label=L"$-\sigma_{xx}$ salt (fixed cell)")
        has_plates && scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, -d.probes.σyy_plate_grid[1:it]*sc.σ ./ 1e6, marker=:circle, markersize=5, color=:gray50, label=L"$-\sigma_{yy}$ plate (fixed cell)")
        axislegend(ax, position=:lt, labelsize=12, framevisible=false)

        # ax  = Axis(fig[0,1], xlabel="Axial shortening", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        # scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.τII[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20

        # ax  = Axis(fig[0,2], xlabel="time [hrs]", ylabel="τII [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        # scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.τII[1:it]*sc.σ./1e6 )
        # scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.τIIW[1:it]*sc.σ./1e6, marker=:star5, markersize=20 )
        # scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.τIIE[1:it]*sc.σ./1e6, marker=:star5, markersize=20 )
        # scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.τIIS[1:it]*sc.σ./1e6, marker=:diamond, markersize=20 )
        # scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.τIIN[1:it]*sc.σ./1e6, marker=:diamond, markersize=20 )

        ax = Axis(fig[0, 2], xlabel="Axial shortening", ylabel="-τ/σ", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        lines!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, ones(it)*0.8, linestyle=:dash, color=:gray, label=L"$0.8$")
        lines!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, ones(it)*tand(pl.ϕ[2]), linestyle=:dash, color=:gray, label=L"$\tan\phi$")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.app_fric[1:it], color=:black, label="apparent (gouge mean)")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.app_fric_lab[1:it], marker=:rect, color=c_app, label=@sprintf("apparent (apparatus) = %.4f", d.probes.app_fric_lab[it]))
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.app_fric_lab_grid[1:it], marker=:circle, markersize=5, color=:gray50, label="apparent (apparatus, fixed cells)")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.true_fric[1:it], marker=:utriangle, color=:black, label="true (centre)") #, markersize=20  
        tanα = cosd(pl.ψ[2])*sind(pl.ϕ[2]) / (1 - sind(pl.ψ[2])*sind(pl.ϕ[2]))
        lines!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, ones(it)*tanα, linestyle=:dot, color=:gray, label=L"$\tan\alpha$")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probe_salt.σxy[1:it] ./ d.probe_salt.σyy[1:it], marker=:star5, color=c_salt, label="salt")
        has_plates && scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probe_plate.σxy[1:it] ./ d.probe_plate.σyy[1:it], marker=:star5, color=c_plate, label="plate")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probe_MC1.σxy[1:it] ./ d.probe_MC1.σyy[1:it], marker=:xcross, color=c_MC1, label="probe 1 (rock)")
        scatter!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probe_MC2.σxy[1:it] ./ d.probe_MC2.σyy[1:it], marker=:xcross, color=c_MC2, label="probe 2 (gouge)")
        ax.title = @sprintf("apparatus: -τ/σ = %.4f", d.probes.app_fric_lab[it])
        Legend(fig[0, 3], ax, framevisible=false, labelsize=12, rowgap=0, halign=:left)

        ax = Axis(fig[1, 2], xlabel="time [hrs]", ylabel="σ1 angle gouge", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
        scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.minθ[1:it]*180/π, marker=:dtriangle, color=c_sub, label="min")
        scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.meanθ[1:it]*180/π, marker=:circle, color=c_sub, label="mean")
        scatter!(ax, d.probes.t[1:it]*sc.t/3600, d.probes.maxθ[1:it]*180/π, marker=:utriangle, color=c_sub, label="max")
        Legend(fig[1, 3], ax, framevisible=false, labelsize=12, halign=:left)

        # Time step history (steps saved by v8 have no Δt)
        # if hasproperty(d.probes, :Δt)
        #     ax = Axis(fig[4, 1:2], xlabel="Axial shortening", ylabel="Δt [s]", yscale=log10, height=120, xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz,
        #               title=@sprintf("Δt = %.3g s", d.probes.Δt[it]*sc.t))
        #     scatterlines!(ax, d.probes.t[1:it]*d.ε̇yy*d.Ly*sc.L*1e3, d.probes.Δt[1:it]*sc.t, color=:black, markersize=4)
        # end

        colsize!(fig.layout, 3, Auto(0.3))
        return fig
    end
end

# Saved step as a NamedTuple, ready for plot_shear_apparatus
load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)

# Copy a saved state (nested NamedTuples/Tuples of arrays) into the preallocated fields
function restore_state!(dst, src)
    for k in keys(src)
        if src[k] isa Union{NamedTuple, Tuple}
            restore_state!(dst[k], src[k])
        else
            n = min(length(dst[k]), length(src[k])) # probes: nt may differ between runs
            copyto!(dst[k], 1, src[k], 1, n)
        end
    end
end

# niter: max. number of nonlinear (Picard + Newton) iterations per step
# ϵ_nl_last: last resort tolerance: if not converged (abs. res. > ϵ_nl) after niter iterations, the step is still accepted (with a warning) if abs. res. < ϵ_nl_last, otherwise the run stops
# Pic2Newt: Picard iterations until the relative residual drops below Pic2Newt, then Newton (Inf: Newton only)
# niter_KSP: max. number of KSP (GCR) iterations per linear solve
# line_search: :backtracking (first α reducing the residual) or :grid (best of 6 α in [0.05, 1])
# restart: step to resume from (0: start from scratch); a checkpoint is written every ncheck steps (0: never), the 2 most recent are kept
# nsave: figure and output file (step_XXXX.jld2) every nsave steps
# plates: true: steel plates at top/bottom, Vx prescribed there (sample bonded to the pistons, v8);
#         false: the sample sits on the top/bottom boundaries with τxy = 0 (perfectly lubricated pistons, it can slide laterally),
#         the axial stress is the mean σyy over the sample along the top/bottom boundaries (load cell)
# Adaptive time stepping: if a step fails, it is restarted from the start of the step with Δt/Δt_red_fac, at most ntry times (then the run stops);
#         after a successful step, Δt is multiplied by Δt_inc_fac, up to Δt_max (nothing: Δt0). Δt is also limited by the CFL condition
# nt, time_max: the run stops after nt steps or at time_max [s], whichever comes first (time_max = Inf: nt steps)
# Gouge plasticity: friction angle ϕ_gouge + Δϕ_gouge tanh(εp/εh_gouge) [°] and cohesion C_gouge + ΔC_gouge tanh(εp/εh_gouge) [Pa], εp = ∫λ̇dt.
#         Defaults: cohesion hardening fitted to the lab apparent friction (30-50°, low-res runs). No hardening (previous defaults):
#         ϕ_gouge=36.13, C_gouge=0.67e6, ΔC_gouge=0.0. Dilation angle of the gouge: ψ_gouge [°] (default 15).
# Rate-and-state friction of the gouge (rate_state = true): the Drucker-Prager slope sinϕ_gouge becomes
#         μ(V, Ω) = aRS asinh(V/(2V0RS) exp((sinϕ_gouge + bRS Ω)/aRS)), V = Δx λ̇ (slip rate across one cell), aging law for Ω = ln(V0RS θ/LRS).
#         The rate-and-state branch is regularised by a radiation damping ηRS = Δx G_rock/(2cs) (bounds the slip rate, no inertia in the model);
#         gouge in tension (P ≤ 0, e.g. at the contact with the salt) uses the static cap with ηvp.
#         The last tip_RS [m] of the gouge at each sample edge (against the salt) is phase 5: static gouge, or velocity-strengthening
#         rate-and-state (aRS_tip > bRS_tip) if tip_VS = true, which stops ruptures smoothly instead of an abrupt arrest.
#         G_salt: finite shear modulus of the salt (Maxwell time η/G) so that it responds elastically during fast slip events.
#         save_fields: steps (e.g. 450:700) at which light 2D snapshots fields_XXXXX.jld2 (λ̇, P, τII, Ω, phases) are written.
#         Time step: Δt ≤ fRS_w LRS/Vmax (weakening) and the change of Ω over a step is limited to ΔΩmax (healing / nucleation), Δt ≥ Δt_min
@views function main(nc, nt, Δt0, θgouge, hgouge; outdir=nothing, nsave=10, niter=10, restart=0, ncheck=100, Pic2Newt=Inf, line_search=:backtracking, niter_KSP=60, plates=false, ϵ_nl_last=1e-5,
                     Δt_red_fac=2.0, Δt_inc_fac=1.25, ntry=4, Δt_max=nothing, time_max=Inf, ϕ_gouge=34.0, C_gouge=1.25e6, Δϕ_gouge=0.0, ΔC_gouge=2.8e6, εh_gouge=1.0, ψ_gouge=15.0,
                     rate_state=true, tip_RS=2e-3, tip_VS=false, aRS_tip=0.020, G_salt=1e60, bRS_tip=0.008, aRS=0.008, bRS=0.014, LRS=1e-6, V0RS=1e-7, fRS_w=1.0, ΔΩmax=0.2, Δt_min=1e-6, nsave_probes=nothing, save_fields=nothing,
                     het_frac=0.0, het_ℓ=3e-3, het_Δϕ=-4.0, het_ΔC=0.0, het_seed=1, rough_A=0.0, rough_λ=5e-3, rough_matched=false, rough_seed=2)
    #--------------------------------------------#

    # Scaling
    sc = (σ=1e9, L=1e-3, t=1e6)

    # Parameters
    w_sample  = 44e-3/sc.L                   # rock sample width
    t_salt    = 6e-3/sc.L                    # salt thickness on each side (was 3 mm)
    width     = w_sample + 2*t_salt
    height    = 200e-3/sc.L
    t_plate   = 7e-3/sc.L                    # plate thickness (top/bottom)
    y_plate   = 1e-3/sc.L                    # plate centre, from the top/bottom boundary
    t_salt_y  = 0.0                          # horizontal salt layers between the plates and the sample (plates = true only, 0: none)
    hgouge    = hgouge/sc.L
    Δt0       = Δt0/sc.t
    Δt_max    = (Δt_max === nothing ? Δt0*sc.t : Δt_max)/sc.t
    L         = (x=width, y=height)
    θgouge    = (θgouge) / 180*π
    ε̇yy       = 1.31e-6*sc.t
    Pbg       = 1e7/sc.σ

    𝐱_probe_salt = (width/2 - t_salt/2, height/2) # apparatus response (plate and salt), centre of the salt
    𝐱_probe_plate = (0.000/sc.L, 0.002/sc.L) # apparatus response (plate and salt)
    # Mohr-circle probes, placed along the gouge normal n = (cosθ, sinθ) (= y′, row 2 of Rot)
    n_gouge     = (cos(θgouge), sin(θgouge))
    δ_contact   = 4*max(width/nc.x, height/nc.y) # 4 cells outside the gouge contact: the bilinear stencil (±1 cell) stays out of mixed rock/gouge cells
    𝐱_probe_MC1 = (0.0 + (hgouge/2 + δ_contact)*n_gouge[1], height/2 + (hgouge/2 + δ_contact)*n_gouge[2]) # sandstone, next to the contact
    𝐱_probe_MC2 = (0.0, height/2) # gouge centre

    # Boundary loading type
    config = :EW_stress
    # config = :free_slip
    D_BC = @SMatrix([ε̇yy 0.;
                     0 -ε̇yy])
    σ_BC = @SMatrix([-Pbg 0.;
                     0 -Pbg])

      # Material parameters
    nphases                   = 6 # rock, gouge, salt, plate, gouge ends (static: no rate-and-state), gouge 2 (heterogeneity: ϕ + het_Δϕ, C + het_ΔC)
    plasticity                = DruckerPragerCap # DruckerPrager: no tension cap (then comment the Pt line below)
    materials                 = initialize_materials(nphases; compressible=true, plasticity)  #Dzz=-0.0*ε̇yy
    materials.n              .= [1.0,   1.0,    1.0,    1.0, 1.0, 1.0]                                                        # Power law exponent
    materials.η0             .= [1e48,  1e28,   1e10,   1e48, 1e28, 1e28]  ./ sc.σ ./ sc.t                                    # Reference viscosity 
    materials.G              .= [3e10,  1e9,    G_salt, 1e12, 1e9, 1e9]  ./ sc.σ                                             # Shear modulus (G_salt = 1e60: purely viscous salt)
    materials.β              .= [1e-11, 1e-10,  1e-12,  1e-12, 1e-10, 1e-10] .* sc.σ                                        # Compressibility
    materials.plasticity.C   .= [45e6,  C_gouge, 15e60,  15e60, C_gouge, C_gouge] ./ sc.σ                                        # Cohesion
    materials.plasticity.ϕ   .= [36.13, ϕ_gouge, 36.13,  36.13, ϕ_gouge, ϕ_gouge]                                                       # Friction angle
    materials.plasticity.ψ   .= [0.0,   ψ_gouge, 0.0,    0.0, ψ_gouge, ψ_gouge]                                                      # Dilation angle
    materials.plasticity.Pt  .= [-10e6, -0.5e6, -1e60,  -1e60, -0.5e6, -0.5e6] ./ sc.σ                                         # Tensile strength (DruckerPragerCap), must be > -C/tanϕ
    materials.plasticity.ηvp .= [1e10,  1e8,    5e11,   5e11, 1e8, 1e8] .* 10/sc.σ ./ sc.t                                   # Viscoplastic regularisation
    materials.plasticity.Δϕ  .= [0.0,   Δϕ_gouge, 0.0,  0.0, Δϕ_gouge, Δϕ_gouge]                                                       # Strain hardening: ϕ + Δϕ tanh(εp/εh) [°] (0: none)
    materials.plasticity.ΔC  .= [0.0,   ΔC_gouge, 0.0,  0.0, ΔC_gouge, ΔC_gouge] ./ sc.σ                                              # Strain hardening: C + ΔC tanh(εp/εh) [Pa] (0: none)
    materials.plasticity.εh  .= [1.0,   εh_gouge, 1.0,  1.0, εh_gouge, εh_gouge]                                                       # Plastic strain scale of the hardening
    ΔRS = min(width/nc.x, height/nc.y)                                                                              # width of the slipping band: one cell
    if rate_state
        G_rock, ρ_rock = 3e10, 2600.0
        η_rad = ΔRS*sc.L * G_rock / (2*sqrt(G_rock/ρ_rock))                                                         # radiation damping [Pa s]
        tip = tip_VS ? 1.0 : 0.0                                                                                    # tips: velocity strengthening (a > b) or static
        materials.plasticity.aRS  .= [0.0, aRS,  0.0, 0.0, tip*aRS_tip, aRS]
        materials.plasticity.bRS  .= [0.0, bRS,  0.0, 0.0, tip*bRS_tip, bRS]
        materials.plasticity.LRS  .= [1.0, LRS/sc.L,  1.0, 1.0, tip_VS ? LRS/sc.L : 1.0, LRS/sc.L]
        materials.plasticity.V0RS .= [1.0, V0RS/(sc.L/sc.t), 1.0, 1.0, tip_VS ? V0RS/(sc.L/sc.t) : 1.0, V0RS/(sc.L/sc.t)]
        materials.plasticity.DRS  .= [1.0, ΔRS,  1.0, 1.0, tip_VS ? ΔRS : 1.0, ΔRS]
        materials.plasticity.ηRS  .= [0.0, η_rad / sc.σ / sc.t, 0.0, 0.0, tip*η_rad / sc.σ / sc.t, η_rad / sc.σ / sc.t]                                       # rate-and-state branch only (ηvp kept for the tension cap)
        tip_VS && @printf("Velocity-strengthening tips (phase 5): a = %.4f, b = %.4f\n", aRS_tip, bRS_tip)
        @printf("Rate-and-state gouge: a = %.4f, b = %.4f, L = %.2e m, V0 = %.2e m/s, D = %.2e m, ηvp (radiation damping) = %.2e Pa s\n", aRS, bRS, LRS, V0RS, ΔRS*sc.L, η_rad)
    end
    materials.plasticity.ϕ[6] += het_Δϕ                                                                                # gouge 2 (heterogeneity)
    materials.plasticity.C[6] += het_ΔC / sc.σ
    het_frac > 0 && @printf("Heterogeneous gouge: fraction %.2f of gouge 2 (Δϕ = %.1f°, ΔC = %.2f MPa), correlation length %.2f mm\n", het_frac, het_Δϕ, het_ΔC/1e6, het_ℓ*1e3)
      # materials.plasticity.soft .= [0.0, 0.0, 0.0, 0.0]             # softening ?
    #                            rock   gouge   salt    plates
    preprocess!(materials)

    @show π*180/π/4 + materials.plasticity.ϕ[2]/2
    @show π*180/π/4 + (materials.plasticity.ϕ[2] + materials.plasticity.ψ[2])/4

    # Solver parameters
    # Pic2Newt = 1e10   # more than 1.0 - always Newton
    iter_params = IterParams(niter=niter, ϵ_nl=1e-8, ϵ_nl_last=ϵ_nl_last, solver_type=:GCR, Pic2Newt=Pic2Newt, line_search=line_search, niter_KSP=niter_KSP,
                             Δt_red_fac=Δt_red_fac, Δt_inc_fac=Δt_inc_fac, ntry=ntry,
                             nt_max=nt, time_max=time_max/sc.t)

    # Discretisation
    Δ = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    x = (min=(-L.x/2), max=L.x/2)
    y = (min=0.0, max=L.y)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Allocations
    a = Allocs(nc, config, x, y, Δ, nphases)
    if !plates
        # Top/bottom: τxy = 0 instead of the prescribed Vx. With normal stress on the sides, nothing would fix the x translation:
        # Vx is pinned at a single bottom node, at x = 0 (the net horizontal force is zero, so it carries ~no load)
        a.type.Vx[inx_Vx, 2]     .= :tangential_stress
        a.type.Vx[inx_Vx, end-1] .= :tangential_stress
        a.type.Vx[inx_Vx[argmin(abs.(a.X.v.x))], 2] = :Dirichlet_tangent
    end
    # additional Allocations
    εp = a.εp # accumulated plastic strain (∫λ̇dt), used by the solver for strain hardening
    Ω  = a.Ω  # rate-and-state variable, used by the solver for rate-and-state friction (lagged: start of the step)
    pl = materials.plasticity
    sp = (c= ones(size_c...), v= ones(size_v...))
    C  = (c=zeros(size_c...), v=zeros(size_v...))

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1, 1]*a.X.vx_e.x .+ D_BC[1, 2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1]*a.X.vy_e.x .+ D_BC[2, 2]*a.X.vy_e.y'
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

      # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx]     .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1] .+ (a.type.Vx[1, iny_Vx] .== :normal_stress) .* σ_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1] .+ (a.type.Vx[end, iny_Vx] .== :normal_stress) .* σ_BC[1, 1]
        BC.Vx[inx_Vx, 2]     .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[end])
        BC.Vy[inx_Vy, 2]     .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2] .+ (a.type.Vy[inx_Vy, 1] .== :normal_stress) .* σ_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2] .+ (a.type.Vy[inx_Vy, end] .== :normal_stress) .* σ_BC[2, 2]
        BC.Vy[2, iny_Vy]     .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[1] .+ D_BC[2, 2]*a.X.v.y)
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[end] .+ D_BC[2, 2]*a.X.v.y)
    end

    # Markers
    nxcell = (4, 4)
    max_xcell = 25
    min_xcell = 10
    args = 1
    particles, tm = TensorMarkers(backend, a, nxcell, max_xcell, min_xcell)
    adv = Markers(backend, a, particles, tm, nc, nphases; args)
    phases, = adv.particle_args

    # Probes are carried by passive markers (advected with the flow, see end of time loop)
    𝐱_probes = (𝐱_probe_salt, 𝐱_probe_plate, 𝐱_probe_MC1, 𝐱_probe_MC2)
    probe_markers = init_passive_markers(backend, (Float64[p[1] for p in 𝐱_probes], Float64[p[2] for p in 𝐱_probes]))

    # Material geometry
    gouge = (
        Rectangle((0.0/sc.L, L.y/2), hgouge, 20.0e-2/sc.L; θ=θgouge),
    )
    salt_x = (
        Rectangle((-L.x/2 + t_salt/2, L.y/2), t_salt, L.y; θ=0),
        Rectangle((L.x/2 - t_salt/2, L.y/2), t_salt, L.y; θ=0),
    )
    # Against the inner face of the plates, full width (the sample is shortened by t_salt_y at each end)
    salt_y = (
        Rectangle((0.0, y_plate + t_plate/2 + t_salt_y/2), L.x, t_salt_y; θ=0),
        Rectangle((0.0, L.y - y_plate - t_plate/2 - t_salt_y/2), L.x, t_salt_y; θ=0),
    )
    salt = plates && t_salt_y > 0 ? (salt_x..., salt_y...) : salt_x
    plate = !plates ? () : (
        Rectangle((0.0/sc.L, y_plate), 1.1e-1/sc.L, t_plate; θ=0),
        Rectangle((0.0/sc.L, L.y - y_plate), 1.1e-1/sc.L, t_plate; θ=0),
    )
    # seed = (
    #     Ellipse((0.0, 0.0), 0.1/sc.L, 0.1/sc.L; θ = 0),
    # )

    # Gouge: |ξ| < h/2 in the gouge frame (ξ along the normal n_gouge, η along the gouge), walls displaced by rough profiles of RMS rough_A
    # (rough_matched: both walls have the same profile, i.e. a wavy layer of constant thickness; otherwise independent walls)
    t_gouge = (-n_gouge[2], n_gouge[1])
    r_top = rough_A > 0 ? random_profile_1D(rough_λ/sc.L, rough_seed) : η -> 0.0
    r_bot = rough_A > 0 && !rough_matched ? random_profile_1D(rough_λ/sc.L, rough_seed + 1000) : r_top
    function ingouge(𝐱)
        ξ = (𝐱[1] - 0.0)*n_gouge[1] + (𝐱[2] - L.y/2)*n_gouge[2]
        η = (𝐱[1] - 0.0)*t_gouge[1] + (𝐱[2] - L.y/2)*t_gouge[2]
        A = rough_A/sc.L
        return abs(η) < 10e-2/sc.L && -hgouge/2 + A*r_bot(η) < ξ < hgouge/2 + A*r_top(η)
    end
    rough_A > 0 && @printf("Rough gouge walls: RMS %.2f mm, wavelengths %.1f–%.1f mm, %s\n", rough_A*1e3, rough_λ*1e3/2, rough_λ*1e3*2, rough_matched ? "matched (wavy layer)" : "independent walls")
    # Heterogeneity: gouge markers where a correlated Gaussian field exceeds its (1 - het_frac) quantile are gouge 2
    if het_frac > 0
        fhet = random_field_2D(het_ℓ/sc.L, het_seed)
        rng  = Random.Xoshiro(het_seed + 1)
        smp  = sort([fhet((L.x*(rand(rng) - 0.5), L.y*rand(rng))) for _ in 1:20000])
        qhet = smp[clamp(round(Int, (1 - het_frac)*length(smp)), 1, length(smp))]
        isalt = 𝐱 -> fhet(𝐱) > qhet
    else
        isalt = 𝐱 -> false
    end
    set_material_geometry!(phases, particles, ingouge, salt, plate, rate_state ? w_sample/2 - tip_RS/sc.L : Inf, isalt)

    # Set phase ratios on the grid
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)
    Set_Grid_Phases!(a)
    fig_markers = visualise_markers(particles, phases, sc, x, y, 0)
    outdir === nothing ? display(fig_markers) : save(joinpath(outdir, "markers_0000.png"), fig_markers)

    # check 
    for I in CartesianIndices(a.phase_ratios.c)
        s = sum(a.phase_ratios.c[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.c at $I: sum = $s, values = $(a.phase_ratios.c[I])"
        end
    end

    ΔCc = materials.plasticity.C[1]/5 .* rand(size(C.c)...)
    ΔCv = materials.plasticity.C[1]/5 .* rand(size(C.v)...)

    a.Pt  .= Pbg #*rand(size(Pt)...)
    a.Pt0 .= a.Pt
    centroid2particle!(tm.p.P, a.Pt, particles) # markers carry the pressure from step to step (Pt0): start from the initial pressure
    a.Pti .= a.Pt

    #--------------------------------------------#

    rvec        = zeros(length(iter_params.α))
    err         = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    probes      = (Δt=zeros(nt), Vmax=zeros(nt), Ωmean=zeros(nt), τII=zeros(nt),τIIW=zeros(nt), τIIE=zeros(nt), τIIS=zeros(nt), τIIN=zeros(nt), true_fric=zeros(nt), app_fric=zeros(nt), app_fric_lab=zeros(nt), t=zeros(nt), εxx=zeros(nt), εyy=zeros(nt), σyyN=zeros(nt), σyyS=zeros(nt), σxxW=zeros(nt), σxxE=zeros(nt), PW=zeros(nt), PE=zeros(nt), minθ=zeros(nt), maxθ=zeros(nt), meanθ=zeros(nt),
                   σxx_salt_grid=zeros(nt), σyy_plate_grid=zeros(nt), P_salt_grid=zeros(nt), app_fric_lab_grid=zeros(nt)) # *_grid: old (v7) way, fixed cells
    probe_salt  = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt), x=zeros(nt), y=zeros(nt))
    probe_plate = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt), x=zeros(nt), y=zeros(nt))
    probe_MC1   = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt), x=zeros(nt), y=zeros(nt))
    probe_MC2   = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt), x=zeros(nt), y=zeros(nt))
    to          = TimerOutput()

    # Swap old values 
    update_old_stress!(a)
    #--------------------------------------------#

    time = 0

    # Checkpointing: everything carried from one step to the next (grid fields, markers incl. stresses, probes).
    # Markers are CellArrays: their underlying data arrays are saved/restored in place.
    state = (;
        grid    = (; V=a.V, Pt=a.Pt, Pt0=a.Pt0, Pti=a.Pti, τ=a.τ, τ0=a.τ0, λ̇=a.λ̇, ε̇=a.ε̇, ΔPt=a.ΔPt, εp, Ω, sp),
        markers = (; coords=map(c -> c.data, particles.coords), index=particles.index.data,
                     args=map(c -> c.data, adv.particle_args), probes=probe_markers.coords),
        history = (; probes, probe_salt, probe_plate, probe_MC1, probe_MC2),
    )
    checkpoint_dir  = outdir === nothing ? (@__DIR__) : outdir
    checkpoint_file(it) = joinpath(checkpoint_dir, @sprintf("checkpoint_%05d.jld2", it))
    Δt       = Δt0                  # current time step (adaptive)
    if restart > 0
        restore_state!(state, load(checkpoint_file(restart), "state"))
        time = load(checkpoint_file(restart), "time")
        Δt   = jldopen(f -> haskey(f, "Δt") ? f["Δt"] : Δt0, checkpoint_file(restart)) # older checkpoints: no Δt
        # Phase ratios from the restored markers
        Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)
        Set_Grid_Phases!(a)
        @info "Restarting after step $(restart) from $(checkpoint_file(restart))"
    end
    # Grid fields at the start of the step, restored when a step fails. Markers need no backup: the solver fails before they are advected
    grid_old = deepcopy(state.grid)

    t_end = iter_params.time_max * (1 - 1e-10) # end of the run (tolerance: round-off in the sum of the Δt)
    time >= t_end && @info "Restart: time_max already reached"
    for it=restart+1:iter_params.nt_max
        time >= t_end && break

        # Adaptive time step, limited by the CFL condition and by time_max (the last step ends exactly at time_max)
        Vmax   = max(maximum(abs.(a.V.x)), maximum(abs.(a.V.y)))
        Δt_CFL = Vmax > 0 ? 0.5*min(Δ.x, Δ.y)/Vmax : Inf
        restore_state!(grid_old, state.grid)

        # Main loop: solver and advection -------------------------------------------------------------------
        for itry = 0:iter_params.ntry
            Δ = (x=Δ.x, y=Δ.y, t=min(Δt, Δt_CFL, iter_params.time_max - time))
            try
                @time main_loop(a, adv, tm, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
                break
            catch e
                # Diagnostics of the failed attempt (state at the last nonlinear iteration)
                outdir === nothing || jldsave(joinpath(outdir, @sprintf("failed_%05d_try%d.jld2", it, itry)); it, itry, Δt=Δ.t, sc,
                    Rx=a.R.x, Ry=a.R.y, Rp=a.R.p, λ̇c=a.λ̇.c, λ̇v=a.λ̇.v, Pt=a.Pt, τII=a.τ.II, phc=a.phases.c, Ωc=Ω.c, Ωv=Ω.v, εpc=εp.c,
                    xc=a.X.c_e.x, yc=a.X.c_e.y, xv=a.X.v.x, yv=a.X.v.y)
                (e isa Union{InterruptException, UndefVarError, MethodError} || itry == iter_params.ntry) && rethrow() # Ctrl-C or bug in the code: no retry
                @printf("Time step %04d: FAILED with Δt = %.3e s (%s), restart from the start of the step with Δt/%.2f (try %d/%d)\n", it, Δ.t*sc.t, sprint(showerror, e), iter_params.Δt_red_fac, itry+1, iter_params.ntry)
                restore_state!(state.grid, grid_old)
                Δt = Δ.t / iter_params.Δt_red_fac
            end
        end
        time += Δ.t
        done  = time >= t_end # last step: always saved (figure, step file, checkpoint)
        probes.Δt[it] = Δ.t
        Δt = min(Δ.t * iter_params.Δt_inc_fac, Δt_max) # try a larger time step for the next one
        Set_Grid_Phases!(a)

        # Update platic strain
        εp.c .+= Δ.t .* a.λ̇.c
        εp.v .+= Δ.t .* a.λ̇.v

        # Rate-and-state: update Ω with the slip rate of the step (V = Δx λ̇), then limit the next time step
        if rate_state
            Lg, V0g, Dg = pl.LRS[2], pl.V0RS[2], pl.DRS[2]
            ΔΩ_step, Vmax_g, Ωsum, nsum = 0.0, 0.0, 0.0, 0
            for (Ωf, λ̇f, ph) in ((Ω.c, a.λ̇.c, a.phase_ratios.c), (Ω.v, a.λ̇.v, a.phase_ratios.v))
                for I in eachindex(Ωf)
                    ph[I][2] + ph[I][6] + (tip_VS ? ph[I][5] : 0.0) > 0.5 || continue # rate-and-state gouge cells
                    V   = Dg * λ̇f[I]
                    Ωn  = update_rate_state(Ωf[I], V, Δ.t, Lg, V0g)
                    ΔΩ_step = max(ΔΩ_step, abs(Ωn - Ωf[I]))
                    Vmax_g  = max(Vmax_g, V)
                    Ωf[I] = Ωn
                    Ωsum += Ωn; nsum += 1
                end
            end
            probes.Vmax[it]  = Vmax_g
            probes.Ωmean[it] = Ωsum / max(nsum, 1)
            Δt_w = fRS_w * Lg / Vmax_g                                    # weakening
            Δt_Ω = ΔΩ_step > 0 ? Δ.t * ΔΩmax / ΔΩ_step : Inf              # change of Ω per step
            Δt   = clamp(min(Δt, Δt_w, Δt_Ω), Δt_min/sc.t, Δt_max)
            @printf("Rate-and-state: Vmax = %.3e m/s, max|ΔΩ| = %.3f, mean Ω = %.3f, next Δt = %.3e s\n", Vmax_g*sc.L/sc.t, ΔΩ_step, probes.Ωmean[it], Δt*sc.t)
        end

        # Softening
        sp.c .= softening.(εp.c, 1.0, 0.2, 0.0, 0.1)
        sp.v .= softening.(εp.v, 1.0, 0.2, 0.0, 0.1)
        @show extrema(sp.c)
        @show extrema(sp.v)

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = av2D(a.τ.xy)

        isrock  = zeros(Bool, size_c)
        isgouge = zeros(Bool, size_c)
        [isrock[I]  = a.phase_ratios.c[I][1]≈1.0 for I in eachindex(a.phase_ratios.c)]
        [isgouge[I] = a.phase_ratios.c[I][2] + a.phase_ratios.c[I][6] ≈ 1.0 for I in eachindex(a.phase_ratios.c)]

        τII_rock = a.τ.II[isrock]
        P_rock   =   a.Pt[isrock]
        λ̇_rock   =  a.λ̇.c[isrock]

        τII_gouge = a.τ.II[isgouge]
        P_gouge   =   a.Pt[isgouge]
        λ̇_gouge   =  a.λ̇.c[isgouge]

        # Compute apparent and true friction locally
        app_fric  = zeros(size(a.Pt))
        true_fric = zeros(size(a.Pt))
        Rot = @SMatrix[cos(π/2 - θgouge) -sin(π/2 - θgouge) 0; sin(π/2 - θgouge) cos(π/2 - θgouge) 0; 0 0 1.0] # x′ along the gouge (dipping right), y′ normal to it
        σ′xy_sum, σ′yy_sum, true_fric_sum = 0.0, 0.0, 0.0 # gouge-mean friction as a ratio of sums: a mean of local ratios blows up in cells where σ′yy → 0
        app_sum, true_sum  = 0, 0
        ind_mid_x = Int64(floor((nc.x+2)/2))
        ind_mid_y = Int64(floor((nc.y+2)/2))

        for i in inx_c, j in iny_c
            if a.phases.c[i, j] == 2 || a.phases.c[i, j] == 6 # gouge
                app_sum += 1
                σ = @SMatrix[-a.Pt[i, j]+a.τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -a.Pt[i, j]+a.τ.yy[i, j] 0.; 0. 0. -a.Pt[i, j]+(-a.τ.xx[i, j]-a.τ.yy[i, j])]
                # Compute apparent friction
                σ′ = Rot * σ * Rot'
                app_fric[i, j] = -σ′[1, 2] / σ′[2, 2]
                σ′xy_sum += σ′[1, 2]
                σ′yy_sum += σ′[2, 2]
                # Compute true friction
                if a.λ̇.c[i, j] > abs(ε̇yy) && i==ind_mid_x && j==ind_mid_y
                    ph     = a.phases.c[i, j]
                    cxcosϕ = materials.plasticity.C[ph] * materials.plasticity.cosϕ[ph]
                    ηvp    = materials.plasticity.ηvp[ph]
                    A      = 1/a.Pt[i, j] * (a.τ.II[i, j] - cxcosϕ - ηvp*a.λ̇.c[i, j])
                    if A <= 1
                        true_fric[i, j] = tand(asind(A))
                        true_fric_sum += true_fric[i, j]
                        true_sum += 1
                    end
                end
            end
        end

        # Store probes data
        # Probe cells: centroid cell containing each passive marker (used for MC2, not interpolated: the gouge is only a few cells wide)
        pxm, pym = probe_markers.coords[1], probe_markers.coords[2]
        𝐈_probe_salt, 𝐈_probe_plate, 𝐈_probe_MC1, 𝐈_probe_MC2 = ntuple(k -> (argmin(abs.(a.X.c_e.x .- pxm[k])), argmin(abs.(a.X.c_e.y .- pym[k]))), 4)
        for (k, probe) in enumerate((probe_salt, probe_plate, probe_MC1, probe_MC2))
            probe.x[it] = pxm[k]
            probe.y[it] = pym[k]
        end
        ind_mid_x     = Int64(floor(nc.x/2))
        ind_mid_y     = Int64(floor(nc.y/2))

        σyy             = a.τ.yy .- a.Pt
        σxx             = a.τ.xx .- a.Pt
        σzz             = .-(a.τ.xx .+ a.τ.yy) .- a.Pt
        probes.t[it]    = time
        probes.τII[it]  = mean(a.τ.II)
        probes.τIIW[it] = a.τ.II[2, ind_mid_y]
        probes.τIIE[it] = a.τ.II[end-1, ind_mid_y]
        probes.τIIS[it] = a.τ.II[ind_mid_x, 2]
        probes.τIIN[it] = a.τ.II[ind_mid_x, end-1]
        probes.PW[it]   = a.Pt[2, ind_mid_y]
        probes.PE[it]   = a.Pt[end-1, ind_mid_y]
        probes.σxxW[it] = a.τ.xx[2, ind_mid_y]     - a.Pt[2, ind_mid_y]
        probes.σxxE[it] = a.τ.xx[end-1, ind_mid_y] - a.Pt[end-1, ind_mid_y]
        probes.σyyS[it] = mean(a.τ.yy[inx_c, 2]     .- a.Pt[inx_c, 2])
        probes.σyyN[it] = mean(a.τ.yy[inx_c, end-1] .- a.Pt[inx_c, end-1])
        # Load cell (no plates): mean σyy over the sample (rock + gouge cells) along the top and bottom boundaries
        insample(i, j) = a.phases.c[i, j] in (1, 2, 5, 6)
        σyy_load = mean([σyy[i, j] for j in (iny_c[1], iny_c[end]) for i in inx_c if insample(i, j)])

        # Apparatus probes (salt, plate) and rock probe (MC1): bilinear interpolation of the centroid fields at the passive markers,
        # so the value does not jump when a marker crosses a cell face
        probe_fields = ntuple(_ -> zeros(length(pxm)), 6)
        grid2particle!(probe_fields, (a.X.c_e.x, a.X.c_e.y), (σxx, σyy, σzz, τxyc, a.Pt, a.τ.II), probe_markers)
        for (k, probe) in ((1, probe_salt), (2, probe_plate), (3, probe_MC1))
            probe.σxx[it] = probe_fields[1][k]
            probe.σyy[it] = probe_fields[2][k]
            probe.σzz[it] = probe_fields[3][k]
            probe.σxy[it] = probe_fields[4][k]
            probe.P[it]   = probe_fields[5][k]
            probe.τII[it] = probe_fields[6][k]
        end
        plates || (probe_plate.σyy[it] = σyy_load) # no plates: the "plate" axial stress is the load-cell value

        probe_MC2.σxx[it] = σxx[𝐈_probe_MC2...]
        probe_MC2.σyy[it] = σyy[𝐈_probe_MC2...]
        probe_MC2.σzz[it] = σzz[𝐈_probe_MC2...]
        probe_MC2.σxy[it] = τxyc[𝐈_probe_MC2...]
        probe_MC2.P[it]   = a.Pt[𝐈_probe_MC2...]
        probe_MC2.τII[it] = a.τ.II[𝐈_probe_MC2...]

        # σd = probes.σyyS[it] - probes.σxxW[it]
        # σm = -probes.PW[it]
        σd = 2 * 1/2*(mean(τII_gouge))
        σm = 1 * 1/2*(mean(P_gouge))
        τr = σd/2 * sin(2*(π/2 - θgouge))
        τn = σm + σd/2 * cos(2*(π/2 - θgouge))
        # probes.app_fric[it] =  τr / τn
        probes.app_fric[it] = -σ′xy_sum / σ′yy_sum
        probes.true_fric[it] = true_fric_sum / true_sum

        # σ  = @SMatrix[probes.σxxW[it] 0.0 0.; 0.0 probes.σyyS[it] 0.; 0. 0. 1/2*(probes.σxxW[it] + probes.σyyS[it])]
        σ = @SMatrix[probe_salt.σxx[it] 0.0 0.;
            0. probe_plate.σyy[it] 0.;
            0. 0. -probe_salt.P[it]]

        σ′ = Rot * σ * Rot'
        probes.app_fric_lab[it] = -σ′[1, 2] / σ′[2, 2] # = τ/σn of the gouge-plane point on the apparatus Mohr circle

        # Old (v7) way, for comparison: centroid cell at the initial (fixed) probe positions, no interpolation
        𝐈_salt_grid  = argmin(abs.(a.X.c_e.x .- 𝐱_probe_salt[1])),  argmin(abs.(a.X.c_e.y .- 𝐱_probe_salt[2]))
        𝐈_plate_grid = argmin(abs.(a.X.c_e.x .- 𝐱_probe_plate[1])), argmin(abs.(a.X.c_e.y .- 𝐱_probe_plate[2]))
        probes.σxx_salt_grid[it]  = σxx[𝐈_salt_grid...]
        probes.σyy_plate_grid[it] = plates ? σyy[𝐈_plate_grid...] : σyy_load
        probes.P_salt_grid[it]    = a.Pt[𝐈_salt_grid...]
        σ = @SMatrix[probes.σxx_salt_grid[it] 0.0 0.;
            0. probes.σyy_plate_grid[it] 0.;
            0. 0. -probes.P_salt_grid[it]]
        σ′ = Rot * σ * Rot'
        probes.app_fric_lab_grid[it] = -σ′[1, 2] / σ′[2, 2]

        # Stress angles
        i1 = argmin(abs.(a.X.c_e.x .- (-0.01/sc.L)))
        i2 = argmin(abs.(a.X.c_e.x .- (0.01/sc.L)))
        j1 = argmin(abs.(a.X.c_e.y .- (0.075/sc.L)))
        j2 = argmin(abs.(a.X.c_e.y .- (0.125/sc.L)))
        θawayBC = a.τ.θ[i1:i2, j1:j2]
        probes.minθ[it] = minimum(θawayBC[in.(a.phases.c[i1:i2, j1:j2], Ref((2, 6)))])
        probes.meanθ[it] = mean(θawayBC[in.(a.phases.c[i1:i2, j1:j2], Ref((2, 6)))])
        probes.maxθ[it] = maximum(θawayBC[in.(a.phases.c[i1:i2, j1:j2], Ref((2, 6)))])

        # Everything the figure needs (also what is saved to disk)
        d = (; it, time, sc, Ly=L.y, θgouge, ε̇yy, Rot=Matrix(Rot), plates,
            xc=a.X.c.x, yc=a.X.c.y, xv_min=minimum(a.X.v.x), xv_max=maximum(a.X.v.x),
            λ̇c=a.λ̇.c[inx_c, iny_c], phc=a.phases.c[inx_c, iny_c], θc=a.τ.θ[inx_c, iny_c], εpc=a.εp.c[inx_c, iny_c],
            ind_mid_x, I_salt=𝐈_probe_salt, I_plate=𝐈_probe_plate, I_MC1=𝐈_probe_MC1, I_MC2=𝐈_probe_MC2, i1, i2, j1, j2,
            P_rock, τII_rock, λ̇_rock, P_gouge, τII_gouge, λ̇_gouge,
            plasticity=(C=materials.plasticity.C, ϕ=materials.plasticity.ϕ, ψ=materials.plasticity.ψ, sinϕ=materials.plasticity.sinϕ, cosϕ=materials.plasticity.cosϕ, ηvp=materials.plasticity.ηvp,
                        Δϕ=materials.plasticity.Δϕ, ΔC=materials.plasticity.ΔC, εh=materials.plasticity.εh,
                        cap=materials.plasticity isa DruckerPragerCap ? (; (k => getfield(materials.plasticity, k) for k in (:Pt, :py, :Ry, :pd, :τd, :a))...) : nothing),
            probes, probe_salt, probe_plate, probe_MC1, probe_MC2)

        # Visualise / save
        if outdir === nothing
            display(plot_shear_apparatus(d))
        elseif it % nsave == 0 || it == iter_params.nt_max || done
            save(joinpath(outdir, @sprintf("figure_%05d.png", it)), plot_shear_apparatus(d))
            jldsave(joinpath(outdir, @sprintf("step_%05d.jld2", it)); d...)
        end
        if outdir !== nothing && save_fields !== nothing && it in save_fields # light 2D snapshots, e.g. across a slip event
            jldsave(joinpath(outdir, @sprintf("fields_%05d.jld2", it)); it, time, Δt=Δ.t, sc, xc=a.X.c.x, yc=a.X.c.y,
                λ̇c=a.λ̇.c[inx_c, iny_c], Pt=a.Pt[inx_c, iny_c], τII=a.τ.II[inx_c, iny_c], Ωc=Ω.c[inx_c, iny_c], phc=a.phases.c[inx_c, iny_c],
                μ_app=probes.app_fric_lab[it], Vmax=rate_state ? probes.Vmax[it] : NaN)
        end
        if outdir !== nothing && nsave_probes !== nothing && (it % nsave_probes == 0 || done)
            jldsave(joinpath(outdir, "probes.jld2"); it, sc, ε̇yy, Ly=L.y, θgouge, probes, probe_salt, probe_plate)
        end

        # Move the probes with the flow of this step (same velocity and Δt as the phase markers)
        @views V = (a.V.x[2:(end-1), 2:(end-1)], a.V.y[2:(end-1), 2:(end-1)])
        advection!(probe_markers, RungeKutta4(), V, adv.grid_vi, Δ.t)

        # Checkpoint (end of step: markers already advected)
        if ncheck > 0 && (it % ncheck == 0 || done)
            jldsave(checkpoint_file(it); it, time, Δt, state)
            # Keep only the 2 most recent checkpoints
            old = checkpoint_file(it - 2ncheck)
            it - 2ncheck > 0 && isfile(old) && rm(old)
        end
        done && break
    end
    display(to)
end

# let
#     nt = 1000
#     Δt = 2.5e1
#     θ  = 15.0
#     h  = 2e-3
#     main((x = 280, y = 1000), nt, Δt, θ, h) # matches friction
# end

let
    n        = (x = 70, y = 250) # low resolution test
    nt_max   = 20000              # max. number of steps
    time_max = 12500.0            # [s] end of the run (if reached before nt_max steps)
    Δt = 2.5e1
    θ  = [30]
    h  = [3e-3]
    for θi in θ, hi in h
        outdir = joinpath(@__DIR__, @sprintf("_ShearApparatus_v10_Tet%02d_H%02d_LR", round(Int, θi), round(Int, hi*1e3)))
        mkpath(outdir)
        main(n, nt_max, Δt, θi, hi;
        outdir, time_max,
        restart=0, nsave=200, ncheck=200, nsave_probes=20, niter=25,
        Pic2Newt=1.8, niter_KSP=40, ϵ_nl_last=1e-10,
        Δt_red_fac=1.75, Δt_inc_fac=1.25, ntry=6)
    end
end
