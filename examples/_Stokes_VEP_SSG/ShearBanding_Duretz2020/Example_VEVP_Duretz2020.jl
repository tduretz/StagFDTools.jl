#---------------------------------------------------------------------------------------
# Crustal-scale shear banding with a visco-elasto-viscoplastic (V-E-VP) rheology
# Duretz, de Borst, Yamato & Le Pourhiet (2020), GRL, https://doi.org/10.1029/2019GL086027
# Reference MATLAB code: M2Di2_VEVP_CrustalShearBanding (https://bitbucket.org/lraess/m2di)
#
# 100 x 30 km slice of Westerly granite (power-law creep, T-dependent) with a weak circular
# seed at its centre, pure shear compression or extension, free slip on all sides, gravity.
# Incompressible, non-associated (ψ = 0) Drucker-Prager with linear viscoplastic regularisation.
# The setup is perfectly symmetric and symmetry is never enforced: the run prints the
# mirror asymmetry of the solution at every step.
#---------------------------------------------------------------------------------------
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
using CairoMakie

# Relative mirror asymmetry x -> -x (odd: field changes sign under the mirror, e.g. Vx or τxy)
asymmetry(A; odd=false) = maximum(abs.(A .- (odd ? -1 : 1) .* reverse(A, dims=1))) / maximum(abs, A)

@views function main(nc; mode=:compression, nt=100, Δt=4e11, ηvp=1e21, niter=15, ϵ_nl=5e-10, save_dir=nothing)

    save_dir === nothing || mkpath(save_dir)

    @assert isodd(nc.x) "nc.x must be odd: the pressure is pinned in the top centroid lying on the symmetry axis"

    # Scaling
    sc = (σ=1e7, L=1e3, t=1e15)

    # Boundary loading: pure shear, Vx = -ε̇ x, Vy = ε̇ y in compression (opposite in extension)
    ε̇bg  = (mode == :compression ? 1.0 : -1.0) * 1e-15 * sc.t
    D_BC = @SMatrix([ -ε̇bg 0.0;
                       0.0  ε̇bg ])

    # Materials: 1 = Westerly granite (Hansen & Carter, 1983), 2 = weak seed
    # Creep law ε̇II = A exp(-Q/(RT)) (τII/2)^n  <=>  η = η0 exp(Q/(nR) (1/T - 1/Tref)) ε̇II^(1/n-1)
    nphases = 2
    A    = [3.1623e-26, 1e-20]  # [Pa^-n s^-1]
    n    = [3.3, 1.0]
    Q    = [186.5e3, 0.0]       # [J/mol]
    Tref = 773.15               # [K]
    η0   = @. A^(-1/n) * exp(Q / (n * 8.314462618 * Tref))
    ϕ    = atand(0.6)
    C    = [5e7, 1e4]           # [Pa], yield τII = C + P sinϕ as in the M2Di code (cohesion term without cosϕ)

    materials = initialize_materials(nphases; plasticity=DruckerPrager, compressible=false)
    materials.g    .= [0.0, -10.0] / (sc.L / sc.t^2)
    materials.ρ    .= [2700, 2700] / (sc.σ * sc.t^2 / sc.L^2)
    materials.n    .= n
    materials.η0   .= @. η0 / (sc.σ * sc.t^(1/n))
    materials.Ea   .= Q
    materials.Tref .= Tref
    materials.G    .= [1e10, 1e10] / sc.σ
    materials.plasticity.C   .= C ./ cosd(ϕ) / sc.σ
    materials.plasticity.ϕ   .= [ϕ, 0.0]
    materials.plasticity.ψ   .= [0.0, 0.0]
    materials.plasticity.ηvp .= [ηvp, ηvp] / (sc.σ * sc.t)
    preprocess!(materials)

    # Grid
    L    = (x=100e3/sc.L, y=30e3/sc.L)
    rad  = 2e3/sc.L
    Δ    = (x=L.x/nc.x, y=L.y/nc.y, t=Δt/sc.t)
    x    = (min=-L.x/2, max=L.x/2)
    y    = (min=-L.y,   max=0.0)
    X    = GenerateGrid(x, y, Δ, nc)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Node types: free slip on all sides, pressure fixed in the top centroid on the symmetry axis
    # (incompressible with prescribed normal velocities: P is otherwise defined up to a constant)
    type = Fields(fill(:out, size_x), fill(:out, size_y), fill(:out, size_c))
    set_boundaries_template!(type, :free_slip, nc)
    ipin, jpin = (nc.x + 1) ÷ 2 + 1, nc.y + 1
    type.Pt[ipin, jpin] = :constant

    number = Fields(fill(0, size_x), fill(0, size_y), fill(0, size_c))
    Numbering!(number, type, nc)
    nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]), @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])),
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1; 1 1 1]), @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0; 0 1 0]), @SMatrix([0 0; 1 1; 0 0]), @SMatrix([1])),
    )
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)),
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)),
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt)),
    )
    𝐊  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏  = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r  = zeros(nVx + nVy + nPt)

    # Fields
    R    = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V    = (x=zeros(size_x...), y=zeros(size_y...))
    Vi   = (x=zeros(size_x...), y=zeros(size_y...))
    η    = (c=ones(size_c...),  v=ones(size_v...))
    ξ    = (c=ones(size_c...),  v=ones(size_v...))
    λ̇    = (c=zeros(size_c...), v=zeros(size_v...))
    G    = (c=zeros(size_c...), v=zeros(size_v...))
    β    = (c=zeros(size_c...), v=zeros(size_v...))
    ρ    = (c=zeros(size_c...), v=zeros(size_v...))
    ε̇    = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...), θ=zeros(size_c...))
    τ0   = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...))
    τ    = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...), θ=zeros(size_c...))
    Pt   = zeros(size_c...)
    Pti  = zeros(size_c...)
    Pt0  = zeros(size_c...)
    ΔPt  = (c=zeros(size_c...), Vx=zeros(size_x...), Vy=zeros(size_y...))
    𝐷    = (c=[@MMatrix(zeros(4, 4)) for _ in 1:size_c[1], _ in 1:size_c[2]], v=[@MMatrix(zeros(4, 4)) for _ in 1:size_v[1], _ in 1:size_v[2]])
    𝐷_ctl= (c=[@MMatrix(zeros(4, 4)) for _ in 1:size_c[1], _ in 1:size_c[2]], v=[@MMatrix(zeros(4, 4)) for _ in 1:size_v[1], _ in 1:size_v[2]])

    # Geotherm: 20 °C at the surface, 15 °C/km (ghost nodes included)
    T = (c=[293.15 - 15e-3 * yc * sc.L for _ in X.c_e.x, yc in X.c_e.y],
         v=[293.15 - 15e-3 * yv * sc.L for _ in X.v_e.x, yv in X.v_e.y])

    # Weak seed in the middle of the domain
    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))
    phases.c[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y' .+ L.y/2).^2) .< rad^2] .= 2
    phases.v[inx_v, iny_v][(X.v.x.^2 .+ (X.v.y' .+ L.y/2).^2) .< rad^2] .= 2
    phase_ratios = InitialisePhaseRatios(phases, nphases)
    compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, nphases)

    # Initial guess: pure shear velocity, lithostatic pressure (0 in the pinned top centroid)
    V.x[inx_Vx, iny_Vx] .= D_BC[1,1] .* X.v.x .+ D_BC[1,2] .* X.c.y'
    V.y[inx_Vy, iny_Vy] .= D_BC[2,1] .* X.c.x .+ D_BC[2,2] .* X.v.y'
    Pt .= [-materials.ρ[1] * materials.g[2] * (-yc - Δ.y/2) for _ in X.c_e.x, yc in X.c_e.y]

    # Boundary condition values (free slip: zero tangential stress)
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))

    α     = collect(LinRange(0.1, 1.0, 10))
    rvec  = zeros(length(α))
    iters = zeros(Int, nt)
    asym  = zeros(nt)
    nl    = (x=zeros(niter), y=zeros(niter))

    for it = 1:nt
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        ϵ0, ϵ, iter = 0.0, 0.0, 0
        while iter < niter
            iter += 1
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ; T)
            ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G,    materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
            nl.x[iter] = norm(R.x[inx_Vx, iny_Vx]) / sqrt(nVx)
            nl.y[iter] = norm(R.y[inx_Vy, iny_Vy]) / sqrt(nVy)
            ϵ = max(nl.x[iter], nl.y[iter])
            (iter == 1) && (ϵ0 = ϵ)
            @printf("    iter %02d: abs. res. = %1.3e --- rel. res. = %1.3e\n", iter, ϵ, ϵ / ϵ0)
            (ϵ / ϵ0 < ϵ_nl || ϵ < ϵ_nl) && break

            # Newton step
            AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, ξ, materials, number, pattern, type, BC, nc, Δ)
            AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G,    materials, number, pattern, type, BC, nc, Δ)
            AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, ρ, materials, number, pattern, type, BC, nc, Δ)
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt
            SetRHS!(r, R, number, type, nc)
            mechanical_solver!(dx, M, r, 𝐊, 𝐐, 𝐐ᵀ, 𝐏, 𝐊, 𝐐, 𝐐ᵀ, 𝐏; solver=:PH, ϵ_l=1e-11)
            imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ; T)
            UpdateSolution!(V, Pt, α[imin] * dx, number, type, nc)
        end
        (ϵ / ϵ0 < ϵ_nl || ϵ < ϵ_nl) || error(@sprintf("Step %d: Newton did not converge in %d iterations (rel. res. = %1.3e)", it, niter, ϵ / ϵ0))
        Pt .+= ΔPt.c

        iters[it] = iter
        asym[it]  = max(asymmetry(V.x[inx_Vx, iny_Vx]; odd=true), asymmetry(Pt[inx_c, iny_c]), asymmetry(ε̇.II[inx_c, iny_c]))
        @printf("Step %03d (%s): %02d iterations, max λ̇ = %1.2e s⁻¹, mirror asymmetry (Vx, P, ε̇II) = %1.2e\n",
            it, mode, iter, maximum(λ̇.c) / sc.t, asym[it])
    

        # ---------------------------------------------------------------- Figures
        xc, yc   = X.c.x * sc.L / 1e3, X.c.y * sc.L / 1e3
        ε̇II      = ε̇.II[inx_c, iny_c] / sc.t
        j10      = argmin(abs.(yc .+ 10))                         # profile at y = -10 km
        rock     = phases.c[inx_c, iny_c] .== 1
        depth    = repeat(yc', nc.x)[rock]
        Plith    = @. 2700 * 10 * (-yc * 1e3 - Δ.y * sc.L / 2)    # lithostatic pressure (top centroid pinned to 0)
        sinϕ     = sind(ϕ)
        τ_DP     = mode == :compression ? @.((C[1] + Plith * sinϕ) / (1 - sinϕ)) : @.((C[1] + Plith * sinϕ) / (1 + sinϕ))
        P_DP     = mode == :compression ? Plith .+ τ_DP : Plith .- τ_DP

        fig = Figure(size=(1200, 900), fontsize=16)
        ax  = Axis(fig[1, 1:3], aspect=DataAspect(), title="$(mode), $(nc.x)×$(nc.y) cells, t = $(@sprintf("%1.2f", nt * Δt / (1e6 * 365.25 * 24 * 3600))) My", xlabel="x [km]", ylabel="y [km]")
        hm  = heatmap!(ax, xc, yc, log10.(ε̇II), colormap=:roma, colorrange=(-16, -14))
        hlines!(ax, yc[j10], color=:white, linestyle=:dash)
        Colorbar(fig[1, 4], hm, label=L"\log_{10} \dot{\varepsilon}_{II}")

        ax  = Axis(fig[2, 1:3], title="Profile at y = $(@sprintf("%1.1f", yc[j10])) km (dashed: mirrored)", xlabel="x [km]", ylabel=L"\log_{10} \dot{\varepsilon}_{II}")
        lines!(ax, xc, log10.(ε̇II[:, j10]))
        lines!(ax, xc, log10.(reverse(ε̇II[:, j10])), linestyle=:dash)

        g   = fig[3, 1:4] = GridLayout()
        ax  = Axis(g[1, 1], title="Newton iterations", xlabel="Time step", ylabel="Iterations")
        scatter!(ax, 1:nt, iters)
        ax  = Axis(g[1, 2], title=L"\tau_{II}", xlabel=L"\tau_{II}\ \mathrm{[MPa]}", ylabel="y [km]")
        scatter!(ax, τ.II[inx_c, iny_c][rock] * sc.σ / 1e6, depth, markersize=2, color=:black)
        lines!(ax, τ_DP / 1e6, yc, color=:green)
        ax  = Axis(g[1, 3], title=L"P", xlabel=L"P\ \mathrm{[MPa]}", ylabel="y [km]")
        scatter!(ax, Pt[inx_c, iny_c][rock] * sc.σ / 1e6, depth, markersize=2, color=:black)
        lines!(ax, P_DP / 1e6, yc, color=:green)
        lines!(ax, Plith / 1e6, yc, color=:gray, linestyle=:dash)
        display(fig)
        save_dir === nothing || save(joinpath(save_dir, @sprintf("Duretz2020_%s_%dx%d_step%04d.png", mode, nc.x, nc.y, it)), fig)
    end

    # Animation of the png sequence (ffmpeg with a generated palette for clean colours)
    if save_dir !== nothing
        frames = joinpath(save_dir, "Duretz2020_$(mode)_$(nc.x)x$(nc.y)_step%04d.png")
        gif    = joinpath(save_dir, "Duretz2020_$(mode)_$(nc.x)x$(nc.y).gif")
        run(`ffmpeg -y -loglevel error -framerate 10 -i $frames -vf "scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" $gif`)
    end

    return (; iters, asym, ε̇II=ε̇.II[inx_c, iny_c] / sc.t, xc=X.c.x * sc.L / 1e3, yc=X.c.y * sc.L / 1e3)
end

let
    nc = (x=201, y=61) # Figure 2a resolution (nc.x odd)
    main(nc; mode=:compression, save_dir = "/Users/tduretz/REPO/StagFDTools/examples/_Stokes_VEP_SSG/ShearBanding_Duretz2020/compression")
    main(nc; mode=:extension,   save_dir = "/Users/tduretz/REPO/StagFDTools/examples/_Stokes_VEP_SSG/ShearBanding_Duretz2020/extension")
end
