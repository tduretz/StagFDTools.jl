# Coupled (V, P) dynamic relaxation for 2D Stokes: original vs robust pseudo-time step selection
#
# Iteration structure (both schemes): damped second-order (heavy-ball) update for V, first-order update for P
#     dV/dτ ← α dV/dτ + β D⁻¹R_V ;  V += dτ_V dV/dτ
#     P     += dτ_P · η · (-∇⋅V)
#
# :original — dτ_V from Gershgorin incl. a "+2/Δx" pressure term, dτ_P = Δx/2 (fixed),
#             separate damping cVx, cVy from per-field Rayleigh quotients, no damping until the first estimate.
# :robust   — dτ_V from Gershgorin of the viscous block D⁻¹A only,
#             one damping c for the whole velocity field from a D-weighted Rayleigh quotient (updated every nupd its),
#             pressure rate tied to the damping: dτ_P = κ·c·dτ (κ < 1).
#             Reason: for heavy-ball V + first-order P, each mode behaves like s³ + c s² + a s + k b² = 0
#             (a: viscous, b²: grad-div coupling, k: pressure rate), stable iff c·a > k·b² (Routh-Hurwitz).
#             With an η-scaled pressure a ≳ (4/3) b², hence k = κ c with κ < 4/3 stays stable whatever the
#             resolution or viscosity contrast, while P relaxes at a rate comparable to V.
#             Default κ = 0.125: converged in all tests, ncx = 63…1023, η_incl = 1e-3…1e6 (smooth and sharp).
#             κ = 0.25 is up to 2x faster for ncx ≤ 511 but diverges slowly at ncx = 1023 (η_incl = 1e3, 1e6):
#             the Rayleigh estimate of λmin is inflated by the pressure update, c grows, so does k = κc → feedback.
#             κ = 0.5 already fails for strong inclusions at low resolution.
# Many inclusions (setup=:multi, 30 random circles, η_incl and 1/η_incl): the η-scaled pressure Schur complement
# becomes ill-conditioned (cond 1e3 for ±1e2, 1e5 for ±1e3 vs 2-15 for one inclusion) and the default stalls.
#   γ̂ > 0 adds an augmented-Lagrangian penalty γ = γ̂η (τ += γ∇⋅V); the per-mode bound becomes κ < a/b² + γ̂,
#   use κ = θ(γ̂ + 1/3). ncx = 63/127: ±1e2 converges with γ̂ = 64…256, θ = 0.5…0.75 (100-200 iter/nx);
#   ±1e3 still not converged after 600 iter/nx (err ~ 2e-3). m_AA > 0 (Anderson on P per window): no gain.
using CairoMakie, Printf, Statistics, LinearAlgebra, Random
# Macros
@views av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])

function smooth!(A2, A, fact)
    @. A2[2:end-1,2:end-1] = A[2:end-1,2:end-1] + 1.0/4.1/fact*((-2*A[2:end-1,2:end-1].+A[3:end,2:end-1].+A[1:end-2,2:end-1]) + (-2*A[2:end-1,2:end-1].+A[2:end-1,3:end].+A[2:end-1,1:end-2]))
    return
end

# Gershgorin bounds of the Jacobi-preconditioned viscous block (same stencil as Stokes2D_DYREL_Coupled_CompareRaess22.jl)
function Gershgorin_Stokes2D(ηc, ηv, Δx, Δy, ncx, ncy; γ=zero(ηc))
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]; γW = γ[1:end-1,:]
    ηE    = ηc[2:end-0,:]; γE = γ[2:end-0,:]
    nE, nW = (4 // 3) .* ηE .+ γE, (4 // 3) .* ηW .+ γW    # normal-stress coefficients
    xE, xW = (2 // 3) .* ηE .- γE, (2 // 3) .* ηW .- γW    # cross (∂y Vy) coefficients
    Cxx   = abs.(nE ./ Δx .^ 2) + abs.(nW ./ Δx .^ 2) + abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (nE ./ Δx + nW ./ Δx) ./ Δx)
    Cxy   = abs.(xE ./ (Δx .* Δy) - ηN ./ (Δx .* Δy)) + abs.(xE ./ (Δx .* Δy) - ηS ./ (Δx .* Δy)) + abs.(ηN ./ (Δx .* Δy) - xW ./ (Δx .* Δy)) + abs.(ηS ./ (Δx .* Δy) - xW ./ (Δx .* Δy))
    DVx   = -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (nE ./ Δx + nW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]; γS = γ[:,1:end-1]
    ηN    = ηc[:,2:end-0]; γN = γ[:,2:end-0]
    nN, nS = (4 // 3) .* ηN .+ γN, (4 // 3) .* ηS .+ γS
    xN, xS = (2 // 3) .* ηN .- γN, (2 // 3) .* ηS .- γS
    Cyy   = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(nN ./ Δy .^ 2) + abs.(nS ./ Δy .^ 2) + abs.((nN ./ Δy + nS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    Cyx   = abs.(ηE ./ (Δx .* Δy) - xN ./ (Δx .* Δy)) + abs.(ηE ./ (Δx .* Δy) - xS ./ (Δx .* Δy)) + abs.(xN ./ (Δx .* Δy) - ηW ./ (Δx .* Δy)) + abs.(xS ./ (Δx .* Δy) - ηW ./ (Δx .* Δy))
    DVy   = (nN ./ Δy + nS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx   = (Cxx .+ Cxy) ./ DVx              # viscous block only
    λmaxVy   = (Cyx .+ Cyy) ./ DVy
    λmaxVx_o = (Cxx .+ Cxy .+ 2/Δx) ./ DVx      # original (incl. pressure term with mismatched units)
    λmaxVy_o = (Cyx .+ Cyy .+ 2/Δy) ./ DVy
    return DVx, DVy, λmaxVx, λmaxVy, λmaxVx_o, λmaxVy_o
end

# Stable pressure rate: κ < 1/λmax(S), S = η^½ B A⁻¹ Bᵀ η^½ (η-scaled pressure Schur complement).
# λmax(S) = 3/4 for constant η and grows with the largest viscosity jump between neighbouring cells,
# r = max η_cell/η_neighbour, saturating at 3 (exact eigenvalues on 31² grids, smooth and sharp inclusions,
# η_incl = 1e-6…1e6). Upper envelope of those data: λ̂(r) = 3 - 2.25/r².
# NB: this is a necessary bound, not a good tuning rule. Benchmarks (ncx = 63…255, η_incl = 2…1e6, smooth/sharp):
#  - κ = 0.25 (≈ 0.75/λmax(S) for the worst case λmax(S) = 3) converged everywhere and was fastest or close to it;
#  - κ = θ/λ̂(r) with θ = 0.25 was stable but 2-4x slower; with θ = 0.5/0.75 larger κ for mild contrasts
#    gave no speed-up and failed at ncx = 255 for η_incl = 2 (κ ≈ 0.4-0.6 < 1/λmax(S) ≈ 1.1).
# Use κ = :auto only for experiments.
@views function κ_auto(ηc, θ)
    r = max(maximum(ηc[2:end,:]./ηc[1:end-1,:]), maximum(ηc[1:end-1,:]./ηc[2:end,:]),
            maximum(ηc[:,2:end]./ηc[:,1:end-1]), maximum(ηc[:,1:end-1]./ηc[:,2:end]))
    λ̂ = 3 - 2.25/r^2
    return θ/λ̂, r
end

@views function Stokes2D_coupled(ncx; ncy=ncx, Musi=1e-3, setup=:single, n_inc=30, seed=1, inc_margin=-Inf, scheme=:robust, κ=0.125, θ=0.75, nsm=10, c_fact=1.0, λest=:raw, λupd=:last, λgrow=2.0, λcap=10.0, γ̂=0.0, m_AA=0, AA_every=0, nupd=100, nupd_orig=1000, nout=100, ϵ=1e-5, itmax_fac=400, do_plot=false, verbose=false)
    # Physics
    Lx, Ly  = 10., 10.       # domain size
    εbg     = 1.0            # background strain-rate
    # Numerics
    iterMax = itmax_fac*ncx
    Δx, Δy  = Lx/ncx, Ly/ncy
    # Arrays
    Pt      = zeros(ncx  ,ncy  )
    ∇V      = zeros(ncx  ,ncy  )
    Vx      = zeros(ncx+1,ncy+2)
    Vy      = zeros(ncx+2,ncy+1)
    Exx     = zeros(ncx  ,ncy  )
    Eyy     = zeros(ncx  ,ncy  )
    Exy     = zeros(ncx+1,ncy+1)
    Txx     = zeros(ncx  ,ncy  )
    Tyy     = zeros(ncx  ,ncy  )
    Txy     = zeros(ncx+1,ncy+1)
    Rx      = zeros(ncx-1,ncy  )
    Ry      = zeros(ncx  ,ncy-1)
    Rp      = zeros(ncx  ,ncy  )
    Rx0     = zeros(ncx-1,ncy  )
    Ry0     = zeros(ncx  ,ncy-1)
    dVxdt   = zeros(ncx-1,ncy  )
    dVydt   = zeros(ncx  ,ncy-1)
    dVx     = zeros(ncx-1,ncy  )
    dVy     = zeros(ncx  ,ncy-1)
    dPt     = zeros(ncx  ,ncy  )
    ηc      = zeros(ncx  ,ncy  )
    ηv      = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx ,ncy  )
    # Coordinates
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    # Inclusions in a matrix of viscosity 1, smoothed with nsm passes
    ηc_sharp .= 1.0
    if setup == :single      # one circular inclusion of viscosity Musi
        ηc_sharp[xc.^2 .+ (yc').^2 .< 1.0^2] .= Musi
    elseif setup == :multi   # n_inc random circles, alternately Musi and 1/Musi (later ones overwrite overlaps)
        rng = MersenneTwister(seed)
        for k in 1:n_inc
            xi, yi = (rand(rng)-0.5)*Lx, (rand(rng)-0.5)*Ly
            ri     = 0.2 + 0.6*rand(rng)
            # inc_margin ≥ 0: keep every circle at least inc_margin away from the walls
            xi     = clamp(xi, -Lx/2 + ri + inc_margin, Lx/2 - ri - inc_margin)
            yi     = clamp(yi, -Ly/2 + ri + inc_margin, Ly/2 - ri - inc_margin)
            ηc_sharp[(xc.-xi).^2 .+ (yc'.-yi).^2 .< ri^2] .= isodd(k) ? Musi : 1/Musi
        end
    else
        error("Unknown setup $(setup)")
    end
    ηc       .= ηc_sharp
    for ism=1:nsm
        smooth!(ηc, ηc_sharp, 1.0)
        ηc_sharp, ηc = ηc, ηc_sharp
    end
    ηv[2:end-1,2:end-1] .= av(ηc)
    # Pseudo-time steps
    # Augmented-Lagrangian penalty γ = γ̂η (robust scheme only): adds γ∇(∇⋅V) to the momentum equation
    γc  = scheme == :robust ? γ̂ .* ηc : zero(ηc)
    DVx, DVy, λmaxVx, λmaxVy, λmaxVx_o, λmaxVy_o = Gershgorin_Stokes2D(ηc, ηv, Δx, Δy, ncx, ncy; γ=γc)
    if scheme == :original
        dtVx = 2.0 ./ sqrt.(λmaxVx_o) * 0.98
        dtVy = 2.0 ./ sqrt.(λmaxVy_o) * 0.98
        dtPt = 2 * maximum(ηc) / maximum((2/Δx + 2/Δy) .* ηc)   # = Δx/2 for Δx = Δy
        cVx  = zeros(ncx-1,ncy); cVy = zeros(ncx,ncy-1)          # no damping before the first estimate
    elseif scheme == :robust
        dtVx = 2.0 ./ sqrt.(λmaxVx) * 0.98
        dtVy = 2.0 ./ sqrt.(λmaxVy) * 0.98
        dτ   = min(minimum(dtVx), minimum(dtVy))                 # pseudo-time step used for P
        κ == :auto && ((κ, r) = κ_auto(ηc, θ))
        # initial damping from the lowest Laplacian mode, refined on the fly
        λmin0 = π^2*(1/Lx^2 + 1/Ly^2) / (2/Δx^2 + 2/Δy^2)
        λmin  = λmin0
        c     = 2*sqrt(λmin)*c_fact
    else
        error("Unknown scheme $(scheme)")
    end
    # Initial condition: pure shear
    Vx .=   εbg.*xv .+ 0 .*yce'
    Vy .= .-εbg.*yv' .+ 0 .*xce
    errVx0 = errVy0 = errPt0 = 1.0
    # Anderson acceleration of P: each window of AA_every iterations is a fixed-point map x -> g(x)
    nP_AA  = length(Pt); mA = max(m_AA, 1)
    ΔG_AA, ΔF_AA = zeros(nP_AA, mA), zeros(nP_AA, mA)
    x_AA, g_prev, f_prev = vec(copy(Pt)), zeros(nP_AA), zeros(nP_AA)
    k_AA   = 0
    AA_every == 0 && (AA_every = ncx)
    iter, err = 0, 1.0
    err_evo = Float64[]; it_evo = Int[]
    while err > ϵ && iter < iterMax
        iter += 1
        Rx0  .= Rx
        Ry0  .= Ry
        # Boundaries (free slip)
        Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
        Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
        # Kinematics and stress
        ∇V  .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
        Exx .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
        Eyy .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
        Exy .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
        Txx .= 2.0.*ηc.*Exx .+ γc.*∇V    # γc = 0 unless augmented Lagrangian
        Tyy .= 2.0.*ηc.*Eyy .+ γc.*∇V
        Txy .= 2.0.*ηv.*Exy
        # Jacobi-preconditioned residuals; pressure residual scaled by η
        Rx  .= (1.0./DVx).*(.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
        Ry  .= (1.0./DVy).*(.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
        Rp  .= .-∇V.*ηc
        # Errors: momentum relative to the initial residual, divergence relative to the background strain rate
        # (the initial pure-shear field is divergence free, so the initial ∇⋅V cannot serve as a reference)
        if iter == 1 || mod(iter, nout) == 0
            errVx = norm(DVx.*Rx)/sqrt(length(Rx)); errVy = norm(DVy.*Ry)/sqrt(length(Ry)); errPt = norm(∇V)/sqrt(length(∇V))
            if iter == 1 errVx0 = errVx; errVy0 = errVy; errPt0 = abs(εbg) end
            err = max(errVx/errVx0, errVy/errVy0, errPt/errPt0)
            push!(err_evo, err); push!(it_evo, iter)
            verbose && @printf("iter = %06d, iter/nx = %6.1f, err = %1.3e [Vx=%1.3e Vy=%1.3e P=%1.3e]\n", iter, iter/ncx, err, errVx/errVx0, errVy/errVy0, errPt/errPt0)
            (isfinite(err) && err < 1e10) || break
        end
        if scheme == :original
            # Per-field damping from Euclidean Rayleigh quotients (as in the original script)
            if mod(iter, nupd_orig) == 0
                λminVx = abs(sum(dVx.*(Rx .- Rx0))) / sum(dVx.*dVx)
                λminVy = abs(sum(dVy.*(Ry .- Ry0))) / sum(dVy.*dVy)
                cVx .= 2*sqrt(λminVx)
                cVy .= 2*sqrt(λminVy)
            end
            dVxdt .= (2 .- cVx.*dtVx)./(2 .+ cVx.*dtVx).*dVxdt .+ 2 .*dtVx./(2 .+ cVx.*dtVx).*Rx
            dVydt .= (2 .- cVy.*dtVy)./(2 .+ cVy.*dtVy).*dVydt .+ 2 .*dtVy./(2 .+ cVy.*dtVy).*Ry
            dVx   .= dVxdt.*dtVx
            dVy   .= dVydt.*dtVy
            Vx[2:end-1,2:end-1] .+= dVx
            Vy[2:end-1,2:end-1] .+= dVy
            Pt                  .+= dtPt.*Rp
        else
            # One damping for the whole velocity field: D-weighted Rayleigh quotient of the last increment
            if mod(iter, nupd) == 0 && iter > 1
                # residual change of the last iteration; :viscous removes the part due to the pressure
                # increment (ΔR_V = -A ΔV - ∇ΔP), so that only the viscous block is probed.
                # :raw (default) keeps it: robust in all tests; :viscous was up to 2.5x faster for strong
                # contrasts but under-damped (slow oscillatory convergence) for a weak inclusion.
                if λest == :viscous
                    num = abs(sum(dVx.*(DVx.*(Rx .- Rx0) .+ (dPt[2:end,:] .- dPt[1:end-1,:])./Δx)) +
                              sum(dVy.*(DVy.*(Ry .- Ry0) .+ (dPt[:,2:end] .- dPt[:,1:end-1])./Δy)))
                else
                    num = abs(sum(dVx.*DVx.*(Rx .- Rx0)) + sum(dVy.*DVy.*(Ry .- Ry0)))
                end
                den  = sum(DVx.*dVx.^2) + sum(DVy.*dVy.^2)
                if den > 0
                    # :last (default) uses the latest estimate; :min keeps the running minimum (tested, NOT
                    # recommended: c far too small); :limited lets the estimate rise by at most λgrow per update;
                    # :capped bounds it by λcap × the lowest-Laplacian-mode estimate λmin0
                    λnew = num/den
                    λmin = λupd == :min     ? min(λmin, λnew) :
                           λupd == :limited ? min(λnew, λgrow*λmin) :
                           λupd == :capped  ? min(λnew, λcap*λmin0) : λnew
                    c    = 2*sqrt(λmin)*c_fact
                end
            end
            dVxdt .= (2 .- c.*dtVx)./(2 .+ c.*dtVx).*dVxdt .+ 2 .*dtVx./(2 .+ c.*dtVx).*Rx
            dVydt .= (2 .- c.*dtVy)./(2 .+ c.*dtVy).*dVydt .+ 2 .*dtVy./(2 .+ c.*dtVy).*Ry
            dVx   .= dVxdt.*dtVx
            dVy   .= dVydt.*dtVy
            Vx[2:end-1,2:end-1] .+= dVx
            Vy[2:end-1,2:end-1] .+= dVy
            dPt                  .= (κ*c*dτ).*Rp     # pressure rate tied to the damping
            Pt                  .+= dPt
            if m_AA > 0 && mod(iter, AA_every) == 0
                g = vec(copy(Pt)); f = g .- x_AA      # g(x) after one window, fixed-point residual f
                if k_AA > 0
                    slot = (k_AA - 1) % m_AA + 1
                    ΔG_AA[:, slot] .= g .- g_prev
                    ΔF_AA[:, slot] .= f .- f_prev
                end
                g_prev .= g; f_prev .= f; k_AA += 1
                n_AA = min(k_AA - 1, m_AA)
                if n_AA > 0
                    θ_AA = ΔF_AA[:, 1:n_AA] \ f        # least squares: θ = argmin ‖f − ΔF⋅θ‖
                    if all(isfinite, θ_AA)
                        Pt .= reshape(g .- ΔG_AA[:, 1:n_AA]*θ_AA, size(Pt))
                    else
                        k_AA = 0
                    end
                end
                x_AA .= vec(Pt)
            end
        end
    end
    converged = err <= ϵ
    if do_plot
        fig = Figure(size=(1000, 800))
        ax  = Axis(fig[1, 1], title="Pt", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yc, Pt, colormap=:inferno); Colorbar(fig[1, 2], hm)
        ax  = Axis(fig[1, 3], title="Vy", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yv, Vy[2:end-1,:], colormap=:inferno); Colorbar(fig[1, 4], hm)
        ax  = Axis(fig[2, 1], title="log10 ηc", xlabel="x", ylabel="y", aspect=DataAspect())
        hm  = heatmap!(ax, xc, yc, log10.(ηc), colormap=:inferno); Colorbar(fig[2, 2], hm)
        ax  = Axis(fig[2, 3], title="Convergence ($(scheme))", xlabel="iterations / nx", ylabel="log10 error")
        lines!(ax, it_evo./ncx, log10.(err_evo))
        hlines!(ax, log10(ϵ), color=:black, linestyle=:dash)
        display(fig)
    end
    return (; iter, converged, err, Pt, ηc, xc, yc)
end

# Benchmark: iterations / nx over resolution and inclusion viscosity
function benchmark(; ncxs=(63, 127, 255), Musis=(1e-3, 1e3), schemes=(:original, :robust), kwargs...)
    @printf("\n%9s | %6s | %10s | %12s | %s\n", "scheme", "ncx", "η_incl", "iter / nx", "status")
    for scheme in schemes, Musi in Musis, ncx in ncxs
        t0  = time()
        out = Stokes2D_coupled(ncx; Musi, scheme, kwargs...)
        @printf("%9s | %6d | %10.0e | %12.1f | %s (%.1f s)\n", scheme, ncx, Musi, out.iter/ncx,
            out.converged ? "converged" : (isfinite(out.err) ? @sprintf("NOT converged (err=%.1e)", out.err) : "diverged"), time()-t0)
    end
end

benchmark()
