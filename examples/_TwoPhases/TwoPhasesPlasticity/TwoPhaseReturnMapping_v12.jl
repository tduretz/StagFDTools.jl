using CairoMakie, LinearAlgebra, JLD2, StaticArrays
using StagFDTools#: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
using DifferentiationInterface
import ForwardDiff: ForwardDiff
import ForwardDiff as FD

const benchmark = true

# Intends to implement constitutive updates as in RheologicalCalculator

# use the trial to determine corrected pressures

# for pratical allocation free implemetation with Albert

@inline mynorm(x) = sum(xi^2 for xi in x)

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + 1/2*(-x[1]-x[2])^2 + x[3]^2) 

# Drucker-Prager
F(τII, Pt, Pf, λ̇, α1, c, sinϕ, cosϕ, ηvp) = τII - (Pt - α1*Pf)*sinϕ - c*cosϕ - λ̇*ηvp

# Non-linear bulk viscosity
bulk_viscosity(ϕ, η0, m) = η0*abs(ϕ)^m

# The porosity rate is non-linear because bulk parameters (ηΦ) is a function of Φ
function PorosityRate(Φ, Pt, Pf, λ̇, Pt0, Pf0, p)
    KΦ, ξ0, m, sinψ, Δt  = p.KΦ, p.ηΦ, p.m, p.sinψ, p.Δt
    ηΦ      = bulk_viscosity(Φ, ξ0, m)
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = ((dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sinψ) * 1
    return dΦdt, ηΦ
end

# Equation of states solid and fluid
function EOS(Pt, Pf, Φ, Pt0, Pf0, Φ0, p)
    Ks, Kf, Δt = p.Ks, p.Kf, p.Δt
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dlnρfdt = dPfdt / Kf
    # Approximation in Yarushina ≈
    dPsdt   = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    # Exact, but non linear
    # dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1/Ks * dPsdt 
    return dlnρsdt, dlnρfdt
end

# This function compute plastic pressure increments as function of multiplier
# It allows to write up corrected pressure as P_corr = Pt_trial + ΔP
function ΔPΦ_residual(x, λ̇, Pt0, Pf0, Φ0, p)

    Pt, Pf  = x[1], x[2]
    Φ       = Φ0 # one could make phi a variable too

    # Porosity rate
    dΦdt, ηΦ = PorosityRate(Φ, Pt, Pf, λ̇, Pt0, Pf0, p)

    # Equations of state
    dlnρsdt, dlnρfdt = EOS(Pt, Pf, Φ, Pt0, Pf0, Φ0, p)

    return @SVector [ 
        dlnρsdt - dΦdt/(1-Φ),
        (Φ*dlnρfdt + dΦdt),
    ]
end

function ΔPΦ(λ̇, Pt0, Pf0, Φ0, p)

    x   = @SVector[0.0, 0.0 ]
    r0  = 1.0
    tol = 1e-13

    for iter=1:10
        r, J  = value_and_jacobian(x -> ΔPΦ_residual(x, λ̇, Pt0, Pf0, Φ0, p), AutoForwardDiff(), x)
        x -= J\r
        nr = mynorm(r)
        # @show J
        if iter==1 && nr>1e-17
            r0 = nr
        end
        r = nr/r0
        if r<tol
            break
        end
    end
    return x
end

function residual_two_phase_trial(x, ε̇II_eff, Pt_trial, Pf_trial, Φ_trial, divVs, divqD, Pt0, Pf0, Φ0, p)
    
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]
    G, KΦ, Ks, Kf, c, sinϕ, cosϕ, sinψ, ηvp, ηv, ηΦ, Δt = p.G, p.KΦ, p.Ks, p.Kf, p.C, p.sinϕ, p.cosϕ, p.sinψ, p.ηvp, p.ηs, p.ηΦ, p.Δt
    eps   = -1e-13
    ηe    = G*Δt 
    ηve   = inv(1/ηv + 1/ηe)
    α1    = !(p.single_phase)
    
    # Check yield
    f       = F(τII, Pt, Pf, λ̇, α1, c, sinϕ, cosϕ, ηvp) 

    # Porosity rate
    dΦdt    = PorosityRate(Φ, Pt, Pf, λ̇, Pt0, Pf0, p)[1]

    # Φ1 = Φ0 + dΦdt*Δt


    # # Form 1 - requires one additional solve: here it's done by hand
    # ΔP = SA[
    #     KΦ .* sinψ .* Δt .* Φ1 .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ1 + Kf .* KΦ .* Δt - Kf .* Φ1 .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ1 + Ks .* Φ1 .* ηΦ + KΦ .* Φ1 .* ηΦ),
    #     Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ1 - Kf .* KΦ .* Δt + Kf .* Φ1 .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ1 - Ks .* Φ1 .* ηΦ - KΦ .* Φ1 .* ηΦ)
    # ]
    # rpt = Pt - (Pt_trial + ΔP[1])
    # rpf = Pf - (Pf_trial + ΔP[2])

    # !!!! It would be better to have this version working 
    # Form 2 - requires one additional solve: one more nested AD loop
    # ΔP  = ΔPΦ(λ̇, 0.0*Pt0, 0.0*Pf0, Φ0, p)
    # rpt = Pt - (Pt_trial + ΔP[1])
    # rpf = Pf - (Pf_trial + ΔP[2])

    # Form 3 - needs to build full continuity does not give the correct P trial dependence
    dPfdt   = (Pf - Pf0) / Δt
    dPtdt   = (Pt - Pt0) / Δt 
    dlnρfdt = dPfdt / Kf
    # dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    # dlnρsdt = 1/Ks * dPsdt 
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    rpt = dlnρsdt - dΦdt/(1-Φ) + divVs
    rpf = Φ*dlnρfdt + dΦdt     + Φ*divVs + divqD

    return @SVector[ 
        ε̇II_eff   -  τII/2/ηve - λ̇/2*(f>=eps),
        rpt,
        rpf,
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps),
        Φ    - (Φ0 + dΦdt*Δt),
    ]
end

function StressVector_trial(ϵ̇::SVector{N,T}, divVs, divqD, τ0, Pt0, Pf0, Φ0, p) where {N,T}

    ε̇_eff  = SVector{3,T}(ϵ̇[i] for i in 1:3)
    Pt, Pf = ϵ̇[4], ϵ̇[5]
    ηv, G, Δt, c, sinϕ, cosϕ, ηvp = p.ηs, p.G, p.Δt, p.C, p.sinϕ, p.cosϕ, p.ηvp
    
    # Rheology update
    ηe        = G*Δt 
    ε̇II_eff   = invII(ε̇_eff) 
    ηve       = inv(1/ηv + 1/ηe) 
    τII_trial = 2 * ηve*ε̇II_eff
    Pt_trial  = Pt
    Pf_trial  = Pf
    Φ_trial   = Φ0
    λ̇_trial   = 0.0
    α1        = !p.single_phase

    # Initial residual
    nr, nr0, tol, r = 0.0, 0.0, 1e-13, 1

    # Check yield
    f       = F(τII_trial, Pt_trial, Pf_trial, λ̇_trial, α1, c, sinϕ, cosϕ, ηvp) 

    # Initial guess
    x = @SVector([τII_trial, Pt_trial, Pf_trial, 0.0, Φ_trial])

    if f>-tol
        # @info "plastic"
        # This is the proper return mapping with plasticity

        for iter=1:10
            r, J = value_and_jacobian(x -> residual_two_phase_trial(x, ε̇II_eff, Pt_trial, Pf_trial, Φ_trial, divVs, divqD, Pt0, Pf0, Φ0, p), AutoForwardDiff(), x)
            # display(J.derivs[1])
            x -= J\r
            if iter==1 
                nr0 = norm(J)
            end
            nr = norm(r)/nr0
            # @show iter, nr, r
            if nr<tol
                break
            end
        end
    end

    τII, Pt, Pf, λ̇, Φ1   = x[1], x[2], x[3], x[4], x[5]

    # Recompute components and get out of here
    τ = @. ε̇_eff * τII / ε̇II_eff

    #### All checks!!!
    typeof(λ̇)==Float64 && @info "Post solve residual"
    typeof(λ̇)==Float64 && @show r

    KΦ, ηΦ, sinψ, Δt = p.KΦ, p.ηΦ, p.sinψ, p.Δt
    Kf, Ks = p.Kf, p.Ks 

    #### Check residual with trial state pressures: it is also zero !!!
    typeof(λ̇)==Float64 && @info "Trial pressure formulation: general form"
    dPtdt   = (Pt_trial - Pt0) / Δt
    dPfdt   = (Pf_trial - Pf0) / Δt
    dΦdt    = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf_trial - Pt_trial)
    Φ       = Φ_trial
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    f1      = dlnρsdt   - dΦdt/(1-Φ) +   divVs
    f2      = Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD
    typeof(f1)==Float64 && @show f1, f2

    ### Check residuals with corrected pressures: should be zero !

    # General form
    typeof(λ̇)==Float64 && @info "True pressure formulation: general form"
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf - Pt) + λ̇*sinψ
    Φ       = Φ0 + Δt * dΦdt 
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    f1      = dlnρsdt   - dΦdt/(1-Φ) +   divVs
    f2      = Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD
    typeof(f1)==Float64 && @show f1, f2

    # Specific form 
    typeof(λ̇)==Float64 && @info "True pressure formulation: specific form 1"
    Kd = (1-Φ)*(1/KΦ + 1/Ks)^-1
    α  = 1 - Kd/Ks
    B  = (1/Kd - 1/Ks) / (1/Kd - 1/Ks + Φ*(1/Kf - 1/Ks))
    f1 = divVs     + 1/Kd*(dPtdt -   α*dPfdt) - 1/(1-Φ)*λ̇*sinψ + (Pt-Pf)/((1-Φ)*ηΦ)
    f2 = divqD     - α/Kd*(dPtdt - 1/B*dPfdt) + 1/(1-Φ)*λ̇*sinψ - (Pt-Pf)/((1-Φ)*ηΦ)
    typeof(f1)==Float64 && @show f1, f2

    # Specific form (rederived)
    typeof(λ̇)==Float64 && @info "True pressure formulation: specific form 2"
    f1 = divVs    + (1/Ks)/(1-Φ) * (dPtdt - Φ*dPfdt) + (1/KΦ)/(1-Φ) * (dPtdt - dPfdt) + (Pt-Pf)/((1-Φ)*ηΦ) - 1/(1-Φ)*λ̇*sinψ
    f2 = divqD    - (dPtdt - dPfdt)/KΦ + Φ*dPfdt/Kf + Φ*divVs - (Pt-Pf)/ηΦ + λ̇*sinψ
    typeof(f1)==Float64 && @show f1, f2

    return @SVector[τ[1], τ[2], τ[3], Pt, Pf, λ̇, Φ1, nr]
end

function two_phase_return_mapping()

    sc = (σ=1e7, t=1e10, L=1e3)

    # Kinematics
    ε̇     = SA[2e-15,-2e-15, 0].*sc.t
    divVs =  0*1e-14 .*sc.t
    divqD = -0*1e-14 .*sc.t

    # Initial conditions
    Pt   = 1e6/sc.σ
    Pf   = 1e6/sc.σ 
    τ    = SA[0.0, -0.0, 0]./sc.σ
    Φ    = 0.05 

    # Parameters
    nt = 30
    
    params = (
        G       = 3e10/sc.σ,
        Ks      = 1e11/sc.σ,
        KΦ      = 1e10/sc.σ,
        Kf      = 1e9/sc.σ,
        C       = 1e7 /sc.σ,
        sinϕ    = sind(35.0),
        cosϕ    = cosd(35.0),
        sinψ    = sind(10.0),
        ηvp     = 0/sc.σ/sc.t,
        ηs      = 1e22/sc.σ/sc.t,
        ηΦ      = 2e22/sc.σ/sc.t,
        m       = 0.0,
        Δt      = 1e10/sc.t,
        single_phase = false,
    )  

    # Probes
    probes = (
        τ  = zeros(nt),
        Pt = zeros(nt),
        Pf = zeros(nt),
        Pe = zeros(nt),
        t  = zeros(nt),
        λ̇  = zeros(nt),
        Φ  = zeros(nt),
        r  = zeros(nt),
    )

    D_ctl_trial = @MMatrix zeros(5,5)
    D_ctl_div   = @MMatrix zeros(5,5)
    C           = @MMatrix zeros(5,5)
    C[diagind(C)] .= 1.0

    # Time loop
    for it=1:nt

        # @info "Step $(it)"

        # Old guys
        Pt0    = Pt
        Pf0    = Pf
        τ0     = τ
        Φ0     = Φ

        # Effective deviatoric strain rate
        ε̇_eff      = ε̇ + τ0/(2*params.G*params.Δt)

        #########################################################
        # Trial pressure determination, in practive this comes from the solver
        # No need to optimise anything here
        # @info "Determine trial pressure"
        K_s     = params.Ks
        K_f     = params.Kf
        K_phi   = params.KΦ
        eta_phi = params.ηΦ
        dt      = params.Δt
        phi_0   = Φ0
        phi     = Φ0
        for it=1:10
            Pt = (-K_f .* K_phi .* K_s .* divVs .* dt .^ 2 - K_f .* K_phi .* K_s .* divqD .* dt .^ 2 - K_f .* K_phi .* Pf0 .* dt .* phi + K_f .* K_phi .* Pt0 .* dt - K_f .* K_phi .* divVs .* dt .* eta_phi .* phi .^ 2 - K_f .* K_phi .* divqD .* dt .* eta_phi .* phi - K_f .* K_s .* divVs .* dt .* eta_phi - K_f .* K_s .* divqD .* dt .* eta_phi - K_f .* Pt0 .* eta_phi .* phi + K_f .* Pt0 .* eta_phi + K_phi .* K_s .* Pf0 .* dt .* phi + K_phi .* K_s .* divVs .* dt .* eta_phi .* phi .^ 2 - K_phi .* K_s .* divVs .* dt .* eta_phi .* phi + K_phi .* Pt0 .* eta_phi .* phi + K_s .* Pt0 .* eta_phi .* phi) ./ (-K_f .* K_phi .* dt .* phi + K_f .* K_phi .* dt - K_f .* eta_phi .* phi + K_f .* eta_phi + K_phi .* K_s .* dt .* phi + K_phi .* eta_phi .* phi + K_s .* eta_phi .* phi)
            Pf = (-K_f .* K_phi .* K_s .* divVs .* dt .^ 2 - K_f .* K_phi .* K_s .* divqD .* dt .^ 2 - K_f .* K_phi .* Pf0 .* dt .* phi + K_f .* K_phi .* Pt0 .* dt - K_f .* K_phi .* divVs .* dt .* eta_phi .* phi - K_f .* K_phi .* divqD .* dt .* eta_phi - K_f .* K_s .* divVs .* dt .* eta_phi - K_f .* K_s .* divqD .* dt .* eta_phi - K_f .* Pf0 .* eta_phi .* phi + K_f .* Pf0 .* eta_phi + K_phi .* K_s .* Pf0 .* dt .* phi + K_phi .* Pf0 .* eta_phi .* phi + K_s .* Pf0 .* eta_phi .* phi) ./ (-K_f .* K_phi .* dt .* phi + K_f .* K_phi .* dt - K_f .* eta_phi .* phi + K_f .* eta_phi + K_phi .* K_s .* dt .* phi + K_phi .* eta_phi .* phi + K_s .* eta_phi .* phi)
            Φ  = (K_phi .* dt .* (Pf - Pt) + K_phi .* eta_phi .* phi_0 + eta_phi .* (Pf - Pf0 - Pt + Pt0)) ./ (K_phi .* eta_phi)
        end
        #########################################################
        # Rheological calculation: this is the crucial part
        # Need to optimise everything here
        # @info "Calculate rheology as function of trial pressures"

        # Local input array includes trial pressures 
        ϵ̇          = @SVector[ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], Pt, Pf]

        # Stress evaluation
        @time σ  = StressVector_trial(ϵ̇, divVs, divqD, τ0, Pt0, Pf0, Φ0, params)
        τ, Pt, Pf  = SA[σ[1], σ[2], σ[3]], σ[4], σ[5]
        λ̇, Φ, nr = σ[6], σ[7], σ[8]
        # @show τ, Pt, Pf, Φ

        # Consistent tangent by deriving stress w.r.t. input vector 
        @time pseudo_J = jacobian(ϵ̇->StressVector_trial(ϵ̇,  divVs, divqD, τ0, Pt0,  Pf0, Φ0, params), AutoForwardDiff(), ϵ̇) # pseudo_J because the 3 last rows are not really part of ∂τ/∂ϵ̇
        J = SMatrix{5,5}(pseudo_J[i,j] for i in 1:5, j in 1:5) # get rid of last 3 rows

        #########################################################

        # Probes
        probes.t[it]  = it*params.Δt
        probes.τ[it]  = invII(τ)
        probes.Pt[it] = Pt
        probes.Pf[it] = Pf
        probes.Pe[it] = Pt - Pf
        probes.λ̇[it]  = λ̇ 
        probes.Φ[it]  = Φ
        probes.r[it]  = nr
    end

    function figure()
        data = benchmark && load("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_remix.jld2")

        fig = Figure(fontsize = 20, size = (600, 800) )     
        ax1 = Axis(fig[1,1], title="Deviatoric stress",  xlabel=L"$t$ [yr]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)
        lines!(ax1, probes.t[1:nt]*sc.t, probes.τ[1:nt]*sc.σ./1e6)
        benchmark && scatter!(ax1, data["probes"].t[1:nt], data["probes"].τ[1:nt]./1e6, marker=:xcross)

        ax2 = Axis(fig[2,1], title="Pressure",  xlabel=L"$t$ [yr]",  ylabel=L"$P$ [MPa]", xlabelsize=20, ylabelsize=20)
        lines!(ax2, probes.t[1:nt]*sc.t, probes.Pt[1:nt]*sc.σ)
        lines!(ax2, probes.t[1:nt]*sc.t, probes.Pf[1:nt]*sc.σ)
        benchmark && scatter!(ax2, data["probes"].t[1:nt], data["probes"].Pt[1:nt], marker=:xcross)
        benchmark && scatter!(ax2, data["probes"].t[1:nt], data["probes"].Pf[1:nt], marker=:xcross)

        ax3 = Axis(fig[3,1], title="Plastic multiplier",  xlabel=L"$t$ [yr]",  ylabel=L"$\dot{\lambda}$ [1/s]", xlabelsize=20, ylabelsize=20)    
        lines!(ax3, probes.t[1:nt]*sc.t, probes.λ̇[1:nt]/sc.t)
        benchmark && scatter!(ax3, data["probes"].t[1:nt], data["probes"].λ̇[1:nt], marker=:xcross)

        ax4 = Axis(fig[4,1], title="Porosity",  xlabel=L"$t$ [yr]",  ylabel=L"$\phi$", xlabelsize=20, ylabelsize=20)    
        lines!(ax4, probes.t[1:nt]*sc.t, probes.Φ[1:nt])
        benchmark && scatter!(ax4, data["probes"].t[1:nt], data["probes"].Φ[1:nt], marker=:xcross)

        ax5 = Axis(fig[5,1], title="Residual",  xlabel=L"$t$ [yr]",  ylabel=L"$r$", xlabelsize=20, ylabelsize=20)    
        scatter!(ax5, probes.t[1:nt]*sc.t, log10.(probes.r[1:nt]))
        display(fig)
    end
    with_theme(figure, theme_latexfonts())
end

two_phase_return_mapping()
# julia> @time two_phase_return_mapping()
#    0.000278 seconds (526 allocations: 85.266 KiB)
