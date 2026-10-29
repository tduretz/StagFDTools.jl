using CairoMakie, LinearAlgebra, JLD2, StaticArrays
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian

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

    x   = @MVector[0.0, 0.0 ]
    r0  = 1.0
    tol = 1e-13

    for iter=1:10
        J = forwarddiff_jacobian(ΔPΦ_residual, x, Const(λ̇), Const(Pt0), Const(Pf0), Const(Φ0), Const(p))
        x .-= J.derivs[1]\J.val
        nr = mynorm(J.val)
        @show J.val
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
    α1    = if p.single_phase
                0.0
            else
                1.0
            end 
    
    # Check yield
    f       = F(τII, Pt, Pf, λ̇, α1, c, sinϕ, cosϕ, ηvp) 

    # Porosity rate
    dΦdt    = PorosityRate(Φ0, Pt, Pf, λ̇, Pt0, Pf0, p)[1]

    # Form 1 - requires one additional solve: here it's done by hand
    ΔP = [
        KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ),
        Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    ]
    rpt = Pt - (Pt_trial + ΔP[1])
    rpf = Pf - (Pf_trial + ΔP[2])

    # !!!! It would be better to have this version working 
    # Form 2 - requires one additional solve: one more nested AD loop
    # ΔP  = ΔPΦ(λ̇, 0.0*Pt0, 0.0*Pf0, Φ0, p)
    # rpt = Pt - (Pt_trial + ΔP[1])
    # rpf = Pf - (Pf_trial + ΔP[2])

    # Form 3 - needs to build full continuity does not give the correct P trial dependence
    # dPfdt   = (Pf - Pf0) / Δt
    # dlnρfdt = dPfdt / Kf
    # dPsdt   = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    # # dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    # dlnρsdt = 1/Ks * dPsdt 
    # rpt = dlnρsdt - dΦdt/(1-Φ) + divVs
    # rpf = Φ*dlnρfdt + dΦdt     + Φ*divVs + divqD

    return @SVector[ 
        ε̇II_eff   -  τII/2/ηve - λ̇/2*(f>=eps),
        rpt,
        rpf,
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps),
        Φ    - (Φ0 + dΦdt*Δt),
    ]
end

function StressVector_trial(ϵ̇, divVs, divqD, τ0, Pt0, Pf0, Φ0, p)

    ε̇_eff, Pt, Pf = ϵ̇[1:3], ϵ̇[4], ϵ̇[5]
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
    α1        = if p.single_phase
        0.0
    else
        1.0
    end 

    # Initial residual
    r, r0, tol = 0.0, 0.0, 1e-13

    # Check yield
    f       = F(τII_trial, Pt_trial, Pf_trial, λ̇_trial, α1, c, sinϕ, cosϕ, ηvp) 

    # Initial guess
    x = @MVector([τII_trial, Pt_trial, Pf_trial, 0.0, Φ_trial])

    if f>-tol
        @info "plastic"
        # This is the proper return mapping with plasticity

        for iter=1:10
            J = forwarddiff_jacobian(residual_two_phase_trial, x, Const(ε̇II_eff), Const(Pt_trial), Const(Pf_trial), Const(Φ_trial), Const(divVs), Const(divqD), Const(Pt0), Const(Pf0), Const(Φ0), Const(p))
            # display(J.derivs[1])
            x .-= J.derivs[1]\J.val
            if iter==1 
                r0 = norm(J.val)
            end
            r = norm(J.val)/r0
            # @show iter, r
            if r<tol
                break
            end
        end
    end

    τII, Pt, Pf, λ̇, Φ1   = x[1], x[2], x[3], x[4], x[5]

     # Recompute components and get out of here
    τ = ε̇_eff .* τII./ε̇II_eff

    return @SVector[τ[1], τ[2], τ[3], Pt, Pf], λ̇, Φ1, r 
end

function two_phase_return_mapping()

    sc = (σ=1e7, t=1e10, L=1e3)

    # Kinematics
    ε̇     = [2e-15,-2e-15, 0].*sc.t
    divVs =  0*1e-14 .*sc.t
    divqD = -0*1e-14 .*sc.t

    # Initial conditions
    Pt   = 1e6/sc.σ
    Pf   = 1e6/sc.σ 
    τ    = [0.0, -0.0, 0]./sc.σ
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

        @info "Step $(it)"

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
        @info "Determine trial pressure"
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
        @info "Calculate rheology as function of trial pressures"

        # Local input array includes trial pressures 
        ϵ̇          = @SVector[ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], Pt, Pf]

        # Stress evaluation
        σ, λ̇, Φ, r = StressVector_trial(ϵ̇, divVs, divqD, τ0, Pt0, Pf0, Φ0, params)
        τ, Pt, Pf  = σ[1:3], σ[4], σ[5]
        @show τ, Pt, Pf, Φ

        # Consistent tangent by deriving stress w.r.t. input vector 
        J = forwarddiff_jacobian(StressVector_trial, ϵ̇,  Const(divVs), Const(divqD), Const(τ0), Const(Pt0),  Const(Pf0), Const(Φ0), Const(params))

        @views D_ctl_trial[:,1] .= J.derivs[1][1][1]
        @views D_ctl_trial[:,2] .= J.derivs[1][2][1]
        @views D_ctl_trial[:,3] .= J.derivs[1][3][1]
        @views D_ctl_trial[:,4] .= J.derivs[1][4][1]
        @views D_ctl_trial[:,5] .= J.derivs[1][5][1]

        display(D_ctl_trial)

        #########################################################

        # Probes
        probes.t[it]  = it*params.Δt
        probes.τ[it]  = invII(τ)
        probes.Pt[it] = Pt
        probes.Pf[it] = Pf
        probes.Pe[it] = Pt - Pf
        probes.λ̇[it]  = λ̇ 
        probes.Φ[it]  = Φ
        probes.r[it]  = r
    end

    function figure()
        fig = Figure(fontsize = 20, size = (600, 800) )     
        ax1 = Axis(fig[1,1], title="Deviatoric stress",  xlabel=L"$t$ [yr]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)
        lines!(ax1, probes.t[1:nt]*sc.t, probes.τ[1:nt]*sc.σ)

        ax2 = Axis(fig[2,1], title="Pressure",  xlabel=L"$t$ [yr]",  ylabel=L"$P$ [MPa]", xlabelsize=20, ylabelsize=20)
        lines!(ax2, probes.t[1:nt]*sc.t, probes.Pt[1:nt]*sc.σ)
        lines!(ax2, probes.t[1:nt]*sc.t, probes.Pf[1:nt]*sc.σ)

        ax3 = Axis(fig[3,1], title="Plastic multiplier",  xlabel=L"$t$ [yr]",  ylabel=L"$\dot{\lambda}$ [1/s]", xlabelsize=20, ylabelsize=20)    
        lines!(ax3, probes.t[1:nt]*sc.t, probes.λ̇[1:nt]/sc.t)
   
        ax4 = Axis(fig[4,1], title="Porosity",  xlabel=L"$t$ [yr]",  ylabel=L"$\phi$", xlabelsize=20, ylabelsize=20)    
        lines!(ax4, probes.t[1:nt]*sc.t, probes.Φ[1:nt])

        ax5 = Axis(fig[5,1], title="Residual",  xlabel=L"$t$ [yr]",  ylabel=L"$r$", xlabelsize=20, ylabelsize=20)    
        scatter!(ax5, probes.t[1:nt]*sc.t, log10.(probes.r[1:nt]))
        display(fig)
    end
    with_theme(figure, theme_latexfonts())
end

two_phase_return_mapping()