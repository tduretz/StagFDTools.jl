using CairoMakie, LinearAlgebra, JLD2, StaticArrays
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian

# Intends to implement constitutive updates as in RheologicalCalculator

# use the trial to determine corrected pressures

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + 1/2*(-x[1]-x[2])^2 + x[3]^2) 

function residual_two_phase(x, ε̇II_eff, Pt_trial, Pf_trial, Φ_trial, Pt0, Pf0, Φ0, p)
    
    G, KΦ, Ks, Kf, C, ϕ, ψ, ηvp, ηv, ηΦ, Δt = p.G, p.KΦ, p.Ks, p.Kf, p.C, p.ϕ, p.ψ, p.ηvp, p.ηs, p.ηΦ, p.Δt
    eps   = -1e-13
    ηe    = G*Δt 
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]
    ηve  = inv(1/ηv + 1/ηe)
    sinψ = sind(ψ)
    
    # Pressure corrections
    ΔPt = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
    ΔPf = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    
    # Check yield
    # f       = τII - (1-Φ)*C*cosd(ϕ) - (Pt - Pf)*sind(ϕ)
    f       = τII - C*cosd(ϕ) - (Pt - Pf)*sind(ϕ)

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sind(ψ)*(f>=eps)


    return [ 
        ε̇II_eff   -  τII/2/ηve - λ̇/2*(f>=eps),
        Pt - (Pt_trial + ΔPt),
        Pf - (Pf_trial + ΔPf),
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps),
        Φ    - (Φ0 + dΦdt*Δt),
    ]
end

function StressVector(ϵ̇, τ0, Pt0, Pf0, Φ0, params)

    ε̇_eff = ϵ̇[1:3]
    divVs = ϵ̇[4]
    divqD = ϵ̇[5]

    ηv, G, Δt, C, ϕ = params.ηs, params.G, params.Δt, params.C, params.ϕ
    ηe    = G*Δt 

    ε̇II_eff = invII(ε̇_eff) 
    
    # Rheology update
    ηve = inv(1/ηv + 1/ηe) 
    τII_trial = 2 * ηve*ε̇II_eff

    # Predict pressures from trial state (comes from global solver)
    K_s     = params.Ks
    K_f     = params.Kf
    K_phi   = params.KΦ
    eta_phi = params.ηΦ
    dt = params.Δt
    phi_0   = Φ0
    Pt_trial, Pf_trial, Φ_trial = Pt0, Pf0, Φ0

    for it=1:10
        Pf  = Pf_trial
        Pt  = Pt_trial
        phi =  Φ_trial
        Pt_trial = (-K_f .* K_phi .* K_s .* divVs .* dt .^ 2 - K_f .* K_phi .* K_s .* divqD .* dt .^ 2 - K_f .* K_phi .* Pf0 .* dt .* phi + K_f .* K_phi .* Pt0 .* dt - K_f .* K_phi .* divVs .* dt .* eta_phi .* phi .^ 2 - K_f .* K_phi .* divqD .* dt .* eta_phi .* phi - K_f .* K_s .* divVs .* dt .* eta_phi - K_f .* K_s .* divqD .* dt .* eta_phi - K_f .* Pt0 .* eta_phi .* phi + K_f .* Pt0 .* eta_phi + K_phi .* K_s .* Pf0 .* dt .* phi + K_phi .* K_s .* divVs .* dt .* eta_phi .* phi .^ 2 - K_phi .* K_s .* divVs .* dt .* eta_phi .* phi + K_phi .* Pt0 .* eta_phi .* phi + K_s .* Pt0 .* eta_phi .* phi) ./ (-K_f .* K_phi .* dt .* phi + K_f .* K_phi .* dt - K_f .* eta_phi .* phi + K_f .* eta_phi + K_phi .* K_s .* dt .* phi + K_phi .* eta_phi .* phi + K_s .* eta_phi .* phi)
        Pf_trial = (-K_f .* K_phi .* K_s .* divVs .* dt .^ 2 - K_f .* K_phi .* K_s .* divqD .* dt .^ 2 - K_f .* K_phi .* Pf0 .* dt .* phi + K_f .* K_phi .* Pt0 .* dt - K_f .* K_phi .* divVs .* dt .* eta_phi .* phi - K_f .* K_phi .* divqD .* dt .* eta_phi - K_f .* K_s .* divVs .* dt .* eta_phi - K_f .* K_s .* divqD .* dt .* eta_phi - K_f .* Pf0 .* eta_phi .* phi + K_f .* Pf0 .* eta_phi + K_phi .* K_s .* Pf0 .* dt .* phi + K_phi .* Pf0 .* eta_phi .* phi + K_s .* Pf0 .* eta_phi .* phi) ./ (-K_f .* K_phi .* dt .* phi + K_f .* K_phi .* dt - K_f .* eta_phi .* phi + K_f .* eta_phi + K_phi .* K_s .* dt .* phi + K_phi .* eta_phi .* phi + K_s .* eta_phi .* phi)
        Φ_trial  = (K_phi .* dt .* (Pf - Pt) + K_phi .* eta_phi .* phi_0 + eta_phi .* (Pf - Pf0 - Pt + Pt0)) ./ (K_phi .* eta_phi)
    end

    r = 0.0

    # Check yield
    f       = τII_trial - (1-Φ_trial)*C*cosd(ϕ) - (Pt_trial - Pf_trial)*sind(ϕ)

    x = [τII_trial, Pt_trial, Pf_trial, 0.0, Φ_trial]

    R = residual_two_phase(x, ε̇II_eff, Pt_trial, Pf_trial, Φ_trial, Pt0, Pf0, Φ0, params)

    @show R

    if f>-1e-13 
        @info "plastic"

        # This is the proper return mapping with plasticity
        r0  = 1.0
        tol = 1e-15

        for iter=1:10
            J = forwarddiff_jacobian(residual_two_phase, x, Const(ε̇II_eff), Const(Pt_trial), Const(Pf_trial), Const(Φ_trial), Const(Pt0), Const(Pf0), Const(Φ0), Const(params))
            # display(J.derivs[1])
            x .-= J.derivs[1]\J.val
            if iter==1 
                r0 = norm(J.val)
            end
            r = norm(J.val)/r0
            @show iter, r
            if r<tol
                break
            end
        end

    end

    # Recompute components
    τII, Pt, Pf, λ̇, Φ1   = x[1], x[2], x[3], x[4], x[5]

    τ = ε̇_eff .* τII./ε̇II_eff

    KΦ, ηΦ, ψ, Δt = params.KΦ, params.ηΦ, params.ψ, params.Δt
    Kf, Ks = params.Kf, params.Ks 

    #### Check residual with trial state pressures: it is also zero !!!
    dPtdt   = (Pt_trial - Pt0) / Δt
    dPfdt   = (Pf_trial - Pf0) / Δt
    dΦdt    = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf_trial - Pt_trial) 
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ_trial) *(dPtdt - Φ_trial*dPfdt) / Ks
    f1=dlnρsdt   - dΦdt/(1-Φ_trial) +   divVs
    f2=Φ_trial*dlnρfdt + dΦdt       + Φ_trial*divVs + divqD
    @show f1, f2

    ### Check residuals with corrected pressures: should be zero !

    # Looks like Φ_trial has to be the one here
    Φ       = Φ_trial

    # General form
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf - Pt) + λ̇*sind(ψ) 
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    f1=dlnρsdt   - dΦdt/(1-Φ) +   divVs
    f2=Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD
    @show f1, f2

    # Specific form 
    Kd = (1-Φ)*(1/KΦ + 1/Ks)^-1
    α  = 1 - Kd/Ks
    B  = (1/Kd - 1/Ks) / (1/Kd - 1/Ks + Φ*(1/Kf - 1/Ks))
    f1 = divVs     + 1/Kd*(dPtdt -   α*dPfdt) - 1/(1-Φ)*λ̇*sind(ψ) + (Pt-Pf)/((1-Φ)*ηΦ)
    f2 = divqD     - α/Kd*(dPtdt - 1/B*dPfdt) + 1/(1-Φ)*λ̇*sind(ψ) - (Pt-Pf)/((1-Φ)*ηΦ)
    @show f1, f2

    # Specific form (rederived)
    f1 = divVs    + (1/Ks)/(1-Φ) * (dPtdt - Φ*dPfdt) + (1/KΦ)/(1-Φ) * (dPtdt - dPfdt) + (Pt-Pf)/((1-Φ)*ηΦ) - 1/(1-Φ)*λ̇*sind(ψ)
    f2 = divqD    - (dPtdt - dPfdt)/KΦ + Φ*dPfdt/Kf + Φ*divVs - (Pt-Pf)/ηΦ + λ̇*sind(ψ)
    @show f1, f2

    return [τ[1], τ[2], τ[3], Pt, Pf], λ̇, Φ1, r 
end


function two_phase_return_mapping()

    # @load "v6.jld2" probes
    # probes_v6  = probes

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
        KΦ      = 1e10/sc.σ,
        Ks      = 1e11/sc.σ,
        Kf      = 1e9/sc.σ,
        C       = 1e7 /sc.σ,
        ϕ       = 35.0,
        ψ       = 10.0,
        ηvp     = 0/sc.σ/sc.t,
        ηs      = 1e22/sc.σ/sc.t,
        ηΦ      = 2e22/sc.σ/sc.t,
        Δt      = 1e10/sc.t,
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

    # Time loop
    for it=1:nt

        @info "Step $(it)"

        # Old guys
        Pt0 = Pt
        Pf0 = Pf
        τ0  = τ
        Φ0  = Φ
        
        # Invariants
        ε̇_eff      = ε̇ + τ0/(2*params.G*params.Δt)
        ϵ̇          = [ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], divVs, divqD]
        σ, λ̇, Φ, r = StressVector(ϵ̇, τ0, Pt0, Pf0, Φ0, params)
        τ, Pt, Pf  = σ[1:3], σ[4], σ[5]

        @show τ, Pt, Pf, Φ


        # # Consistent tangent
        # J = forwarddiff_jacobian(StressVector1, ϵ̇, Const(τ0), Const(P0), Const(params))
        # display(J.derivs[1])

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

        data = load("./examples/_TwoPhases/TwoPhasesPlasticity/VE_loading_homogeneous.jld2")

        @show data["probes"].τ

        fig = Figure(fontsize = 20, size = (600, 800) )     
        ax1 = Axis(fig[1,1], title="Deviatoric stress",  xlabel=L"$t$ [yr]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)
        lines!(ax1, probes.t[1:nt]*sc.t, probes.τ[1:nt]*sc.σ)
        scatter!(ax1, data["probes"].t[1:nt], data["probes"].τ[1:nt], marker=:xcross)
        scatter!(ax1, data["probes"].t[1:nt], data["probes"].τII[1:nt], marker=:xcross)
        # scatter!(ax1, probes_v6.t[1:nt]*sc.t, probes_v6.τ[1:nt]*sc.σ)

        ax2 = Axis(fig[2,1], title="Pressure",  xlabel=L"$t$ [yr]",  ylabel=L"$P$ [MPa]", xlabelsize=20, ylabelsize=20)
        lines!(ax2, probes.t[1:nt]*sc.t, probes.Pt[1:nt]*sc.σ)
        lines!(ax2, probes.t[1:nt]*sc.t, probes.Pf[1:nt]*sc.σ)
        scatter!(ax2, data["probes"].t[1:nt], data["probes"].Pt[1:nt], marker=:xcross)
        scatter!(ax2, data["probes"].t[1:nt], data["probes"].Pf[1:nt], marker=:xcross)
        # scatter!(ax2, probes_v6.t[1:nt]*sc.t, probes_v6.Pt[1:nt]*sc.σ)
        # scatter!(ax2, probes_v6.t[1:nt]*sc.t, probes_v6.Pf[1:nt]*sc.σ)
        # ylims!(ax2, 1e5, 2e6)
        
        ax3 = Axis(fig[3,1], title="Plastic multiplier",  xlabel=L"$t$ [yr]",  ylabel=L"$\dot{\lambda}$ [1/s]", xlabelsize=20, ylabelsize=20)    
        lines!(ax3, probes.t[1:nt]*sc.t, probes.λ̇[1:nt]/sc.t)
        scatter!(ax3, data["probes"].t[1:nt], data["probes"].λ̇[1:nt], marker=:xcross)
        # scatter!(ax3, probes_v6.t[1:nt]*sc.t, probes_v6.λ̇[1:nt]/sc.t)

        ax4 = Axis(fig[4,1], title="Porosity",  xlabel=L"$t$ [yr]",  ylabel=L"$\phi$", xlabelsize=20, ylabelsize=20)    
        lines!(ax4, probes.t[1:nt]*sc.t, probes.Φ[1:nt])
        scatter!(ax4, data["probes"].t[1:nt], data["probes"].Φ[1:nt], marker=:xcross)
        # scatter!(ax4, probes_v6.t[1:nt]*sc.t, probes_v6.Φ[1:nt])

        # ylims!(ax4, 0, 0.1)

        ax5 = Axis(fig[5,1], title="Residual",  xlabel=L"$t$ [yr]",  ylabel=L"$r$", xlabelsize=20, ylabelsize=20)    
        scatter!(ax5, probes.t[1:nt]*sc.t, log10.(probes.r[1:nt]))
        display(fig)
    end
    with_theme(figure, theme_latexfonts())
    # display(probes.Pt)
    # display(probes.Pf)

end

two_phase_return_mapping()