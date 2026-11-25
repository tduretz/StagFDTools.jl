using CairoMakie, LinearAlgebra, JLD2
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian

# Intends to implement constitutive updates as in RheologicalCalculator

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + 1/2*(-x[1]-x[2])^2 + x[3]^2) 

function residual_two_phase_trial(x, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0, p)
    G, KΦ, Ks, Kf, C, ϕ, ψ, ηvp, ηv, ηΦ, Δt = p.G, p.KΦ, p.Ks, p.Kf, p.C, p.ϕ, p.ψ, p.ηvp, p.ηs, p.ηΦ, p.Δt
    eps   = -1e-13
    ηe    = G*Δt 
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    f       = -1e100
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf - Pt)
    Φ       = Φ0 + dΦdt*Δt
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    ηve = (1-Φ)*inv(1/ηv + 1/ηe)

    return [ 
        ε̇II_eff   -  τII/2/ηve,
        dlnρsdt   - dΦdt/(1-Φ) +   divVs,
        Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD,
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps)
    ]
end

function residual_two_phase(x, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0, p)
    G, KΦ, Ks, Kf, C, ϕ, ψ, ηvp, ηv, ηΦ, Δt = p.G, p.KΦ, p.Ks, p.Kf, p.C, p.ϕ, p.ψ, p.ηvp, p.ηs, p.ηΦ, p.Δt
    eps   = -1e-13
    ηe    = G*Δt 
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    f       = τII - C*cosd(ϕ) - (Pt - Pf)*sind(ϕ)
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    @show λ̇*sind(ψ)
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sind(ψ)*(f>=eps)
    Φ       = Φ0 + dΦdt*Δt
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    # Kd = (1-Φ)*(1/KΦ + 1/Ks)^-1
    # α  = 1 - Kd/Ks
    # B  = (1/Kd - 1/Ks) / (1/Kd - 1/Ks + Φ*(1/Kf - 1/Ks))

    # Most pristine form 
    # fpt1 = dlnρsdt   - dΦdt/(1-Φ) +   divVs
    # fpf1 = Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD

    # # Equation from Yarushina (2015) adding dilation bu educated guess :D
    # fpt2 = divVs     + 1/Kd*(dPtdt -   α*dPfdt) - 1/(1-Φ)*λ̇*sind(ψ)*(f>=eps) + (Pt-Pf)/((1-Φ)*ηΦ)
    # fpf2 = divqD     - α/Kd*(dPtdt - 1/B*dPfdt) + 1/(1-Φ)*λ̇*sind(ψ)*(f>=eps) - (Pt-Pf)/((1-Φ)*ηΦ)

    # # Equations self-rederived from Yarushina (2015) adding dilation
    # fpt3 = divVs    + (1/Ks)/(1-Φ) * (dPtdt - Φ*dPfdt) + (1/KΦ)/(1-Φ) * (dPtdt - dPfdt) + (Pt-Pf)/((1-Φ)*ηΦ) - 1/(1-Φ)*λ̇*sind(ψ)*(f>=eps)
    # fpf3 = divqD    - (dPtdt - dPfdt)/KΦ + Φ*dPfdt/Kf + Φ*divVs - (Pt-Pf)/ηΦ +   λ̇*sind(ψ)*(f>=eps)

    ηve = (1-Φ)*inv(1/ηv + 1/ηe)

    return [ 
        ε̇II_eff   -  τII/2/ηve - λ̇/2*(f>=eps),
        dlnρsdt   - dΦdt/(1-Φ) +   divVs,
        Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD,
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps)
    ]
end

function StressVector(ϵ̇, τ0, Pt0, Pf0, Φ0, params)

    ε̇_eff = ϵ̇[1:3]
    divVs = ϵ̇[4]
    divqD = ϵ̇[5]

    ε̇II_eff = invII(ε̇_eff) 
    τII     = invII(τ0)

    # Rheology update
    x = [τII, Pt0, Pf0, 0.0]
    Φ = 0.0
    r = 0.0

    # for iter=1:10
    #     J_enz = forwarddiff_jacobian(residual_two_phase, x, Const(ε̇II_eff), Const(divVs), Const(divqD), Const(Pt0), Const(Pf0), Const(Φ0), Const(params))
    #     display(J_enz.derivs[1])
    #     @show J_enz.derivs[1][2][1]
    #         J = zeros(4,4)

    #     J[1,:] .= J_enz.derivs[1][1][1]
    #     J[2,:] .= J_enz.derivs[1][2][1]
    #     J[3,:] .= J_enz.derivs[1][3][1]
    #     J[4,:] .= J_enz.derivs[1][4][1]
        
    #     display(J)
    #     display(J_enz.val[1])
    #     f = J_enz.val[1]
    #     # Φ = J_enz.val[2]
    #     x .-= J\f
    #     @show norm(f)
    #     if norm(f)<1e-10
    #         break
    #     end
    #     error("stop")
    # end

    # This is the proper retun mapping with plasticity
    r0  = 1.0
    tol = 1e-9

    for iter=1:10
        J = forwarddiff_jacobian(residual_two_phase, x, Const(ε̇II_eff), Const(divVs), Const(divqD), Const(Pt0), Const(Pf0), Const(Φ0), Const(params))
        # display(J.derivs[1])
        x .-= J.derivs[1]\J.val
        if iter==1 
            r0 = norm(J.val)
        end
        r = norm(J.val)/r0
        # @show r
        if r<tol
            break
        end
    end

    # Recompute components
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]

    # This is just a calculation of viscoelastic trial state
    for iter=1:10
        J = forwarddiff_jacobian(residual_two_phase_trial, x, Const(ε̇II_eff), Const(divVs), Const(divqD), Const(Pt0), Const(Pf0), Const(Φ0), Const(params))
        # display(J.derivs[1])
        x .-= J.derivs[1]\J.val
        # @show norm(J.val)
        if norm(J.val)<1e-10
            break
        end
    end
    Pt_t, Pf_t = x[2], x[3]

    τ = ε̇_eff .* τII./ε̇II_eff
    KΦ, ηΦ, ψ, Δt = params.KΦ, params.ηΦ, params.ψ, params.Δt
    Kf, Ks = params.Kf, params.Ks 

    # Check residual using trial state pressures: it is also zero !!!
    dPtdt   = (Pt_t - Pt0) / Δt
    dPfdt   = (Pf_t - Pf0) / Δt
    dΦdt    = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf_t - Pt_t) 
    Φ       = Φ0 + dΦdt*Δt
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    f1=dlnρsdt   - dΦdt/(1-Φ) +   divVs
    f2=Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD
    @show f1, f2

    # Check residual should be zero
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf - Pt) + λ̇*sind(ψ) 
    Φ       = Φ0 + dΦdt*Δt
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    f1=dlnρsdt   - dΦdt/(1-Φ) +   divVs
    f2=Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD
    @show f1, f2

    Kd = (1-Φ)*(1/KΦ + 1/Ks)^-1
    α  = 1 - Kd/Ks
    B  = (1/Kd - 1/Ks) / (1/Kd - 1/Ks + Φ*(1/Kf - 1/Ks))
    f1 = divVs     + 1/Kd*(dPtdt -   α*dPfdt) - 1/(1-Φ)*λ̇*sind(ψ) + (Pt-Pf)/((1-Φ)*ηΦ)
    f2 = divqD     - α/Kd*(dPtdt - 1/B*dPfdt) + 1/(1-Φ)*λ̇*sind(ψ) - (Pt-Pf)/((1-Φ)*ηΦ)
    @show f1, f2

    f1 = divVs    + (1/Ks)/(1-Φ) * (dPtdt - Φ*dPfdt) + (1/KΦ)/(1-Φ) * (dPtdt - dPfdt) + (Pt-Pf)/((1-Φ)*ηΦ) - 1/(1-Φ)*λ̇*sind(ψ)
    f2 = divqD    - (dPtdt - dPfdt)/KΦ + Φ*dPfdt/Kf + Φ*divVs - (Pt-Pf)/ηΦ + λ̇*sind(ψ)
    @show f1, f2

    return [τ[1], τ[2], τ[3], Pt, Pf], λ̇, Φ, r 
end


function two_phase_return_mapping()

    sc = (σ=1e7, t=1e10, L=1e3)

    # Kinematics
    ε̇     = [1e-15, -1e-15, 0].*sc.t
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
        KΦ      = 1e9/sc.σ,
        Ks      = 1e11/sc.σ,
        Kf      = 1e10/sc.σ,
        C       = 1e7 /sc.σ,
        ϕ       = 1*35.0,
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
        lines!(ax1, probes.t*sc.t, probes.τ*sc.σ)
        scatter!(ax1, data["probes"].t, data["probes"].τ)

        ax2 = Axis(fig[2,1], title="Pressure",  xlabel=L"$t$ [yr]",  ylabel=L"$P$ [MPa]", xlabelsize=20, ylabelsize=20)
        lines!(ax2, probes.t*sc.t, probes.Pt*sc.σ)
        lines!(ax2, probes.t*sc.t, probes.Pf*sc.σ)
        scatter!(ax2, data["probes"].t, data["probes"].Pt)
        scatter!(ax2, data["probes"].t, data["probes"].Pf)
        # ylims!(ax2, 1e5, 2e6)
        
        ax3 = Axis(fig[3,1], title="Plastic multiplier",  xlabel=L"$t$ [yr]",  ylabel=L"$\dot{\lambda}$ [1/s]", xlabelsize=20, ylabelsize=20)    
        lines!(ax3, probes.t*sc.t, probes.λ̇/sc.t)
        scatter!(ax3, data["probes"].t, data["probes"].λ̇)

        ax4 = Axis(fig[4,1], title="Porosity",  xlabel=L"$t$ [yr]",  ylabel=L"$\phi$", xlabelsize=20, ylabelsize=20)    
        lines!(ax4, probes.t*sc.t, probes.Φ)
        scatter!(ax4, data["probes"].t, data["probes"].Φ)
        # ylims!(ax4, 0, 0.1)

        ax5 = Axis(fig[5,1], title="Residual",  xlabel=L"$t$ [yr]",  ylabel=L"$r$", xlabelsize=20, ylabelsize=20)    
        scatter!(ax5, probes.t*sc.t, log10.(probes.r))
        display(fig)
    end
    with_theme(figure, theme_latexfonts())
    # display(probes.Pt)
    # display(probes.Pf)

end

two_phase_return_mapping()