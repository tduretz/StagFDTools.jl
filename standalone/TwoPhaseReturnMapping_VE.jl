using CairoMakie, LinearAlgebra#, ForwardDiff
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian

# Intends to implement constitutive updates as in RheologicalCalculator

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + x[3]^2) 

function residual_two_phase(x, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0, p)
    G, KΦ, Ks, Kf, C, ϕ, ψ, ηvp, ηs, ηΦ, Δt = p.G, p.KΦ, p.Ks, p.Kf, p.C, p.ϕ, p.ψ, p.ηvp, p.ηs, p.ηΦ, p.Δt
    elastic, plastic = p.elastic, p.plastic
    eps   = -1e-13
    ηe    = G*Δt 
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    f       = τII  - C*cosd(ϕ) - (Pt - Pf)*sind(ϕ)
    if plastic==false
        f = -1e100
    end
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = elastic*1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (Pf - Pt) + plastic*λ̇*sind(ψ)*(f>=eps)
    Φ       = Φ0 + dΦdt*Δt
    dlnρfdt = elastic * (dPfdt / Kf)
    dlnρsdt = elastic * (1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks)

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
    # fpf3 = divqD    - (dPtdt - dPfdt)/KΦ + Φ*dPfdt/Kf + Φ*divVs - (Pt-Pf)/ηΦ +   1/(1-Φ)*λ̇*sind(ψ)*(f>=eps)

    ηs_eff = ηs#ηs*(1-Φ) + (Φ)*ηs/100

    return [ 
        ε̇II_eff   -  (τII)/2/ηs_eff  - elastic*(τII)/2/ηe - plastic*λ̇*(f>=eps),
        elastic*dlnρsdt   - dΦdt/(1-Φ) +   divVs,
        elastic*Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD,
        # divVs + (Pt-Pf)/((1-Φ)*ηΦ),
        # divqD - (Pt-Pf)/((1-Φ)*ηΦ),
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
    τ = ε̇_eff .* τII./ε̇II_eff
    KΦ, ηΦ, ψ, Δt = params.KΦ, params.ηΦ, params.ψ, params.Δt
    elastic, plastic = params.elastic, params.plastic
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = elastic * (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + plastic * λ̇*sind(ψ) 
    Φ       = Φ0 + dΦdt*Δt
    return [τ[1], τ[2], τ[3], Pt, Pf], λ̇, Φ, r 
end


function two_phase_return_mapping()

    sc = (σ=1e7, t=1e10, L=1e3)

    # Kinematics
    ε̇     = [1e-15, -1e-15, 0].*sc.t
    divVs =  1e-14 .*sc.t
    divqD = -1e-14 .*sc.t

    # Initial conditions
    Pt   = 1e1/sc.σ
    Pf   = 1e1/sc.σ 
    τ    = [0.0, -0.0, 0]./sc.σ
    Φ    = 0.1 

    # Parameters
    nt = 100000

    params = (
        G       = 3e11/sc.σ,
        KΦ      = 1e9/sc.σ,
        Ks      = 1e11/sc.σ,
        Kf      = 1e10/sc.σ,
        C       = 1e100/sc.σ,
        ϕ       = 35.0,
        ψ       = 30.0,
        ηvp     = 10.0*0/sc.σ/sc.t,
        ηs      = 1e21/sc.σ/sc.t,
        ηΦ      = 1e21/sc.σ/sc.t,
        Δt      = 1e11/sc.t,
        elastic = true,
        plastic = false,
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
        ε̇_eff      = ε̇ + params.elastic*τ0/(2*params.G*params.Δt)
        ϵ̇          = [ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], divVs, divqD]
        σ, λ̇, Φ, r = StressVector(ϵ̇, τ0, Pt0, Pf0, Φ0, params)
        τ, Pt, Pf = σ[1:3], σ[4], σ[5]

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
        fig = Figure(fontsize = 20, size = (600, 800) )     
        ax1 = Axis(fig[1,1], title="Deviatoric stress",  xlabel=L"$t$ [yr]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)
        scatter!(ax1, probes.t*sc.t, probes.τ*sc.σ)
        ax2 = Axis(fig[2,1], title="Pressure",  xlabel=L"$t$ [yr]",  ylabel=L"$P$ [MPa]", xlabelsize=20, ylabelsize=20)
        scatter!(ax2, probes.t*sc.t, probes.Pt*sc.σ)
        scatter!(ax2, probes.t*sc.t, probes.Pf*sc.σ)
        ax4 = Axis(fig[3,1], title="Porosity",  xlabel=L"$t$ [yr]",  ylabel=L"$\phi$", xlabelsize=20, ylabelsize=20)    
        scatter!(ax4, probes.t*sc.t, probes.Φ)
        # ax3 = Axis(fig[3,1], title="Plastic multiplier",  xlabel=L"$t$ [yr]",  ylabel=L"$\dot{\lambda}$ [1/s]", xlabelsize=20, ylabelsize=20)    
        # scatter!(ax3, probes.t, probes.λ̇)
        ax4 = Axis(fig[4,1], title="Residual",  xlabel=L"$t$ [yr]",  ylabel=L"$r$", xlabelsize=20, ylabelsize=20)    
        scatter!(ax4, probes.t*sc.t, log10.(probes.r))
        display(fig)
    end
    with_theme(figure, theme_latexfonts())
    # display(probes.Pt)
    display(probes.Φ)

end

two_phase_return_mapping()