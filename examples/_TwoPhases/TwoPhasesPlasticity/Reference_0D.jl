using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs
import Statistics:mean

function residual_two_phase_P2(x, ηve, Δt, ε̇II_eff, τII_trial, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )
     
    τII, ΔPt, ΔPf, λ̇ = x[1], x[2], x[3], (x[4])
    α1 = single_phase ? 0.0 : 1.0 

    # Pressure corrections: closed form
    # ΔPt_1 = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
    # ΔPf   = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    
    # Pressure corrections: numerics (nested AD)
    # ΔPt_1, ΔPf = ΔP(Pt_trial, Pf_trial, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ, m,  KΦ, Ks, Kf, sinψ, Δt)

    # Check yield
    f = if single_phase
            τII - C*cosϕ - Pt*sinϕ 
        else
            F(τII, Pt_trial+ΔPf, Pf_trial+ΔPf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)
        end

    # Make ΔPt the unknown
    Pt = Pt_trial + ΔPt
    Pf = Pf_trial + ΔPf

    # Porosity rate
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dlnρfdt = dPfdt / Kf
    # dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    Φ, dΦdt = Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ, m, λ̇, sinψ, Δt)  
    # dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1/Ks * ( dPsdt ) 

    # ΔPt = if single_phase
    #     Ks .* sinψ .* Δt .* λ̇
    #     else
    #         ΔPt_1
    #     end

    return @SVector [ 
        ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        # τII - (τII_trial - ηve*λ̇),
        # Pt - (Pt_trial + ΔPt),
        # Pf - (Pf_trial + ΔPf),
        dlnρsdt   - dΦdt/(1-Φ),
        (Φ*dlnρfdt + dΦdt     )/ηΦ,
        f, 
    ]
end

function LocalRheology_P2(ε̇::SVector{N, D}, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) where {N, D}

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt = ε̇[4]
    Pf = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    m    = materials.m[phases]
    η0   = materials.η0[phases]
    G    = materials.G[phases]
    C    = materials.plasticity.C[phases]
    ηΦ0   = materials.ξ0[phases]
    KΦ   = materials.KΦ[phases]
    Ks   = materials.Ks[phases]
    Kf   = materials.Kf[phases]

    ηvp  = materials.plasticity.ηvp[phases]
    sinψ = materials.plasticity.sinψ[phases]    
    sinϕ = materials.plasticity.sinϕ[phases] 
    cosϕ = materials.plasticity.cosϕ[phases]  

    # ηvep, λ̇, Pt, Pf, τII, Φ, f  = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    
    α1 = materials.single_phase ? zero(D) : one(D)

    # Initial guess
    η         = η0 * ε̇II_eff^(1 / n - 1 )
    ηve       = inv(1/η + 1/(G*Δ.t))
    τII       = 2*ηve*ε̇II_eff
    ηvep      = ηve

    Φ = if materials.single_phase
        zero(D)
    else
        # Trial porosity: closed form
        # Φ = (KΦ * Δ.t * (Pf - Pt) + KΦ * Φ0 * ηΦ0 + ηΦ0 * (Pf - Pf0 - Pt + Pt0)) / (KΦ * ηΦ0)
    
        # Trial porosity: numerics (nested AD)
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, 0.0, sinψ, Δ.t)[1]
    end

    # Check yield
    λ̇  = zero(D)

    #############################

    f  = F(τII, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    x = @SVector [τII, 0*Pt, 0*Pf, λ̇]
    plastic_correction = false

    nr   = D(1.0)
    nr0  = D(1.0)
    tol  = D(1e-10)

    # Return mapping
    if f > D(-1e-13)

        # @show f, τII, Pt, Pf, λ̇
        # @show ηve, ε̇II_eff
        # @show ε̇

        plastic_correction = true
        # This is the proper return mapping with plasticity
        for iter=1:10
            R, J = fd_value_and_jacobian(residual_two_phase_P2, x, ηve, Δ.t, ε̇II_eff, τII,       Pt,       Pf,       divVs, divqD, Φ,       Pt0, Pf0, Φ0, ηΦ0, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, materials.single_phase)
            x   -= J \ R
            nr   = norm(R)
            if iter==1 
                nr0 = nr
            end
            # if x[4] < 0
            #     @show nr/nr0, nr, x[4]
            #     @show ε̇
            #     @show η, G
            #     @show ηve, Δ.t, ε̇II_eff, τII,       Pt,       Pf,       divVs, divqD, Φ,       Pt0, Pf0, Φ0, ηΦ0, m, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp
            # end
            nr/nr0 < tol && break
        end
    end

    τII, dPt, dPf, λ̇ = x[1], x[2], x[3], x[4]
    Pt = Pt + dPt
    Pf = Pf + dPf

    Φ = if materials.single_phase
        zero(D)
    else
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δ.t)[1]
    end

    dΦdt = if materials.single_phase
        zero(D)
    else
        Porosity(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δ.t)[2]
    end

    # Φ, dΦdt = if materials.linearizeΦ ||  materials.single_phase
    #     TΦ      = promote_type_to_dual(Pt_loc, Pf_loc)
    #     Φ       = TΦ(Φ0)
    #     dΦdt    = TΦ(zeros(3,3))
    #     Φ, dΦdt 
    # else
    #     Φ, dΦdt = compute_Φ_and_dΦdt(Φ0, Pt, Pf, Pt0, Pf0, KΦ, ηΦ0, m, λ̇, sinψ, Δ.t)
    #     Φ, dΦdt 
    # end

    # EOS
    dPtdt   = (Pt - Pt0) / Δ.t
    dPfdt   = (Pf - Pf0) / Δ.t
    dlnρfdt = dPfdt / Kf
    # dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1/Ks * ( dPsdt ) 

    #############################

    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    # Optional: check f
    if plastic_correction
        f    = F(τII, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α1)
        # @show f
    end

    return ηvep, λ̇, Pt, Pf, τII, Φ, f, dlnρsdt, dlnρfdt 
end


@views function main(nc, nt, n_nt; 
    homo=false, niter=100, Φini=5e-2, ηvp=0.0)

    sc = (σ=1e7, t=1e10, L=1e3)
    ky = 1e3*365*24*3600

    # Time steps
    Δt     = 1e10/sc.t / n_nt 

    rad     = 1e3/sc.L 
    Pt_ini  = 1e6/sc.σ
    Pf_ini  = 1e6/sc.σ
    ε̇       = 2e-15.*sc.t
    τ_ini   = 0*(sind(35)*(Pt_ini-Pf_ini) + 0*1e7/sc.σ*cosd(35))  

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )

    τxx_ini = τ_ini*D_BC[1,1]/ε̇
    τyy_ini = τ_ini*D_BC[2,2]/ε̇

    ε̇kk, divqD = 0.0, 0.0

    # Material parameters
    nphases = 2
    materials = initialize_materials_TwoPhases(nphases,
        oneway       = false,
        compressible = true,
        linearizeΦ   = false, 
        single_phase = false,
        conservative = false,
        plasticity   = DruckerPrager,
    )

    materials.n     .= [  1.0,    1.0 ]
    materials.m     .= [  0.0,    0.0 ]
    materials.n_CK  .= [  0.0,    0.0 ]
    materials.η0    .= [ 1e22,   1e19 ]/sc.σ/sc.t 
    materials.ξ0    .= [ 2e22,   2e22 ]/sc.σ/sc.t
    materials.G     .= [ 3e10,   3e10 ]./sc.σ 
    materials.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials.Ks    .= [ 1e11,   1e11 ]./sc.σ
    materials.KΦ    .= [ 1e10,   1e10 ]./sc.σ
    materials.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials.k_ηf0 .= [1e-15,  1e-15 ]./(sc.L^2/sc.σ/sc.t)
    materials.plasticity.ϕ   .= [ 35.,     35. ]
    materials.plasticity.ψ   .= [ 10.,     10. ] .* 1
    materials.plasticity.C   .= [ 1e7,     1e7 ]./sc.σ
    materials.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t 
    preprocess!(materials)

    Φ0      = Φini
    # Φ0 = (materials.KΦ[1] .* Δt0 .* (Pf_ini - Pt_ini)) ./ (materials.KΦ[1] .* materials.ξ0[1])
    @show Φ0
    # error()
    Φ_ini   = Φ0

    probes = (
        Pe  = zeros(nt),
        Pt  = zeros(nt),
        Pf  = zeros(nt),
        τ   = zeros(nt),
        Φ   = zeros(nt),
        λ̇   = zeros(nt),
        t   = zeros(nt),
        τII = zeros(nt),
    )

    τII, λ̇ = 0.0, 0.0
    Pt, Pf = Pt_ini, Pf_ini
    τxx, τyy, τxy =  0.0, 0.0, 0.0
    ΔPt, ΔPf = 0.0, 0.0
    Φ = Φ0
    ρs, ρf = materials.ρs[1], materials.ρf[1]
    Δ = (t=Δt,)

    for it=1:nt

        @printf("\nStep %04d\n", it)

        # Swap old values 
        Pt0  = Pt
        Pf0  = Pf
        τxx0 = τxx
        τyy0 = τyy
        τxy0 = τxy
        Φ0   = Φ 
        ρs0  = ρs
        ρf0  = ρf

        # Trial deviatoric stress
        ε̇xx_eff = ε̇ + τxx0/(2*materials.G[1]*Δt)
        ε̇yy_eff =-ε̇ + τyy0/(2*materials.G[1]*Δt)

        # # Trial pressures - not needed with LocalRheology_P2 !!!
        # div = @SVector[ε̇kk, divqD]
        # x   = Pressures(div, Pt0, Pf0, Φ0, materials.KΦ[1],  materials.Ks[1],  materials.Kf[1], materials.ξ0[1], Δ.t)
        # Pt, Pf, Φ = x[1], x[2], x[3]

        # # Correction
        # ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )
        # η, λ̇, Pt1, Pf1, τII, Φ, f = LocalRheology_P(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)

        ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )
        η, λ̇, Pt1, Pf1, τII, Φ, f = LocalRheology_P2(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)

        #--------------------------------------------#

        # Include plasticity corrections
        Pt  = Pt .+ (Pt1-Pt)
        Pf  = Pf .+ (Pf1-Pf)
        τxx = 2 * η * ε̇vec[1]
        τyy = 2 * η * ε̇vec[2]
        τxy = 2 * η * ε̇vec[3]

        @show τxx, τyy, τII
     
        #--------------------------------------------#
        probes.Pe[it]   = (Pt .- Pf)*sc.σ
        probes.Pt[it]   = Pt*sc.σ
        probes.Pf[it]   = Pf*sc.σ
        probes.τ[it]    = τII*sc.σ
        probes.Φ[it]    = Φ
        probes.λ̇[it]    = λ̇/sc.t
        probes.t[it]    = it*Δt*sc.t

        #-------------------------------------------# 

        @info τ_ini*sc.σ
        @show τxx_ini*sc.σ, τyy_ini*sc.σ

        # fname = @sprintf("PoroVEP_%03d.jld2",  it)
        # save("./examples/_TwoPhases/TwoPhasesPlasticity/results/$(fname)", "X", X, "sc", sc, "probes", probes,
        # "λ̇", λ̇, "P", P, "τ", τ, "ε̇", ε̇, "V", V, "η", η, "Φ", Φ, "εp", εp, "niter", niter, "err", err ) 
      
        data = load("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_remix2.jld2")
        probes2D = data["probes"]

        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 600) )    
            step = 10
            ftsz = 15
            eps  = 1e-10
            # ax    = Axis(fig[1,1], title=L"$\dot{\lambda}$ [1/s]", xlabel=L"x", ylabel=L"y")
            ax    = Axis(fig[1,1], title=L"$\tau_\text{II}$", xlabel=L"$t$ [ky]", ylabel=L"$\tau_\text{II}$ [MPa]")
            lines!(ax, probes.t[1:it]/ky, probes.τ[1:it]/1e6)
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.τ[1:it]/1e6)
            ax    = Axis(fig[2,1], title=L"$P$", xlabel=L"$t$ [ky]", ylabel=L"$P$ [MPa]")
            lines!(ax, probes.t[1:it]/ky, probes.Pt[1:it]/1e6)
            lines!(ax, probes.t[1:it]/ky, probes.Pf[1:it]/1e6)
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.Pt[1:it]/1e6)
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.Pf[1:it]/1e6)
            ax    = Axis(fig[1,2], title=L"$\dot{\lambda}$", xlabel=L"$t$ [ky]", ylabel=L"$\dot{\lambda}$ [1/s]")
            lines!(ax, probes.t[1:it]/ky, probes.λ̇[1:it])
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.λ̇[1:it])
            ax    = Axis(fig[2,2], title=L"$\Phi$", xlabel=L"$t$ [ky]", ylabel=L"$\Phi$ [-]")
            lines!(ax, probes.t[1:it]/ky, probes.Φ[1:it])
            scatter!(ax, probes2D.t[1:it]/ky, probes2D.Φ[1:it])


            ax    = Axis(fig[3:4,1:2], title=L"$$Yield", xlabel=L"$P^\text{eff}$ [MPa]", ylabel=L"$\tau$ [MPa]")
            Peff = (probes.Pt[1:it].-probes.Pf[1:it])
            pmin = Peff[it] - 1.1*Peff[it] 
            pmax = Peff[it] + 0.1*Peff[it]  + 0.01
            p    = LinRange(pmin, pmax, 5)
            C    = materials.plasticity.C[1] * sc.σ 
            cosϕ = materials.plasticity.cosϕ[1] 
            sinϕ = materials.plasticity.sinϕ[1]
            τ    = C*cosϕ .+ p*sinϕ
            lines!(ax, p./1e6, τ./1e6)
            scatter!(ax, Peff/1e6, probes2D.τ[1:it]./1e6)

            display(fig) 
        end
        with_theme(figure, theme_latexfonts())
        #-------------------------------------------# 
    end
    #--------------------------------------------#
    return 
end

function Run()
    # Homogeneous test
    n_nx = 1
    n_nt = 1
    nc   = (x=n_nx*50, y=n_nx*25)
    nt   = 40*n_nt
    main(nc, nt, n_nt, homo=true, niter=2)
end

Run()