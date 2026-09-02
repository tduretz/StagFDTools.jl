using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs
import Statistics:mean

@views function main(D_BC, nc, nt, n_nt; 
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
    D_BC = D_BC .* ε̇ 

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

    ε̇xx = D_BC[1,1] - 1/3*tr(D_BC)
    ε̇yy = D_BC[2,2] - 1/3*tr(D_BC)
    ε̇kk = tr(D_BC)

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
        ε̇xx_eff = ε̇xx + τxx0/(2*materials.G[1]*Δt)
        ε̇yy_eff = ε̇yy + τyy0/(2*materials.G[1]*Δt)

        # OLD STYLE 

        # Trial pressures - not needed with LocalRheology_P2 !!!
        div = @SVector[ε̇kk, divqD]
        x   = Pressures(div, Pt0, Pf0, Φ0, materials.KΦ[1],  materials.Ks[1],  materials.Kf[1], materials.ξ0[1], Δ.t)
        Pt, Pf, Φ = x[1], x[2], x[3]

        # Correction
        ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )
        η, λ̇, Pt1, Pf1, τII, Φ, f = LocalRheology_P(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)

        # NEW STYLE 

        # ε̇vec = @SVector( [ε̇xx_eff; ε̇yy_eff; 0.0; Pt; Pf] )
        # η, λ̇, Pt1, Pf1, τII, Φ, f = LocalRheology_P2(ε̇vec, ε̇kk, divqD, Pt0, Pf0, Φ0, materials, 1, Δ)

        #--------------------------------------------#

        # Include plasticity corrections
        Pt  = Pt1# .+ (Pt1-Pt)
        Pf  = Pf1# .+ (Pf1-Pf)
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
      
        if abs(D_BC[2,2])>1e-13
            data = load("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DP_PS.jld2")
        else
            data = load("./examples/_TwoPhases/TwoPhasesPlasticity/VEP_loading_homogeneous_DP_tens.jld2")
        end
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
    D_BC = @SMatrix([1 0; 0 -1] )
    main(D_BC, nc, nt, n_nt, homo=true, niter=2)
end

Run()