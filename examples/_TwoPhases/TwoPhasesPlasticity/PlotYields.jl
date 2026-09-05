using StagFDTools, StagFDTools.TwoPhases, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf, JLD2, TimerOutputs, ForwardDiff
import Statistics:mean

@views function main(D_BC, nc, nt, n_nt; 
    homo=false, niter=20, Φini=5e-2, ηvp=0.0)

    sc = (σ=1e7, t=1e10, L=1e3)
    ky = 1e3*365*24*3600

    visualization = true
    free_clims    = true

    # Linear solver
    solver      = :GCR
    GCR_restart = 25
    GCR_maxit   = 100
    ϵ_l         = 1e-11
    Pic2Newt    = 1.8#0.1   # more than 1.0 - always Newton

    # Non-linear solver
    ϵ_nl    = 1e-8
    α       = LinRange(0.05, 1.0, 5)

    # Time steps
    Δt     = 1e10/sc.t / n_nt 

    rad     = 1e3/sc.L 
    Pt_ini  = 1e6/sc.σ
    Pf_ini  = 1e6/sc.σ
    Pf_bot  = 60e6/sc.σ
    ε̇       = 1e-15.*sc.t
    τ_ini   = 0*(sind(35)*(Pt_ini-Pf_ini) + 0*1e7/sc.σ*cosd(35))  

    # Velocity gradient matrix
    D_BC = D_BC .* ε̇ 

    τxx_ini = τ_ini*D_BC[1,1]/ε̇
    τyy_ini = τ_ini*D_BC[2,2]/ε̇

    # Material parameters
    nphases = 2

    materials_T = initialize_materials_TwoPhases(nphases,
        plasticity     = Tensile,
    )
    materials_T.n     .= [  1.0,    1.0 ]
    materials_T.m     .= [  0.0,    0.0 ]
    materials_T.n_CK  .= [  0.0,    0.0 ]
    materials_T.η0    .= [ 1e25,   1e19 ]/sc.σ/sc.t 
    materials_T.ξ0    .= [ 2e25,   2e22 ]/sc.σ/sc.t
    materials_T.G     .= [ 3e10,   3e10 ]./sc.σ 
    materials_T.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_T.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_T.Ks    .= [ 1e11,   1e11 ]./sc.σ
    materials_T.KΦ    .= [ 1e10,   1e10 ]./sc.σ
    materials_T.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials_T.k_ηf0 .= [1e-20,  1e-20 ]./(sc.L^2/sc.σ/sc.t)
    materials_T.Φ0    .= [Φini,  Φini]
    materials_T.plasticity.ϕ   .= [ 35.,     35. ]
    materials_T.plasticity.ψ   .= [ 10.,     10. ] 
    materials_T.plasticity.C   .= [ 1e7,     1e7 ]./sc.σ
    materials_T.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t 
    materials_T.plasticity.Pt  .= [ -1e6,     -1e6 ]./sc.σ 
    preprocess!(materials_T)

    materials_DP = initialize_materials_TwoPhases(nphases,
        plasticity     = DruckerPrager,
    )
    materials_DP.n     .= [  1.0,    1.0 ]
    materials_DP.m     .= [  0.0,    0.0 ]
    materials_DP.n_CK  .= [  0.0,    0.0 ]
    materials_DP.η0    .= [ 1e25,   1e19 ]/sc.σ/sc.t 
    materials_DP.ξ0    .= [ 2e25,   2e22 ]/sc.σ/sc.t
    materials_DP.G     .= [ 3e10,   3e10 ]./sc.σ 
    materials_DP.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_DP.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_DP.Ks    .= [ 1e11,   1e11 ]./sc.σ
    materials_DP.KΦ    .= [ 1e10,   1e10 ]./sc.σ
    materials_DP.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials_DP.k_ηf0 .= [1e-20,  1e-20 ]./(sc.L^2/sc.σ/sc.t)
    materials_DP.Φ0    .= [Φini,  Φini]
    materials_DP.plasticity.ϕ   .= [ 35.,     35. ]
    materials_DP.plasticity.ψ   .= [ 10.,     10. ] 
    materials_DP.plasticity.C   .= [ 1e7,     1e7 ]./sc.σ
    materials_DP.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t 
    preprocess!(materials_DP)


    materials_DPH = initialize_materials_TwoPhases(nphases,
        plasticity     = DruckerHyperbolic,
    )
    materials_DPH.n     .= [  1.0,    1.0 ]
    materials_DPH.m     .= [  0.0,    0.0 ]
    materials_DPH.n_CK  .= [  0.0,    0.0 ]
    materials_DPH.η0    .= [ 1e25,   1e19 ]/sc.σ/sc.t 
    materials_DPH.ξ0    .= [ 2e25,   2e22 ]/sc.σ/sc.t
    materials_DPH.G     .= [ 3e10,   3e10 ]./sc.σ 
    materials_DPH.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_DPH.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_DPH.Ks    .= [ 1e11,   1e11 ]./sc.σ
    materials_DPH.KΦ    .= [ 1e10,   1e10 ]./sc.σ
    materials_DPH.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials_DPH.k_ηf0 .= [1e-20,  1e-20 ]./(sc.L^2/sc.σ/sc.t)
    materials_DPH.Φ0    .= [Φini,  Φini]
    materials_DPH.plasticity.ϕ   .= [ 35.,     35. ]
    materials_DPH.plasticity.ψ   .= [ 10.,     10. ] 
    materials_DPH.plasticity.C   .= [ 1e7,     1e7 ]./sc.σ
    materials_DPH.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t 
    materials_DPH.plasticity.Pt  .= [ -1e6,     -1e6 ]./sc.σ 
    preprocess!(materials_DPH)


    materials_DPC = initialize_materials_TwoPhases(nphases,
        plasticity     = DruckerPragerCap,
    )
    materials_DPC.n     .= [  1.0,    1.0 ]
    materials_DPC.m     .= [  0.0,    0.0 ]
    materials_DPC.n_CK  .= [  0.0,    0.0 ]
    materials_DPC.η0    .= [ 1e25,   1e19 ]/sc.σ/sc.t 
    materials_DPC.ξ0    .= [ 2e25,   2e22 ]/sc.σ/sc.t
    materials_DPC.G     .= [ 3e10,   3e10 ]./sc.σ 
    materials_DPC.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_DPC.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_DPC.Ks    .= [ 1e11,   1e11 ]./sc.σ
    materials_DPC.KΦ    .= [ 1e10,   1e10 ]./sc.σ
    materials_DPC.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials_DPC.k_ηf0 .= [1e-20,  1e-20 ]./(sc.L^2/sc.σ/sc.t)
    materials_DPC.Φ0    .= [Φini,  Φini]
    materials_DPC.plasticity.ϕ   .= [ 35.,     35. ]
    materials_DPC.plasticity.ψ   .= [ 10.,     10. ] 
    materials_DPC.plasticity.C   .= [ 1e7,     1e7 ]./sc.σ
    materials_DPC.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t 
    materials_DPC.plasticity.Pt  .= [ -1e6,     -1e6 ]./sc.σ 
    preprocess!(materials_DPC)
    
    materials_MCC = initialize_materials_TwoPhases(nphases,
    plasticity     = Golchin2021,
    )
    materials_MCC.n     .= [  1.0,    1.0 ]
    materials_MCC.m     .= [  0.0,    0.0 ]
    materials_MCC.n_CK  .= [  0.0,    0.0 ]
    materials_MCC.η0    .= [ 1e25,   1e19 ]/sc.σ/sc.t 
    materials_MCC.ξ0    .= [ 2e25,   2e22 ]/sc.σ/sc.t
    materials_MCC.G     .= [ 3e10,   3e10 ]./sc.σ 
    materials_MCC.ρs    .= [ 2800,   2800 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_MCC.ρf    .= [ 1000,   1000 ]/(sc.σ*sc.t^2/sc.L^2)
    materials_MCC.Ks    .= [ 1e11,   1e11 ]./sc.σ
    materials_MCC.KΦ    .= [ 1e10,   1e10 ]./sc.σ
    materials_MCC.Kf    .= [  1e9,    1e9 ]./sc.σ
    materials_MCC.k_ηf0 .= [1e-20,  1e-20 ]./(sc.L^2/sc.σ/sc.t)
    materials_MCC.Φ0    .= [Φini,  Φini]
    materials_MCC.plasticity.ϕ   .= [ 35.,     35. ]
    materials_MCC.plasticity.ψ   .= [ 35.,     35. ] 
    materials_MCC.plasticity.C   .= [ 1e7,     1e7 ]./sc.σ
    materials_MCC.plasticity.ηvp .= [ ηvp,     ηvp ]./sc.σ/sc.t 
    materials_MCC.plasticity.Pt  .= [ -1e6,     -1e6 ]./sc.σ 
    materials_MCC.plasticity.Pc  .= [5e7,      5e7]   ./ sc.σ
    materials_MCC.plasticity.a   .= [0.8,      0.8]
    materials_MCC.plasticity.b   .= [0.0,      0.0]
    materials_MCC.plasticity.c   .= [0.8,      0.8]
    preprocess!(materials_MCC)

        # Visualise
        function figure()

            fig = Figure(fontsize = 20, size = (800, 400))

            ax = Axis(
                fig[1, 1],
                aspect = DataAspect(),
                title = L"$$Plasticity models",
                xlabel = L"$p^\text{eff}$ [MPa] ",
                ylabel = L"$\tau_\text{II}$ [MPa]"
            )

            # Axes
            Pe_ax  = [-1.4e7, 5.9e7] ./ sc.σ
            τII_ax = [0, 5e7] ./ sc.σ
            P_ax   = LinRange(minimum(Pe_ax), maximum(Pe_ax), 500)
            τ_ax   = LinRange(minimum(τII_ax), maximum(τII_ax), 500)

            xplot = P_ax .* sc.σ ./ 1e6
            yplot = τ_ax .* sc.σ ./ 1e6

            # Models, colours and labels
            models = (
                # materials_T,
                # materials_DP,
                # materials_DPH,
                # materials_DPC,
                materials_MCC,
            )

            colors = [:black, :blue, :cyan, :green, :red]

            labels = [
                # "Mode 1 (linear)",
                # "Drucker-Prager",
                # "Hyp. Drucker-Prager",
                # "Drucker-Prager Cap.",
                "Mod. Cam-clay",
            ]

            # Evaluate and plot f = 0 and q = 0
            for (model, color) in zip(models, colors)

                f = [
                    F(model.plasticity, τ, P, 0*P, 0.0, 0.0, 1)
                    for P in P_ax, τ in τ_ax
                ]

                dQdp = zeros(length(P_ax),length(τ_ax))
                for i in eachindex(P_ax), j in eachindex(τ_ax)

                    P  = P_ax[i]
                    Pf = 0.1 * P_ax[i]
                    τ = τ_ax[j]

                    dQdpt = ForwardDiff.derivative(P  -> Q(model.plasticity, τ, P, Pf, 0.0, 0.0, 1), P )
                    dQdpf = ForwardDiff.derivative(Pf -> Q(model.plasticity, τ, P, Pf, 0.0, 0.0, 1), Pf)
                    dQdp[i,j]  = dQdpt - dQdpf

                end
                

                mat = materials_MCC

                if model == mat
                    N    = 100
                    fmin = 1.0
                    fmax = 3.0 
                    inds = findall(x -> fmin < x < fmax, f)
                    samples = rand(inds, N)

                    I = getindex.(samples, 1)
                    J = getindex.(samples, 2)
                    P_t = P_ax[I]
                    τ_t = τ_ax[J]
                    scatter!(ax, P_t.* sc.σ ./ 1e6, τ_t.* sc.σ ./ 1e6)
                
                    pf = 1e7/sc.σ *ones(size(P_t)) 
                    # pe = pt - pf
                    pe = P_t
                    p̄ = P_t + pf

                    pl   = mat.plasticity
                    ph  = 1
                    λ̇   = 0.0
                    Φ   = 1e-2

                    p̄c  = zeros(N)
                    pfc = zeros(N)
                    pec = zeros(N)
                    τc  = zeros(N)
                    suc = zeros(Bool, N)

                    
                    for i = 1:length(pe)
                        τII = τ_t[i]
                        Pt  = p̄[i]
                        Pf  = pf[i]
                        Pe  = Pt - Pf
                        f_trial    = F(pl, τII,  Pt, Pf, Φ, λ̇, ph)
                        x = @SVector [τII, Pt, Pf, λ̇, Φ]
                        x0 =copy(x)
                        plastic_correction = false

                        nr   = 1.0
                        nr0  = 1.0
                        tol  = 1e-14

                        τc[i]  = x[1]
                        p̄c[i]  = x[2] 
                        pfc[i] = x[3]
                        pec[i] = x[2] - x[3]

                        # Return mapping
                        if f_trial > -1e-13
                            plastic_correction = true
                            ηv  = mat.η0[ph]
                            ηe  = mat.G[ph]* Δt
                            ηve = inv(1/ηv + 1/ηe)
                            ε̇II_eff = 1/2
                            divVs, divqD = 0.0, 0.0
                            Pt0, Pf0 =   1.5*Pt, 0.5*Pf
                            Φ0 = Φ
                            KΦ, Ks, Kf = mat.KΦ[ph], mat.Ks[ph], mat.Kf[ph]
                            ξ0, m = mat.ξ0[ph],  mat.m[ph]

                            # This is the proper return mapping with plasticity
                            args = (ηve, Δt, ε̇II_eff, τII,       Pt,       Pf,       divVs, divqD, Φ,       Pt0, Pf0, Φ0, KΦ, Ks, Kf, ξ0, m, pl, ph, mat.single_phase)
                            
                            for iter=1:50
                                r, J = fd_value_and_jacobian(StagFDTools.TwoPhases.residual_two_phase_P, x, args...)
                                Δx   = -J \ r
                                α    = StagFDTools.TwoPhases.bt_line_search(Δx, J, x, r, args, α=1.0, ρ=0.5, c=1.0e-4, α_min=1.0e-8)
                                x   += α*Δx
                                nr   = StagFDTools.TwoPhases.mynorm(r)
                                if iter==1 
                                    nr0 = nr
                                end
                                # if iter==50
                                #     error("Local iteration failed: nr=$(nr) nr0=$(nr0) f = $(f_trial) x0 = $(x0), α = $(α) ")
                                # end
                                if nr/nr0 < tol && x[4]>0
                                    # if  x[4]<0
                                    #     print("Negative multiplier!!!")

                                    # end
                                    # @show x[4]

                                    suc[i] = true
                                    τc[i]  = x[1]
                                    p̄c[i]  = x[2] 
                                    pfc[i] = x[3]
                                    pec[i] = p̄c[i] - pfc[i]
                                    scatter!(ax, pec[i].* sc.σ ./ 1e6, τc[i].* sc.σ ./ 1e6, color=:red)
                                    break
                                end
                            end

                            if !suc[i]
                                scatter!(ax, pec[i].* sc.σ ./ 1e6, τc[i].* sc.σ ./ 1e6, color=:green, marker=:cross)
                            end           
                            
                        end
                    end
                    @show sum(suc)                
                end

                q = [
                    Q(model.plasticity, τ, P, 0*P, 0.0, 0.0, 1)
                    for P in P_ax, τ in τ_ax
                ]

                contour!(
                    ax, xplot, yplot, f,
                    levels = [0.0],
                    linewidth = 2,
                    color = color
                )
                contour!(
                    ax, xplot, yplot, dQdp,
                    levels = [0.0],
                    linewidth = 2,
                    color = color
                )

                contour!(
                    ax, xplot, yplot, q,
                    levels = [0.0],
                    linewidth = 2,
                    linestyle = :dash,
                    color = color
                )
            end

            # Legend
            f_elements = [
                LineElement(color = c, linewidth = 2, linestyle = :solid)
                for c in colors
            ]

            q_elements = [
                LineElement(color = c, linewidth = 2, linestyle = :dash)
                for c in colors
            ]

            Legend(
                fig[1, 2],
                [f_elements, q_elements],
                [labels, labels],
                [L"F = 0", L"Q = 0"],
                framevisible = false,
                tellheight = false,
                tellwidth = false
            )

            colgap!(fig.layout, 1, 30)            

            # axislegend(ax, position=:rb)
            return fig
        end
        fig = visualization && with_theme(figure, theme_latexfonts())
        display(fig)

        save("/Users/tduretz/PowerFolders/_manuscripts/TwoPhasePressure/_PoroVEP/figures/yields.png", fig, px_per_unit = 4)

        #-------------------------------------------# 

    # end

    #--------------------------------------------#
end

function Run()

    # Homogeneous test
    # n_nx = 1
    # n_nt = 1
    # nc   = (x=n_nx*50, y=n_nx*25)
    # nt   = 40*n_nt
    # main(nc, nt, n_nt, homo=true, niter=2)

    ###################################

    # # Does not complete successfully - crashes at step 10
    # n_nx = 16
    # n_nt = 1
    # nc   = (x=n_nx*50, y=n_nx*25)
    # nt   = 40*n_nt
    # main(nc, nt, n_nt);

    ###################################

    # # Resolution test dt
    # n_nx = 4
    # n_nt = 8
    # nc   = (x=n_nx*50, y=n_nx*25)
    # nt   = Int64(40*n_nt)
    # main(nc, nt, n_nt);

    ###################################

    # # with eta_vp
    n_nx = 2
    n_nt = 1
    nc   = (x=n_nx*50, y=n_nx*25)
    nt   = 100#4*n_nt
    D_BC = @SMatrix([1 0; 0 -1] )
    main(D_BC, nc, nt, n_nt; ηvp=0*1e21, homo=false); #1e20
    
end

Run();

