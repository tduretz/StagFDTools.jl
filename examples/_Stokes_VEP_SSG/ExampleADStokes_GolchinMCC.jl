using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics:mean
using DifferentiationInterface
using TimerOutputs, CairoMakie

function line(p, K, Δt, η_ve, ψ, p1, t1)
    p2 = p1 + K*Δt*sind(ψ)
    t2 = t1 - η_ve
    a  = (t2-t1)/(p2-p1)
    b  = t2 - a*p2
    return a*p + b
end

@views function main(nc, θgouge)
    #--------------------------------------------#

    # Scaling
    sc  = (σ=1e9, L=1, t=1e6)

    # Parameters
    width     = 1.0/sc.L
    height    = 1.0/sc.L
    thickness = 0.2/sc.L
    θgouge    = (90-θgouge) /180*π
    ε̇xx       = 1e-6*sc.t
    Pbg       = 1/2*5e7/sc.σ

    # Boundary loading type
    # config = :NS_Neumann
    config = :free_slip

    # # mode 1
    # nt     = 85*2
    # Δt0    = 5e0/sc.t
    # D_BC   = @SMatrix( [  ε̇xx  0.;
    #                       0  -ε̇xx*0 ])

    # mode 2
    nt     = 85
    Δt0       = 5e1/sc.t
    D_BC   = @SMatrix( [  ε̇xx  0.;
                          0  -ε̇xx ])

    # Material parameters
    nphases = 2
    materials = initialize_materials(nphases; plasticity=Golchin2021, compressible=true)
    materials.g   .= [0.0,  0.0]
    materials.ρ   .= [0.0,  0.0]
    materials.n   .= [1.0,  1.0]
    materials.η0  .= [1e48, 1e28] ./ sc.σ ./ sc.t
    materials.G   .= [1e10, 1e9]  ./ sc.σ
    materials.β   .= [1e-11, 1e-10] .* sc.σ
    materials.plasticity.C   .= [10e6,  10e6]  ./ sc.σ
    materials.plasticity.ϕ   .= [35.0,  35.0]
    materials.plasticity.ψ   .= [15.0,  15.0]
    materials.plasticity.ηvp .= [5e9,   5e9]   ./ sc.σ ./ sc.t
    materials.plasticity.Pc  .= [6e7,   6e7]   ./ sc.σ
    materials.plasticity.σT  .= [5e6,   5.0e6] ./ sc.σ
    materials.plasticity.a   .= [0.5,   0.5]
    materials.plasticity.b   .= [0.0,   0.0]
    materials.plasticity.c   .= [0.5,   0.5]
    preprocess!(materials)

    # Geometry
    L     = (x=width/sc.L, y=height/sc.L)

    # Solver parameters
    iter_params = IterParams(niter=25, ϵ_nl=1e-9, α=LinRange(0.05, 1.0, 10))

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Discretisation
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Set material geometry
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= 0.1^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= 0.1^2] .= 2
    FillPhaseRatios!(a)

    a.Pt  .= Pbg#*rand(size(a.Pt)...)
    a.Pt0 .= a.Pt
    a.Pti .= a.Pt

    #--------------------------------------------#

    rvec   = zeros(length(iter_params.α))
    err    = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    probes = (τII = zeros(nt), fric = zeros(nt), t = zeros(nt), εxx=zeros(nt), εyy=zeros(nt), σyyN=zeros(nt), σyyS=zeros(nt), σxxW=zeros(nt), σxxE=zeros(nt))
    to     = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Post process stress and strain rate
        τII_rock  = a.τ.II[inx_c,iny_c][a.phases.c[inx_c,iny_c].==1]
        P_rock    =   a.Pt[inx_c,iny_c][a.phases.c[inx_c,iny_c].==1]
        λ̇_rock    =  a.λ̇.c[inx_c,iny_c][a.phases.c[inx_c,iny_c].==1]

        # τII_gouge = a.τ.II[inx_c,iny_c][a.phases.c[inx_c,iny_c].==2]
        # P_gouge   =  a.Pt[inx_c,iny_c][a.phases.c[inx_c,iny_c].==2]

        # Principal stress
        σ1 = (x = zeros(size(a.Pt)), y = zeros(size(a.Pt)), v = zeros(size(a.Pt)))
        τxyc = 0.25*(a.τ.xy[1:end-1,1:end-1] .+ a.τ.xy[2:end-0,1:end-1] .+ a.τ.xy[1:end-1,2:end-0] .+ a.τ.xy[2:end-0,2:end-0])

        for i in inx_c, j in iny_c
            σ  = @SMatrix[-a.Pt[i,j]+a.τ.xx[i,j] τxyc[i,j] 0.; τxyc[i,j] -a.Pt[i,j]+a.τ.yy[i,j] 0.; 0. 0. -a.Pt[i,j]+(-a.τ.xx[i,j]-a.τ.yy[i,j])]
            v  = eigvecs(σ)
            σp = eigvals(σ)
            σ1
            scale = sqrt(v[1,1]^2 + v[2,1]^2)
            σ1.x[i,j] = v[1,1]/scale
            σ1.y[i,j] = v[2,1]/scale
            σ1.v[i] = σp[1]
        end

        # Store probes data
        probes.t[it]    = it*Δ.t
        probes.τII[it]  = mean(a.τ.II[inx_c, iny_c])
        probes.σxxW[it] = a.τ.xx[2,     Int64(floor(nc.y/2))] - a.Pt[2,     Int64(floor(nc.y/2))]
        probes.σxxE[it] = a.τ.xx[end-1, Int64(floor(nc.y/2))] - a.Pt[end-1, Int64(floor(nc.y/2))]
        probes.σyyS[it] = a.τ.yy[Int64(floor(nc.x/2)),     2] - a.Pt[Int64(floor(nc.x/2)),     2]
        probes.σyyN[it] = a.τ.yy[Int64(floor(nc.x/2)), end-1] - a.Pt[Int64(floor(nc.x/2)), end-1]

        i_midx = Int64(floor(nc.x))
        probes.fric[it] = mean(.-τxyc[i_midx, end-3]./(-a.Pt[i_midx, end-3] .+ a.τ.yy[i_midx, end-3]))

        @show minimum(a.Pt)*sc.σ,  maximum(a.Pt)*sc.σ

        # Visualise
        function figure()
            ftsz = 25
            fig = Figure(size=(1000, 1000))
            empty!(fig)
            ax  = Axis(fig[1:2,1], aspect=DataAspect(), title="Plastic Strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            eps   = 1e-1
            # field = a.Pt[inx_c,iny_c] .* sc.σ
            field = log10.((a.λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            hm = heatmap!(ax, a.X.c.x.*sc.L, a.X.c.y.*sc.L, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x.*sc.L, a.X.c.y.*sc.L,  a.phases.c[inx_c,iny_c], color=:white)
            Colorbar(fig[3, 1], hm, label = L"$\dot\lambda$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            Vxc = (0.5*(a.V.x[1:end-1,2:end-1] + a.V.x[2:end,2:end-1]))[2:end-1,2:end-1].*sc.L/sc.t
            Vyc = (0.5*(a.V.y[2:end-1,1:end-1] + a.V.y[2:end-1,2:end]))[2:end-1,2:end-1].*sc.L/sc.t
            step = 20
            arrows2d!(ax, a.X.c.x[1:step:end].*sc.L, a.X.c.y[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=500000.4, color=:white)
            # arrows2d!(ax, xc[1:st:end], yc[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.04, linewidth=2, color=:white)
            xlims!(ax, minimum(a.X.v.x).*sc.L, maximum(a.X.v.x).*sc.L)
            # ax  = Axis(fig[1,2], xlabel="Displacement", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:nt]/sc.t, probes.τII[1:nt]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σxxW[1:nt]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σxxE[1:nt]*sc.σ./1e6, marker=:star5, markersize=20 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σyyN[1:nt]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σyyS[1:nt]*sc.σ./1e6 )
            ax  = Axis(fig[1,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            scatter!(ax, 1:iter, log10.(err.x[1:iter]./err.x[1]) )
            scatter!(ax, 1:iter, log10.(err.y[1:iter]./err.y[1]) )
            scatter!(ax, 1:iter, log10.(err.p[1:iter]./err.p[1]) )
            ylims!(ax, -15, 1)
            ax  = Axis(fig[2,2], title=L"$$Stress space", xlabel=L"$P$", ylabel=L"$\tau_{II}$", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            P_ax       = LinRange(-10/1e3, 100/1e3, 100)
            # τ_ax_rock = materials.C[1]*materials.cosϕ[1] .+ P_ax.*materials.sinϕ[1]
            # lines!(ax, P_ax*sc.σ/1e6, τ_ax_rock*sc.σ/1e6, color=:black)

            # Plot yield
            m = materials.plasticity
            P_ax       = LinRange(-m.σT[1]+1e-4, 80/1e3, 100)
            τ_ax       = LinRange( 0, 60/1e3, 100)
            f_max       = zeros(length(P_ax), length(τ_ax))
            f_min       = zeros(length(P_ax), length(τ_ax))
            q          = zeros(length(P_ax), length(τ_ax))
            yieldf = Golchin2021()
            for i in eachindex(P_ax), j in eachindex(τ_ax)
                p = (M=m.M[1], N=m.N[1], Pt=-m.σT[1], Pc=m.Pc[1], α=m.a[1], β=m.b[1], γ=m.c[1], ηvp=m.ηvp[1])
                f_max[i,j] = Yield(@SVector([τ_ax[j], P_ax[i], maximum(a.λ̇.c)]), p, yieldf)
                f_min[i,j] = Yield(@SVector([τ_ax[j], P_ax[i], 0.0]), p, yieldf)
                q[i,j] = Potential(@SVector([τ_ax[j], P_ax[i], 0.0]), p, yieldf)
            end
            contour!(ax, P_ax*sc.σ/1e6, τ_ax*sc.σ/1e6, f_max*sc.σ./1e6, levels=[0., 0.0], color=:red)
            contour!(ax, P_ax*sc.σ/1e6, τ_ax*sc.σ/1e6, f_min*sc.σ./1e6, levels=[0., 0.0], color=:black)
            contour!(ax, P_ax*sc.σ/1e6, τ_ax*sc.σ/1e6, q*sc.σ./1e6, levels=[0., 0.0], color=:red, linestyle=:dash)

            cosΨ, sinΨ, C, σT = m.cosϕ[1], m.sinϕ[1], m.sinϕ[1], m.σT[1]
            B = C * cosΨ - σT*sinΨ
            dQdtau = @. τII_rock /sqrt(τII_rock^2 + B^2)
            scatter!(ax, (P_rock .+ 0*sinΨ .* λ̇_rock.*m.ηvp[1])*sc.σ/1e6, (τII_rock .+ 0*dQdtau.*λ̇_rock.*m.ηvp[1])*sc.σ/1e6, color=:black )

            # τ_ax_gouge = materials.C[2]*materials.cosϕ[2] .+ P_ax.*materials.sinϕ[2]
            # lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:red)
            # scatter!(ax, P_gouge*sc.σ/1e6, τII_gouge*sc.σ/1e6, color=:red )
            display(fig)
        end
        with_theme(figure, theme_latexfonts())
    end

    display(to)

end

let
    main((x = 100, y = 100), 60)
end
