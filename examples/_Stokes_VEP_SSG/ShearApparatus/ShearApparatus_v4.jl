using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics:mean
using DifferentiationInterface
using TimerOutputs, CairoMakie

# comment rand() in initial pressure
# make large VP
# made :free_slip

@views function main(nc, θgouge)
    #--------------------------------------------#

    # Scaling
    sc  = (σ=1e9, L=1, t=1e6)

    # Parameters
    width     = 1.0/sc.L
    height    = 1.5/sc.L
    thickness = 0.1/sc.L
    θgouge    = (90-θgouge) /180*π
    Δt0       = 5e1/sc.t
    ε̇xx       = 1e-6*sc.t
    Pbg       = 1e8/sc.σ

    # Boundary loading type
    config = :EW_stress
    D_BC   = @SMatrix( [  ε̇xx  0.;
                          0  -ε̇xx ])
    σ_BC   = @SMatrix( [ -Pbg  0.;
                          0  -Pbg ])

    # Material parameters
    nphases                   = 4
    materials                 = initialize_materials( nphases, compressible = true, plasticity = DruckerPrager )
    materials.n              .= [  1.0,    1.0,     1.0,    1.0]             # Power law exponent
    materials.η0              .= [ 1e48,   1e28,    1e10,   1e48]./sc.σ./sc.t # Reference viscosity
    materials.G               .= [ 1e10,    5e9,    1e60,   1e10]./sc.σ       # Shear modulus
    materials.β                .= [1e-11,  1e-10,   1e-12,  1e-12].*sc.σ       # Compressibility
    materials.plasticity.C    .= [  1e8,    1e5,   15e60,  15e60]./sc.σ       # Cohesion
    materials.plasticity.ϕ    .= [  40.,    30.,     35.,    35.]             # Friction angle
    materials.plasticity.ψ    .= [  0.0,    5.0,     0.0,    0.0]             # Dilation angle
    materials.plasticity.ηvp  .= [ 1e11,   1e11,    1e11,   1e11].*0.5/sc.σ./sc.t # Viscoplastic regularisation
    #                              rock    gouge     salt   plates
    preprocess!(materials)

    # Geometry
    L     = (x=width/sc.L, y=height/sc.L)
    gouge = (
        Rectangle((0.0/sc.L, 0.0/sc.L+L.y/2), thickness/sc.L, 2.0/sc.L; θ = θgouge),
    )
    salt = (
        Rectangle((-.5/sc.L, 0.0/sc.L+L.y/2), 0.5/sc.L, 2.0/sc.L; θ = 0),
        Rectangle((0.5/sc.L, 0.0/sc.L+L.y/2), 0.5/sc.L, 2.0/sc.L; θ = 0),
    )
    plate = (
        Rectangle((0.0/sc.L, 0.7/sc.L+L.y/2), 1.1/sc.L, 0.1/sc.L; θ = 0),
        Rectangle((0.0/sc.L,-0.7/sc.L+L.y/2), 1.1/sc.L, 0.1/sc.L; θ = 0),
    )

    # Time steps
    nt    = 500

    # Newton solver
    iter_params = IterParams(niter=20, γ=1e5, ϵ_l=1e-11, ϵ_nl=1e-9, inexact=false, solver_type=:GCR, α=LinRange(0.05, 1.0, 6))

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Discretisation
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x   = (min=-L.x/2, max=L.x/2)
    y   = (min=0.0, max=L.y)

    # Markers
    nmpc  = (x=4, y=4)
    noise = false

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases, nmpc, noise)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]  .+ (a.type.Vx[     1, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]  .+ (a.type.Vx[   end, iny_Vx] .== :normal_stress) .* σ_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]  .+ (a.type.Vy[inx_Vy,     1 ] .== :normal_stress) .* σ_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]  .+ (a.type.Vy[inx_Vy,   end ] .== :normal_stress) .* σ_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Set material geometry (sharp phases, used for masks & plotting)
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([a.X.c_e.x[i], a.X.c_e.y[j]])

        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                a.phases.c[i, j] = 2
            end
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                a.phases.c[i, j] = 3
            end
        end
        for igeom in eachindex(plate) # Plate: phase 4
            if inside(𝐱, plate[igeom])
                a.phases.c[i, j] = 4
            end
        end
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([a.X.v_e.x[i], a.X.v_e.y[j]])

        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                a.phases.v[i, j] = 2
            end
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                a.phases.v[i, j] = 3
            end
        end
        for igeom in eachindex(plate) # Plate: phase 4
            if inside(𝐱, plate[igeom])
                a.phases.v[i, j] = 4
            end
        end
    end

    # Assign marker phases from the same geometry (drives the soft phase ratios)
    for I in eachindex(a.m.phase)
        𝐱 = @SVector([a.m.Xm[I], a.m.Ym[I]])
        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                a.m.phase[I] = 2
            end
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                a.m.phase[I] = 3
            end
        end
        for igeom in eachindex(plate) # Plate: phase 4
            if inside(𝐱, plate[igeom])
                a.m.phase[I] = 4
            end
        end
    end

    # Set phase ratios
    SetPhaseRatios!(a.phase_ratios, a.m, a.X.c_e.x, a.X.c_e.y, a.X.v_e.x, a.X.v_e.y, Δ, nphases)

    # check
    for I in CartesianIndices(a.phase_ratios.c)
        s = sum(a.phase_ratios.c[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.c at $I: sum = $s, values = $(a.phase_ratios.c[I])"
        end
    end

    a.Pt  .= Pbg #*rand(size(a.Pt)...)
    a.Pt0 .= a.Pt
    a.Pti .= a.Pt

    #--------------------------------------------#

    rvec   = zeros(length(iter_params.α))
    err    = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    probes = (τII = zeros(nt), τIIW = zeros(nt), τIIE = zeros(nt), τIIS = zeros(nt), τIIN = zeros(nt), true_fric = zeros(nt), app_fric = zeros(nt), t = zeros(nt), εxx=zeros(nt), εyy=zeros(nt), σyyN=zeros(nt), σyyS=zeros(nt), σxxW=zeros(nt), σxxE=zeros(nt), PW=zeros(nt), PE=zeros(nt), minθ=zeros(nt), maxθ=zeros(nt), meanθ=zeros(nt))
    to     = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = 0.25 * (a.τ.xy[1:end-1, 1:end-1] .+ a.τ.xy[2:end-0, 1:end-1] .+ a.τ.xy[1:end-1, 2:end-0] .+ a.τ.xy[2:end-0, 2:end-0])

        isrock      = zeros(Bool, size_c)
        isgouge     = zeros(Bool, size_c)
        [isrock[I]  = a.phase_ratios.c[I][1]≈1.0 for I in eachindex(a.phase_ratios.c)]
        [isgouge[I] = a.phase_ratios.c[I][2]≈1.0 for I in eachindex(a.phase_ratios.c)]

        τII_rock  = a.τ.II[isrock]
        P_rock    =   a.Pt[isrock]
        λ̇_rock    =  a.λ̇.c[isrock]

        τII_gouge = a.τ.II[isgouge]
        P_gouge   =   a.Pt[isgouge]
        λ̇_gouge   =  a.λ̇.c[isgouge]

        # Compute apparent and true friction locally
        app_fric       = zeros(size(a.Pt))
        true_fric      = zeros(size(a.Pt))
        Rot            = @SMatrix[cos(π/2 - θgouge) sin(π/2 - θgouge) 0; -sin(π/2 - θgouge) cos(π/2 - θgouge) 0; 0 0 1.0]
        app_fric_sum   = 0.0
        true_fric_sum  = 0.0
        app_sum        = 0
        true_sum       = 0

        for i in inx_c, j in iny_c
            if a.phases.c[i,j] == 2 # a.λ̇.c[i,j] > 1e-10
                app_sum     += 1
                σ  = @SMatrix[-a.Pt[i,j]+a.τ.xx[i,j] τxyc[i,j] 0.; τxyc[i,j] -a.Pt[i,j]+a.τ.yy[i,j] 0.; 0. 0. -a.Pt[i,j]+(-a.τ.xx[i,j]-a.τ.yy[i,j])]
                # Compute apparent friction
                σ′             = Rot * σ * Rot'
                app_fric[i,j]  = σ′[1,2] / σ′[2,2]
                app_fric_sum  += app_fric[i,j]
                # Compute true friction
                if a.λ̇.c[i,j] > 1e-10
                    true_sum      += 1
                    ph             = a.phases.c[i,j]
                    cxcosϕ         = materials.plasticity.C[ph] * materials.plasticity.cosϕ[ph]
                    ηvp            = materials.plasticity.ηvp[ph]
                    true_fric[i,j] = tand(asind( 1/a.Pt[i,j] * (a.τ.II[i,j] - cxcosϕ - ηvp*a.λ̇.c[i,j])  ))
                    true_fric_sum += true_fric[i,j]
                end
            end
        end

        # Store probes data
        σyy = a.τ.yy .- a.Pt
        ind_mid_x = Int64(floor(nc.x/2))
        ind_mid_y = Int64(floor(nc.y/2))
        probes.t[it]    = it*Δ.t
        probes.τII[it]  = mean(a.τ.II)
        probes.τIIW[it] = a.τ.II[2,     ind_mid_y]
        probes.τIIE[it] = a.τ.II[end-1, ind_mid_y]
        probes.τIIS[it] = a.τ.II[ind_mid_x,     2]
        probes.τIIN[it] = a.τ.II[ind_mid_x, end-1]
        probes.PW[it]   = a.Pt[2,     ind_mid_y]
        probes.PE[it]   = a.Pt[end-1, ind_mid_y]
        probes.σxxW[it] = a.τ.xx[2,     ind_mid_y] - a.Pt[2,     ind_mid_y]
        probes.σxxE[it] = a.τ.xx[end-1, ind_mid_y] - a.Pt[end-1, ind_mid_y]
        probes.σyyS[it] = mean(a.τ.yy[inx_c,     2] .- a.Pt[inx_c,     2])
        probes.σyyN[it] = mean(a.τ.yy[inx_c, end-1] .- a.Pt[inx_c, end-1])
        probes.app_fric[it] = app_fric_sum / app_sum
        probes.true_fric[it] =  true_fric_sum / true_sum

        # Stress angles
        probes.minθ[it]  = minimum(a.τ.θ[a.phases.c.==2])
        probes.meanθ[it] =    mean(a.τ.θ[a.phases.c.==2])
        probes.maxθ[it]  = maximum(a.τ.θ[a.phases.c.==2])

        # Visualise
        function figure()
            ftsz = 25
            fig = Figure(size=(1000, 1000))
            empty!(fig)

            # Split heatmap of the apparatus
            ax  = Axis(fig[1:3,1], aspect=DataAspect(), title="Apparent / True friction", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            eps   = 1e-1
            field1 = app_fric .+ eps
            field2 = true_fric .+ eps
            hm1 = heatmap!(ax, a.X.c.x[1:ind_mid_x].*sc.L, a.X.c.y.*sc.L, field1[1:ind_mid_x,:], colormap=:bluesreds, colorrange=(minimum(field1)-eps, maximum(field1)+eps))
            hm2 = heatmap!(ax, a.X.c.x[ind_mid_x:end].*sc.L, a.X.c.y.*sc.L, field2[ind_mid_x:end,:], colormap=:bluesreds, colorrange=(minimum(field2)-eps, maximum(field2)+eps))
            contour!(ax, a.X.c.x.*sc.L, a.X.c.y.*sc.L,  a.phases.c[inx_c,iny_c], color=:white)
            Colorbar(fig[4, 1], hm1, label = L"$\phi^\text{app}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            Colorbar(fig[4, 2], hm2, label = L"$\phi^\text{true}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            step = 10
            arrows2d!(ax, a.X.c.x[1:step:end], a.X.c.y[1:step:end], cos.(a.τ.θ)[inx_c,iny_c][1:step:end,1:step:end], sin.(a.τ.θ)[inx_c,iny_c][1:step:end,1:step:end], lengthscale=0.04, color=:white, tiplength = 0)
            xlims!(ax, minimum(a.X.v.x).*sc.L, maximum(a.X.v.x).*sc.L)
            lines!(ax, a.X.c.x[ind_mid_x].*sc.L *  ones(size(a.X.c.y)), a.X.c.y.*sc.L, color=:white, linewidth=4)

            # Zoom on the gouge
            ax  = Axis(fig[2,2], aspect=DataAspect(), title="Plastic Strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            eps   = 1e-1
            field = log10.((a.λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            hm = heatmap!(ax, a.X.c.x.*sc.L, a.X.c.y.*sc.L, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, a.X.c.x.*sc.L, a.X.c.y.*sc.L,  a.phases.c[inx_c,iny_c], color=:white)
            arrows2d!(ax, a.X.c.x[1:step:end], a.X.c.y[1:step:end], cos.(a.τ.θ)[inx_c,iny_c][1:step:end,1:step:end], sin.(a.τ.θ)[inx_c,iny_c][1:step:end,1:step:end], lengthscale=0.04, color=:white, tiplength = 0)
            xlims!(ax, -0.35, 0.35)
            ylims!(ax, 0.5, 1.0)

            ax  = Axis(fig[3,2], title=L"$$Stress space", xlabel=L"$P$", ylabel=L"$\tau_{II}$", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            P_ax       = LinRange(minimum(P_rock), maximum(P_rock), 100)
            τ_ax_rock  = materials.plasticity.C[1] * materials.plasticity.cosϕ[1] .+ P_ax.*materials.plasticity.sinϕ[1]
            τ_ax_gouge = materials.plasticity.C[2] * materials.plasticity.cosϕ[2] .+ P_ax.*materials.plasticity.sinϕ[2]
            lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:black)
            lines!(ax, P_ax*sc.σ/1e6, τ_ax_rock*sc.σ/1e6, color=:black)
            scatter!(ax,  P_rock*sc.σ/1e6, ( τII_rock .- λ̇_rock.* materials.plasticity.ηvp[2])*sc.σ/1e6, color=:blue )
            scatter!(ax, P_gouge*sc.σ/1e6, (τII_gouge .- λ̇_gouge.*materials.plasticity.ηvp[2])*sc.σ/1e6, color= :red )

            ax  = Axis(fig[0,1], xlabel="Displacement", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            scatter!(ax, probes.t[1:it]*ε̇xx*L.y*sc.L, probes.PW[1:it]*sc.σ./1e6, marker=:diamond )
            scatter!(ax, probes.t[1:it]*ε̇xx*L.y*sc.L, probes.PE[1:it]*sc.σ./1e6, marker=:diamond )
            scatter!(ax, probes.t[1:it]*ε̇xx*L.y*sc.L, probes.σxxW[1:it]*sc.σ./1e6, marker=:star5 )
            scatter!(ax, probes.t[1:it]*ε̇xx*L.y*sc.L, probes.σxxE[1:it]*sc.σ./1e6, marker=:star5 )
            scatter!(ax, probes.t[1:it]*ε̇xx*L.y*sc.L, probes.σyyN[1:it]*sc.σ./1e6, marker=:circle )
            scatter!(ax, probes.t[1:it]*ε̇xx*L.y*sc.L, probes.σyyS[1:it]*sc.σ./1e6, marker=:circle )

            ax  = Axis(fig[0,2], xlabel="time [hrs]", ylabel="-τxy/σyy", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            lines!(ax, probes.t[1:it]*sc.t/3600, ones(it)*tand(materials.plasticity.ϕ[2]), linestyle=:dash, color=:gray )
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.app_fric[1:it] )
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.true_fric[1:it], marker=:star5)

            ax  = Axis(fig[1,2], xlabel="time [hrs]", ylabel="σ1 angle gouge", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.minθ[1:it]*180/π )
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.meanθ[1:it]*180/π )
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.maxθ[1:it]*180/π )

            display(fig)

        end
        with_theme(figure, theme_latexfonts())
    end
    display(to)
end

let
    main((x = 200, y = 300), 60)
end
