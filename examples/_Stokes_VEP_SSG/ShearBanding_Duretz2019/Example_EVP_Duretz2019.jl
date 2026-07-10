#---------------------------------------------------------------------------------------
# Compute deformation field with VEVP rheology and benchmark with M2Di code from Duretz et al., 2019
#---------------------------------------------------------------------------------------
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs
using MAT
using CairoMakie

function invariants(Δ, τ, ε̇, inx_c, iny_c, εII)
    
    τxyc = av2D(τ.xy)
    τII  = sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + (-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
    ε̇xyc = av2D(ε̇.xy)
    ε̇II  = sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + (-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )
    
    # Strain increment
    εII .+= ε̇II.*Δ.t
    
    return τII, ε̇II, εII
end

function cumulated_strain(ε̇, ε̇kk, εII, inx_c, iny_c, Δ)
    
    ε̇xx = ε̇.xx .+ 1/3*ε̇kk
    ε̇yy = ε̇.yy .+ 1/3*ε̇kk
    ε̇zz = 0. .+ 1/3*ε̇kk
    ε̇xy = ε̇.xy
    ε̇II  = sqrt.( 0.5.*ε̇xx[inx_c,iny_c].^2 + 0.5*ε̇yy[inx_c,iny_c].^2 .+ 0.5*ε̇zz^2 .+ ε̇xy[inx_c,iny_c].^2 )
    
    # Strain increment
    εII .+= ε̇II.*Δ.t
    
    return εII
end

function section(εII, Δ, nc, materials, L)

    θ = 45. - (materials.plasticity.ϕ[1] + materials.plasticity.ψ[1])/4
    line = Δ.y*nc.y*0.5
    N    = Int64(round(line/Δ.y))

    C = zeros(2, N)
    C[1, :] .= 0.0
    C[2, :]  = LinRange(-line*0.5, line*0.5, N)

    Rot = [cosd(θ) -sind(θ); sind(θ) cosd(θ)]

    # Sample εII along the rotated line, averaging over a few offsets
    n_elem = 2
    ε_sum  = zeros(N)
    for k = -n_elem:n_elem
        D = copy(C)
        D[1, :] .+= k*Δ.x
        D′ = Rot * D
        indx′ = Int64.(round.(D′[1, :]./Δ.x .+ nc.x*0.5 .+ 0.5))
        indy′ = Int64.(round.(D′[2, :]./Δ.y .+ nc.y*0.5 .+ 0.5))
        for m = 1:N
            ε_sum[m] += εII[indx′[m], indy′[m]]
        end
    end
    ε_prof = ε_sum ./ (2*n_elem + 1)

    # Rotate C so the displayed line matches the sampling direction
    C = Rot * C

    return ε_prof, C
end

function MatlabCheck(materials, res, nt)
    
    # 1) Import Variables
    # path = @__DIR__
    # folder = /r51_vp
    path= "path"
    @show path
    m_res = res + 1
    tstep = nt
    name = @sprintf("TimeEvol%04d_Res%d.mat", tstep, m_res)
    filename = joinpath(path, name)
    @show filename
    @show keys(matopen(filename))

    if isfile(filename)
        var = matread(filename)

        # 2) Name Variables
        Δ    = (x = var["dx"],  y = var["dy"], t = var["dt"])
        x    = (min = var["xmin"],  max = var["xmax"])
        y    = (min = var["ymin"],  max = var["ymax"])
        nc   = (x = Int64(var["ncx"]), y = Int64(var["ncy"]))
        xc   = LinRange(x.min, x.max, nc.x)
        yc   = LinRange(y.min, y.max, nc.y)
        L    = (x = var["Lx"], y = var["Ly"])
        εII  = var["Eii"]
        ε̇II  = εII ./ Δ.t
        incr = var["incr0"]

        # 3) Build section across shear band and create tuples
        m_sec = section(εII, Δ, nc, materials, L)
        m = (incr=incr, Δ=Δ, εII=εII, ε̇II=ε̇II, xc = xc, yc = yc, ε_prof=m_sec[1], C = m_sec[2])
        return m
    else
        return nothing
    end
end

@views function main(nc, flag, res)
    #--------------------------------------------#

    # Markers
    nmpc = (x = 4, y = 4)
    noise = false

    # Scaling
    sc = (σ = 1, L = 1, t = 1)

    # Boundary loading type
    config = :free_slip
    ε̇bg = (5e-10) * sc.t
    D_BC   = @SMatrix( [ -ε̇bg 0.;
                          0  ε̇bg ]) 
    bulk_rate = D_BC[4]
    ε̇kk = tr(D_BC) # whatch out

    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; plasticity=DruckerPrager,compressible=true)
    
    # Parameters
    params_bg = (ρ=1.0, n=1.0, η0=2e50, G=1.0, C=1.74e-4, ϕ=30., ηvp=2.5e2, β=0.5, ψ=10.)
    params_in = (ρ=1.0, n=1.0, η0=2e40, G=0.25, C=1.74e-4, ϕ=30., ηvp=2.5e2, β=0.5, ψ=10.)
    r_incl = 5e-2

    materials.g .= [0. , 0.]
    materials.ρ .= [params_bg.ρ , params_in.ρ]
    materials.n .= [params_bg.n , params_in.n]
    materials.η0 .= [params_bg.η0 , params_in.η0]
    materials.G .= [params_bg.G , params_in.G]
    materials.β .= [params_bg.β , params_in.β]
    materials.plasticity.C .= [params_bg.C , params_in.C]
    materials.plasticity.ϕ .= [params_bg.ϕ , params_in.ϕ]
    materials.plasticity.ηvp .= [params_bg.ηvp , params_in.ηvp]
    materials.plasticity.ψ .= [params_bg.ψ , params_in.ψ]

    preprocess!(materials)

    iter_params = IterParams(niter=10, ϵ_nl=1e-10, α = LinRange(0.05, 1.0, 10))

    # Time steps and bulk strain intervals
    Δt0    = 1e4/sc.t
    nt     = 60
    if flag.strain_int
        ε_bulk = LinRange(1e-4,3e-4,5)
        d = 1
    end

    # Intialise field
    L = (x=1.0, y=0.7)
    Δ = (x=L.x / nc.x, y=L.y / nc.y, t=Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

   # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    # a.X bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)


    # Initial velocity & pressure field
    @views a.V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.c.x' 
    @views a.V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*a.X.c.x .+ D_BC[2,2]*a.X.v.y'
    @views a.Pt[inx_c, iny_c ]  .= 0.                 
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.y .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # NO Markers
    # Material geometry
    ccord = (x=-L.x/2, y=-L.y/2)
    @views a.phases.c[inx_c, iny_c][((a.X.c.x .-ccord.x).^2 .+ ((a.X.c.y').-ccord.y).^2) .<= (r_incl)^2] .= 2
    @views a.phases.v[inx_v, iny_v][((a.X.v.x .-ccord.x).^2 .+ ((a.X.v.y').-ccord.y).^2) .<= (r_incl)^2] .= 2
    FillPhaseRatios!(a)

    #------------------------------------------------------------------#

    # Post-processing and plotting initialisation
    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()
    εII  = zeros(nc.x,nc.y)
    if flag.strain_evo
        fig7 = Figure(size=(700, 300))
        ax7 = Axis(fig7[1, 1], xlabel="x", ylabel="εᵢᵢ [10⁻³]", title="StagFD")
    end
    #-----------------------------------------------------------------#

    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)


        #--------------------------------------------#

        # (τII, ε̇II, εII) = invariants(Δ, τ, ε̇, inx_c, iny_c, εII)
        εII = cumulated_strain(a.ε̇, ε̇kk, εII, inx_c, iny_c, Δ)
        
        #--------------------------------------------#
        # Plot fields
         if flag.fields
            fig_fields = Figure(size=(1000, 800))
            ax1 = Axis(fig_fields[1, 1], aspect=DataAspect(), title = "time step $it")
            ax2 = Axis(fig_fields[1, 3], aspect=DataAspect(), title = "time step $it")
            ax3 = Axis(fig_fields[2, 1], aspect=DataAspect(), title = "time step $it")
            ax4 = Axis(fig_fields[2, 3], aspect=DataAspect(), title = "time step $it")

            hm1 = heatmap!(ax1, a.X.v.x, a.X.c.y, (V.x[inx_Vx,iny_Vx]').*1e9./sc.t)
            hm2 = heatmap!(ax2, a.X.c.x, a.X.c.y, (Pt[inx_c,iny_c]').*1e4.*sc.σ; colormap=:turbo)
            hm3 = heatmap!(ax3, a.X.c.x, a.X.c.y, log10.(εII)'; colormap=:coolwarm)
            hm4 = heatmap!(ax4, a.X.c.x, a.X.c.y, ((τII').*1e4.*sc.σ); colormap=:turbo)
            Colorbar(fig_fields[1, 2], hm1, label="Vx [10⁻⁹ m s⁻¹]")
            Colorbar(fig_fields[1, 4], hm2, label="P [10⁻⁴ Pa]")
            Colorbar(fig_fields[2, 2], hm3, label="log10(ε̇)")
            Colorbar(fig_fields[2, 4], hm4, label="τ [10⁻⁴ Pa]")


            xlims!(ax1, extrema(a.X.c.x)...)
            xlims!(ax2, extrema(a.X.c.x)...)
            xlims!(ax3, extrema(a.X.c.x)...)
            xlims!(ax4, extrema(a.X.c.x)...)
            # if flag.Matlab && m !== nothing
            #     fig_cmp = Figure(size=(1000, 400))
            #     ax_cmp1 = Axis(fig_cmp[1, 1], title="εII", aspect=DataAspect())
            #     ax_cmp2 = Axis(fig_cmp[1, 2], title="εII from M2Di", aspect=DataAspect())
            #     heatmap!(ax_cmp1, a.X.c.x, a.X.c.y, log10.(εII)'; colormap=:coolwarm)
            #     heatmap!(ax_cmp2, m.xc, m.yc, log10.(m.εII)'; colormap=:coolwarm)
            #     xlims!(ax_cmp1, extrema(a.X.c.x)...)
            #     xlims!(ax_cmp2, extrema(m.xc)...)
            #     display(fig_cmp)
            # else
            #     display(fig_fields)
            # end

            #z0 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright)
            #z0 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
            #z0 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
            #z0 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
            # dislpay(z0)
        end
        #--------------------------------------------#
        # PLot time evolution of accumulated strain
        if flag.strain_evo
            (ε̇_prof, C) = section(ε̇II, Δ, nc,materials,L)
            (P_prof, C) = section(ε̇II, Δ, nc,materials,L)
            (ε_prof, C) = section(εII, Δ, nc,materials,L)
            if flag.strain_int
                cur_ε = bulk_rate*Δ.t*it
                @show(cur_ε )
                if cur_ε ≈ ε_bulk[d]
                    lines!(ax7, C[2,:], (ε_prof)*1e3, label = "$(@sprintf("%0.1f", cur_ε*1e4)) [10⁻⁴]")
                    axislegend(ax7)
                    # if flag.Matlab
                    #     if m !== nothing
                    #         lines!(ax8, m.C[2,:], (m.ε_prof)*1e3, label = "$(@sprintf("%0.1f", m.incr*it*1e4)) [10⁻⁴]")
                    #         axislegend(ax8)
                    #         display(fig8)
                    #     end
                    # else
                    #     display(fig7)
                    # end 
                    d += 1
                    if d > 5
                        d = 5
                    end
                end
            end
        end
    end

    #--------------------------------------------#
    # Compare resolutions
    (ε_prof, C) = section(εII, Δ, nc, materials,L)
    (ε̇_prof, C) = section(a.ε̇.II, Δ, nc, materials,L)
    (P_prof, C) = section(a.Pt, Δ, nc, materials,L)
    # if flag.Matlab
    #     m_outp = MatlabCheck(materials, res,nt)
    #     if m_outp !== nothing
    #         (m_ε_prof, m_C) = m_outp
    #     end
    # else
    #     m_ε_prof = 1
    #     m_C = 1
    # end
    display(to)

    return (ε = ε_prof, ε̇ = ε̇_prof, P = P_prof), C, εII, a.X.c.x, a.X.c.y
end

#                                       M A I N    

let
    resolution = [51, 101] #, 201, 401]
    n = length(resolution)

    εprofiles = Vector{Vector{Float64}}(undef, n)
    ε̇profiles = Vector{Vector{Float64}}(undef, n)
    Pprofiles = Vector{Vector{Float64}}(undef, n)
    cuts     = Vector{Matrix{Float64}}(undef, n)
    grids_x  = Vector{AbstractVector{Float64}}(undef, n)
    grids_y  = Vector{AbstractVector{Float64}}(undef, n)
    eps_mats = Vector{Matrix{Float64}}(undef, n)

    for (i, res) in enumerate(resolution)
        flag = (strain_evo = false, fields = false, strain_int = false)
        (prof, C, εII, xc, yc) = main((x = res, y = res), flag, res)
        εprofiles[i] = prof.ε
        ε̇profiles[i] = prof.ε̇
        Pprofiles[i] = prof.P
        cuts[i]     = C
        grids_x[i]  = xc
        grids_y[i]  = yc
        eps_mats[i] = εII
    end

    # Figure 1
    ncols = min(2, n)
    nrows = cld(n, ncols)
    fig_heat = Figure(size = (500 * ncols + 100, 450 * nrows))

    for idx in 1:n
        row    = div(idx - 1, ncols) + 1
        col    = ((idx - 1) % ncols) + 1
        ax_col = 2col - 1 
        cb_col = 2col

        ax = Axis(fig_heat[row, ax_col], title  = "$(resolution[idx])²", xlabel = "𝑥", ylabel = "𝑦", aspect = DataAspect())

        xc, yc, εmat = grids_x[idx], grids_y[idx], eps_mats[idx]
        hm = heatmap!(ax, xc, yc, log10.(εmat); colormap = :lajolla, colorrange = (-3.6, -2.6))
        Colorbar(fig_heat[row, cb_col], hm)

        C = cuts[idx]
        lines!(ax, C[1, :], C[2, :], color = :white, linewidth = 2)
    end

    display(fig_heat)

    # Figure 2
    fig_prof = Figure(size = (700, 500))
    axε = Axis(fig_prof[1, 1], xlabel = "𝑥", ylabel = "εII [10⁻³]")
    axε̇ = Axis(fig_prof[2, 1], xlabel = "𝑥", ylabel = "ε̇II [10⁻⁹]")
    axP = Axis(fig_prof[3, 1], xlabel = "𝑥", ylabel = "P [10⁻⁴]")
    

    for i in 1:n
        C     = cuts[i]
        εprof =  εprofiles[i]
        s     = vec(sqrt.(sum((C .- C[:, 1]).^2, dims = 1)))
        s = s .- s[end] / 2
        lines!(axε, s, εprof .* 1e3, label = "$(resolution[i])²")

    end
    for i in 1:n
        C     = cuts[i]
        ε̇prof = ε̇profiles[i]
        s     = vec(sqrt.(sum((C .- C[:, 1]).^2, dims = 1)))
        s = s .- s[end] / 2 
        lines!(axε̇, s, ε̇prof .* 1e9, label = "$(resolution[i])²")
    end
    for i in 1:n
        C     = cuts[i]
        Pprof = Pprofiles[i]
        s     = vec(sqrt.(sum((C .- C[:, 1]).^2, dims = 1)))
        s = s .- s[end] / 2 
        lines!(axP, s, Pprof .* 1e4, label = "$(resolution[i])²")
    end

    axislegend(axP)
    axislegend(axε)
    axislegend(axε̇)
    display(fig_prof)
end



###########################################################################################
##########################################################################################