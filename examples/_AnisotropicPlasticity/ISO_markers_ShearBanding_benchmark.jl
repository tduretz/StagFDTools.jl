#---------------------------------------------------------------------------------------
# Compute deformation field with VEVP rheology and benchmark with M2Di code from Duretz et al., 2018
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

function section(εII, Δ, nc, materials, L)

    # Angle of the section: Roscoe angle
    θ = 45. - (materials.plasticity.ϕ[1] + materials.plasticity.ψ[1])/4
    # θrad = deg2rad(θ)

    # Section initialisation
    line = Δ.y*nc.y*0.5
    C = zeros(2,Int64(round(line/Δ.y)))
    C[1,:] .= L.x*0.5 - L.x*0.5 
    C[2,:]  = LinRange(-line*0.5, line*0.5,Int64(round(line/Δ.y)))

    # Rotation matrix
    # Rot = [cos(θrad) -sin(θrad); sin(θrad) cos(θrad)]
    Rot = [cosd(θ) -sind(θ); sind(θ) cosd(θ)]


    # Find mean for a smooth line
    n_elem = 2
    ε_sum  = zeros(Int64(round(line/Δ.y)))
    for k = -n_elem:n_elem

        D = copy(C)
        D[1,:] .+= k*Δ.x
        D′ = Rot * D

        indx′ = Int64.(round.(D′[1,:]./Δ.x .+ nc.x*0.5 .+ 0.5))
        indy′ = Int64.(round.(D′[2,:]./Δ.y .+ nc.y*0.5 .+ 0.5))

        for m = 1 : Int64(round(line/Δ.y))
            i = indx′[m]
            j = indy′[m]

            ε_sum[m] += εII[i,j]
        end
    end

    n_val = n_elem*2+1
    ε_prof = ε_sum./n_val
    
    return ε_prof, C

end

function MatlabCheck(materials, res, nt)
    
    # 1) Import Variables
    # path = @__DIR__
    # folder = /r51_vp
    path= "/Users/filippozarabara/Documents/PHD/CODES/MATLAB/Shearbands_codes/r51_vp"
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
    noise = true

    # Scaling
    sc = (σ = 1, L = 1, t = 1)

    # Boundary loading type
    config = :free_slip
    ε̇bg = (5e-11) * sc.t
    D_BC   = @SMatrix( [ -ε̇bg 0.;
                          0  ε̇bg ]) 
    bulk_rate = D_BC[4]

    # Materials initialization
    materials = initialize_materials(2; plasticity=DruckerPrager,compressible=true)
    
    # Parameters
    params_bg = (ρ=1.0, n=1.0, η0=2e50, G=1.0, C=1.74e-4, ϕ=30., ηvp=2e3, β=0.5, ψ=10., ε̇=5e-11, rad=25e-4)
    params_in = (ρ=1.0, n=1.0, η0=2e40, G=0.25, C=1.74e-4, ϕ=30., ηvp=2e3, β=0.5, ψ=10.)

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

    # Time steps and bulk strain intervals
    Δt0    = 1e5/sc.t
    nt     = 40
    if flag.strain_int
        ε_bulk = LinRange(1e-4,3e-4,5)
        d = 1
    end

    # Newton solver
    niter = 10
    ϵ_nl  = 1e-10
    α     = LinRange(0.05, 1.0, 10)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    set_boundaries_template!(type, config, nc)

    #--------------------------------------------#

    # Equation numbering
    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    #--------------------------------------------#

    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏  = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r  = zeros(nVx + nVy + nPt)

    #--------------------------------------------#

    # Intialise field
    L   = (x=1.0, y=0.7)
    x   = (min=-L.x/2, max=L.x/2)
    y   = (min=-L.y/2, max=L.y/2)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    εII     = zeros(nc.x, nc.y)
    ξ       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    G       = (c  = zeros(size_c...), v  = zeros(size_v...))
    β       = (c  = zeros(size_c...), v  = zeros(size_v...))
    ρ       = (c  = zeros(size_c...), v  = zeros(size_v...))

    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    ΔPt     = (c=zeros(size_c...), Vx = zeros(size_x...), Vy = zeros(size_y...))

    Dc      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    # Mesh coordinates
    Grid = GenerateGrid(x,y,Δ,nc)

    # Initial velocity & pressure field
    @views V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*Grid.v.x .+ D_BC[1,2]*Grid.c.x' 
    @views V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*Grid.c.x .+ D_BC[2,2]*Grid.v.y'
    @views Pt[inx_c, iny_c ]  .= 0.                 
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*Grid.v.x .+ D_BC[1,2]*Grid.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*Grid.v.y .+ D_BC[1,2]*Grid.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*Grid.v.x[1]   .+ D_BC[2,2]*Grid.v.y)
        BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*Grid.v.x[end] .+ D_BC[2,2]*Grid.v.y)
    end

    #------------------------------------------------------------------#
    # Markers 

    # Initialise marker field
    m = InitialiseParticleField(nc, nmpc, L, Δ, materials, noise)
    phase_ratios, phase_weights = InitialisePhaseRatios(m, ε̇)
    mphase = ones(Int64,m.num...)

    # Set material geometry: circle
    ccord = (x=-L.x/2, y=-L.y/2)
    mphase[((m.Xm .- ccord.x).^2 .+ (m.Ym .- ccord.y).^2) .<= (25e-4)] .= 2

    # Set phase ratios
    PhaseRatios!(phase_ratios, phase_weights, m, mphase, Grid.c_e.x, Grid.c_e.y, Grid.v_e.x, Grid.v_e.y, Δ)
    # check 
    for I in CartesianIndices(phase_ratios.c)
        s = sum(phase_ratios.c[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.center at $I: sum = $s, values = $(phase_ratios.center[I])"
        end
    end
    # Cut ghost cells
    phase_ratios = (
        c   = phase_ratios.c[2:end-1,2:end-1],
        v   = phase_ratios.v[2:end-1,2:end-1],
    )

    #------------------------------------------------------------------#

    # Post-processing and plotting initialisation
    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    to   = TimerOutput()
    εII  = zeros(nc.x,nc.y)
    if flag.strain_evo
        fig7 = Figure(size=(700, 300))
        ax7 = Axis(fig7[1, 1], xlabel="x", ylabel="εᵢᵢ [10⁻³]", title="StagFD")
        if flag.Matlab
            fig8 = Figure(size=(700, 300))
            ax8 = Axis(fig8[1, 1], xlabel="x", ylabel="εᵢᵢ [10⁻³]", title="M2Di code")
        end
    end
    #-----------------------------------------------------------------#

    for it=1:nt

        @printf("Step %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)
        
        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        # Compute bulk and shear moduli
        compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, size_c, size_v, m.nphases)

        for iter=1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(R.p[inx_c,iny_c])/sqrt(nPt)

            @show  max(err.x[iter], err.y[iter], err.p[iter])

            max(err.x[iter], err.y[iter]) < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, ξ, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, ρ, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt
            
            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = @views -r[1:size(𝐊,1)]
            fp   = @views -r[size(𝐊,1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            @views dx[1:size(𝐊,1)]     .= u
            @views dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)

        end

        # Update pressure
        Pt .+= ΔPt.c

        #--------------------------------------------#

        (τII, ε̇II, εII) = invariants(Δ, τ, ε̇, inx_c, iny_c, εII)

        if flag.Matlab
            m= MatlabCheck(materials, res, it)
            @show size(εII)
            @show size(m.εII)
        end
        
        #--------------------------------------------#
        # Plot fields
         if flag.fields
            fig_fields = Figure(size=(1000, 800))
            ax1 = Axis(fig_fields[1, 1], aspect=DataAspect())
            ax2 = Axis(fig_fields[1, 3], aspect=DataAspect())
            ax3 = Axis(fig_fields[2, 1], aspect=DataAspect())
            ax4 = Axis(fig_fields[2, 3], aspect=DataAspect())

            hm1 = heatmap!(ax1, Grid.v.x, Grid.c.y, (V.x[inx_Vx,iny_Vx]').*1e9./sc.t)
            hm2 = heatmap!(ax2, Grid.c.x, Grid.c.y, (Pt[inx_c,iny_c]').*1e4.*sc.σ; colormap=:turbo)
            hm3 = heatmap!(ax3, Grid.c.x, Grid.c.y, log10.(εII)'; colormap=:coolwarm)
            hm4 = heatmap!(ax4, Grid.c.x, Grid.c.y, ((τII').*1e4.*sc.σ); colormap=:turbo)
            Colorbar(fig_fields[1, 2], hm1, label="Vx [10⁻⁹ m s⁻¹]")
            Colorbar(fig_fields[1, 4], hm2, label="P [10⁻⁴ Pa]")
            Colorbar(fig_fields[2, 2], hm3, label="log10(ε̇)")
            Colorbar(fig_fields[2, 4], hm4, label="τ [10⁻⁴ Pa]")


            xlims!(ax1, extrema(Grid.c.x)...)
            xlims!(ax2, extrema(Grid.c.x)...)
            xlims!(ax3, extrema(Grid.c.x)...)
            xlims!(ax4, extrema(Grid.c.x)...)
            if flag.Matlab && m !== nothing
                fig_cmp = Figure(size=(1000, 400))
                ax_cmp1 = Axis(fig_cmp[1, 1], title="εII", aspect=DataAspect())
                ax_cmp2 = Axis(fig_cmp[1, 2], title="εII from M2Di", aspect=DataAspect())
                heatmap!(ax_cmp1, Grid.c.x, Grid.c.y, log10.(εII)'; colormap=:coolwarm)
                heatmap!(ax_cmp2, m.xc, m.yc, log10.(m.εII)'; colormap=:coolwarm)
                xlims!(ax_cmp1, extrema(Grid.c.x)...)
                xlims!(ax_cmp2, extrema(m.xc)...)
                display(fig_cmp)
            else
                display(fig_fields)
            end

            #z0 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright)
            #z0 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
            #z0 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
            #z0 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
            # dislpay(z0)
        end
        #--------------------------------------------#
        # PLot time evolution of accumulated strain
        if flag.strain_evo
            (ε_prof, C) = section(εII, Δ, nc,materials,L)
            if flag.strain_int
                cur_ε = bulk_rate*Δ.t*it
                @show(cur_ε )
                if cur_ε ≈ ε_bulk[d]
                    lines!(ax7, C[2,:], (ε_prof)*1e3, label = "$(@sprintf("%0.1f", cur_ε*1e4)) [10⁻⁴]")
                    axislegend(ax7)
                    if flag.Matlab
                        if m !== nothing
                            lines!(ax8, m.C[2,:], (m.ε_prof)*1e3, label = "$(@sprintf("%0.1f", m.incr*it*1e4)) [10⁻⁴]")
                            axislegend(ax8)
                            display(fig8)
                        end
                    else
                        display(fig7)
                    end 
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
    if flag.Matlab
        m_outp = MatlabCheck(materials, res,nt)
        if m_outp !== nothing
            (m_ε_prof, m_C) = m_outp
        end
    else
        m_ε_prof = 1
        m_C = 1
    end
    display(to)

    return ε_prof, C, m_ε_prof, m_C
end

#---------------------------------------------------------------------------------------
#                                       M A I N    
#---------------------------------------------------------------------------------------

let 
    resolution = [100]
    NY = 69 
    # z5 = plot(xlabel="x", ylabel="εᵢᵢ [10⁻³]", size = (700,300), title = "Accumulated strain across shear bands" )

    for i in eachindex(resolution)

        res = resolution[i]
        flag = (strain_evo=true, Matlab=false, fields=true, strain_int=true )

        (ε_prof, C, m_ε_prof, m_C) = main((x = res, y = res), flag, res)
        # plot!(z5,C[2,:],(ε_prof)*1e3, label="$(res)²")
        # if flag.Matlab
        #     plot!(z5,m_C[2,:],(m_ε_prof)*1e3, label="$(res)² from M2Di")
        # end

    end

    # display(z5)

end



###########################################################################################
##########################################################################################