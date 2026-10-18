using DistributedSparseGrids
import DistributedSparseGrids: AbstractCollocationPoint, AbstractHierarchicalCollocationPoint, AbstractHierarchicalSparseGrid, numlevels, coord, pt_idx, i_multi, level, scaling_weight, fval
using Colors
import Colors: distinguishable_colors, RGB, N0f8, colormap
using Printf
using GLMakie
using Distributed

tol = 4.5e-1 #sparse grid refinement tolerance
initlevel = 4 #sparse grid initial level
maxlevel = 10 #sparse grid max refinement depth

if nprocs() == 1
    addprocs(4) # add additional workers
end

@everywhere begin
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
using DifferentiationInterface
using TimerOutputs
using StaticArrays
end

function Makie.scatter!(ax::Axis3, sg::SG; markersize=10) where {CT,CP<:AbstractCollocationPoint{3,CT},HCP<:AbstractHierarchicalCollocationPoint{3,CP},SG<:AbstractHierarchicalSparseGrid{3,HCP}}
    colors = cols = map(x->RGBA{Float64}(x.r,x.g,x.b,1.0), distinguishable_colors(numlevels(sg)+1, [RGB(1,1,1)])[2:end])
    nlevel = numlevels(sg)
    xvals = Vector{Vector{CT}}(undef,nlevel)
    yvals = Vector{Vector{CT}}(undef,nlevel)
    zvals = Vector{Vector{CT}}(undef,nlevel)
    clr = Vector{Vector{RGB{N0f8}}}(undef,nlevel)
    for l = 1:nlevel
        xvals[l] = Vector{CT}()
        yvals[l] = Vector{CT}()
        zvals[l] = Vector{CT}()
        clr[l] = Vector{RGB{N0f8}}()
    end
    for hcpt in sg
        l = level(hcpt)
        push!(xvals[l],coord(hcpt,1))
        push!(yvals[l],coord(hcpt,2))
        push!(zvals[l],coord(hcpt,3))
        push!(clr[l],colors[level(hcpt)])
    end
    for i = 1:nlevel
        mw = markersize-foldl((x,y)->x+2.0/(y),1:i)
        Makie.scatter!(ax, xvals[i], yvals[i], zvals[i], markersize=mw, color=clr[i])
    end
    return nothing
end

@everywhere begin

function lin_func(x,xmin,ymin,xmax,ymax)
    a = (ymax-ymin)/(xmax-xmin)
    b = ymax-a*xmax
    return a*x+b
end

@views function main(x)
    #--------------------------------------------#

    # Resolution
    nc = (x = 100, y = 100)

    # Boundary loading type
    config = :free_slip
    # config = :EW_Neumann
    D_BC   = @SMatrix( [  1. 0.;
                          0  -1 ])

    # transform stochastic state space variables to physical state space 
    x1 = lin_func(x[1],-1.0,1.5,1.0,2.5) #[-1,1]->[1.5,2.5]
    x2 = lin_func(x[2],-1.0,1.5,1.0,2.5) #[-1,1]->[1.5,2.5]
    x3 = lin_func(x[3],-1.0,1.0,1.0,3.0) #[-1,1]->[1.0,3.0]

    # Material parameters
    nphases = 3
    materials = initialize_materials(nphases)
    materials.η0 .= [10.0^x1,  10^x2,  10^x3 ]  
    materials.n  .= [x1,    x2,   x3 ]
    materials.G  .= [1e20,   1e20,  1e20]
    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Solver parameters
    niter   = 20    # max. number of non-linear iters
    γ       = 1e5   # penalty viscosity
    ϵ_l     = 1e-11 # linear solver tolerance
    ϵ_nl    = 1e-8  # non-linear solver tolerance
    inexact = false  # inexact Newton
    solver  = :GCR  # :GCR or :PH
    α       = LinRange(0.05, 1.0, 6)

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
    M_PC = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊    = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐊_PC = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐    = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐_PC = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ   = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐐ᵀ_PC= ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏    = ExtendableSparseMatrix(nPt, nPt)
    𝐏_PC = ExtendableSparseMatrix(nPt, nPt)
    dx   = zeros(nVx + nVy + nPt)
    r    = zeros(nVx + nVy + nPt)

    #--------------------------------------------#
    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
     x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y, max=0.0)
    X = GenerateGrid(x, y, Δ, nc)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    ξ       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    G       = (c  = zeros(size_c...), v  = zeros(size_v...))
    β       = (c  = zeros(size_c...), v  = zeros(size_v...))
    ρ       = (c  = zeros(size_c...), v  = zeros(size_v...))
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...), θ = zeros(size_c...) )
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
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    @views V.x .= D_BC[1,1]*X.vx_e.x .+ D_BC[1,2]*X.vx_e.y' 
    @views V.y .= D_BC[2,1]*X.vy_e.x .+ D_BC[2,2]*X.vy_e.y'
    @views Pt[inx_c, iny_c ]  .= 10.                 
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * X.v.x .+ D_BC[1, 2] * X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * X.v.x .+ D_BC[1, 2] * X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * X.v.x[1] .+ D_BC[2, 2] * X.v.y)
        BC.Vy[end-1, iny_Vy] .= (type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * X.v.x[end] .+ D_BC[2, 2] * X.v.y)
    end

    # Set material geometry and phase ratios
    @views phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    @views phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2
    @views phases.v[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    @views phases.v[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries
    @views phases.c[[2,end-1], :] .= 3  # Use linear material along Neumann boundaries
    @views phases.c[:, [2,end-1]] .= 3  # Use linear material along Neumann boundaries
    phase_ratios = InitialisePhaseRatios(phases, nphases)
    # phase_ratios = (
    #     c   = phase_ratios.c[2:end-1,2:end-1],
    #     v   = phase_ratios.v[2:end-1,2:end-1],
    # )

    #--------------------------------------------#

    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    to   = TimerOutput()

    #--------------------------------------------#

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

        compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, nphases)

        @printf("Time step %04d (nthreads = %03d)\n", it, Threads.nthreads())
        iter, ϵ0, ϵ = 0, 0.0, 0.0
        niter = 10

        @time while iter<niter

            iter +=1
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
            ϵ =  max(err.x[iter], err.y[iter])
            (iter == 1) && (ϵ0 = ϵ)
            ϵ < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                # Jacobian
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, ξ, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, ρ, materials, number, pattern, type, BC, nc, Δ)
                # Preconditioner
                AssembleContinuity2D!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, ξ, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, ρ, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt
            # Picard preconditioner
            𝐊_PC  .= [M_PC.Vx.Vx M_PC.Vx.Vy; M_PC.Vy.Vx M_PC.Vy.Vy]
            𝐐_PC  .= [M_PC.Vx.Pt; M_PC.Vy.Pt]
            𝐐ᵀ_PC .= [M_PC.Pt.Vx M_PC.Pt.Vy]
            𝐏_PC  .= M_PC.Pt.Pt
            #--------------------------------------------#
     
            # Inexact Newton-Raphson
            ϵ_l = inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : ϵ_l
            @printf("Abs. res. = %02e --- Rel. res = %02e  --- ϵ_l = %1.2e\n", ϵ, ϵ/ϵ0, ϵ_l)

            # Direct-iterative solver
            @timeit to "Linear solve" begin
                mechanical_solver!( dx, M, r, 𝐊, 𝐐, 𝐐ᵀ, 𝐏, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC; solver=solver, ηb=γ, ϵ_l=ϵ_l, niter_l=10, restart=20) 
            end

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" begin
                imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, ξ, ρ, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)
            end
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
        end
        # Update pressure
        Pt .+= ΔPt.c
    end
    return V, ε̇, τ, inx_Vx, iny_Vx, inx_c, iny_c, xv, xc, yc
end
end

V, ε̇, τ, inx_Vx, iny_Vx, inx_c, iny_c, xv, xc, yc = main([-1.0,1.0,0.5])

function sparse_grid(N::Int,pointprops,nlevel=initlevel)
    CT=Float64 
    RT=Matrix{Float64} # return type
    # define collocation point
    CPType = CollocationPoint{N,CT}
    # define hierarchical collocation point
    HCPType = HierarchicalCollocationPoint{N,CPType,RT}
    # init grid
    asg = init(AHSG{N,HCPType},pointprops)
    #set of all collocation points
    cpts = Set{HierarchicalCollocationPoint{N,CPType,RT}}(collect(asg))
    # fully refine grid nlevel-1 times
    for i = 1:nlevel-1
        union!(cpts,generate_next_level!(asg))
    end
    return asg
end

#init sparse grid
asg = sparse_grid(3, @SVector [1,1,1]) 

# objective function
@everywhere function fun1(x::SVector{N,CT},ID::String) where {N,CT}
    V, ε̇, τ, inx_Vx, iny_Vx, inx_c, iny_c, xv, xc, yc = main(x)
    return Matrix{Float64}(log10.(abs.(ε̇.II[inx_c,iny_c]))')
end

# initialize weights
distributed_init_weights!(asg, fun1, workers())

# adaptive refinement
while true
    cpts = generate_next_level!(asg, tol, maxlevel)
    if isempty(cpts)
        break
    end
    distributed_init_weights!(asg, collect(cpts), fun1, workers())
end

GLMakie.activate!()
fig = Figure(size=(1200,800), fontsize=14)

view = fig[1,1] = GridLayout()
controls = fig[2,1] = GridLayout()

# Observables
sl_x = Slider(controls[1, 1], range = -1.0:0.01:1.0, startvalue = -0.5, update_while_dragging=true, linewidth=14)
sl_y = Slider(controls[2, 1], range = -1.0:0.01:1.0, startvalue = 0.5, update_while_dragging=true, linewidth=14)
sl_z = Slider(controls[3, 1], range = -1.0:0.01:1.0, startvalue = 0.6, update_while_dragging=true, linewidth=14)
label1 = map!(Observable{Any}(),sl_x.value) do x
    x1 = @sprintf("%.2f", lin_func(x,-1.0,1.5,1.0,2.5))
    "η0₁=10^$x1, n₁=$x1"
end
label2 = map!(Observable{Any}(),sl_y.value) do x
    x2 = @sprintf("%.2f", lin_func(x,-1.0,1.5,1.0,2.5))
    "η0₂=10^$x2, n₂=$x2"
end
label3 = map!(Observable{Any}(),sl_z.value) do x
    x3 = @sprintf("%.2f", lin_func(x,-1.0,1.0,1.0,3.0))
    "η0₃=10^$x3, n₃=$x3"
end
_ε̇II = interpolate(asg, [0.0,0.0,0.0])
interpres = map!(Observable{Any}(),sl_x.value,sl_y.value,sl_z.value) do x,y,z
    global _ε̇II
    interpolate!(_ε̇II, asg, [x,y,z])
    return _ε̇II
end

Label(controls[1, 2], label1)
Label(controls[2, 2], label2)
Label(controls[3, 2], label3)
ax1 = Axis(view[1,1], title="ε̇II", aspect=DataAspect())
hm3 = heatmap!(ax1, xc, yc, interpres; colormap=:coolwarm, colorrange=(-0.4,0.4))
xlims!(ax1, extrema(xc))
Colorbar(view[1,2], hm3)
xytickformat = vals -> [@sprintf("%.2f", lin_func(val,-1.0,1.5,1.0,2.5)) for val in vals]
ztickformat = vals -> [@sprintf("%.2f", lin_func(val,-1.0,1.0,1.0,3.0)) for val in vals]
ax2 = Axis3(view[1,3], xtickformat=xytickformat, ytickformat=xytickformat, ztickformat=ztickformat)
scatter!(ax2, asg)
scatter!(ax2, sl_x.value, sl_y.value, sl_z.value, color=:red, markersize=20)
display(fig)
