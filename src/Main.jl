using JustPIC

abstract type AbstractSolver end

# Keeping a Solver struct if one wants to add solvers
struct JustPICAdvection{}
    particles::P
    grid_vi::G
    xvi::X
    particle_args::PA
    phase_ratios::PR
end
struct Solver{t,n,p,M} <: AbstractSolver
    type::t
    number::n
    pattern::p
    M::M
    M_PC::M
    𝐊::ExtendableSparseMatrix
    𝐊_PC::ExtendableSparseMatrix
    𝐐::ExtendableSparseMatrix
    𝐐_PC::ExtendableSparseMatrix
    𝐐ᵀ::ExtendableSparseMatrix
    𝐐ᵀ_PC::ExtendableSparseMatrix
    𝐏::ExtendableSparseMatrix
    𝐏_PC::ExtendableSparseMatrix
    dx::Vector{Float64}
    r::Vector{Float64}
end
struct Allocs{RNT,VNT,FNT,SNT,TNT,PNT,DNT,DC,DV,PHNT,G,S<:AbstractSolver}
    solv::S
    R::RNT
    V::VNT
    Vi::VNT
    η::FNT
    ξ::FNT
    λ̇::FNT
    G::FNT
    β::FNT
    ρ::FNT
    ε̇::SNT
    τ0::TNT
    τ::SNT
    Pt::Matrix{Float64}
    Pti::Matrix{Float64}
    Pt0::Matrix{Float64}
    ΔPt::PNT
    Dc::DC
    Dv::DV
    𝐷::DNT
    D_ctl_c::DC
    D_ctl_v::DV
    𝐷_ctl::DNT
    phases::PHNT
    phase_ratios::FNT
    X::G
end

const _allocs_own_fields = fieldnames(Allocs)
function Base.getproperty(a::Allocs, s::Symbol)
    s ∈ _allocs_own_fields && return getfield(a, s)
    return getproperty(getfield(a, :solv), s)
end

function JustPICAdvection(backend, a::Allocs, nxcell, max_xcell, min_xcell, nc, args)
    grid_vx = (a.X.v.x, a.X.c_e.y)
    grid_vy = (a.X.c_e.x, a.X.v.y)
    xvi = (a.X.v.x, a.X.v.y)
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy)
    particle_args = init_cell_arrays(particles, Val(args))
    phase_ratios = JustPIC._2D.PhaseRatios(backend, nphases, values(nc))
    return JustPICAdvection(particles, (grid_vx, grid_vy), xvi, particle_args, phase_ratios)
end

function allocate(nc, config, x, y, Δ)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c,
    inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    type = Fields(
        fill(:out, (nc.x + 3, nc.y + 4)),
        fill(:out, (nc.x + 4, nc.y + 3)),
        fill(:out, (nc.x + 2, nc.y + 2)),
    )
    set_boundaries_template!(type, config, nc)

    number = Fields(fill(0, size_x), fill(0, size_y), fill(0, size_c))
    Numbering!(number, type, nc)

    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),
            @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),
            @SMatrix([1 1 1; 1 1 1])),
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),
            @SMatrix([1 1 1; 1 1 1; 1 1 1]),
            @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0; 0 1 0]),
            @SMatrix([0 0; 1 1; 0 0]),
            @SMatrix([1]))
    )

    nVx = maximum(number.Vx)
    nVy = maximum(number.Vy)
    nPt = maximum(number.Pt)

    R = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V = (x=zeros(size_x...), y=zeros(size_y...))
    Vi = (x=zeros(size_x...), y=zeros(size_y...))
    η = (c=ones(size_c...), v=ones(size_v...))
    ξ = (c=ones(size_c...), v=ones(size_v...))
    λ̇ = (c=zeros(size_c...), v=zeros(size_v...))
    G = (c=zeros(size_c...), v=zeros(size_v...))
    β = (c=zeros(size_c...), v=zeros(size_v...))
    ρ = (c=zeros(size_c...), v=zeros(size_v...))
    ε̇ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...),
        II=zeros(size_c...), θ=zeros(size_c...))
    τ0 = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...))
    τ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...),
        II=zeros(size_c...), θ=zeros(size_c...))
    Pt = zeros(size_c...)
    Pti = zeros(size_c...)
    Pt0 = zeros(size_c...)
    ΔPt = (c=zeros(size_c...), Vx=zeros(size_x...), Vy=zeros(size_y...))

    Dc = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    Dv = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷 = (c=Dc, v=Dv)
    D_ctl_c = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    D_ctl_v = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷_ctl = (c=D_ctl_c, v=D_ctl_v)
    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))
    phase_ratios = (c=zeros(size_c...), v=zeros(size_v...))
    X = GenerateGrid(x, y, Δ, nc)

    return type, number, pattern, nVx, nVy, nPt,
    R, V, Vi, η, ξ, λ̇, G, β, ρ, ε̇, τ0, τ,
    Pt, Pti, Pt0, ΔPt, Dc, Dv, 𝐷, D_ctl_c, D_ctl_v, 𝐷_ctl, phases, phase_ratios, X
end

function allocate_matrices(nVx, nVy, nPt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)),
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)),
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊 = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐 = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏 = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r = zeros(nVx + nVy + nPt)
    return M, 𝐊, 𝐐, 𝐐ᵀ, 𝐏, dx, r
end

function Allocs(nc, config, x, y, Δ)
    type, number, pattern, nVx, nVy, nPt,
    R, V, Vi, η, ξ, λ̇, G, β, ρ, ε̇, τ0, τ,
    Pt, Pti, Pt0, ΔPt, Dc, Dv, 𝐷, D_ctl_c, D_ctl_v, 𝐷_ctl, phases, phase_ratios, X =
        allocate(nc, config, x, y, Δ)

    M, 𝐊, 𝐐, 𝐐ᵀ, 𝐏, dx, r = allocate_matrices(nVx, nVy, nPt)
    M_PC, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC, _, _ = allocate_matrices(nVx, nVy, nPt)

    solv = Solver(type, number, pattern,
        M, M_PC, 𝐊, 𝐊_PC, 𝐐, 𝐐_PC, 𝐐ᵀ, 𝐐ᵀ_PC, 𝐏, 𝐏_PC, dx, r)

    return Allocs(solv, R, V, Vi, η, ξ, λ̇, G, β, ρ, ε̇, τ0, τ,
        Pt, Pti, Pt0, ΔPt, Dc, Dv, 𝐷, D_ctl_c, D_ctl_v, 𝐷_ctl, phases, phase_ratios, X)
end

function _assemble!(a::Allocs, materials, BC, nc, Δ)
    # Jacobian
    AssembleContinuity2D!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.β, a.ξ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_x!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_y!(a.M, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷_ctl, a.G, a.ρ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    # Picard preconditioner
    AssembleContinuity2D!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_x!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
    AssembleMomentum2D_y!(a.M_PC, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ,
        materials, a.number, a.pattern, a.type, BC, nc, Δ)
end

function update_solution!(a::Allocs, materials, BC, phase_ratios, nc, Δ, to,
    rvec, iter, ϵ0, ϵ, iter_params)
    a.𝐊 .= [a.M.Vx.Vx a.M.Vx.Vy; a.M.Vy.Vx a.M.Vy.Vy]
    a.𝐐 .= [a.M.Vx.Pt; a.M.Vy.Pt]
    a.𝐐ᵀ .= [a.M.Pt.Vx a.M.Pt.Vy]
    a.𝐏 .= a.M.Pt.Pt
    a.𝐊_PC .= [a.M_PC.Vx.Vx a.M_PC.Vx.Vy; a.M_PC.Vy.Vx a.M_PC.Vy.Vy]
    a.𝐐_PC .= [a.M_PC.Vx.Pt; a.M_PC.Vy.Pt]
    a.𝐐ᵀ_PC .= [a.M_PC.Pt.Vx a.M_PC.Pt.Vy]
    a.𝐏_PC .= a.M_PC.Pt.Pt

    ϵ_l = iter_params.inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : iter_params.ϵ_l
    @printf("Abs. res. = %02e --- Rel. res = %02e  --- ϵ_l = %1.2e\n", ϵ, ϵ / ϵ0, ϵ_l)

    @timeit to "Linear solve" begin
        mechanical_solver!(a.dx, a.M, a.r, a.𝐊, a.𝐐, a.𝐐ᵀ, a.𝐏, a.𝐊_PC, a.𝐐_PC, a.𝐐ᵀ_PC, a.𝐏_PC; solver=iter_params.solver_type, ηb=iter_params.γ, ϵ_l=ϵ_l, niter_l=10, restart=20)
    end

    @timeit to "Line search" begin
        imin = LineSearch!(rvec, iter_params.α, a.dx, a.R, a.V, a.Pt, a.ε̇, a.τ, a.Vi, a.Pti, a.ΔPt, a.Pt0, a.τ0, a.λ̇, a.η, a.G, a.β, a.ξ, a.ρ, a.𝐷, a.𝐷_ctl, a.number, a.type, BC, materials, phase_ratios, nc, Δ)
    end
    UpdateSolution!(a.V, a.Pt, iter_params.α[imin] * a.dx, a.number, a.type, nc)
end

function Solve!(a::Allocs, materials, BC, phase_ratios, nc, Δ, to,
    rvec, iter, ϵ0, ϵ, iter_params)
    @timeit to "Assembly" _assemble!(a, materials, BC, nc, Δ)
    update_solution!(a, materials, BC, phase_ratios, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params)
end

function main_solver!(a::Allocs, it, materials, BC, phase_ratios, nc, Δ, to,
    nphases, iter_params)

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter),
        y=zeros(iter_params.niter),
        p=zeros(iter_params.niter))

    a.τ0.xx .= a.τ.xx
    a.τ0.yy .= a.τ.yy
    a.τ0.xy .= a.τ.xy
    a.Pt0 .= a.Pt

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c,
    inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    nVx = maximum(a.number.Vx)
    nVy = maximum(a.number.Vy)
    nPt = maximum(a.number.Pt)

    compute_grid_fields!(a.G, a.β, a.ρ, a.ξ, materials, phase_ratios, nc, nphases)

    @printf("Time step %04d (nthreads = %03d)\n", it, Threads.nthreads())
    iter, ϵ0, ϵ = 0, 0.0, 0.0

    @time while iter < iter_params.niter
        iter += 1
        @printf("Iteration %04d\n", iter)

        @timeit to "Residual" begin
            TangentOperator!(a.𝐷, a.𝐷_ctl, a.τ, a.τ0, a.ε̇, a.λ̇, a.η, a.G, a.V, a.Pt, a.Pt0, a.ΔPt, a.type, BC, materials, phase_ratios, Δ)
            ResidualContinuity2D!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.β, a.ξ, materials, a.number, a.type, BC, nc, Δ)
            ResidualMomentum2D_x!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, materials, a.number, a.type, BC, nc, Δ)
            ResidualMomentum2D_y!(a.R, a.V, a.Pt, a.Pt0, a.ΔPt, a.τ0, a.𝐷, a.G, a.ρ, materials, a.number, a.type, BC, nc, Δ)
        end

        err.x[iter] = @views norm(a.R.x[inx_Vx, iny_Vx]) / sqrt(nVx)
        err.y[iter] = @views norm(a.R.y[inx_Vy, iny_Vy]) / sqrt(nVy)
        err.p[iter] = @views norm(a.R.p[inx_c, iny_c]) / sqrt(nPt)
        ϵ = max(err.x[iter], err.y[iter])
        (iter == 1) && (ϵ0 = ϵ)
        ϵ < iter_params.ϵ_nl && break

        SetRHS!(a.r, a.R, a.number, a.type, nc)
        Solve!(a, materials, BC, phase_ratios, nc, Δ, to, rvec, iter, ϵ0, ϵ, iter_params)
    end

    a.Pt .+= a.ΔPt.c

    return iter, err
end

main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params) = main_loop(a, it, materials, BC, phase_ratios, nc, Δ, to, nphases, iter_params)
# No advection
function main_loop(a::Allocs, it, materials, BC, phase_ratios::Nothing, nc, Δ, to, nphases, iter_params)
    @printf("Step %04d\n", it)
    return main_solver!(a, it, materials, BC, phase_ratios, nc, Δ, to, nphases, iter_params)
end

# JustPIC advection
function main_loop(a::Allocs, it, materials, BC, phase_ratios::PhaseRatios, nc, Δ, to, nphases, iter_params)

    main_solver!(a, it, materials, BC, tphase_ratios, nc, Δ, to, nphases, iter_params)

    @timeit to "Advection" begin
        advection!(adv.particles, RungeKutta4(), (a.V.x, a.V.y), adv.grid_vi, Δ.t)
        move_particles!(adv.particles, adv.xvi, adv.particle_args)
    end
end
