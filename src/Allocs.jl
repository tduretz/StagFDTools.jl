abstract type AbstractAdvection end

struct Markers{P,G,X,PA,PR} <: AbstractAdvection
    particles::P
    grid_vi::G
    xvi::X
    particle_args::PA
    phase_ratios::PR
end

struct TensorMarkers{np,nt,nd,Tp,Tt,Td} <: AbstractAdvection
    p::NamedTuple{np,Tp}
    t::NamedTuple{nt,Tt}
    Δ::NamedTuple{nd,Td}
end
struct Allocs{t,n,p,M,RNT,VNT,FNT,SNT,TNT,TAUNT,DENT,ONT,PNT,DNT,DC,DV,PHNT,PRNT,G}
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
    τ::TAUNT
    D::DENT
    ω::ONT
    ω0::ONT
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
    phase_ratios::PRNT
    X::G
end

# Method with no advection of stresses -----------------------
function Markers(backend, a::Allocs, nxcell::Union{Number,NTuple{N,Integer}}, max_xcell, min_xcell, nc, nphases; args=1) where {N}
    # args defaulted to 1 = phase
    grid_vx = (a.X.v.x, a.X.c_e.y)
    grid_vy = (a.X.c_e.x, a.X.v.y)
    xi_vel = (grid_vx, grid_vy)
    xvi = (a.X.v.x, a.X.v.y)
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xi_vel)
    particle_args = init_cell_arrays(particles, Val(args))
    phase_ratios = JustPIC.PhaseRatios(backend, nphases, values(nc))
    return Markers(particles, xi_vel, xvi, particle_args, phase_ratios)
end

# Method with stress advection --------------------------------
function Markers(backend, a::Allocs, particles, tm::TensorMarkers, nc, nphases; args=1)
    # args defaulted to 1 = phase
    grid_vx = (a.X.v.x, a.X.c_e.y)
    grid_vy = (a.X.c_e.x, a.X.v.y)
    xi_vel = (grid_vx, grid_vy)
    xvi = (a.X.v.x, a.X.v.y)
    particle_args = (init_cell_arrays(particles, Val(args))..., tensor_args(tm)...)
    phase_ratios = JustPIC.PhaseRatios(backend, nphases, values(nc))
    return Markers(particles, xi_vel, xvi, particle_args, phase_ratios)
end

# Helper for TensorConstructor
function initialise_markers(backend, a::Allocs, nxcell::Union{Number,NTuple{N,Integer}}, max_xcell, min_xcell) where {N}
    grid_vx = (a.X.v.x, a.X.c_e.y)
    grid_vy = (a.X.c_e.x, a.X.v.y)
    xi_vel = (grid_vx, grid_vy)
    return init_particles(backend, nxcell, max_xcell, min_xcell, xi_vel)
end

# !! Need to create a dispatch for no-upper advected (just Jaumann)
function TensorConstructor(particles)
    τxx, τyy, P, ε̇xx, ε̇yy = init_cell_arrays(particles, Val(5))
    τxy, ω, ε̇xy = init_cell_arrays(particles, Val(3))
    p = (τxx=τxx, τyy=τyy, P=P, τxy=τxy, ω=ω)
    Δ = (τxx=copy(τxx), τyy=copy(τyy), P=copy(P), τxy=copy(τxy), ω=copy(ω))
    t = (τxx=copy(τxx), τyy=copy(τyy), P=copy(P), τxy=copy(τxy), ω=copy(ω), ε̇xx=ε̇xx, ε̇yy=ε̇yy, ε̇xy=ε̇xy)
    return TensorMarkers(p, t, Δ)
end

function TensorMarkers(backend, a::Allocs, nxcell::Union{Number,NTuple{N,Integer}}, max_xcell, min_xcell) where {N}
    particles = initialise_markers(backend, a, nxcell, max_xcell, min_xcell)
    tm = TensorConstructor(particles)
    return particles, tm
end

function allocate(nc, config, x, y, Δ, nphases)
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
    τ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), xy_c=zeros(size_c...),
        II=zeros(size_c...), θ=zeros(size_c...))
    D = (τxx=zeros(size_c...), τyy=zeros(size_c...), τxy=zeros(size_v...), ω=zeros(size_v...),
        Pt=zeros(size_c))
    ω = (xy_c=zeros(size_c...), xy_v=zeros(size_v...))
    ω0 = (xy_c=zeros(size_c...), xy_v=zeros(size_v...))
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
    phase_ratios = (c=[zeros(nphases) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)],
        v=[zeros(nphases) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)])
    X = GenerateGrid(x, y, Δ, nc)

    return type, number, pattern, nVx, nVy, nPt,
    R, V, Vi, η, ξ, λ̇, G, β, ρ, ε̇, τ0, τ, D, ω, ω0,
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

function Allocs(nc, config, x, y, Δ, nphases)
    type, number, pattern, nVx, nVy, nPt,
    R, V, Vi, η, ξ, λ̇, G, β, ρ, ε̇, τ0, τ, D, ω, ω0,
    Pt, Pti, Pt0, ΔPt, Dc, Dv, 𝐷, D_ctl_c, D_ctl_v, 𝐷_ctl, phases, phase_ratios, X =
        allocate(nc, config, x, y, Δ, nphases)

    M, 𝐊, 𝐐, 𝐐ᵀ, 𝐏, dx, r = allocate_matrices(nVx, nVy, nPt)
    M_PC, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC, _, _ = allocate_matrices(nVx, nVy, nPt)

    return Allocs(type, number, pattern,
        M, M_PC, 𝐊, 𝐊_PC, 𝐐, 𝐐_PC, 𝐐ᵀ, 𝐐ᵀ_PC, 𝐏, 𝐏_PC, dx, r, R, V, Vi, η, ξ, λ̇, G, β, ρ, ε̇, τ0, τ, D, ω, ω0,
        Pt, Pti, Pt0, ΔPt, Dc, Dv, 𝐷, D_ctl_c, D_ctl_v, 𝐷_ctl, phases, phase_ratios, X)
end
