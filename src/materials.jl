abstract type AbstractPlasticity end

Base.@kwdef struct VonMises{T} <: AbstractPlasticity
    C::T = Float64[]
    cosϕ::T = Float64[]
    ηvp::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end
Base.@kwdef struct DruckerPrager{T} <: AbstractPlasticity
    C::T = Float64[]
    ϕ::T = Float64[]
    ψ::T = Float64[]
    ηvp::T = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct DruckerPrager1{T} <: AbstractPlasticity
    C::T = Float64[]
    ϕ::T = Float64[]
    ψ::T = Float64[]
    ηvp::T = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct DruckerHyperbolic{T} <: AbstractPlasticity
    σT::T = Float64[]
    C::T = Float64[]
    ϕ::T = Float64[]
    ψ::T = Float64[]
    ηvp::T = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct DruckerAniso{T} <: AbstractPlasticity
    δ::T = Float64[]
    C::T = Float64[]
    ϕ::T = Float64[]
    ψ::T = Float64[]
    ηvp::T = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
end

Base.@kwdef struct Tensile{T} <: AbstractPlasticity
    σT::T = Float64[]
    C::T = Float64[]
    ϕ::T = Float64[]
    ψ::T = Float64[]
    ηvp::T = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
end

Base.@kwdef struct Golchin2021{T} <: AbstractPlasticity
    C::T = Float64[]
    ϕ::T = Float64[]
    ψ::T = Float64[]
    ηvp::T = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
    M::T = Float64[]
    N::T = Float64[]
    Pc::T = Float64[]
    a::T = Float64[]
    b::T = Float64[]
    c::T = Float64[]
    σT::T = Float64[]
end

Base.@kwdef struct Kiss2023{T} <: AbstractPlasticity
    C::T = Float64[]
    ϕ::T = Float64[]
    ψ::T = Float64[]
    ηvp::T = Float64[]
    cosϕ::T = Float64[]
    sinϕ::T = Float64[]
    sinψ::T = Float64[]
    cosψ::T = Float64[]
    σT::T = Float64[]
    δσT::T = Float64[]
    P1::T = Float64[]
    τ1::T = Float64[]
    P2::T = Float64[]
    τ2::T = Float64[]
end

Base.@kwdef struct Materials{T, P <: AbstractPlasticity}
    g::T = [0.0, 0.0]
    ρ::T = Float64[]
    n::T = Float64[]
    η0::T = Float64[]
    ξ0::T = Float64[]
    G::T = Float64[]
    β::T = Float64[]
    B::T = Float64[]
    plasticity::P = NoPlasticity()
    compressible::Bool = false
    phase_avg::Symbol = :arithmetic
end

struct NoPlasticity <: AbstractPlasticity end

initialize(::Type{VonMises}, n::Integer) = VonMises(
    C = fill(1.0e50, n),
    cosϕ = ones(n),
    ηvp = zeros(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{DruckerPrager}, n::Integer) = DruckerPrager(
    C = fill(1.0e50, n),
    ϕ = zeros(n),
    ψ = zeros(n),
    ηvp = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{DruckerPrager1}, n::Integer) = DruckerPrager1(
    C = fill(1.0e50, n),
    ϕ = zeros(n),
    ψ = zeros(n),
    ηvp = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{DruckerHyperbolic}, n::Integer) = DruckerHyperbolic(
    σT = zeros(n),
    C = fill(1.0e50, n),
    ϕ = zeros(n),
    ψ = zeros(n),
    ηvp = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{DruckerAniso}, n::Integer) = DruckerAniso(
    δ = ones(n),
    C = fill(1.0e50, n),
    ϕ = zeros(n),
    ψ = zeros(n),
    ηvp = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n)
)

initialize(::Type{Golchin2021}, n::Integer) = Golchin2021(
    C = fill(1.0e50, n),
    ϕ = zeros(n),
    ψ = zeros(n),
    ηvp = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n),
    M = zeros(n),
    N = zeros(n),
    Pc = zeros(n),
    a = zeros(n),
    b = zeros(n),
    c = zeros(n),
    σT = zeros(n),
)

initialize(::Type{Kiss2023}, n::Integer) = Kiss2023(
    C = fill(1.0e50, n),
    ϕ = zeros(n),
    ψ = zeros(n),
    ηvp = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n),
    σT = zeros(n),
    δσT = zeros(n),
    P1 = zeros(n),
    τ1 = zeros(n),
    P2 = zeros(n),
    τ2 = zeros(n),
)

initialize(::Type{Tensile}, n::Integer) = Tensile(
    σT = zeros(n),
    C = fill(1.0e50, n),
    ϕ = zeros(n),
    ψ = zeros(n),
    ηvp = zeros(n),
    cosϕ = ones(n),
    sinϕ = zeros(n),
    sinψ = zeros(n),
    cosψ = zeros(n),
)

initialize(::Type{NoPlasticity}, ::Integer) = NoPlasticity()

function initialize_materials(
        nphases::Integer;
        plasticity = NoPlasticity(),
        compressible::Bool = false,
        phase_avg::Symbol = :arithmetic
    )
    P = plasticity isa Type ? plasticity : typeof(plasticity)
    return Materials(
        ρ = ones(nphases),
        n = ones(nphases),
        η0 = ones(nphases),
        ξ0 = 1.0e50 * ones(nphases),
        G = 1.0e50 * ones(nphases),
        β = 1.0e-50 * ones(nphases),
        B = ones(nphases),
        plasticity = initialize(P, nphases),
        compressible = compressible,
        phase_avg = phase_avg
    )
end

function preprocess!(dp::DruckerPrager)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
    return @. dp.cosψ = cosd(dp.ψ)
end

function preprocess!(dp::DruckerPrager1)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
    return @. dp.cosψ = cosd(dp.ψ)
end

function preprocess!(dh::DruckerHyperbolic)
    @. dh.cosϕ = cosd(dh.ϕ)
    @. dh.sinϕ = sind(dh.ϕ)
    @. dh.sinψ = sind(dh.ψ)
    return @. dh.cosψ = cosd(dh.ψ)
end

function preprocess!(da::DruckerAniso)
    @. da.cosϕ = cosd(da.ϕ)
    @. da.sinϕ = sind(da.ϕ)
    @. da.sinψ = sind(da.ψ)
    return @. da.cosψ = cosd(da.ψ)
end

function preprocess!(g::Golchin2021)
    @. g.cosϕ = cosd(g.ϕ)
    @. g.sinϕ = sind(g.ϕ)
    @. g.sinψ = sind(g.ψ)
    @. g.cosψ = cosd(g.ψ)
    @. g.M = 6 * sind(g.ϕ) / (3 - sind(g.ϕ))
    return @. g.N = 6 * sind(g.ψ) / (3 - sind(g.ψ))

end

function preprocess!(k::Kiss2023)
    @. k.cosϕ = cosd(k.ϕ)
    @. k.sinϕ = sind(k.ϕ)
    @. k.sinψ = sind(k.ψ)
    @. k.P1 = -(k.σT - k.δσT)
    @. k.τ1 = k.δσT
    @. k.P2 = -(k.σT - k.C * cosd(k.ϕ)) / (1.0 - sind(k.ϕ))
    return @. k.τ2 = k.P2 + k.σT
end

function preprocess!(t::Tensile)
    @. t.cosϕ = cosd(t.ϕ)
    @. t.sinϕ = sind(t.ϕ)
    @. t.sinψ = sind(t.ψ)
    return @. t.cosψ = cosd(t.ψ)
end

function preprocess!(mat::Materials)
    @. mat.B = (2 * mat.η0)^(-mat.n)
    return preprocess!(mat.plasticity)
end

preprocess!(::VonMises) = nothing

preprocess!(::NoPlasticity) = nothing

preprocess(x) = (preprocess!(x); x)
