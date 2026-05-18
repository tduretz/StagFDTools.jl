abstract type AbstractPlasticity end

Base.@kwdef struct VonMises <: AbstractPlasticity
    C::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
    cosψ::Vector{Float64} = Float64[]
end
Base.@kwdef struct DruckerPrager <: AbstractPlasticity
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
    cosψ::Vector{Float64} = Float64[]
end

Base.@kwdef struct DruckerPrager1 <: AbstractPlasticity
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
    cosψ::Vector{Float64} = Float64[]
end

Base.@kwdef struct DruckerHyperbolic <: AbstractPlasticity
    σT::Vector{Float64} = Float64[]
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
    cosψ::Vector{Float64} = Float64[]
end

Base.@kwdef struct DruckerAniso <: AbstractPlasticity
    δ::Vector{Float64} = Float64[]
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
    cosψ::Vector{Float64} = Float64[]
end

Base.@kwdef struct Tensile <: AbstractPlasticity
    σT::Vector{Float64} = Float64[]
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
end

Base.@kwdef struct Golchin2021 <: AbstractPlasticity
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
    cosψ::Vector{Float64} = Float64[]
    M::Vector{Float64} = Float64[]
    N::Vector{Float64} = Float64[]
    Pc::Vector{Float64} = Float64[]
    a::Vector{Float64} = Float64[]
    b::Vector{Float64} = Float64[]
    c::Vector{Float64} = Float64[]
    σT::Vector{Float64} = Float64[]
end

Base.@kwdef struct Kiss2023 <: AbstractPlasticity
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
    σT::Vector{Float64} = Float64[]
    δσT::Vector{Float64} = Float64[]
    P1::Vector{Float64} = Float64[]
    τ1::Vector{Float64} = Float64[]
    P2::Vector{Float64} = Float64[]
    τ2::Vector{Float64} = Float64[]
end

Base.@kwdef struct Materials{P<:AbstractPlasticity}
    g::Vector{Float64} = [0.0, 0.0]
    ρ::Vector{Float64} = Float64[]
    n::Vector{Float64} = Float64[]
    η0::Vector{Float64} = Float64[]
    ξ0::Vector{Float64} = Float64[]
    G::Vector{Float64} = Float64[]
    β::Vector{Float64} = Float64[]
    B::Vector{Float64} = Float64[]
    plasticity::P = NoPlasticity()
    compressible::Bool = false
    phase_avg::Symbol = :arithmetic
end

struct NoPlasticity <: AbstractPlasticity end

initialize(::Type{VonMises}, n::Integer) = VonMises(
    C=1e50 * ones(n),
    cosϕ=ones(n),
    ηvp=0. * ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n)
)

initialize(::Type{DruckerPrager}, n::Integer) = DruckerPrager(
    C=1e50 * ones(n),
    ϕ=0. * ones(n),
    ψ=0. * ones(n),
    ηvp=0. * ones(n),
    cosϕ=ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n)
)

initialize(::Type{DruckerPrager1}, n::Integer) = DruckerPrager1(
    C=1e50 * ones(n),
    ϕ=0. * ones(n),
    ψ=0. * ones(n),
    ηvp=0. * ones(n),
    cosϕ=ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n)
)

initialize(::Type{DruckerHyperbolic}, n::Integer) = DruckerHyperbolic(
    σT=0. * ones(n),
    C=1e50 * ones(n),
    ϕ=0. * ones(n),
    ψ=0. * ones(n),
    ηvp=0. * ones(n),
    cosϕ=ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n)
)

initialize(::Type{DruckerAniso}, n::Integer) = DruckerAniso(
    δ=ones(n),
    C=1e50 * ones(n),
    ϕ=0. * ones(n),
    ψ=0. * ones(n),
    ηvp=0. * ones(n),
    cosϕ=ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n)
)

initialize(::Type{Golchin2021}, n::Integer) = Golchin2021(
    C=1e50 * ones(n),
    ϕ=0. * ones(n),
    ψ=0. * ones(n),
    ηvp=0. * ones(n),
    cosϕ=ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n),
    M=0. * ones(n),
    N=0. * ones(n),
    Pc=0. * ones(n),
    a=0. * ones(n),
    b=0. * ones(n),
    c=0. * ones(n),
    σT=0. * ones(n),
)

initialize(::Type{Kiss2023}, n::Integer) = Kiss2023(
    C=1e50 * ones(n),
    ϕ=0. * ones(n),
    ψ=0. * ones(n),
    ηvp=0. * ones(n),
    cosϕ=ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n),
    σT=0. * ones(n),
    δσT=0. * ones(n),
    P1=0. * ones(n),
    τ1=0. * ones(n),
    P2=0. * ones(n),
    τ2=0. * ones(n),
)

initialize(::Type{Tensile}, n::Integer) = Tensile(
    σT=0. * ones(n),
    C=1e50 * ones(n),
    ϕ=0. * ones(n),
    ψ=0. * ones(n),
    ηvp=0. * ones(n),
    cosϕ=ones(n),
    sinϕ=0. * ones(n),
    sinψ=0. * ones(n),
    cosψ=0. * ones(n),
)

initialize(::Type{NoPlasticity}, ::Integer) = NoPlasticity()

function initialize_materials(nphases::Integer;
    plasticity=NoPlasticity(),
    compressible::Bool=false,
    phase_avg::Symbol=:arithmetic)
    P = plasticity isa Type ? plasticity : typeof(plasticity)
    return Materials(
        ρ=ones(nphases),
        n=ones(nphases),
        η0=ones(nphases),
        ξ0=1e50 * ones(nphases),
        G=1e50 * ones(nphases),
        β=1e-50 * ones(nphases),
        B=ones(nphases),
        plasticity=initialize(P, nphases),
        compressible=compressible,
        phase_avg=phase_avg
    )
end

function preprocess!(dp::DruckerPrager)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
    @. dp.cosψ = cosd(dp.ψ)
end

function preprocess!(dp::DruckerPrager1)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
    @. dp.cosψ = cosd(dp.ψ)
end

function preprocess!(dh::DruckerHyperbolic)
    @. dh.cosϕ = cosd(dh.ϕ)
    @. dh.sinϕ = sind(dh.ϕ)
    @. dh.sinψ = sind(dh.ψ)
    @. dh.cosψ = cosd(dh.ψ)
end

function preprocess!(da::DruckerAniso)
    @. da.cosϕ = cosd(da.ϕ)
    @. da.sinϕ = sind(da.ϕ)
    @. da.sinψ = sind(da.ψ)
    @. da.cosψ = cosd(da.ψ)
end

function preprocess!(g::Golchin2021)
    @. g.cosϕ = cosd(g.ϕ)
    @. g.sinϕ = sind(g.ϕ)
    @. g.sinψ = sind(g.ψ)
    @. g.cosψ = cosd(g.ψ)
    @. g.M = 6 * sind(g.ϕ) / (3 - sind(g.ϕ))
    @. g.N = 6 * sind(g.ψ) / (3 - sind(g.ψ))

end

function preprocess!(k::Kiss2023)
    @. k.cosϕ = cosd(k.ϕ)
    @. k.sinϕ = sind(k.ϕ)
    @. k.sinψ = sind(k.ψ)
    @. k.P1 = -(k.σT - k.δσT)
    @. k.τ1 = k.δσT
    @. k.P2 = -(k.σT - k.C * cosd(k.ϕ)) / (1.0 - sind(k.ϕ))
    @. k.τ2 = k.P2 + k.σT
end

function preprocess!(t::Tensile)
    @. t.cosϕ = cosd(t.ϕ)
    @. t.sinϕ = sind(t.ϕ)
    @. t.sinψ = sind(t.ψ)
    @. t.cosψ = cosd(t.ψ)
end

function preprocess!(mat::Materials)
    @. mat.B = (2 * mat.η0)^(-mat.n)
    preprocess!(mat.plasticity)
end

preprocess!(::VonMises) = nothing

preprocess!(::NoPlasticity) = nothing

preprocess(x) = (preprocess!(x); x)
