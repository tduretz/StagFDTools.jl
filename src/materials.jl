abstract type AbstractPlasticity end

Base.@kwdef struct DruckerPrager <: AbstractPlasticity
    C::Vector{Float64} = Float64[]
    ϕ::Vector{Float64} = Float64[]
    ψ::Vector{Float64} = Float64[]
    ηvp::Vector{Float64} = Float64[]
    cosϕ::Vector{Float64} = Float64[]
    sinϕ::Vector{Float64} = Float64[]
    sinψ::Vector{Float64} = Float64[]
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
    plasticity::P = DruckerPrager()
    compressible::Bool = false
end

initialize(::Type{DruckerPrager}, n::Integer) = DruckerPrager(
    C=1e50 * ones(n),
    ϕ=ones(n),
    ψ=ones(n),
    ηvp=ones(n),
    cosϕ=ones(n),
    sinϕ=ones(n),
    sinψ=ones(n),
)

initialize(::Type{DruckerHyperbolic}, n::Integer) = DruckerHyperbolic(
    σT=ones(n),
    C=1e50 * ones(n),
    ϕ=ones(n),
    ψ=ones(n),
    ηvp=ones(n),
    cosϕ=ones(n),
    sinϕ=ones(n),
    sinψ=ones(n),
)

initialize(::Type{DruckerAniso}, n::Integer) = DruckerAniso(
    δ=ones(n),
    C=1e50 * ones(n),
    ϕ=ones(n),
    ψ=ones(n),
    ηvp=ones(n),
    cosϕ=ones(n),
    sinϕ=ones(n),
    sinψ=ones(n),
)

initialize(::Type{Golchin2021}, n::Integer) = Golchin2021(
    C=1e50 * ones(n),
    ϕ=ones(n),
    ψ=ones(n),
    ηvp=ones(n),
    cosϕ=ones(n),
    sinϕ=ones(n),
    sinψ=ones(n),
    M=ones(n),
    N=ones(n),
    Pc=ones(n),
    a=ones(n),
    b=ones(n),
    c=ones(n),
    σT=ones(n),
)

initialize(::Type{Kiss2023}, n::Integer) = Kiss2023(
    C=1e50 * ones(n),
    ϕ=ones(n),
    ψ=ones(n),
    ηvp=ones(n),
    cosϕ=ones(n),
    sinϕ=ones(n),
    sinψ=ones(n),
    σT=ones(n),
    δσT=ones(n),
    P1=ones(n),
    τ1=ones(n),
    P2=ones(n),
    τ2=ones(n),
)

initialize(::Type{Nothing}, ::Integer) = nothing

function initialize(::Type{Materials}, nphases::Integer;
    plasticity=DruckerPrager(),
    compressible::Bool=false)
    return Materials(
        ρ=ones(nphases),
        n=ones(nphases),
        η0=ones(nphases),
        ξ0=1e50 * ones(nphases),
        G=1e50 * ones(nphases),
        β=1e-50 * ones(nphases),
        B=ones(nphases),
        plasticity=initialize_plasticity(plasticity, nphases),
        compressible=compressible,
    )
end

initialize_materials(nphases::Integer; kwargs...) = initialize(Materials, nphases; kwargs...)
# initialize_plasticity(n::Integer) = initialize(DruckerPrager, n)
initialize_plasticity(::Type{Nothing}, n::Integer) = initialize(Nothing, n)
initialize_plasticity(::Type{T}, n::Integer) where {T<:AbstractPlasticity} = initialize(T, n)
initialize_plasticity(::Nothing, n::Integer) = initialize(Nothing, n)
initialize_plasticity(p::AbstractPlasticity, n::Integer) = initialize(typeof(p), n)

function preprocess!(dp::DruckerPrager)
    @. dp.cosϕ = cosd(dp.ϕ)
    @. dp.sinϕ = sind(dp.ϕ)
    @. dp.sinψ = sind(dp.ψ)
end

function preprocess!(dh::DruckerHyperbolic)
    @. dh.cosϕ = cosd(dh.ϕ)
    @. dh.sinϕ = sind(dh.ϕ)
    @. dh.sinψ = sind(dh.ψ)
end

function preprocess!(da::DruckerAniso)
    @. da.cosϕ = cosd(da.ϕ)
    @. da.sinϕ = sind(da.ϕ)
    @. da.sinψ = sind(da.ψ)
end

preprocess!(::Nothing) = nothing

function preprocess!(g::Golchin2021)
    @. g.cosϕ = cosd(g.ϕ)
    @. g.sinϕ = sind(g.ϕ)
    @. g.sinψ = sind(g.ψ)
end

function preprocess!(k::Kiss2023)
    @. k.cosϕ = cosd(k.ϕ)
    @. k.sinϕ = sind(k.ϕ)
    @. k.sinψ = sind(k.ψ)
end

function preprocess!(mat::Materials)
    @. mat.B = (2 * mat.η0)^(-mat.n)
    preprocess!(mat.plasticity)
end

preprocess(x) = (preprocess!(x); x)