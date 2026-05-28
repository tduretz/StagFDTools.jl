Base.@kwdef mutable struct Physics
    Poisson::Bool = false
    Stokes::Bool = false
    TwoPhases::Bool = false
    Cosserat::Bool = false
    Thermal::Bool = false
end

# Base.@kwdef mutable struct NumberingPoisson
#     Num     ::Union{Matrix{Int64},  Missing} = missing
#     Type    ::Union{Matrix{Symbol}, Missing} = missing
#     Pattern ::Union{SMatrix,        Missing} = missing
# end

Base.@kwdef mutable struct NumberingPoisson{N}
    num::Union{Matrix{Int64}, Missing} = missing
    type::Union{Matrix{Symbol}, Missing} = missing
    bc_val::Union{Matrix{Float64}, Missing} = missing
    pattern::Union{SMatrix{N, N, Int64}, Missing} = missing
end


struct NumberingPoisson2{T1, T2, T3, T4}
    num::Matrix{T1}
    type::Matrix{T2}
    bc_val::Matrix{T3}
    pattern::T4

    function NumberingPoisson2(ni::NTuple, ::Val{N}) where {N}
        num = zeros(Int64, (ni .+ 2)...)
        bc_val = zeros(Float64, (ni .+ 2)...)
        type = Matrix{Symbol}(undef, (ni .+ 2)...)
        pattern = @MMatrix zeros(Int64, N, N)
        return new{
            eltype(num),
            eltype(type),
            eltype(bc_val),
            typeof(pattern),
        }(num, type, bc_val, pattern)
    end

    function NumberingPoisson2{N}(ni::NTuple) where {N}
        num = zeros(Int64, (ni .+ 2)...)
        bc_val = zeros(Float64, (ni .+ 2)...)
        type = Matrix{Symbol}(undef, (ni .+ 2)...)
        pattern = @MMatrix zeros(Int64, 3, 3)
        return new{
            eltype(num),
            eltype(type),
            eltype(bc_val),
            typeof(pattern),
        }(num, type, bc_val, pattern)
    end
end
