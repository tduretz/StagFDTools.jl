for type in (:SMatrix, :MMatrix)
    @eval begin
        Base.@propagate_inbounds @inline inn(A::($type){M, N}) where {M, N} = ($type){M - 2, N - 2}(A[i + 1, j + 1] for i in 1:(M - 2), j in 1:(N - 2))
        Base.@propagate_inbounds @inline inn_x(A::($type){M, N}) where {M, N} = ($type){M - 2, N}(A[i + 1, j]     for i in 1:(M - 2), j in 1:N)
        Base.@propagate_inbounds @inline inn_y(A::($type){M, N}) where {M, N} = ($type){M, N - 2}(A[i, j + 1]     for i in 1:M,   j in 1:(N - 2))
        Base.@propagate_inbounds @inline av(A::($type){M, N}) where {M, N} = ($type){M - 1, N - 1}((A[i, j] + A[i + 1, j] + A[i, j + 1] + A[i + 1, j + 1]) / 4 for i in 1:(M - 1), j in 1:(N - 1))
        Base.@propagate_inbounds @inline avx(A::($type){M, N}) where {M, N} = ($type){M - 1, N}((A[i, j] + A[i + 1, j]) / 2 for i in 1:(M - 1), j in 1:N)
        Base.@propagate_inbounds @inline avy(A::($type){M, N}) where {M, N} = ($type){M, N - 1}((A[i, j] + A[i, j + 1]) / 2 for i in 1:M, j in 1:(N - 1))
        Base.@propagate_inbounds @inline harm(A::($type){M, N}) where {M, N} = ($type){M - 1, N - 1}(4 * inv(inv(A[i, j]) + inv(A[i + 1, j]) + inv(A[i, j + 1]) + inv(A[i + 1, j + 1]))  for i in 1:(M - 1), j in 1:(N - 1))
        Base.@propagate_inbounds @inline ∂x(A::($type){M, N}) where {M, N} = ($type){M - 1, N}(A[i + 1, j] - A[i, j] for i in 1:(M - 1), j in 1:N)
        Base.@propagate_inbounds @inline ∂x_inn(A::($type){M, N}) where {M, N} = ($type){M - 1, N - 2}(A[i + 1, j] - A[i, j] for i in 1:(M - 1), j in 2:(N - 1))
        Base.@propagate_inbounds @inline ∂y(A::($type){M, N}) where {M, N} = ($type){M, N - 1}(A[i, j + 1] - A[i, j] for i in 1:M, j in 1:(N - 1))
        Base.@propagate_inbounds @inline ∂y_inn(A::($type){M, N}) where {M, N} = ($type){M - 2, N - 1}(A[i, j + 1] - A[i, j] for i in 2:(M - 1), j in 1:(N - 1))
        Base.@propagate_inbounds @inline ∂kk(A::($type){M1, N1}, B::($type){M2, N2}) where {M1, N1, M2, N2} = ($type){M1, N2}(A[i, j + 1] + B[i + 1, j] for i in 1:M1, j in 1:N2)
    end
end

# @albert-de-montserrat: could we make the size of the SVector below variable?
# Ideally it's 2 when working on momentum balance
# But it's one (or scalar) when computing local rheology.
@inline function deviatoric_strain_rate(Dxx, Dxy, Dyx, Dyy)
    ε̇kk = SVector{2}(@. Dxx + Dyy)
    ε̇xx = SVector{2}(@. Dxx - 1 / 3 * ε̇kk)
    ε̇yy = SVector{2}(@. Dyy - 1 / 3 * ε̇kk)
    ε̇xy = SVector{2}(@. 1 / 2 * (Dxy + Dyx))
    return ε̇xx, ε̇yy, ε̇xy, ε̇kk
end


using StaticArrays

@inline function deviatoric_strain_rate(
        Dxx::SVector{N, T},
        Dxy::SVector{N, T},
        Dyx::SVector{N, T},
        Dyy::SVector{N, T}
    ) where {N, T}
    ε̇kk = Dxx .+ Dyy
    ε̇xx = Dxx .- (1 / 3) .* ε̇kk
    ε̇yy = Dyy .- (1 / 3) .* ε̇kk
    ε̇xy = (1 / 2) .* (Dxy .+ Dyx)

    return ε̇xx, ε̇yy, ε̇xy, ε̇kk
end

@inline function deviatoric_strain_rate(Dxx::T, Dxy::T, Dyx::T, Dyy::T) where {T}
    ε̇kk = Dxx + Dyy
    ε̇xx = Dxx - (1 / 3) * ε̇kk
    ε̇yy = Dyy - (1 / 3) * ε̇kk
    ε̇xy = (1 / 2) * (Dxy + Dyx)
    return ε̇xx, ε̇yy, ε̇xy, ε̇kk
end

@inline function effective_strain_rate(
        ε̇xx::SVector{N},
        ε̇yy::SVector{N},
        ε̇xy::SVector{N},
        τ0xx::SVector{N},
        τ0yy::SVector{N},
        τ0xy::SVector{N},
        _2GΔt
    ) where {N}
    return (
        ε̇xx .+ τ0xx .* _2GΔt,
        ε̇yy .+ τ0yy .* _2GΔt,
        ε̇xy .+ τ0xy .* _2GΔt,
    )
end

@inline function effective_strain_rate(
        ε̇xx::T, ε̇yy::T, ε̇xy::T,
        τ0xx::T, τ0yy::T, τ0xy::T,
        _2GΔt
    ) where {T}
    return (
        ε̇xx + τ0xx * _2GΔt,
        ε̇yy + τ0yy * _2GΔt,
        ε̇xy + τ0xy * _2GΔt,
    )
end
