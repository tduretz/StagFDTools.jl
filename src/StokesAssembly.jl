using Base.Threads, SparseArrays

# Assembly storage for the single-phase (Stokes) kernels, following the fixed-pattern
# approach of src/TwoPhases/TwoPhases_v3.jl. The block matrices `M` are nested `Fields`
# and the kernels write `K[a][b][row, col] = val` into one of the storages below.

# Records every position an assembly kernel writes, whatever the value, so that
# entries that happen to be zero in the current state still enter the pattern.
struct PatternBlock
    I::Vector{Int}
    J::Vector{Int}
end

PatternBlock() = PatternBlock(Int[], Int[])

@inline function Base.setindex!(A::PatternBlock, val, i::Integer, j::Integer)
    push!(A.I, i)
    push!(A.J, j)
    return val
end

# Writes into the existing nonzeros of a SparseMatrixCSC. Assembly kernels write each
# row from a single loop iteration, so concurrent writes never touch the same slot
# and need no thread-local storage. Writes accumulate (like the duplicate
# summation of `sparse(I, J, V, m, n, +)`), so values must be zeroed before assembly.
struct FixedPattern{T<:SparseMatrixCSC}
    A::T
end

@inline function Base.setindex!(K::FixedPattern, val, i::Integer, j::Integer)
    A  = K.A
    rv = rowvals(A)
    r  = nzrange(A, j)
    k  = first(r) - 1 + searchsortedfirst(view(rv, r), i)
    (k <= last(r) && rv[k] == i) || error("entry ($i, $j) is not in the sparsity pattern; build it with set_sparsity_pattern!")
    nonzeros(A)[k] += val
    return val
end

# Kernels index per-thread storage (a vector) by thread, and shared storage directly.
@inline _target(K::AbstractVector, tid) = K[tid - nthreads(:interactive)]
@inline _target(K, tid) = K

# Per-thread storage and FixedPattern blocks can be written from several threads;
# anything else (e.g. ExtendableSparseMatrix blocks) is assembled serially.
@inline _threadsafe(K::AbstractVector) = true
@inline _threadsafe(K) = K[1][1] isa FixedPattern

"""
    assembly_loop(f, K, js)

Call `f(j, tid)` for every `j` in `js`: threaded when the storage `K` allows it, serially otherwise.
"""
function assembly_loop(f::F, K, js) where {F}
    if _threadsafe(K)
        @threads for j in js
            f(j, threadid())
        end
    else
        for j in js
            f(j, threadid())
        end
    end
    return nothing
end

# Generic traversal of the nested `Fields` block structure (any number of fields)
@inline _nblocks(M) = fieldcount(typeof(M))
@inline _rebuild(M, xs) = Base.typename(typeof(M)).wrapper(xs...)
_map_blocks(f, M) = _rebuild(M, ntuple(a -> _rebuild(M[a], ntuple(b -> f(M[a][b]), _nblocks(M[a]))), _nblocks(M)))

"""
    pattern_storage(M)

Per-thread `PatternBlock` storage, with the block structure of `M`, to pass to the
`Assemble*2D!` kernels in place of the matrix, before calling `set_sparsity_pattern!`.
"""
pattern_storage(M) = [_map_blocks(_ -> PatternBlock(), M) for _ in 1:nthreads()]

"""
    set_sparsity_pattern!(M, K_pattern)

Replace the structure of every block `M[a][b]` with the positions recorded in
`K_pattern` (from `pattern_storage`), all values set to zero.
"""
function set_sparsity_pattern!(M, K_pattern)
    for a in 1:_nblocks(M), b in 1:_nblocks(M[a])
        A = M[a][b]
        I = reduce(vcat, (K[a][b].I for K in K_pattern))
        J = reduce(vcat, (K[a][b].J for K in K_pattern))
        S = sparse(I, J, zeros(length(I)), size(A)...)
        copyto!(SparseArrays.getcolptr(A), SparseArrays.getcolptr(S))
        copy!(rowvals(A), rowvals(S))
        copy!(nonzeros(A), nonzeros(S))
    end
    return M
end

"""
    fixed_pattern(M)

Wrap each block of `M` so the `Assemble*2D!` kernels write directly into its nonzeros.
The pattern must already hold every written entry (see `set_sparsity_pattern!`).
"""
fixed_pattern(M) = _map_blocks(FixedPattern, M)

"""
    zero_values!(M)

Set all stored values of every block of `M` to zero, keeping the sparsity pattern.
"""
function zero_values!(M)
    for a in 1:_nblocks(M), b in 1:_nblocks(M[a])
        fill!(nonzeros(M[a][b]), 0)
    end
    return M
end
