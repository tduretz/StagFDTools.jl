#---------------------------------------------------------------------------------------
# Allocation-free GCR with the block preconditioner of KSP_GCR_TwoPhases_opt! (Solvers.jl):
#   Cholesky of the d-d block (Jqq) and of the velocity Schur complement J̃vv = Jvv - Jvp Dpp Jpv,
#   Dpp = diag(Jpp)⁻¹ (Jpp is diagonal: Δt ∇⋅V + ΔP/K).
# The global Jacobian A (JacobianCache) has a fixed pattern, so every block, the pattern of J̃vv and the
# Schur product are mapped once onto A.nzval; each solve only gathers values, refactors numerically
# (symbolic analysis reused) and iterates in preallocated storage. Triangular solves use
# cholmod_l_solve2 with persistent workspace (ldiv! with a CHOLMOD factor allocates O(n) per call).
#---------------------------------------------------------------------------------------

const LSS = SparseArrays.LibSuiteSparse
const CM  = SparseArrays.CHOLMOD

# Sub-block of A as its own CSC matrix: M.nzval[k] = A.nzval[idx[k]]
struct BlockMap
    M   :: SparseMatrixCSC{Float64,Int}
    idx :: Vector{Int}
end

function BlockMap(A::SparseMatrixCSC, rows, cols)
    Ai  = SparseMatrixCSC(size(A)..., A.colptr, A.rowval, Float64.(1:nnz(A)))   # nonzeros tagged by their index
    S   = Ai[rows, cols]
    idx = round.(Int, nonzeros(S))
    return BlockMap(SparseMatrixCSC(size(S)..., S.colptr, S.rowval, zeros(length(idx))), idx)
end

function gather!(B::BlockMap, A::SparseMatrixCSC)
    v, a = nonzeros(B.M), nonzeros(A)
    @inbounds for k in eachindex(B.idx)
        v[k] = a[B.idx[k]]
    end
    return B.M
end

@inline function nzindex(A::SparseMatrixCSC, i, j)
    r = A.colptr[j]:A.colptr[j+1]-1
    k = searchsortedfirst(view(A.rowval, r), i)
    (k <= length(r) && A.rowval[r[k]] == i) || error("entry ($i, $j) not in sparsity pattern")
    return r[k]
end

# J̃vv = Jvv - Jvp Dpp Jpv in place: fixed pattern S, with the Jvv entries and the products mapped onto A.nzval
struct SchurMap
    S      :: SparseMatrixCSC{Float64,Int}
    pos_vv :: Vector{Int}   # S index of each Jvv entry ...
    a_vv   :: Vector{Int}   # ... and its A index
    tgt    :: Vector{Int}   # S index of each product term Jvp[i,p] Dpp[p] Jpv[p,j]
    a_vp   :: Vector{Int}
    a_pv   :: Vector{Int}
    p      :: Vector{Int}
    a_pp   :: Vector{Int}   # A index of the Jpp diagonal
    Dpp    :: Vector{Float64}
end

function SchurMap(A::SparseMatrixCSC, iu, ip)
    Jvv, Jvp, Jpv = BlockMap(A, iu, iu), BlockMap(A, iu, ip), BlockMap(A, ip, iu)
    ones_(B) = SparseMatrixCSC(size(B.M)..., B.M.colptr, B.M.rowval, ones(length(B.idx)))
    S = ones_(Jvv) + ones_(Jvp) * ones_(Jpv)      # positive values: no cancellation in the pattern
    fill!(nonzeros(S), 0.0)
    pos_vv = Int[]
    for j in axes(Jvv.M, 2), k in nzrange(Jvv.M, j)
        push!(pos_vv, nzindex(S, rowvals(Jvv.M)[k], j))
    end
    # Rows of Jpv: for each p, the list of (column j, A index)
    np   = length(ip)
    rows = [Tuple{Int,Int}[] for _ in 1:np]
    for j in axes(Jpv.M, 2), k in nzrange(Jpv.M, j)
        push!(rows[rowvals(Jpv.M)[k]], (j, Jpv.idx[k]))
    end
    tgt, a_vp, a_pv, pp = Int[], Int[], Int[], Int[]
    for p in 1:np, k in nzrange(Jvp.M, p), (j, a2) in rows[p]
        push!(tgt, nzindex(S, rowvals(Jvp.M)[k], j)); push!(a_vp, Jvp.idx[k]); push!(a_pv, a2); push!(pp, p)
    end
    a_pp = [nzindex(A, ip[p], ip[p]) for p in 1:np]
    return SchurMap(S, pos_vv, Jvv.idx, tgt, a_vp, a_pv, pp, a_pp, zeros(np))
end

function update!(sm::SchurMap, A::SparseMatrixCSC)
    a, s = nonzeros(A), nonzeros(sm.S)
    @inbounds for k in eachindex(sm.a_pp)
        sm.Dpp[k] = inv(a[sm.a_pp[k]])
    end
    fill!(s, 0.0)
    @inbounds for k in eachindex(sm.pos_vv)
        s[sm.pos_vv[k]] += a[sm.a_vv[k]]
    end
    @inbounds for t in eachindex(sm.tgt)
        s[sm.tgt[t]] -= a[sm.a_vp[t]] * sm.Dpp[sm.p[t]] * a[sm.a_pv[t]]
    end
    return sm.S
end

# Triangular solves with a CHOLMOD factor without allocation (persistent X/Y/E workspace)
mutable struct CholSolve
    b :: Vector{Float64}
    B :: LSS.cholmod_dense_struct
    X :: Base.RefValue{Ptr{LSS.cholmod_dense_struct}}
    Y :: Base.RefValue{Ptr{LSS.cholmod_dense_struct}}
    E :: Base.RefValue{Ptr{LSS.cholmod_dense_struct}}
end

function CholSolve(n)
    b = zeros(n)
    B = LSS.cholmod_dense_struct()
    B.nrow, B.ncol, B.nzmax, B.d = n, 1, n, n
    B.x, B.z, B.xtype, B.dtype = pointer(b), C_NULL, LSS.CHOLMOD_REAL, LSS.CHOLMOD_DOUBLE
    null() = Ref(Ptr{LSS.cholmod_dense_struct}(C_NULL))
    cs = CholSolve(b, B, null(), null(), null())
    finalizer(cs) do c
        for H in (c.X, c.Y, c.E)
            H[] == C_NULL || LSS.cholmod_l_free_dense(H, CM.getcommon(Int))
        end
    end
    return cs
end

function chol_solve!(x::AbstractVector{Float64}, cs::CholSolve, F::CM.Factor{Float64})
    copyto!(cs.b, x)
    GC.@preserve cs begin
        ok = LSS.cholmod_l_solve2(LSS.CHOLMOD_A, F.ptr, Ptr{LSS.cholmod_dense_struct}(pointer_from_objref(cs.B)),
                                  C_NULL, cs.X, C_NULL, cs.Y, cs.E, CM.getcommon(Int))
        ok == 1 || error("cholmod_l_solve2 failed")
        unsafe_copyto!(pointer(x), Ptr{Float64}(unsafe_load(cs.X[]).x), length(x))
    end
    return x
end

# Numeric refactorisation in place: values copied into a persistent CHOLMOD matrix (same CSC layout, stype = 1)
# and cholmod_l_factorize reuses the symbolic analysis of F (cholesky! would copy the matrix on every call)
function chol_sparse(S::SparseMatrixCSC{Float64,Int})
    Sc = CM.Sparse(Hermitian(S))
    s  = unsafe_load(Sc.ptr)
    (s.nzmax == nnz(S) && s.packed == 1 && s.sorted == 1) || error("unexpected CHOLMOD storage")
    return Sc
end

function refactor!(F::CM.Factor{Float64}, Sc::CM.Sparse{Float64}, S::SparseMatrixCSC{Float64,Int})
    GC.@preserve Sc S begin
        unsafe_copyto!(Ptr{Float64}(unsafe_load(Sc.ptr).x), pointer(nonzeros(S)), nnz(S))
        LSS.cholmod_l_factorize(Sc.ptr, F.ptr, CM.getcommon(Int))
    end
    return issuccess(F)
end

struct GCRWorkspace{TF,TS}
    iu :: UnitRange{Int}; ip :: UnitRange{Int}; iq :: UnitRange{Int}
    Jvp :: BlockMap; Jpv :: BlockMap; Jpq :: BlockMap; Jqp :: BlockMap
    Jqu :: BlockMap; Juq :: BlockMap; Jqq :: BlockMap
    schur :: SchurMap
    Fv :: TF; Fq :: TF
    Sv :: TS; Sq :: TS           # persistent CHOLMOD copies of J̃vv and Jqq
    cv :: CholSolve; cq :: CholSolve
    restart :: Int; maxit :: Int
    coupled :: Bool              # include Juq (momentum ← d) in the preconditioner sweep
    its_fresh :: Base.RefValue{Int}   # GCR iterations of the last solve with a fresh preconditioner
    its_last  :: Base.RefValue{Int}   # GCR iterations of the last solve
    n_refresh :: Base.RefValue{Int}   # number of preconditioner updates (factorisations)
    f :: Vector{Float64}; v :: Vector{Float64}; s :: Vector{Float64}
    VV :: Matrix{Float64}; SS :: Matrix{Float64}; Vnorm2 :: Vector{Float64}
    du :: Vector{Float64}; dp :: Vector{Float64}; dq :: Vector{Float64}; q0 :: Vector{Float64}
    r̃u :: Vector{Float64}; r̃p :: Vector{Float64}; tmpp :: Vector{Float64}; tmpq :: Vector{Float64}
end

function GCRWorkspace(J; restart=50, maxit=1000, coupled=true)
    A  = J.A
    iu = first(J.rng[1]):last(J.rng[2]); ip = J.rng[3]; iq = J.rng[4]
    Jvp, Jpv, Jpq, Jqp = BlockMap(A, iu, ip), BlockMap(A, ip, iu), BlockMap(A, ip, iq), BlockMap(A, iq, ip)
    Jqu, Juq, Jqq      = BlockMap(A, iq, iu), BlockMap(A, iu, iq), BlockMap(A, iq, iq)
    schur = SchurMap(A, iu, ip)
    gather!(Jqq, A); update!(schur, A)
    Fq = cholesky(Hermitian(Jqq.M); check=false)       # symbolic analyses, reused afterwards
    Fv = cholesky(Hermitian(schur.S); check=false)
    nu, np, nq = length(iu), length(ip), length(iq); N = nu + np + nq
    return GCRWorkspace(iu, ip, iq, Jvp, Jpv, Jpq, Jqp, Jqu, Juq, Jqq, schur, Fv, Fq, chol_sparse(schur.S), chol_sparse(Jqq.M),
                        CholSolve(nu), CholSolve(nq),
                        restart, maxit, coupled, Ref(0), Ref(0), Ref(0), zeros(N), zeros(N), zeros(N), zeros(N, restart), zeros(N, restart),
                        zeros(restart), zeros(nu), zeros(np), zeros(nq), zeros(nq), zeros(nu), zeros(np), zeros(np), zeros(nq))
end

# Refresh blocks and numeric factorisations from the current A
function update_preconditioner!(w::GCRWorkspace, A)
    foreach(B -> gather!(B, A), (w.Jvp, w.Jpv, w.Jpq, w.Jqp, w.Jqu, w.Juq, w.Jqq))
    update!(w.schur, A)
    ok = refactor!(w.Fq, w.Sq, w.Jqq.M) & refactor!(w.Fv, w.Sv, w.schur.S)
    ok || @warn "GCR preconditioner: Cholesky failed (block not SPD)"
    return w
end

# s = P⁻¹ f: block lower-triangular sweep d → (V,P) → d, P eliminated with the diagonal Dpp (Schur J̃vv).
# More sweeps (block Gauss-Seidel) were tested and do not reduce GCR iterations.
function apply_preconditioner!(w::GCRWorkspace)
    (; f, s, iu, ip, iq, du, dp, dq, q0, r̃u, r̃p, tmpp, tmpq) = w
    fu, fp, fq = view(f, iu), view(f, ip), view(f, iq)
    Dpp = w.schur.Dpp
    copyto!(q0, fq); chol_solve!(q0, w.cq, w.Fq)          # d predictor
    mul!(r̃p, w.Jpq.M, q0); @. r̃p = fp - r̃p
    @. tmpp = Dpp * r̃p
    mul!(r̃u, w.Jvp.M, tmpp); @. r̃u = fu - r̃u
    w.coupled && mul!(r̃u, w.Juq.M, q0, -1.0, 1.0)        # momentum ← d coupling
    copyto!(du, r̃u); chol_solve!(du, w.cv, w.Fv)
    mul!(tmpp, w.Jpv.M, du); @. dp = Dpp * (r̃p - tmpp)
    mul!(tmpq, w.Jqp.M, dp); @. tmpq = fq - tmpq
    mul!(tmpq, w.Jqu.M, du, -1.0, 1.0)
    copyto!(dq, tmpq); chol_solve!(dq, w.cq, w.Fq)
    copyto!(view(s, iu), du); copyto!(view(s, ip), dp); copyto!(view(s, iq), dq)
    return s
end

"""
    gcr_solve!(x, A, b, w::GCRWorkspace; reltol, abstol)

Restarted GCR for A x = b, right-preconditioned with the block preconditioner in `w` (refreshed from A first
if `refresh`, otherwise the lagged preconditioner of an earlier Jacobian is used).
Starts from x = 0. Returns the number of iterations (negative if not converged).
"""
function gcr_solve!(x, A, b, w::GCRWorkspace; reltol=1e-10, abstol=1e-30, refresh=true)
    refresh && (update_preconditioner!(w, A); w.n_refresh[] += 1)
    (; f, v, s, VV, SS, Vnorm2, restart, maxit) = w
    fill!(x, 0.0)
    copyto!(f, b)
    norm0 = norm(f)
    tol   = max(reltol * norm0, abstol)
    norm0 <= tol && return 0
    its = 0
    while its < maxit
        for k in 1:restart
            apply_preconditioner!(w)
            mul!(v, A, s)
            for j in 1:k-1
                Vj, Sj = view(VV, :, j), view(SS, :, j)
                β = dot(Vj, v) / Vnorm2[j]
                axpy!(-β, Vj, v); axpy!(-β, Sj, s)
            end
            den = dot(v, v)
            α   = dot(f, v) / den
            axpy!(α, s, x); axpy!(-α, v, f)
            its += 1
            norm(f) <= tol && return its
            copyto!(view(VV, :, k), v); copyto!(view(SS, :, k), s); Vnorm2[k] = den
            its >= maxit && break
        end
    end
    return -its
end

"""
    lagged_gcr_solve!(x, A, b, w; refresh=:auto, lag_factor=2, its_min=20, kwargs...)

GCR with a lagged preconditioner. `refresh = true` forces an update (e.g. first Newton iteration of a step);
with `:auto` the preconditioner is kept unless the previous solve needed more than
max(lag_factor × iterations with a fresh preconditioner, its_min). A lagged solve that does not converge is
repeated with a fresh preconditioner. Returns the total number of GCR iterations (negative if not converged).
"""
function lagged_gcr_solve!(x, A, b, w::GCRWorkspace; refresh=:auto, lag_factor=2, its_min=20, kwargs...)
    fresh = refresh === true || w.n_refresh[] == 0 ||
            w.its_last[] > max(lag_factor * w.its_fresh[], its_min)
    its = gcr_solve!(x, A, b, w; refresh=fresh, kwargs...)
    if its < 0 && !fresh
        its2  = gcr_solve!(x, A, b, w; refresh=true, kwargs...)
        its   = its2 < 0 ? its2 - abs(its) : its2 + abs(its)
        fresh = true
        w.its_fresh[] = abs(its2)
    elseif fresh
        w.its_fresh[] = abs(its)
    end
    w.its_last[] = abs(its)
    return its
end
