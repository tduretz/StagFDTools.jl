# 2D Poisson problem with a discontinuous coefficient on a Cartesian grid, interface not fitted:
#   ∇·(β∇u) = f,   [u] = a,   [β ∂ₙu] = b   on Γ = {φ = 0},   [q] = q⁺ - q⁻ (φ > 0 minus φ < 0)
# β is constant on each side. Node-centred unknowns, 5-point stencil, Dirichlet boundaries.
# Methods:
#   :standard  harmonic face averaging of β, jumps ignored (only meaningful for a = b = 0)
#   :GFM       ghost fluid method (Liu, Fedkiw & Kang 2000), dimension by dimension: at a crossed
#              face the flux component jump is taken as [β∂ₙu] nₑ (tangential flux jump dropped)
#   :xGFM      as GFM with the full flux component jump [β∇u]·e = [β∂ₙu] nₑ + [β∂ₜu] tₑ, where
#              [β∂ₜu] = [β] ∂ₜu + β⁺ ∂ₛa depends on the solution (one-sided fits); fixed-point
#              iterations on the same symmetric operator (idea of Egan & Gibou 2020)
#   :IIM1      immersed interface method: stencil members across Γ replaced by the extension of
#   :IIM2      the row side's solution, u_k + s([u] + [∇u]·δ (+ ½δᵀ[∇∇u]δ)); one-sided traces
#              (linear / quadratic fits); the traced flux q = β∂ₙu is weighted by the other side's β
#   :IIMz1     IIM in w = βu (the z = ηv form of the Stokes code): Δw = f on each side, [∂ₙw] = b is
#   :IIMz2     known, and [w] = [β]ū + β̄a, [∂ₜw] = [β]∂ₛū + β̄∂ₛa only need traces of the continuous
#              ū = (u⁺ + u⁻)/2 (and ∂ₛū, ∂ₛₛū), averaged with β weights as in Landry's method
#   :IIMh1     hybrid: rows (and fluxes) on the soft side use the u-form extension, rows on the stiff
#   :IIMh2     side the z-form one (the u form amplifies errors by β_stiff, the z form by 1/β_soft)
using LinearAlgebra, SparseArrays, StaticArrays, Printf
import ForwardDiff as FD

side(φ) = φ < 0 ? -1 : 1

# ---------------------------------------------------------------- problem data
function derived(prob)
    (; φ, um, up, β) = prob
    βs(s) = s < 0 ? β.neg : β.pos
    u(x, s) = s < 0 ? um(x) : up(x)
    ∇u(x, s) = FD.gradient(y -> u(y, s), x)
    Δu(x, s) = tr(FD.hessian(y -> u(y, s), x))
    f(x, s) = βs(s) * Δu(x, s)
    nrm(x) = (g = FD.gradient(φ, x); g / norm(g))
    tng(n) = SVector(-n[2], n[1])
    κ(x) = tr(FD.jacobian(nrm, x))
    a(x) = up(x) - um(x)
    b(x) = (n = nrm(x); β.pos * dot(FD.gradient(up, x), n) - β.neg * dot(FD.gradient(um, x), n))
    as(x) = dot(FD.gradient(a, x), tng(nrm(x)))
    ass(x) = (n = nrm(x); t = tng(n); dot(t, FD.hessian(a, x) * t) - κ(x) * dot(FD.gradient(a, x), n))
    bs(x) = dot(FD.gradient(b, x), tng(nrm(x)))
    jf(x) = Δu(x, 1) - Δu(x, -1)                 # [f/β] = [Δu]
    return (; φ, β, βs, u, ∇u, f, nrm, tng, κ, a, b, as, ass, bs, jf)
end

# ---------------------------------------------------------------- linear forms over node values
mutable struct LF
    w::Dict{Int,Float64}
    c::Float64
end
LF(c=0.0) = LF(Dict{Int,Float64}(), c)
function axpy!(d::LF, α, s::LF)
    for (k, v) in s.w
        d.w[k] = get(d.w, k, 0.0) + α * v
    end
    d.c += α * s.c
    return d
end
lin(terms...; c=0.0) = (L = LF(c); for (α, s) in terms; axpy!(L, α, s); end; L)
evalLF(L::LF, U) = sum((v * U[k] for (k, v) in L.w); init=0.0) + L.c

# ---------------------------------------------------------------- grid and geometry
function setup(prob, N)
    P  = derived(prob)
    h  = 2 / N
    xs = range(-1, 1, length=N + 1)
    X(i, j) = SVector(xs[i], xs[j])
    S  = [side(P.φ(X(i, j))) for i in 1:N+1, j in 1:N+1]
    LI = LinearIndices(S)
    # crossings on grid edges: key (i, j, dir) for (i,j)→(i+1,j) (dir 1) or (i,j)→(i,j+1) (dir 2)
    cross = Dict{NTuple{3,Int},Any}()
    for j in 1:N+1, i in 1:N+1, (dir, di, dj) in ((1, 1, 0), (2, 0, 1))
        (i + di <= N + 1 && j + dj <= N + 1) || continue
        S[i, j] == S[i+di, j+dj] && continue
        x0, x1 = X(i, j), X(i + di, j + dj)
        lo, hi = 0.0, 1.0
        f0 = P.φ(x0)
        for _ in 1:60
            m = (lo + hi) / 2
            sign(P.φ(x0 + m * (x1 - x0))) == sign(f0) ? (lo = m) : (hi = m)
        end
        θ  = (lo + hi) / 2
        xΓ = x0 + θ * (x1 - x0)
        n  = P.nrm(xΓ)
        cross[(i, j, dir)] = (x=xΓ, θ=θ, n=n, t=P.tng(n), κ=P.κ(xΓ))
    end
    return (; P, N, h, xs, X, S, LI, cross)
end

# crossing between node (i,j) and neighbour (i+di, j+dj); θ measured from (i,j)
function edge(G, i, j, di, dj)
    dir = di != 0 ? 1 : 2
    c = di + dj > 0 ? G.cross[(i, j, dir)] : G.cross[(i + di, j + dj, dir)]
    θ = di + dj > 0 ? c.θ : 1 - c.θ
    return c, θ
end

# least-squares fit of degree 1 or 2 on nodes of side s nearest to xo, evaluated at xΓ (cell units)
function fit(G, xΓ, xo, s, deg)
    (; N, h, X, S, LI) = G
    ic, jc = round(Int, (xo[1] + 1) / h) + 1, round(Int, (xo[2] + 1) / h) + 1
    cand = [CartesianIndex(i, j) for j in max(1, jc - 5):min(N + 1, jc + 5), i in max(1, ic - 5):min(N + 1, ic + 5) if S[i, j] == s]
    sort!(cand, by=I -> norm(X(I[1], I[2]) - xo))
    row(I) = (d = (X(I[1], I[2]) - xΓ) / h;
              deg == 1 ? SVector(1.0, d[1], d[2]) : SVector(1.0, d[1], d[2], d[1]^2 / 2, d[2]^2 / 2, d[1] * d[2]))
    n0, tol = deg == 1 ? (3, 0.1) : (10, 1e-2)
    for M in n0:length(cand)
        V = [row(I)[k] for I in cand[1:M], k in 1:(deg == 1 ? 3 : 6)]
        σ = svdvals(V)
        σ[end] > tol * σ[1] || continue
        Pm = pinv(V)
        ids = [LI[I] for I in cand[1:M]]
        form(r) = LF(Dict(ids[k] => r[k] for k in 1:M), 0.0)
        g = (form(Pm[2, :] ./ h), form(Pm[3, :] ./ h))
        H = deg == 1 ? nothing : (form(Pm[4, :] ./ h^2), form(Pm[6, :] ./ h^2), form(Pm[5, :] ./ h^2))  # xx, xy, yy
        return (v=form(Pm[1, :]), g=g, H=H)
    end
    error("fit: no well-conditioned node set on side $s near $xΓ")
end

dirform(g, e) = lin((e[1], g[1]), (e[2], g[2]))                       # e·∇u
quadform(H, e1, e2) = lin((e1[1] * e2[1], H[1]), (e1[1] * e2[2] + e1[2] * e2[1], H[2]), (e1[2] * e2[2], H[3]))

# jumps at a crossing as linear forms of the node values: [u_n], [H_nn], [H_nt], [H_tt]
function iim_jumps(G, c, order)
    (; P, h) = G
    βm, βp = P.β.neg, P.β.pos
    L  = h / max(abs(c.n[1]), abs(c.n[2]))
    deg = order == 1 ? 1 : 2
    Fm, Fp = fit(G, c.x, c.x - L * c.n, -1, deg), fit(G, c.x, c.x + L * c.n, 1, deg)
    b, as = P.b(c.x), P.as(c.x)
    # q = β⁻u_n⁻ = β⁺u_n⁺ - b estimated from both sides, each weighted by the other side's β
    # (w⁻ = β⁺/(β⁻+β⁺)): the error of [u_n] ≈ q/β_soft is then not amplified by the contrast
    wm, wp = βp / (βm + βp), βm / (βm + βp)
    q  = lin((wm * βm, dirform(Fm.g, c.n)), (wp * βp, dirform(Fp.g, c.n)); c=-wp * b)
    Un = lin((1 / βp - 1 / βm, q); c=b / βp)
    order == 1 && return (; Un, a=P.a(c.x), as)
    D(F) = lin((1.0, quadform(F.H, c.t, c.n)), (c.κ, dirform(F.g, c.t)))   # d(u_n)/ds
    bs = P.bs(c.x)
    q′  = lin((wm * βm, D(Fm)), (wp * βp, D(Fp)); c=-wp * bs)
    dUn = lin((1 / βp - 1 / βm, q′); c=bs / βp)
    Htt = lin((c.κ, Un); c=P.ass(c.x))
    Hnt = lin((1.0, dUn); c=-c.κ * as)
    Hnn = lin((-1.0, Htt); c=P.jf(c.x))
    return (; Un, Hnn, Hnt, Htt, a=P.a(c.x), as)
end

# extension J(δ) = [u] + [∇u]·δ (+ ½δᵀ[∇∇u]δ) as a linear form
function jump_ext(J, c, δ, order)
    δn, δt = dot(δ, c.n), dot(δ, c.t)
    L = lin((δn, J.Un); c=J.a + δt * J.as)
    order == 2 && (L = lin((1.0, L), (δn^2 / 2, J.Hnn), (δn * δt, J.Hnt), (δt^2 / 2, J.Htt)))
    return L
end

# z-form jumps (w = βu): [w], [∂ₜw], [H_nn], [H_nt], [H_tt] of w as linear forms of the node values u
function iimz_jumps(G, c, order)
    (; P, h) = G
    βm, βp = P.β.neg, P.β.pos
    Δβ, β̄ = βp - βm, (βp + βm) / 2
    L   = h / max(abs(c.n[1]), abs(c.n[2]))
    deg = order == 1 ? 1 : 2
    Fm, Fp = fit(G, c.x, c.x - L * c.n, -1, deg), fit(G, c.x, c.x + L * c.n, 1, deg)
    a, as, b = P.a(c.x), P.as(c.x), P.b(c.x)
    wm, wp = βm / (βm + βp), βp / (βm + βp)            # Landry: η-weighted average of the two sides
    ū  = lin((wm, Fm.v), (wp, Fp.v); c=(wm - wp) * a / 2)
    ūs = lin((wm, dirform(Fm.g, c.t)), (wp, dirform(Fp.g, c.t)); c=(wm - wp) * as / 2)
    W  = lin((Δβ, ū); c=β̄ * a)
    Wt = lin((Δβ, ūs); c=β̄ * as)
    order == 1 && return (; W, Wt, b)
    dss(F) = lin((1.0, quadform(F.H, c.t, c.t)), (-c.κ, dirform(F.g, c.n)))      # d²u/ds² on one side
    ass = P.ass(c.x)
    ūss = lin((wm, dss(Fm)), (wp, dss(Fp)); c=(wm - wp) * ass / 2)
    Htt = lin((Δβ, ūss); c=β̄ * ass + c.κ * b)
    Hnt = lin((-c.κ, Wt); c=P.bs(c.x))
    Hnn = lin((-1.0, Htt); c=P.f(c.x, 1) - P.f(c.x, -1))
    return (; W, Wt, b, Hnn, Hnt, Htt)
end

function jumpz_ext(J, c, δ, order)
    δn, δt = dot(δ, c.n), dot(δ, c.t)
    L = lin((1.0, J.W), (δt, J.Wt); c=δn * J.b)
    order == 2 && (L = lin((1.0, L), (δn^2 / 2, J.Hnn), (δn * δt, J.Hnt), (δt^2 / 2, J.Htt)))
    return L
end

# ---------------------------------------------------------------- assembly and solve
const NBR = ((1, 0), (-1, 0), (0, 1), (0, -1))

function solve(G, method; maxit=200, tol=1e-12, tw=:stiff)
    (; P, N, h, X, S, LI) = G
    nI  = (N - 1)^2
    id(i, j) = (j - 2) * (N - 1) + (i - 1)
    inner(i, j) = 2 <= i <= N && 2 <= j <= N
    Ub  = zeros((N + 1)^2)                           # boundary values (exact)
    for j in 1:N+1, i in 1:N+1
        inner(i, j) || (Ub[LI[i, j]] = P.u(X(i, j), S[i, j]))
    end
    node_of = Dict(LI[i, j] => (i, j) for j in 1:N+1, i in 1:N+1)
    hyb   = method in (:IIMh1, :IIMh2)
    soft  = P.β.neg <= P.β.pos ? -1 : 1               # soft side (hybrid)
    useu(s) = method in (:IIM1, :IIM2) || (hyb && s == soft)
    usez(s) = method in (:IIMz1, :IIMz2) || (hyb && s != soft)
    order = method in (:IIM2, :IIMz2, :IIMh2) ? 2 : 1
    Jc, Jz = Dict{Any,Any}(), Dict{Any,Any}()        # cached IIM jumps per crossing
    jumps(c)  = get!(() -> iim_jumps(G, c, order), Jc, c.x)
    jumpsz(c) = get!(() -> iimz_jumps(G, c, order), Jz, c.x)
    Is, Js, Vs = Int[], Int[], Float64[]
    b0 = zeros(nI)
    addterm!(r, L::LF, α) = begin
        for (k, v) in L.w
            (i, j) = node_of[k]
            inner(i, j) ? (push!(Is, r); push!(Js, id(i, j)); push!(Vs, α * v)) : (b0[r] -= α * v * Ub[k])
        end
        b0[r] -= α * L.c
    end
    node(i, j) = LF(Dict(LI[i, j] => 1.0), 0.0)
    # GFM-type flux pieces: per row, list of (crossing, θ, e, β̂, βK) to build the RHS corrections
    gfm_faces = [Tuple{Int,Any,Float64,SVector{2,Float64},Float64,Float64}[] for _ in 1:nI]
    for j in 2:N, i in 2:N
        r, s0 = id(i, j), S[i, j]
        x0 = X(i, j)
        if useu(s0)
            addterm!(r, node(i, j), -4 / h^2)
            for (di, dj) in NBR
                addterm!(r, node(i + di, j + dj), 1 / h^2)
                S[i+di, j+dj] == s0 && continue
                c, _ = edge(G, i, j, di, dj)
                addterm!(r, jump_ext(jumps(c), c, X(i + di, j + dj) - c.x, order), s0 / h^2)
            end
            b0[r] += P.f(x0, s0) / P.βs(s0)
        elseif usez(s0)
            # Δw = f, w = βu, members across Γ: w_k + s0 J(δ); row scaled by 1/β_s0
            βP = P.βs(s0)
            addterm!(r, node(i, j), -4 / h^2)
            for (di, dj) in NBR
                sK = S[i+di, j+dj]
                addterm!(r, node(i + di, j + dj), P.βs(sK) / βP / h^2)
                sK == s0 && continue
                c, _ = edge(G, i, j, di, dj)
                addterm!(r, jumpz_ext(jumpsz(c), c, X(i + di, j + dj) - c.x, order), s0 / βP / h^2)
            end
            b0[r] += P.f(x0, s0) / βP
        else
            βP = P.βs(s0)
            for (di, dj) in NBR
                sK = S[i+di, j+dj]
                βK = P.βs(sK)
                if sK == s0
                    βf = βP
                elseif method == :standard
                    βf = 2βP * βK / (βP + βK)
                else
                    c, θ = edge(G, i, j, di, dj)
                    βf = 1 / (θ / βP + (1 - θ) / βK)
                    push!(gfm_faces[r], (s0, c, θ, SVector(Float64(di), Float64(dj)), βf, βK))
                end
                addterm!(r, node(i + di, j + dj), βf / h^2)
                addterm!(r, node(i, j), -βf / h^2)
            end
            b0[r] += P.f(x0, s0)
        end
    end
    A = sparse(Is, Js, Vs, nI, nI)
    full(u) = (U = copy(Ub); for j in 2:N, i in 2:N; U[LI[i, j]] = u[id(i, j)]; end; U)
    stats = (nnz=nnz(A), maxrow=maximum(diff(sparse(A').colptr)), asym=norm(A - A', 1) / norm(A, 1))
    if method in (:IIM1, :IIM2, :IIMz1, :IIMz2, :IIMh1, :IIMh2)
        u = A \ b0
        return (U=full(u), A=A, stats=stats, it=0, jumps=jumps, jumpsz=jumpsz, order=order, useu=useu)
    end
    # GFM / xGFM: F_P = β̂ [(u_K - u_P - â)/h - b̂(1-θ)/β_K], â = -s0 a, b̂ = -s0 J·e
    F = cholesky(Symmetric(-A))
    fl = Dict{Any,Any}()                              # cached linear fits for ∂ₜu (xGFM)
    function Jvec(s0, c, U)
        J = P.b(c.x) * c.n
        if method == :xGFM && U !== nothing
            fm, fp = get!(fl, c.x) do
                L = h / max(abs(c.n[1]), abs(c.n[2]))
                (fit(G, c.x, c.x - L * c.n, -1, 1), fit(G, c.x, c.x + L * c.n, 1, 1))
            end
            as = P.as(c.x)
            gm, gp = evalLF(dirform(fm.g, c.t), U), evalLF(dirform(fp.g, c.t), U)
            # ∂ₜu⁻ from both sides (∂ₜu⁺ = ∂ₜu⁻ + ∂ₛa), weighted by β (:stiff), 1/β (:soft) or equally
            wm, wp = tw == :stiff ? (P.β.neg, P.β.pos) : tw == :soft ? (P.β.pos, P.β.neg) : (1.0, 1.0)
            ĝ  = (wm * gm + wp * (gp - as)) / (wm + wp)
            J += ((P.β.pos - P.β.neg) * ĝ + P.β.pos * as) * c.t
        end
        return J
    end
    function rhs(U)
        b = copy(b0)
        method == :standard && return b
        for r in 1:nI, (s0, c, θ, e, βf, βK) in gfm_faces[r]
            â = -s0 * P.a(c.x)
            b̂ = -s0 * dot(Jvec(s0, c, U), e)
            # F_P contains -β̂ â/h - β̂ b̂ (1-θ)/β_K: move to the right-hand side (row: ΣF/h = f)
            b[r] -= (-βf * â / h - βf * b̂ * (1 - θ) / βK) / h
        end
        return b
    end
    u  = F \ (-rhs(nothing))
    it = 0
    if method == :xGFM
        for k in 1:maxit
            un = F \ (-rhs(full(u)))
            d  = norm(un - u, Inf) / max(norm(un, Inf), 1e-300)
            u, it = un, k
            d < tol && break
        end
    end
    return (U=full(u), A=A, stats=stats, it=it, Jvec=Jvec)
end

# ---------------------------------------------------------------- errors
function errors(G, sol, method)
    (; P, N, h, X, S, LI) = G
    U = sol.U
    # errors normalised separately on each side by that side's maximum (the stiff side's fluxes
    # are larger by the contrast and would hide the soft side's errors); the worse side is reported
    k(s) = s < 0 ? 1 : 2
    eu, umax = zeros(2), zeros(2)
    for j in 2:N, i in 2:N
        s, ue = S[i, j], P.u(X(i, j), S[i, j])
        eu[k(s)]   = max(eu[k(s)], abs(U[LI[i, j]] - ue))
        umax[k(s)] = max(umax[k(s)], abs(ue))
    end
    # face fluxes β∂ₑu at face midpoints, from the side of the midpoint
    ef, fmax = zeros(2), zeros(2)
    for j in 1:N+1, i in 1:N+1, (di, dj) in ((1, 0), (0, 1))
        (i + di <= N + 1 && j + dj <= N + 1) || continue
        e  = SVector(Float64(di), Float64(dj))
        xm = X(i, j) + e * h / 2
        sm = side(P.φ(xm))
        ex = P.βs(sm) * dot(P.∇u(xm, sm), e)
        uP, uK = U[LI[i, j]], U[LI[i+di, j+dj]]
        s0, sK = S[i, j], S[i+di, j+dj]
        num = if s0 == sK
            P.βs(s0) * (uK - uP) / h
        elseif method == :standard
            2P.β.neg * P.β.pos / (P.β.neg + P.β.pos) * (uK - uP) / h
        elseif method in (:GFM, :xGFM)
            c, θ = edge(G, i, j, di, dj)
            βP, βK = P.βs(s0), P.βs(sK)
            βf = 1 / (θ / βP + (1 - θ) / βK)
            â, b̂ = -s0 * P.a(c.x), -s0 * dot(sol.Jvec(s0, c, method == :xGFM ? U : nothing), e)
            FP = βf * ((uK - uP - â) / h - b̂ * (1 - θ) / βK)
            sm == s0 ? FP : FP + b̂
        elseif !sol.useu(sm)
            c, _ = edge(G, i, j, di, dj)
            J  = sol.jumpsz(c)
            wext(k, x, sk) = P.βs(sk) * U[k] + (sk == sm ? 0.0 : sm * evalLF(jumpz_ext(J, c, x - c.x, sol.order), U))
            (wext(LI[i+di, j+dj], X(i + di, j + dj), sK) - wext(LI[i, j], X(i, j), s0)) / h
        else
            c, _ = edge(G, i, j, di, dj)
            J  = sol.jumps(c)
            # extend the node on the other side than sm to side sm
            ext(k, x, sk) = sk == sm ? U[k] : U[k] + sm * evalLF(jump_ext(J, c, x - c.x, sol.order), U)
            P.βs(sm) * (ext(LI[i+di, j+dj], X(i + di, j + dj), sK) - ext(LI[i, j], X(i, j), s0)) / h
        end
        ef[k(sm)]   = max(ef[k(sm)], abs(num - ex))
        fmax[k(sm)] = max(fmax[k(sm)], abs(ex))
    end
    return (u=maximum(eu ./ umax), flux=maximum(ef ./ fmax))
end

# ---------------------------------------------------------------- test problems
# T1: circular inclusion in a uniform field (Poisson analogue of Schmid & Podladchikov 2003):
#     a = b = f = 0, ∇u⁻ uniform, tangential flux jump [β]∂ₜu ≠ 0
function T1(βm, βp; R=0.5, c=SVector(0.0123, -0.0217))
    A = 2βp / (βp + βm)
    C = (βp - βm) / (βp + βm)
    return (name="T1 circle, uniform field", φ=x -> norm(x - c) - R,
            um=x -> A * (x[1] - c[1]), up=x -> (x[1] - c[1]) * (1 + C * R^2 / sum(abs2, x - c)), β=(neg=βm, pos=βp))
end
# T2: flower-shaped interface, non-zero jumps a, b and source f
function T2(βm, βp; c=SVector(0.0123, -0.0217))
    φ(x) = (d = x - c; norm(d) - (0.5 + 0.08 * sin(5 * atan(d[2], d[1]))))
    return (name="T2 flower, a, b, f ≠ 0", φ=φ,
            um=x -> sin(2x[1]) * cos(3x[2]) + 1, up=x -> exp(0.5x[1]) * sin(2x[2]) + x[1] * x[2], β=(neg=βm, pos=βp))
end
