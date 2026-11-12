# xGFM-type discretisation of Stokes flow with piecewise constant viscosity η (and bulk viscosity
# ξ, compressible case) on the staggered grid (interface not fitted), solved by GCR
# preconditioned with the Cholesky factor of its symmetric part.
#
# Constitutive law on side s: σ = -p I + 2η(ε̇ - ⅓(∇·v) I), p = -ξ ∇·v (ξ = ∞: incompressible).
# Jump conditions: [v] = 0 and [σ]n = 0. In the (t, n) frame of Γ, ∂ₜvₜ and ∂ₜvₙ are continuous,
# ∂ₙvₜ and ∂ₙvₙ jump. With the continuous normal and shear tractions N, S and the side moduli
# M = ξ + 4η/3, λ = ξ - 2η/3 (M = λ = ∞ if incompressible):
#   ∂ₙvₙ = (N - λ ∂ₜvₜ)/M,   ∂ₙvₜ + ∂ₜvₙ = S/η,
#   σ = N n⊗n + S (n⊗t + t⊗n) + (a N + b ∂ₜvₜ) t⊗t,  a = λ/M = 1 - 2η/M,  b = M - λ²/M = 4η(1 - η/M),
#   N = -(M/ξ) p - 2η ∂ₜvₜ.
# Incompressible: a = 1, b = 4η, M/ξ = 1, 1/M = 0. Only ∂ₜv (continuous, traced from one-sided
# fits, Landry's η-weighted average) is needed besides N and S. The two "fluxes" are closed
# locally, as in the ghost fluid method:
#   - N from the pressure of the centroid (on the centroid's side), or at a vertex (compressible)
#     the mean of the N of the surrounding centroids;
#   - S from the two velocity nodes of a grid segment crossed by Γ, with v piecewise linear along
#     it (∂ₖvᵢ in terms of ∂ₜv, S/η_s and ∂ₙvₙ,s), averaged over the segment:
#     β̂ = 1/(ℓ₋/η₋ + ℓ₊/η₊) (series viscosity), 1/M̂ = ℓ₋/M₋ + ℓ₊/M₊, ā = ℓ₋a₋ + ℓ₊a₊.
# Stresses are evaluated once per centroid (σxx from the Vx pair, σyy from the Vy pair) and per
# vertex (σxy from the Vx or the Vy pair, whichever is closer to the normal, so that the factor
# G/(tₐn_b) stays in [0, 1]); the two rows sharing them see σ_s of their own side (discrete
# traction balance). Continuity at cut centroids: β̂x(∂x vx - Ax) + β̂y(∂y vy - Ay) = 0 (common S
# of the two segments). Exact for piecewise linear solutions.
# Without the ∂ₜv and the compressible N terms the operator is the symmetric staggered Stokes
# operator with series viscosities at cut centroids and vertices: it is the preconditioner.
using StagFDTools, StagFDTools.StokesIIM, LinearAlgebra, SparseArrays, StaticArrays, Printf
import StagFDTools.Stokes: Fields, Numbering!, set_boundaries_template!
import StagFDTools.StokesIIM: node_index, fine_index, node_position, side, tangent
import Statistics: mean

# ---------------------------------------------------------------- linear forms over node values
const Key = Tuple{Symbol,CartesianIndex{2}}
mutable struct Form
    w::Dict{Key,Float64}
    c::Float64
end
Form(c::Real=0.0) = Form(Dict{Key,Float64}(), Float64(c))
unit(f, I) = Form(Dict{Key,Float64}((f, I) => 1.0), 0.0)
function lc(terms...; c=0.0)
    F = Form(c)
    for (α, G) in terms
        α == 0 && continue
        for (k, v) in G.w
            F.w[k] = get(F.w, k, 0.0) + α * v
        end
        F.c += α * G.c
    end
    return F
end

# ---------------------------------------------------------------- context
# material constants of a side (see header)
function side_material(η, ξ)
    ξ == Inf && return (; η, iM=0.0, a=1.0, b=4η, Mξ=1.0, iξ=0.0)
    M = ξ + 4η / 3
    return (; η, iM=1 / M, a=1 - 2η / M, b=4η * (1 - η / M), Mξ=M / ξ, iξ=1 / ξ)
end

# void = :neg or :pos: that side is empty (free surface, η = 0 there, no unknowns; see AssembleXGFM)
function xgfm_context(φ, X, Δ, η, nc; ξ=nothing, nump=nothing, ρ=nothing, g=SVector(0.0, 0.0), void=nothing, void_any=true)
    Γ  = InterfaceCrossings(FineLattice(φ, X, Δ))
    ηtr = void == :neg ? (neg=1e-12 * η.pos, pos=η.pos) : void == :pos ? (neg=η.neg, pos=1e-12 * η.neg) : η
    tr = InterfaceTraces(Γ, φ, X, Δ, ηtr, nc; fit=:linear)
    lookup = Dict((T.p.I, T.p.J, T.p.dir == :x ? 1 : 2) => c for (c, T) in enumerate(tr))
    comp = !isnothing(ξ)
    η    = void == :neg ? (neg=0.0, pos=η.pos) : void == :pos ? (neg=η.neg, pos=0.0) : η
    mats = (neg=side_material(η.neg, comp ? ξ.neg : Inf), pos=side_material(η.pos, comp ? ξ.pos : Inf))
    mat(s) = s < 0 ? mats.neg : mats.pos
    ηof(s) = mat(s).η
    # ∂ₜv at crossing c as linear forms of the velocities (traces act on z = ηv, η of each node's label)
    dtv = map(tr) do T
        fx = Form(); fy = Form()
        for (k, I) in enumerate(T.x.idx)
            fx.w[(:x, I)] = get(fx.w, (:x, I), 0.0) + T.x.t[k] * ηof(T.x.s[k])
        end
        for (k, I) in enumerate(T.y.idx)
            fy.w[(:y, I)] = get(fy.w, (:y, I), 0.0) + T.y.t[k] * ηof(T.y.s[k])
        end
        (x=fx, y=fy)
    end
    φf = (x=φ.x, y=φ.y, p=φ.c, v=φ.v)
    ρof(s) = isnothing(ρ) || isvoid(s) ? 0.0 : s < 0 ? ρ.neg : ρ.pos
    isvoid(s) = (void == :neg && s < 0) || (void == :pos && s > 0)
    return (; X, Δ, η, ηof, mat, comp, tr, lookup, dtv, φf, Γ, nump, ρof, g, isvoid, void_any)
end

sideof(ctx, f, I) = side(ctx.φf[f][I])
nearest_crossing(ctx, x) = argmin(c -> norm(ctx.tr[c].p.x - x), eachindex(ctx.tr))

# crossing on the straight fine-lattice segment between two nodes; if there is none with a trace
# (Γ leaves the domain there), the traced crossing nearest to xm
function segment_crossing(ctx, F0, F1, xm)
    c = segment_crossing(ctx, F0, F1)
    return c > 0 ? c : nearest_crossing(ctx, xm)
end
function segment_crossing(ctx, F0, F1)
    if F0[2] == F1[2]
        for I in min(F0[1], F1[1]):(max(F0[1], F1[1])-1)
            c = get(ctx.lookup, (I, F0[2], 1), 0)
            c > 0 && return c
        end
    else
        for J in min(F0[2], F1[2]):(max(F0[2], F1[2])-1)
            c = get(ctx.lookup, (F0[1], J, 2), 0)
            c > 0 && return c
        end
    end
    return 0
end

# geometry and trace forms at crossing c
function crossing_data(ctx, c)
    p = ctx.tr[c].p
    n, t = p.n, tangent(p.n)
    dx, dy = ctx.dtv[c].x, ctx.dtv[c].y
    dtvt = lc((t[1], dx), (t[2], dy))
    dtvn = lc((n[1], dx), (n[2], dy))
    return (; x=p.x, n, t, dtvt, dtvn)
end

# segment a → b: fraction ℓ on a's side (where the tangent line of Γ at crossing c cuts it; 1 if
# both nodes are on the same side), series viscosity β̂, mean compliance 1/M̂ and mean a = λ/M
function segment_avg(ctx, xa, xb, sa, sb, c)
    ma, mb = ctx.mat(sa), ctx.mat(sb)
    ℓ  = sa == sb || c == 0 ? 1.0 : (p = ctx.tr[c].p; clamp(dot(p.x - xa, p.n) / dot(xb - xa, p.n), 0.0, 1.0))
    cpl = (ℓ > 0 ? ℓ / ma.η : 0.0) + (ℓ < 1 ? (1 - ℓ) / mb.η : 0.0)     # η = 0 (void): β̂ = 0
    return (β=1 / cpl, iM=ℓ * ma.iM + (1 - ℓ) * mb.iM, a=ℓ * ma.a + (1 - ℓ) * mb.a)
end

# normal traction N at a centroid, from its pressure: N = -(M/ξ) p - 2η ∂ₜvₜ (side of the centroid)
# (void centroid: N = 0, the traction of the free surface)
N_centroid(ctx, Ic, sc, dtvt) = ctx.isvoid(sc) ? Form() : (m = ctx.mat(sc); lc((-m.Mξ, unit(:p, Ic)), (-2m.η, dtvt)))

# N at a vertex: mean over the surrounding centroids with a pressure unknown (compressible only)
function N_vertex(ctx, xv, dtvt)
    (; X, Δ) = ctx
    terms = []
    for sx in (-1, 1), sy in (-1, 1)
        Ic = node_index(:p, xv + SVector(sx * Δ.x / 2, sy * Δ.y / 2), X, Δ)
        checkbounds(Bool, ctx.nump, Ic) && ctx.nump[Ic] > 0 || continue
        push!(terms, (1.0, N_centroid(ctx, Ic, sideof(ctx, :p, Ic), dtvt)))
    end
    return lc(((α / length(terms), F) for (α, F) in terms)...)
end

# ---------------------------------------------------------------- closures (full, preconditioner)
# the two face pairs of centroid xc: (Vx pair, Vy pair) with nodes, sides, differences
function centroid_pairs(ctx, xc)
    (; X, Δ) = ctx
    map(((:x, SVector(Δ.x / 2, 0.0), Δ.x), (:y, SVector(0.0, Δ.y / 2), Δ.y))) do (f, e, h)
        Ia, Ib = node_index(f, xc - e, X, Δ), node_index(f, xc + e, X, Δ)
        (; f, e, Ia, Ib, sa=sideof(ctx, f, Ia), sb=sideof(ctx, f, Ib), Δv=lc((1 / h, unit(f, Ib)), (-1 / h, unit(f, Ia))))
    end
end
pair_crossing(ctx, P, xc) = P.sa != P.sb ? segment_crossing(ctx, fine_index(Val(P.f), P.Ia), fine_index(Val(P.f), P.Ib), xc) : 0
pair_avg(ctx, P, xc, c) = segment_avg(ctx, xc - P.e, xc + P.e, P.sa, P.sb, c)

# σxx (dir = 1, Vx pair) or σyy (dir = 2, Vy pair) at centroid xc, seen from side srow
function sigma_nn(ctx, xc, srow, dir)
    Ic = node_index(:p, xc, ctx.X, ctx.Δ)
    sc = sideof(ctx, :p, Ic)
    Ps = centroid_pairs(ctx, xc)
    P, Q = Ps[dir], Ps[3-dir]
    D  = lc((1.0, P.Δv), (1.0, Q.Δv))
    pc = ctx.isvoid(sc) ? Form() : unit(:p, Ic)
    if P.sa == P.sb == sc && (!ctx.comp || Q.sa == Q.sb == sc)
        ηc = ctx.ηof(sc)
        F  = ctx.comp ? lc((-1.0, pc), (2ηc, P.Δv), (-2ηc / 3, D)) : lc((-1.0, pc), (2ηc, P.Δv))
        return F, F
    end
    cP = pair_crossing(ctx, P, xc)
    c  = cP > 0 ? cP : nearest_crossing(ctx, xc)
    g  = crossing_data(ctx, c)
    s  = pair_avg(ctx, P, xc, cP)
    mr = ctx.mat(srow)
    td, nd = g.t[dir], g.n[dir]
    N  = N_centroid(ctx, Ic, sc, g.dtvt)
    cN = nd^2 + td^2 * mr.a - 2s.β * nd^2 * s.iM
    # gravity: ∂ₙN = -ρ g·n differs on the two sides; N at the centroid on the row's side is
    # N_c + (ρ_c - ρ_r) g·n d_c (d_c: signed distance of the centroid to Γ). Exact hydrostatics.
    cg = (ctx.ρof(sc) - ctx.ρof(srow)) * dot(ctx.g, g.n) * dot(g.n, xc - g.x)
    F  = lc((cN, N), (2s.β, P.Δv), (td^2 * mr.b - 2s.β * td^2 + 2s.β * nd^2 * s.a, g.dtvt); c=cN * cg)
    if ctx.comp      # deviatoric term with a common viscosity ≤ min(β̂x, β̂y): symmetric and PSD
        η̄  = min(s.β, pair_avg(ctx, Q, xc, pair_crossing(ctx, Q, xc)).β)
        PC = lc((-1.0, pc), (2s.β, P.Δv), (-2η̄ / 3, D))
    else
        PC = lc((-1.0, pc), (2s.β, P.Δv))
    end
    return F, PC
end

# σxy at vertex xv, seen from side srow
function sigma_xy(ctx, xv, srow)
    (; X, Δ) = ctx
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    Ia, Ib = node_index(:x, xv - ey, X, Δ), node_index(:x, xv + ey, X, Δ)      # Vx pair (y segment)
    Ic, Id = node_index(:y, xv - ex, X, Δ), node_index(:y, xv + ex, X, Δ)      # Vy pair (x segment)
    sa, sb, sc, sd = sideof(ctx, :x, Ia), sideof(ctx, :x, Ib), sideof(ctx, :y, Ic), sideof(ctx, :y, Id)
    Δvx = lc((1 / Δ.y, unit(:x, Ib)), (-1 / Δ.y, unit(:x, Ia)))
    Δvy = lc((1 / Δ.x, unit(:y, Id)), (-1 / Δ.x, unit(:y, Ic)))
    if sa == sb == sc == sd
        F = lc((ctx.ηof(sa), Δvx), (ctx.ηof(sa), Δvy))
        return F, F
    end
    cy = sa != sb ? segment_crossing(ctx, fine_index(Val(:x), Ia), fine_index(Val(:x), Ib), xv) : 0
    cx = sc != sd ? segment_crossing(ctx, fine_index(Val(:y), Ic), fine_index(Val(:y), Id), xv) : 0
    g0 = crossing_data(ctx, cy > 0 ? cy : cx > 0 ? cx : nearest_crossing(ctx, xv))
    usey = g0.n[2]^2 >= g0.n[1]^2                     # segment closer to the normal
    g  = usey ? g0 : (cx > 0 ? crossing_data(ctx, cx) : g0)
    n, t = g.n, g.t
    G  = n[1] * t[2] + n[2] * t[1]
    if usey
        s = segment_avg(ctx, xv - ey, xv + ey, sa, sb, cy)
        K = lc((t[1] * t[2], g.dtvt), (n[1] * t[2] - t[1] * n[2], g.dtvn))
        a, Δv = t[1] * n[2], Δvx
    else
        s = segment_avg(ctx, xv - ex, xv + ex, sc, sd, cx)
        K = lc((t[1] * t[2], g.dtvt), (n[2] * t[1] - t[2] * n[1], g.dtvn))
        a, Δv = t[2] * n[1], Δvy
    end
    # S = (β̂/a)(Δv - K - nₓn_y(N/M̂ - ā ∂ₜvₜ)),  σxy = nₓn_y(N(1 - a_r) - b_r ∂ₜvₜ) + G S
    mr = ctx.mat(srow)
    k  = G / a * s.β
    nn = n[1] * n[2]
    F  = lc((k, Δv), (-k, K), (-nn * mr.b + k * nn * s.a, g.dtvt))
    cN = nn * (1 - mr.a) - k * nn * s.iM
    ctx.comp && cN != 0 && (F = lc((1.0, F), (cN, N_vertex(ctx, xv, g.dtvt))))
    PC = lc((s.β, Δvx), (s.β, Δvy))
    return F, PC
end

# ∇·v (+ p/ξ) at centroid xc
function continuity(ctx, xc)
    Ic = node_index(:p, xc, ctx.X, ctx.Δ)
    sc = sideof(ctx, :p, Ic)
    Px, Py = centroid_pairs(ctx, xc)
    PC = lc((1.0, Px.Δv), (1.0, Py.Δv), (ctx.mat(sc).iξ, unit(:p, Ic)))
    Px.sa == Px.sb == Py.sa == Py.sb == sc && return PC, PC
    cx, cy = pair_crossing(ctx, Px, xc), pair_crossing(ctx, Py, xc)
    g  = crossing_data(ctx, cx > 0 ? cx : cy > 0 ? cy : nearest_crossing(ctx, xc))
    sx, sy = pair_avg(ctx, Px, xc, cx), pair_avg(ctx, Py, xc, cy)
    N  = N_centroid(ctx, Ic, sc, g.dtvt)
    # A_d = t_d² ∂ₜvₜ + n_d² (N/M̂_d - ā_d ∂ₜvₜ)
    A(s, d) = lc((g.t[d]^2 - g.n[d]^2 * s.a, g.dtvt), (g.n[d]^2 * s.iM, N))
    if ctx.void_any && (ctx.isvoid(Px.sa) || ctx.isvoid(Px.sb) || ctx.isvoid(Py.sa) || ctx.isvoid(Py.sb)) || sx.β + sy.β == 0
        # free surface, cut cell: N_Γ = 0 extended to the centroid, N_c + ρ g·n d_c = 0 (exact
        # hydrostatics), written as p_c + ... = 0. With one cut pair only, the continuity row would
        # not involve p_c and nothing would fix the pressure of columns (rows) where Γ runs just
        # below a row of faces (void_any = true, default).
        F = lc((-1.0, N); c=-ctx.ρof(sc) * dot(ctx.g, g.n) * dot(g.n, xc - g.x))
        return F, unit(:p, Ic)
    end
    w  = 2 / (sx.β + sy.β)
    F  = lc((w * sx.β, Px.Δv), (w * sy.β, Py.Δv), (-w * sx.β, A(sx, 1)), (-w * sy.β, A(sy, 2)))
    return F, PC
end

# ---------------------------------------------------------------- assembly
"""
    AssembleXGFM(φ, X, Δ, η, nc, vbc; ξ=nothing, ρ=nothing, g=SVector(0.0, 0.0), void=nothing)

Full operator `M` (non-symmetric), its symmetric part `M_PC`, right-hand side `b` (Dirichlet
velocities `vbc` on the full Vx/Vy arrays), numbering. Unknowns [Vx; Vy; Pt]; momentum rows are
-(∇·σ) (positive definite viscous block), continuity rows +∇·v + p/ξ (= -Gᵀ + C for the
symmetric part). `ξ = (neg, pos)`: bulk viscosities (compressible), `nothing`: incompressible.
`ρ = (neg, pos)`, `g`: densities and gravity (body force ρg of each velocity node's side).
`void = :neg` or `:pos`: free surface, that side is empty (η = 0 limit of the closures, traction
N = S = 0 on Γ): no unknowns there, the series viscosity of the cut segments is zero so that no
void velocity enters the fluid rows, void stress points carry the traction-free stress, and the
continuity row of a centroid with both face pairs cut becomes N_Γ = 0.
"""
function AssembleXGFM(φ, X, Δ, η, nc, vbc; ξ=nothing, ρ=nothing, g=SVector(0.0, 0.0), void=nothing, void_any=true)
    size_x, size_y, size_c = size(φ.x), size(φ.y), size(φ.c)
    type   = Fields(fill(:out, size_x...), fill(:out, size_y...), fill(:out, size_c...))
    number = Fields(zeros(Int, size_x...), zeros(Int, size_y...), zeros(Int, size_c...))
    set_boundaries_template!(type, :all_Dirichlet, nc)
    Numbering!(number, type, nc)
    num  = (x=number.Vx, y=number.Vy, p=number.Pt)
    if !isnothing(void)                  # remove the void nodes and renumber
        vs = void == :neg ? -1 : 1
        for (f, φf) in ((:x, φ.x), (:y, φ.y), (:p, φ.c))
            k = 0
            for I in CartesianIndices(num[f])
                num[f][I] = num[f][I] > 0 && side(φf[I]) != vs ? (k += 1) : 0
            end
        end
    end
    ctx  = xgfm_context(φ, X, Δ, η, nc; ξ, nump=num.p, ρ, g, void, void_any)
    off  = (x=0, y=maximum(num.x), p=maximum(num.x) + maximum(num.y))
    ndof = off.p + maximum(num.p)
    val  = (x=vbc.x, y=vbc.y)
    I1, J1, V1, I2, J2, V2 = Int[], Int[], Float64[], Int[], Int[], Float64[]
    b = zeros(ndof)
    function put!(r, F, PC, α)
        for (Is, Js, Vs, G, rhs) in ((I1, J1, V1, F, true), (I2, J2, V2, PC, false))
            for ((f, I), v) in G.w
                if num[f][I] > 0
                    push!(Is, r); push!(Js, off[f] + num[f][I]); push!(Vs, α * v)
                elseif rhs
                    f == :p && error("pressure node without number")
                    ctx.isvoid(sideof(ctx, f, I)) && abs(v) > 1e-10 && error("void node $f $I in a fluid row ($v)")
                    b[r] -= α * v * val[f][I]
                end
            end
            rhs && (b[r] -= α * G.c)
        end
    end
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    for I in CartesianIndices(num.x)
        (r = num.x[I]) > 0 || continue
        x0, s0 = node_position(Val(:x), I, X), sideof(ctx, :x, I)
        for (xs, sg, h, fun) in ((x0 + ex, -1, Δ.x, x -> sigma_nn(ctx, x, s0, 1)), (x0 - ex, 1, Δ.x, x -> sigma_nn(ctx, x, s0, 1)),
                                 (x0 + ey, -1, Δ.y, x -> sigma_xy(ctx, x, s0)), (x0 - ey, 1, Δ.y, x -> sigma_xy(ctx, x, s0)))
            F, PC = fun(xs)
            put!(off.x + r, F, PC, sg / h)
        end
        b[off.x + r] += ctx.ρof(s0) * g[1]
    end
    for I in CartesianIndices(num.y)
        (r = num.y[I]) > 0 || continue
        x0, s0 = node_position(Val(:y), I, X), sideof(ctx, :y, I)
        for (xs, sg, h, fun) in ((x0 + ex, -1, Δ.x, x -> sigma_xy(ctx, x, s0)), (x0 - ex, 1, Δ.x, x -> sigma_xy(ctx, x, s0)),
                                 (x0 + ey, -1, Δ.y, x -> sigma_nn(ctx, x, s0, 2)), (x0 - ey, 1, Δ.y, x -> sigma_nn(ctx, x, s0, 2)))
            F, PC = fun(xs)
            put!(off.y + r, F, PC, sg / h)
        end
        b[off.y + r] += ctx.ρof(s0) * g[2]
    end
    for I in CartesianIndices(num.p)
        (r = num.p[I]) > 0 || continue
        F, PC = continuity(ctx, node_position(Val(:p), I, X))
        put!(off.p + r, F, PC, 1.0)
    end
    M    = sparse(I1, J1, V1, ndof, ndof)
    M_PC = sparse(I2, J2, V2, ndof, ndof)
    return (; M, M_PC, b, num, off, ndof, ctx)
end

# ---------------------------------------------------------------- solvers
"""
    gcr_xgfm(M, M_PC, b, nu; ηb, ϵ=1e-10, restart=30, maxit=500, lagrange=true, wl=nothing)

GCR on M x = b, preconditioned by one Powell-Hestenes step with the Cholesky factor of
Kuu_PC + G Pb Gᵀ, Pb = (C + 1/ηb)⁻¹ (C: pressure block of `M_PC`, i.e. 1/ξ; Pb = ηb if
incompressible). `lagrange`: fix the pressure level with a multiplier (Σ wl p = 0, incompressible;
`wl`: weights of the pressures, all ones by default).
Returns x and the number of iterations.
"""
function gcr_xgfm(M, M_PC, b, nu; ηb, ϵ=1e-10, restart=30, maxit=500, lagrange=true, wl=nothing)
    # Incompressible: the weighted continuity rows at cut cells change the left null vector, so
    # that the Dirichlet data are not exactly compatible and the singular system would stagnate.
    N   = size(M, 1)
    iu, ip = 1:nu, nu+1:N
    if lagrange
        e  = sparse(collect(ip), ones(Int, length(ip)), isnothing(wl) ? ones(length(ip)) : wl, N, 1)
        Ma = [M e; e' 0]
    else
        Ma = M
    end
    Na  = size(Ma, 1)
    Kuu, Kup, Kpu = M_PC[iu, iu], M_PC[iu, ip], M_PC[ip, iu]
    Pb  = 1 ./ (diag(M_PC[ip, ip]) .+ 1 / ηb)
    Kf  = cholesky(Hermitian(Kuu - Kup * Diagonal(Pb) * Kpu))
    function pc(f)
        su = Kf \ (f[iu] .- Kup * (Pb .* f[ip]))
        sp = Pb .* (f[ip] .- Kpu * su)
        return lagrange ? vcat(su, sp, f[end]) : vcat(su, sp)
    end
    x   = zeros(Na)
    r   = lagrange ? [b; 0.0] : copy(b)
    nb  = norm(b)
    S, V = zeros(Na, restart), zeros(Na, restart)
    its = 0
    while its < maxit
        for k in 1:restart
            its += 1
            s = pc(r)
            v = Ma * s
            for j in 1:k-1
                h = dot(v, V[:, j])
                v .-= h .* V[:, j]
                s .-= h .* S[:, j]
            end
            nv = norm(v)
            v ./= nv; s ./= nv
            α = dot(r, v)
            x .+= α .* s
            r .-= α .* v
            V[:, k] .= v; S[:, k] .= s
            norm(r) / nb < ϵ && return x[1:N], its
            its >= maxit && break
        end
    end
    @warn "GCR not converged: $(norm(r) / nb)"
    return x[1:N], its
end

"""
    SolveXGFM(φ, X, Δ, η, nc, vbc; ξ=nothing, ρ=nothing, g, void=nothing, solver=:gcr, ηb=1e3, ϵ=1e-10, pref=:all, restart=50, maxit=1000)

Solve the xGFM system (`:gcr` or `:lu`; incompressible: Lagrange multiplier for the pressure
level, mean over all centroids (`pref = :all`) or over one side (`:neg`, `:pos`, e.g. zero mean
pressure in sticky air)). Returns v (full arrays), p (on c_e), the assembly and the GCR
iteration count.
"""
function SolveXGFM(φ, X, Δ, η, nc, vbc; ξ=nothing, ρ=nothing, g=SVector(0.0, 0.0), void=nothing, solver=:gcr, ηb=1e3, ϵ=1e-10,
                   pref=:all, restart=50, maxit=1000)
    A = AssembleXGFM(φ, X, Δ, η, nc, vbc; ξ, ρ, g, void)
    its = 0
    lagrange = isnothing(ξ) && isnothing(void)
    # symmetric diagonal scaling: velocities by 1/√η, pressures by √η (η of each node's side), so
    # that all blocks are O(1) whatever the contrast (keeps the preconditioner symmetric)
    sc = ones(A.ndof)
    for (f, φf) in ((:x, φ.x), (:y, φ.y), (:p, φ.c)), I in CartesianIndices(φf)
        A.num[f][I] > 0 || continue
        ηs = side(φf[I]) < 0 ? η.neg : η.pos
        sc[A.off[f] + A.num[f][I]] = f == :p ? sqrt(ηs) : 1 / sqrt(ηs)
    end
    Sd = Diagonal(sc)
    wl = zeros(A.ndof - A.off.p)
    for I in CartesianIndices(φ.c)
        A.num.p[I] > 0 || continue
        wl[A.num.p[I]] = pref == :all || (pref == :neg) == (side(φ.c[I]) < 0) ? 1.0 : 0.0
    end
    if solver == :lu
        if lagrange
            M   = [A.M sparse([zeros(A.off.p); wl]); sparse([zeros(A.off.p); wl]') 0]
            sol = (M \ [A.b; 0.0])[1:A.ndof]
        else
            sol = A.M \ A.b
        end
    else
        y, its = gcr_xgfm(Sd * A.M * Sd, Sd * A.M_PC * Sd, sc .* A.b, A.off.p; ηb, ϵ, lagrange, maxit, restart,
                          wl=wl .* sc[A.off.p+1:end])
        sol = sc .* y
    end
    v = (x=copy(vbc.x), y=copy(vbc.y))
    p = zeros(size(φ.c))
    for (f, a) in ((:x, v.x), (:y, v.y), (:p, p)), I in CartesianIndices(a)
        A.num[f][I] > 0 && (a[I] = sol[A.off[f] + A.num[f][I]])
    end
    return (; v, p, A, its)
end
