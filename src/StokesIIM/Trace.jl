# Interface trace: v_Γ, ∂ₛv_Γ and ∂ₛₛv_Γ at a crossing of Γ as linear combinations of z = ηv
# (s: arc length along Γ, ∂ₛ = t·∇ at x_Γ).
#
# On each side s = ±1 of Γ, nodes of one velocity grid close to x_Γ + s L n (L = one cell along
# n, as in Landry's compute_dv_dtt) define a fit of z:
#   :linear     3 nodes, linear interpolant (Landry, first order)
#   :quadratic  ~10 nodes, quadratic least squares
#   :cubic      ~24 nodes, cubic least squares (one order more accurate derivatives)
# Its value and derivatives at x_Γ give z_s, ∂ₛz_s = t·∇z_s and ∂ₛₛz_s = tᵀ(∇∇z_s)t - κ n·∇z_s
# (second derivative along the curved Γ, κ = ∇·n). Since η is constant on each side, Landry's
# η-weighted average of v_s = z_s/η_s reduces to
#   v_Γ = (z₋ + z₊)/(η₋ + η₊),   ∂ₛv_Γ = (∂ₛz₋ + ∂ₛz₊)/(η₋ + η₊),   ∂ₛₛv_Γ = (∂ₛₛz₋ + ∂ₛₛz₊)/(η₋ + η₊)
# which is exact when z₋ = η₋v, z₊ = η₊v with v continuous across Γ.

tangent(n) = SVector(-n[2], n[1])

# Candidate nodes of side s around xo, sorted by distance (in cell units)
function side_candidates(xo, s, φc, Xc, ranges, Δ, nmax)
    ic = round(Int, (xo[1] - Xc.x[1]) / Δ.x) + 1
    jc = round(Int, (xo[2] - Xc.y[1]) / Δ.y) + 1
    cand = CartesianIndex{2}[]
    for j in max(jc - nmax, first(ranges[2])):min(jc + nmax, last(ranges[2])),
        i in max(ic - nmax, first(ranges[1])):min(ic + nmax, last(ranges[1]))
        side(φc[i, j]) == s && push!(cand, CartesianIndex(i, j))
    end
    xk(I) = SVector(Xc.x[I[1]], Xc.y[I[2]])
    sort!(cand, by=I -> norm((xk(I) - xo) ./ SVector(Δ.x, Δ.y)))
    return cand, xk
end

# Experimental switch: the candidates are sorted by distance and equidistant nodes keep the loop
# order (lower j, then lower i first), so cutting the list at a fixed count selects stencils with
# a directional bias (mirror-symmetric problems get non-mirrored stencils). With
# SYMMETRIC_TIES[] = true, every node tied with the last selected one is also included and the
# fit is a least-squares fit: the stencil then depends on distances only.
const SYMMETRIC_TIES = Ref(false)
function with_ties(cand, N, xk, xo, Δ)     # xo: centre of the distance sort
    d(I) = norm((xk(I) - xo) ./ SVector(Δ.x, Δ.y))
    dN = d(cand[N])
    while N < length(cand) && d(cand[N+1]) <= dN + 1e-9
        N += 1
    end
    return N
end

"""
    OneSidedFit(xΓ, xo, s, φc, Xc, ranges, Δ; nmax=4)

Linear fit (Landry): the 3 nodes of side `s` closest to `xo`, not collinear. Returns their
indices, the weights `λ` of the value at `xΓ` and the gradient weights `g` (∇f ≈ g * f).
"""
function OneSidedFit(xΓ, xo, s, φc, Xc, ranges, Δ; nmax=4)
    # Rows [1, dx, dy] in cell units relative to xo; the third node must not be collinear
    for nm in (nmax, nmax + 2, nmax + 4)
        cand, xk = side_candidates(xo, s, φc, Xc, ranges, Δ, nm)
        row(I) = SVector(1.0, (xk(I)[1] - xo[1]) / Δ.x, (xk(I)[2] - xo[2]) / Δ.y)
        e = SVector(1.0, (xΓ[1] - xo[1]) / Δ.x, (xΓ[2] - xo[2]) / Δ.y)
        if length(cand) >= 3
            k3 = findfirst(k -> abs(det(hcat(row(cand[1]), row(cand[2]), row(cand[k])))) > 0.5, 3:length(cand))
            if !isnothing(k3) && SYMMETRIC_TIES[]
                # all nodes up to the distance of the third one (ties included): least squares
                N   = with_ties(cand, k3 + 2, xk, xo, Δ)
                idx = cand[1:N]
                P   = pinv(reduce(vcat, (row(I)' for I in idx)))
                return (idx=idx, λ=P' * e, g=vcat((P[2, :] ./ Δ.x)', (P[3, :] ./ Δ.y)'))
            end
            if !isnothing(k3)
                idx = [cand[1], cand[2], cand[k3+2]]
                iV  = inv(vcat(row(idx[1])', row(idx[2])', row(idx[3])'))   # a = iV * f
                return (idx=idx, λ=iV' * e, g=vcat((iV[2, :] ./ Δ.x)', (iV[3, :] ./ Δ.y)'))
            end
        end
        # Fallback (a sliver of side s where Γ meets the boundary, fewer than 3 non-collinear
        # nodes even in the widest search): minimum-norm least squares on the nodes available
        if nm == nmax + 4 && !isempty(cand)
            idx = cand[1:min(3, length(cand))]
            P   = pinv([row(I)[c] for I in idx, c in 1:3])
            return (idx=idx, λ=P' * e, g=vcat((P[2, :] ./ Δ.x)', (P[3, :] ./ Δ.y)'))
        end
    end
    error("OneSidedFit: no node on side $s near $xΓ")
end

"""
    QuadraticFit(xΓ, xo, s, φc, Xc, ranges, Δ; npts=10, nmax=5, degree=2)

Quadratic least-squares fit of the `npts` nodes of side `s` closest to `xo` (more if the
6×6 normal matrix is ill-conditioned). Basis in cell units centred at `xΓ`:
[1, dx, dy, dx²/2, dy²/2, dx dy]. `degree=3` adds [dx³/6, dx²dy/2, dx dy²/2, dy³/6] (cubic fit,
one order more accurate for the value and derivatives; use npts ≈ 24). Returns indices, value
weights `λ`, gradient weights `g` (2×N) and Hessian weights `H` (H[a, b, k]) at `xΓ`.
"""
function QuadraticFit(xΓ, xo, s, φc, Xc, ranges, Δ; npts=10, nmax=5, degree=2)
    cand, xk = side_candidates(xo, s, φc, Xc, ranges, Δ, nmax)
    row(I) = (dx = (xk(I)[1] - xΓ[1]) / Δ.x; dy = (xk(I)[2] - xΓ[2]) / Δ.y;
              degree == 2 ? SVector(1.0, dx, dy, dx^2 / 2, dy^2 / 2, dx * dy) :
                            SVector(1.0, dx, dy, dx^2 / 2, dy^2 / 2, dx * dy, dx^3 / 6, dx^2 * dy / 2, dx * dy^2 / 2, dy^3 / 6))
    for N in npts:length(cand)
        V = reduce(vcat, (row(I)' for I in cand[1:N]))
        σ = svdvals(V)
        σ[end] > (degree == 2 ? 1e-2 : 1e-3) * σ[1] || continue   # (cubic terms span a wider range)
        if SYMMETRIC_TIES[]
            N = with_ties(cand, N, xk, xo, Δ)
            V = reduce(vcat, (row(I)' for I in cand[1:N]))
        end
        P = pinv(V)                                          # a = P * f
        g = vcat((P[2, :] ./ Δ.x)', (P[3, :] ./ Δ.y)')
        H = zeros(2, 2, N)
        H[1, 1, :] .= P[4, :] ./ Δ.x^2
        H[2, 2, :] .= P[5, :] ./ Δ.y^2
        H[1, 2, :] .= H[2, 1, :] .= P[6, :] ./ (Δ.x * Δ.y)
        return (idx=cand[1:N], λ=P[1, :], g=g, H=H)
    end
    # too few nodes for a cubic (e.g. where Γ meets the boundary): quadratic fit
    degree == 3 && return QuadraticFit(xΓ, xo, s, φc, Xc, ranges, Δ; npts=min(npts, 16), nmax, degree=2)
    # too few nodes for a quadratic (a sliver where Γ meets the boundary): linear fit, H = 0
    f = OneSidedFit(xΓ, xo, s, φc, Xc, ranges, Δ)
    return (idx=f.idx, λ=Vector(f.λ), g=Matrix(f.g), H=zeros(2, 2, length(f.idx)))
end

"""
    InterfaceTrace(p, φc, Xc, ranges, Δ, η; fit=:linear, nmax=4)

Weights of the interface value and of the first and second derivatives along Γ of one velocity
component at the crossing `p` (from `InterfaceCrossings`), acting on z = ηv of that component's
grid: `v_Γ = Σ v[k] z[idx[k]]`, `∂ₛv_Γ = Σ t[k] z[idx[k]]`, `∂ₛₛv_Γ = Σ tt[k] z[idx[k]]`
(`tt` is zero for `fit=:linear`). `η = (neg=η₋, pos=η₊)` are the viscosities on the φ < 0 and
φ > 0 sides. Also returns the fit centres `xo` for plotting.
"""
# Side weights of the trace (experimental switch, default Landry's η weighting):
#   :eta   v_Γ = (η₋v₋ + η₊v₊)/(η₋ + η₊)            (stiff side dominates)
#   :ieta  v_Γ = (v₋/η₋ + v₊/η₊)/(1/η₋ + 1/η₊)      (soft side dominates)
#   :mean  v_Γ = (v₋ + v₊)/2
# The weights act on z_s = η_s v_s, hence the factor 1/η_s.
const TRACE_WEIGHT = Ref(:eta)
function trace_side_weights(η)
    w = TRACE_WEIGHT[] == :eta  ? (η.neg, η.pos) :
        TRACE_WEIGHT[] == :ieta ? (1 / η.neg, 1 / η.pos) : (1.0, 1.0)
    return (w[1] / sum(w) / η.neg, w[2] / sum(w) / η.pos)
end

function InterfaceTrace(p, φc, Xc, ranges, Δ, η; fit=:linear, nmax=4, borrow=false)
    L  = 1 / max(abs(p.n[1]) / Δ.x, abs(p.n[2]) / Δ.y)
    t  = tangent(p.n)
    xo = (p.x - L * p.n, p.x + L * p.n)
    cm, cp = trace_side_weights(η)
    fm, fp = SideFits(p, xo, φc, Xc, ranges, Δ; fit, nmax, borrow)
    idx = vcat(fm.idx, fp.idx)
    v   = vcat(fm.λ .* cm, fp.λ .* cp)
    dt  = vcat((fm.g' * t) .* cm, (fp.g' * t) .* cp)
    dss(f) = [dot(t, f.H[:, :, k] * t) - p.κ * dot(p.n, f.g[:, k]) for k in eachindex(f.idx)]
    dtt = fit == :linear ? zeros(length(idx)) : vcat(dss(fm) .* cm, dss(fp) .* cp)
    # one-sided weights (compressible jumps): side, ∇z_s = Σ g[:, k] z[idx[k]] over side-s members, same for H
    return (idx=idx, v=Vector(v), t=Vector(dt), tt=dtt, xo=xo,
            s=vcat(fill(-1, length(fm.idx)), fill(1, length(fp.idx))), g=hcat(fm.g, fp.g), H=cat(fm.H, fp.H; dims=3))
end

# Linear, quadratic or cubic fits on both sides, with a Hessian (zero for linear fits).
# `borrow`: if no node of side s lies within reach (a sliver between Γ and a wall, thinner than
# the grid spacing), fit the nodes of the other side instead, still labelled s. This is valid for
# continuous fields acted on as v (stress form: z_s = η_s v with η from the label), not for the
# z-form (whose traces act on the nodes' own z), nor exactly for the pressure.
function SideFits(p, xo, φc, Xc, ranges, Δ; fit=:linear, nmax=4, borrow=false)
    function fitside(s, x)
        if fit == :linear
            f = OneSidedFit(p.x, x, s, φc, Xc, ranges, Δ; nmax)
            (idx=collect(f.idx), λ=Vector(f.λ), g=Matrix(f.g), H=zeros(2, 2, length(f.idx)))
        elseif fit == :cubic
            QuadraticFit(p.x, x, s, φc, Xc, ranges, Δ; npts=24, degree=3)
        else
            QuadraticFit(p.x, x, s, φc, Xc, ranges, Δ)
        end
    end
    map((-1, 1), xo) do s, x
        borrow && isempty(side_candidates(x, s, φc, Xc, ranges, Δ, nmax + 4)[1]) && return fitside(-s, x)
        fitside(s, x)
    end
end

"""
    PressureTrace(p, φc, Xc, ranges, Δ; fit=:linear)

One-sided pressure fits at the crossing `p` (compressible jumps need the traction σn on Γ): on
side `s[k]`, `p_s = Σ λ[k] p[idx[k]]` and `∇p_s = Σ g[:, k] p[idx[k]]` over the side-s members.
"""
function PressureTrace(p, φc, Xc, ranges, Δ; fit=:linear, nmax=4, borrow=false)
    L  = 1 / max(abs(p.n[1]) / Δ.x, abs(p.n[2]) / Δ.y)
    fm, fp = SideFits(p, (p.x - L * p.n, p.x + L * p.n), φc, Xc, ranges, Δ; fit, nmax, borrow)
    return (idx=vcat(fm.idx, fp.idx), s=vcat(fill(-1, length(fm.idx)), fill(1, length(fp.idx))),
            λ=vcat(fm.λ, fp.λ), g=hcat(fm.g, fp.g))
end

"""
    InterfaceTraces(Γ, φ, X, Δ, η, nc; fit=:linear, pressure=false)

`InterfaceTrace` of both velocity components at every crossing of `Γ` inside the physical domain.
Candidate nodes are the Vx and Vy nodes of the domain, boundary nodes included (`Ranges`), and,
with `pressure=true`, the `PressureTrace` on the numbered pressure nodes (field `pt`).
`borrow`: see `SideFits` (used by the stress form for slivers between Γ and a wall).
"""
function InterfaceTraces(Γ, φ, X, Δ, η, nc; fit=:linear, pressure=false, borrow=false)
    ranges = (x=(2:(nc.x+2), 3:(nc.y+2)), y=(3:(nc.x+2), 2:(nc.y+2)), p=(2:(nc.x+1), 2:(nc.y+1)))
    pts = filter(p -> X.v.x[1] <= p.x[1] <= X.v.x[end] && X.v.y[1] <= p.x[2] <= X.v.y[end], Γ.points)
    return [(p=p,
             x=InterfaceTrace(p, φ.x, X.vx_e, ranges.x, Δ, η; fit, borrow),
             y=InterfaceTrace(p, φ.y, X.vy_e, ranges.y, Δ, η; fit, borrow),
             pt=pressure ? PressureTrace(p, φ.c, X.c_e, ranges.p, Δ; fit, borrow) : nothing) for p in pts]
end

"""
    apply_trace(tr, z)

Evaluate a trace from `InterfaceTrace` on the array `z`: returns (v_Γ, ∂ₛv_Γ, ∂ₛₛv_Γ).
"""
function apply_trace(tr, z)
    zk = [z[I] for I in tr.idx]
    return (dot(tr.v, zk), dot(tr.t, zk), dot(tr.tt, zk))
end
