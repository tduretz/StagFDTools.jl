# Interface trace: v_Γ, ∂ₛv_Γ and ∂ₛₛv_Γ at a crossing of Γ as linear combinations of z = ηv
# (s: arc length along Γ, ∂ₛ = t·∇ at x_Γ).
#
# On each side s = ±1 of Γ, nodes of one velocity grid close to x_Γ + s L n (L = one cell along
# n, as in Landry's compute_dv_dtt) define a fit of z:
#   :linear     3 nodes, linear interpolant (Landry, first order)
#   :quadratic  ~10 nodes, quadratic least squares
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

"""
    OneSidedFit(xΓ, xo, s, φc, Xc, ranges, Δ; nmax=4)

Linear fit (Landry): the 3 nodes of side `s` closest to `xo`, not collinear. Returns their
indices, the weights `λ` of the value at `xΓ` and the gradient weights `g` (∇f ≈ g * f).
"""
function OneSidedFit(xΓ, xo, s, φc, Xc, ranges, Δ; nmax=4)
    cand, xk = side_candidates(xo, s, φc, Xc, ranges, Δ, nmax)
    length(cand) >= 3 || error("OneSidedFit: fewer than 3 nodes on side $s near $xΓ")
    # Rows [1, dx, dy] in cell units relative to xo; the third node must not be collinear
    row(I) = SVector(1.0, (xk(I)[1] - xo[1]) / Δ.x, (xk(I)[2] - xo[2]) / Δ.y)
    k3 = findfirst(k -> abs(det(hcat(row(cand[1]), row(cand[2]), row(cand[k])))) > 0.5, 3:length(cand))
    isnothing(k3) && error("OneSidedFit: all nodes collinear on side $s near $xΓ")
    idx = SVector(cand[1], cand[2], cand[k3+2])
    V   = vcat(row(idx[1])', row(idx[2])', row(idx[3])')   # f_k = a₁ + a₂ dx_k + a₃ dy_k
    iV  = inv(V)                                            # a = iV * f
    e   = SVector(1.0, (xΓ[1] - xo[1]) / Δ.x, (xΓ[2] - xo[2]) / Δ.y)
    λ   = iV' * e                                           # f(xΓ) = Σ λ_k f_k
    g   = vcat((iV[2, :] ./ Δ.x)', (iV[3, :] ./ Δ.y)')      # ∇f = g * f
    return (idx=idx, λ=λ, g=g)
end

"""
    QuadraticFit(xΓ, xo, s, φc, Xc, ranges, Δ; npts=10, nmax=5)

Quadratic least-squares fit of the `npts` nodes of side `s` closest to `xo` (more if the
6×6 normal matrix is ill-conditioned). Basis in cell units centred at `xΓ`:
[1, dx, dy, dx²/2, dy²/2, dx dy]. Returns indices, value weights `λ`, gradient weights `g`
(2×N) and Hessian weights `H` (H[a, b, k]).
"""
function QuadraticFit(xΓ, xo, s, φc, Xc, ranges, Δ; npts=10, nmax=5)
    cand, xk = side_candidates(xo, s, φc, Xc, ranges, Δ, nmax)
    row(I) = (dx = (xk(I)[1] - xΓ[1]) / Δ.x; dy = (xk(I)[2] - xΓ[2]) / Δ.y;
              SVector(1.0, dx, dy, dx^2 / 2, dy^2 / 2, dx * dy))
    for N in npts:length(cand)
        V = reduce(vcat, (row(I)' for I in cand[1:N]))
        σ = svdvals(V)
        σ[end] > 1e-2 * σ[1] || continue
        P = pinv(V)                                          # a = P * f
        g = vcat((P[2, :] ./ Δ.x)', (P[3, :] ./ Δ.y)')
        H = zeros(2, 2, N)
        H[1, 1, :] .= P[4, :] ./ Δ.x^2
        H[2, 2, :] .= P[5, :] ./ Δ.y^2
        H[1, 2, :] .= H[2, 1, :] .= P[6, :] ./ (Δ.x * Δ.y)
        return (idx=cand[1:N], λ=P[1, :], g=g, H=H)
    end
    error("QuadraticFit: no well-conditioned set of nodes on side $s near $xΓ")
end

"""
    InterfaceTrace(p, φc, Xc, ranges, Δ, η; fit=:linear, nmax=4)

Weights of the interface value and of the first and second derivatives along Γ of one velocity
component at the crossing `p` (from `InterfaceCrossings`), acting on z = ηv of that component's
grid: `v_Γ = Σ v[k] z[idx[k]]`, `∂ₛv_Γ = Σ t[k] z[idx[k]]`, `∂ₛₛv_Γ = Σ tt[k] z[idx[k]]`
(`tt` is zero for `fit=:linear`). `η = (neg=η₋, pos=η₊)` are the viscosities on the φ < 0 and
φ > 0 sides. Also returns the fit centres `xo` for plotting.
"""
function InterfaceTrace(p, φc, Xc, ranges, Δ, η; fit=:linear, nmax=4)
    L  = 1 / max(abs(p.n[1]) / Δ.x, abs(p.n[2]) / Δ.y)
    t  = tangent(p.n)
    xo = (p.x - L * p.n, p.x + L * p.n)
    ηs = η.neg + η.pos
    if fit == :linear
        fm = OneSidedFit(p.x, xo[1], -1, φc, Xc, ranges, Δ; nmax)
        fp = OneSidedFit(p.x, xo[2],  1, φc, Xc, ranges, Δ; nmax)
        idx = vcat(collect(fm.idx), collect(fp.idx))
        v   = vcat(fm.λ, fp.λ) ./ ηs
        dt  = vcat(fm.g' * t, fp.g' * t) ./ ηs
        dtt = zeros(length(idx))
    else
        fm = QuadraticFit(p.x, xo[1], -1, φc, Xc, ranges, Δ)
        fp = QuadraticFit(p.x, xo[2],  1, φc, Xc, ranges, Δ)
        dss(f) = [dot(t, f.H[:, :, k] * t) - p.κ * dot(p.n, f.g[:, k]) for k in eachindex(f.idx)]
        idx = vcat(fm.idx, fp.idx)
        v   = vcat(fm.λ, fp.λ) ./ ηs
        dt  = vcat(fm.g' * t, fp.g' * t) ./ ηs
        dtt = vcat(dss(fm), dss(fp)) ./ ηs
    end
    return (idx=idx, v=Vector(v), t=Vector(dt), tt=dtt, xo=xo)
end

"""
    InterfaceTraces(Γ, φ, X, Δ, η, nc; fit=:linear)

`InterfaceTrace` of both velocity components at every crossing of `Γ` inside the physical domain.
Candidate nodes are the Vx and Vy nodes of the domain, boundary nodes included (`Ranges`).
"""
function InterfaceTraces(Γ, φ, X, Δ, η, nc; fit=:linear)
    ranges = (x=(2:(nc.x+2), 3:(nc.y+2)), y=(3:(nc.x+2), 2:(nc.y+2)))
    pts = filter(p -> X.v.x[1] <= p.x[1] <= X.v.x[end] && X.v.y[1] <= p.x[2] <= X.v.y[end], Γ.points)
    return [(p=p,
             x=InterfaceTrace(p, φ.x, X.vx_e, ranges.x, Δ, η; fit),
             y=InterfaceTrace(p, φ.y, X.vy_e, ranges.y, Δ, η; fit)) for p in pts]
end

"""
    apply_trace(tr, z)

Evaluate a trace from `InterfaceTrace` on the array `z`: returns (v_Γ, ∂ₛv_Γ, ∂ₛₛv_Γ).
"""
function apply_trace(tr, z)
    zk = [z[I] for I in tr.idx]
    return (dot(tr.v, zk), dot(tr.t, zk), dot(tr.tt, zk))
end
