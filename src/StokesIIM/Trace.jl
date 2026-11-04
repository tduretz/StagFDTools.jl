# Interface trace: v_Γ and ∂ₜv_Γ at a crossing of Γ as linear combinations of z = ηv.
#
# On each side s = ±1 of Γ, the 3 nodes of one velocity grid closest to x_Γ + s L n
# (L = one cell along n, as in Landry's compute_dv_dtt) define a linear interpolant of z.
# Its value and gradient at x_Γ give z_s(x_Γ) and ∂ₜz_s. Since η is constant on each side,
# Landry's η-weighted average of the one-sided velocities v_s = z_s/η_s reduces to
#   v_Γ    = (z₋(x_Γ) + z₊(x_Γ)) / (η₋ + η₊)
#   ∂ₜv_Γ  = (∂ₜz₋   + ∂ₜz₊  ) / (η₋ + η₊)
# which is exact for any z₋ = η₋v, z₊ = η₊v with v linear and continuous across Γ.

tangent(n) = SVector(-n[2], n[1])

"""
    OneSidedFit(xΓ, xo, s, φc, Xc, ranges, Δ; nmax=4)

Select the 3 nodes of the grid `Xc` (coordinates `Xc.x`, `Xc.y`, level set `φc`) on side `s`,
within `ranges`, closest to `xo` and not collinear. Return their indices, the barycentric
weights `λ` of `xΓ` and the gradient weights `g` (∇f ≈ Σ g_k f_k for linear f).
"""
function OneSidedFit(xΓ, xo, s, φc, Xc, ranges, Δ; nmax=4)
    ic = round(Int, (xo[1] - Xc.x[1]) / Δ.x) + 1
    jc = round(Int, (xo[2] - Xc.y[1]) / Δ.y) + 1
    cand = CartesianIndex{2}[]
    for j in max(jc - nmax, first(ranges[2])):min(jc + nmax, last(ranges[2])),
        i in max(ic - nmax, first(ranges[1])):min(ic + nmax, last(ranges[1]))
        side(φc[i, j]) == s && push!(cand, CartesianIndex(i, j))
    end
    xk(I) = SVector(Xc.x[I[1]], Xc.y[I[2]])
    sort!(cand, by=I -> norm((xk(I) - xo) ./ SVector(Δ.x, Δ.y)))
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
    InterfaceTrace(p, φc, Xc, ranges, Δ, η; nmax=4)

Weights of the interface value and tangential derivative of one velocity component at the
crossing `p` (from `InterfaceCrossings`), acting on z = ηv of that component's grid:
`v_Γ = Σ v[k] z[idx[k]]`, `∂ₜv_Γ = Σ t[k] z[idx[k]]`. `η = (neg=η₋, pos=η₊)` are the
viscosities on the φ < 0 and φ > 0 sides. Also returns the fit centres `xo` for plotting.
"""
function InterfaceTrace(p, φc, Xc, ranges, Δ, η; nmax=4)
    L  = 1 / max(abs(p.n[1]) / Δ.x, abs(p.n[2]) / Δ.y)
    t  = tangent(p.n)
    xo = (p.x - L * p.n, p.x + L * p.n)
    fm = OneSidedFit(p.x, xo[1], -1, φc, Xc, ranges, Δ; nmax)
    fp = OneSidedFit(p.x, xo[2],  1, φc, Xc, ranges, Δ; nmax)
    ηs = η.neg + η.pos
    idx = vcat(fm.idx, fp.idx)
    v   = vcat(fm.λ, fp.λ) ./ ηs
    dt  = vcat(fm.g' * t, fp.g' * t) ./ ηs
    return (idx=idx, v=v, t=dt, xo=xo)
end

"""
    InterfaceTraces(Γ, φ, X, Δ, η, nc)

`InterfaceTrace` of both velocity components at every crossing of `Γ` inside the physical domain.
Candidate nodes are the Vx and Vy nodes of the domain, boundary nodes included (`Ranges`).
"""
function InterfaceTraces(Γ, φ, X, Δ, η, nc)
    ranges = (x=(2:(nc.x+2), 3:(nc.y+2)), y=(3:(nc.x+2), 2:(nc.y+2)))
    pts = filter(p -> X.v.x[1] <= p.x[1] <= X.v.x[end] && X.v.y[1] <= p.x[2] <= X.v.y[end], Γ.points)
    return [(p=p,
             x=InterfaceTrace(p, φ.x, X.vx_e, ranges.x, Δ, η),
             y=InterfaceTrace(p, φ.y, X.vy_e, ranges.y, Δ, η)) for p in pts]
end

"""
    apply_trace(tr, z)

Evaluate a trace from `InterfaceTrace` on the array `z`: returns (v_Γ, ∂ₜv_Γ).
"""
apply_trace(tr, z) = (sum(tr.v[k] * z[tr.idx[k]] for k in 1:6), sum(tr.t[k] * z[tr.idx[k]] for k in 1:6))
