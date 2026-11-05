using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
import Random
using DifferentiationInterface
using TimerOutputs


# Grid -> marker stress interpolation that preserves discrete force balance.
#
# After the Stokes solve, stresses are known on the grid (τxx, τyy, P at centroids, τxy at vertices).
# Markers receive them by:
#   (3) standard bilinear interpolation  ->  s̃
#   (4) bilinear interpolation + equilibrium projection on the grid:
#       s = s̃ - W⁻¹Bᵀ (B W⁻¹ Bᵀ)⁻¹ (B s̃ - f)
#       where B = D Π, Π is the marker -> grid stress map and D the staggered discrete divergence.
#       s is the closest field to s̃ (in the W-norm) whose marker -> grid projection is exactly in
#       discrete force balance, i.e. it produces no spurious forces when fed back to the solver.
#   (7) Airy stress function reconstruction: in each cell, a polynomial φ_k(x, y) of degree p is fitted (least
#       squares) to the grid stresses of the surrounding 3×3 cells, with σxx = ∂²φ/∂y², σyy = ∂²φ/∂x²,
#       σxy = -∂²φ/∂x∂y. The stress at the markers of the cell is then exactly divergence-free (pointwise,
#       ∇·σ = 0 with g = 0), whatever the quality of the fit, but traction jumps across cell edges.
#   (8) as (7), with φ = Σ_k w_k φ_k blended by a quadratic B-spline partition of unity (affine gauges of the φ_k
#       aligned first). Airy(φ) is divergence-free for any φ, so markers stay in exact equilibrium; φ is C¹ across
#       cell edges, so traction is also continuous there.
#       For (7) and (8), only σ is reconstructed: P and τ are recovered with τzz taken from the bilinear interpolation.
#
# Marker-level diagnostics:
#   - sub-grid residual D_sub Π_sub s on a staggered sub-grid with one marker per sub-cell
#   - independent check: divergence of a linear least-squares fit of σ through the markers of each cell
#     (cells cut by the interface are skipped since σ_tt legitimately jumps there)
#   - pointwise: central finite-difference divergence of the reconstructed continuous field at each marker
#--------------------------------------------#
# Bilinear interpolation of a field F defined on a regular grid (x0, y0, Δx, Δy) to point (xm, ym)
function bilinear(F, x0, y0, Δx, Δy, xm, ym)
    i  = clamp(floor(Int, (xm - x0) / Δx) + 1, 1, size(F, 1) - 1)
    j  = clamp(floor(Int, (ym - y0) / Δy) + 1, 1, size(F, 2) - 1)
    wx = (xm - (x0 + (i - 1) * Δx)) / Δx
    wy = (ym - (y0 + (j - 1) * Δy)) / Δy
    return (1 - wx) * (1 - wy) * F[i, j] + wx * (1 - wy) * F[i+1, j] +
           (1 - wx) *      wy  * F[i, j+1] + wx *      wy  * F[i+1, j+1]
end

# Marker -> grid map: each node gets the bilinear-weighted average of the markers around it
# Nodes are (x0 + (i-1)Δx, y0 + (j-1)Δy), i = 1:nx, j = 1:ny; nodes with active[i,j] = false get empty rows
function marker_to_grid_matrix(xm, ym, x0, y0, Δx, Δy, nx, ny, active)
    I, J, V = Int[], Int[], Float64[]
    LI = LinearIndices((nx, ny))
    for m in eachindex(xm)
        i0 = floor(Int, (xm[m] - x0) / Δx) + 1
        j0 = floor(Int, (ym[m] - y0) / Δy) + 1
        for j in j0:j0+1, i in i0:i0+1
            (1 <= i <= nx && 1 <= j <= ny && active[i, j]) || continue
            w = (1 - abs(xm[m] - (x0 + (i - 1) * Δx)) / Δx) * (1 - abs(ym[m] - (y0 + (j - 1) * Δy)) / Δy)
            w > 0 || continue
            push!(I, LI[i, j]); push!(J, m); push!(V, w)
        end
    end
    Π = sparse(I, J, V, nx * ny, length(xm))
    wsum = vec(sum(Π, dims=2))
    return spdiagm(1 ./ max.(wsum, eps())) * Π
end

# Staggered discrete divergence of the total stress σ = τ - P I, (∂σxx/∂x + ∂σxy/∂y, ∂σxy/∂x + ∂σyy/∂y)
# on interior Vx and Vy nodes (free slip: boundary normal velocities are prescribed and carry no equation).
# Returns the blocks such that  Rx = Gx_c*σxx + Gy_v*τxy,  Ry = Gy_c*σyy + Gx_v*τxy
function divergence_operators(nc, Δ)
    C, Vt = LinearIndices((nc.x, nc.y)), LinearIndices((nc.x + 1, nc.y + 1))
    nVx, nVy = (nc.x - 1) * nc.y, nc.x * (nc.y - 1)
    # Two-point difference operator assembled from triplets: row k gets +1/h at column ip, -1/h at column im
    diff_op(rows, ip, im, h, nrow, ncol) = sparse([rows; rows], [ip; im], [fill(1 / h, length(rows)); fill(-1 / h, length(rows))], nrow, ncol)
    Ix  = vec(LinearIndices((nc.x - 1, nc.y)))
    ijx = vec([(i, j) for i in 2:nc.x, j in 1:nc.y])
    Gx_c = diff_op(Ix, [C[i, j] for (i, j) in ijx],    [C[i-1, j] for (i, j) in ijx], Δ.x, nVx, nc.x * nc.y)
    Gy_v = diff_op(Ix, [Vt[i, j+1] for (i, j) in ijx], [Vt[i, j] for (i, j) in ijx],  Δ.y, nVx, (nc.x + 1) * (nc.y + 1))
    Iy  = vec(LinearIndices((nc.x, nc.y - 1)))
    ijy = vec([(i, j) for i in 1:nc.x, j in 2:nc.y])
    Gy_c = diff_op(Iy, [C[i, j] for (i, j) in ijy],    [C[i, j-1] for (i, j) in ijy], Δ.y, nVy, nc.x * nc.y)
    Gx_v = diff_op(Iy, [Vt[i+1, j] for (i, j) in ijy], [Vt[i, j] for (i, j) in ijy],  Δ.x, nVy, (nc.x + 1) * (nc.y + 1))
    return Gx_c, Gy_v, Gy_c, Gx_v
end

# Discrete force balance residual of grid stresses
function grid_residual(ops, τxx_c, τyy_c, τxy_v, P_c)
    Gx_c, Gy_v, Gy_c, Gx_v = ops
    Rx = Gx_c * (vec(τxx_c) .- vec(P_c)) .+ Gy_v * vec(τxy_v)
    Ry = Gy_c * (vec(τyy_c) .- vec(P_c)) .+ Gx_v * vec(τxy_v)
    return Rx, Ry
end

# Equilibrium constraint rows for marker stresses s = [τxx; τyy; τxy; P]: B s = Δx * D Π s (scaled by Δx for conditioning)
function equilibrium_constraint(ops, Πc, Πv, Δ, Nm)
    Gx_c, Gy_v, Gy_c, Gx_v = ops
    Z  = spzeros(size(Gx_c, 1), Nm)
    Zy = spzeros(size(Gy_c, 1), Nm)
    return Δ.x .* [Gx_c*Πc  Z        Gy_v*Πv  -Gx_c*Πc;
                   Zy       Gy_c*Πc  Gx_v*Πv  -Gy_c*Πc]
end

# Closest s to s̃ (W-norm) such that B s = f
function project(B, s̃, Winv, f)
    S = B * Winv * B'
    λ = cholesky(Symmetric(S + 1e-12 * mean(diag(S)) * I)) \ (B * s̃ .- f)
    return s̃ .- Winv * (B' * λ)
end

# Exponents (a, b) of the monomials ξᵃηᵇ of φ, 2 ≤ a + b ≤ p (affine terms produce no stress)
airy_exponents(p) = [(a, n - a) for n in 2:p for a in 0:n]

# ∂^(da+db) (ξᵃηᵇ) / ∂ξ^da ∂η^db
dmono(a, b, da, db, ξ, η) = (a < da || b < db) ? 0.0 :
    prod(a-da+1:a; init=1.0) * prod(b-db+1:b; init=1.0) * ξ^(a - da) * η^(b - db)

# Airy stress (σxx, σyy, σxy) of φ = Σ c ξᵃηᵇ in local coordinates ξ = (x - xc)/Δx, η = (y - yc)/Δy
airy_rows(exps, ξ, η, Δ) = ([dmono(a, b, 0, 2, ξ, η) / Δ.y^2 for (a, b) in exps],
                            [dmono(a, b, 2, 0, ξ, η) / Δ.x^2 for (a, b) in exps],
                            [-dmono(a, b, 1, 1, ξ, η) / (Δ.x * Δ.y) for (a, b) in exps])

# Collection of per-cell stress functions φ_k(x, y) = Σ C[:, k] ξᵃηᵇ + A[1, k] + A[2, k] ξ + A[3, k] η,
# with ξ, η local to the centre of cell k
struct AiryField{E,T,D,N}
    exps::E
    C::Matrix{Float64}   # polynomial coefficients (degree 2..p), one column per cell
    A::Matrix{Float64}   # affine part (no stress, only matters for blending)
    xc::T
    yc::T
    Δ::D
    nc::N
end

# (7) Per-cell least-squares fit of φ_k to the grid stresses of the surrounding (2r+1)² cells
# Returns the field and the rms misfit of each cell fit
function airy_fit(X, Δ, nc, τxx_c, τyy_c, τxy_v, P_c; p=5, r=1)
    exps = airy_exponents(p)
    LI   = LinearIndices((nc.x, nc.y))
    C    = zeros(length(exps), nc.x * nc.y)
    misfit = zeros(nc.x, nc.y)
    for j in 1:nc.y, i in 1:nc.x
        xc, yc = X.c.x[i], X.c.y[j]
        # Patch of (2r+1)² cells, shifted inwards at the boundaries
        i0 = clamp(i - r, 1, nc.x - 2r); j0 = clamp(j - r, 1, nc.y - 2r)
        A, d = Vector{Vector{Float64}}(), Float64[]
        for jj in j0:j0+2r, ii in i0:i0+2r
            rxx, ryy, _ = airy_rows(exps, (X.c.x[ii] - xc) / Δ.x, (X.c.y[jj] - yc) / Δ.y, Δ)
            push!(A, rxx); push!(d, τxx_c[ii, jj] - P_c[ii, jj])
            push!(A, ryy); push!(d, τyy_c[ii, jj] - P_c[ii, jj])
        end
        for jj in j0:j0+2r+1, ii in i0:i0+2r+1
            _, _, rxy = airy_rows(exps, (X.v.x[ii] - xc) / Δ.x, (X.v.y[jj] - yc) / Δ.y, Δ)
            push!(A, rxy); push!(d, τxy_v[ii, jj])
        end
        M = reduce(vcat, transpose.(A))
        c = M \ d
        C[:, LI[i, j]] .= c
        misfit[i, j] = norm(M * c .- d) / sqrt(length(d))
    end
    return AiryField(exps, C, zeros(3, nc.x * nc.y), X.c.x, X.c.y, Δ, nc), misfit
end

# ∂^(da+db) φ_k / ∂x^da ∂y^db at (x, y), physical coordinates
function phi_deriv(F::AiryField, k, da, db, x, y; affine=true)
    i, j = mod1(k, F.nc.x), cld(k, F.nc.x)
    ξ, η = (x - F.xc[i]) / F.Δ.x, (y - F.yc[j]) / F.Δ.y
    v = 0.0
    for (n, (a, b)) in enumerate(F.exps)
        v += F.C[n, k] * dmono(a, b, da, db, ξ, η)
    end
    if affine
        (da, db) == (0, 0) && (v += F.A[1, k] + F.A[2, k] * ξ + F.A[3, k] * η)
        (da, db) == (1, 0) && (v += F.A[2, k])
        (da, db) == (0, 1) && (v += F.A[3, k])
    end
    return v / (F.Δ.x^da * F.Δ.y^db)
end

# Unblended stress: φ of the cell containing (x, y)
function airy_sigma_cell(F::AiryField, x, y)
    i = clamp(floor(Int, (x - F.xc[1]) / F.Δ.x + 0.5) + 1, 1, F.nc.x)
    j = clamp(floor(Int, (y - F.yc[1]) / F.Δ.y + 0.5) + 1, 1, F.nc.y)
    k = i + (j - 1) * F.nc.x
    return (phi_deriv(F, k, 0, 2, x, y), phi_deriv(F, k, 2, 0, x, y), -phi_deriv(F, k, 1, 1, x, y))
end

# Quadratic B-spline (unit knot spacing) and its first two derivatives
function bspline2(t)
    s = abs(t)
    s <= 0.5 && return (0.75 - t^2, -2t, -2.0)
    s <= 1.5 && return (0.5 * (1.5 - s)^2, -(1.5 - s) * sign(t), 1.0)
    return (0.0, 0.0, 0.0)
end

# Partition-of-unity weights along one direction: quadratic B-splines centred on the n cell centres x0 + (i-1)h,
# the ghost splines beyond the boundaries being merged into the boundary cells (keeps Σw = 1 and C¹).
# Returns [(index, (w, ∂w, ∂²w))]
function pu_weights_1d(x, x0, h, n)
    out = Tuple{Int,NTuple{3,Float64}}[]
    ic  = round(Int, (x - x0) / h) + 1
    for i in ic-1:ic+1
        b, db, d2b = bspline2((x - (x0 + (i - 1) * h)) / h)
        w = (b, db / h, d2b / h^2)
        b == 0 && continue
        k = clamp(i, 1, n)
        idx = findfirst(e -> e[1] == k, out)
        isnothing(idx) ? push!(out, (k, w)) : (out[idx] = (k, out[idx][2] .+ w))
    end
    return out
end

# ∂^(ax+ay) φ / ∂x^ax ∂y^ay of the blended φ = Σ_k w_k φ_k (Leibniz rule), ax, ay ≤ 2
function phi_blended_deriv(F::AiryField, wx, wy, ax, ay, x, y)
    v = 0.0
    for (kx, Wx) in wx, (ky, Wy) in wy
        k = kx + (ky - 1) * F.nc.x
        for bx in 0:ax, by in 0:ay
            v += binomial(ax, bx) * binomial(ay, by) * Wx[bx+1] * Wy[by+1] * phi_deriv(F, k, ax - bx, ay - by, x, y)
        end
    end
    return v
end

# (8) Blended stress: σ = Airy(Σ_k w_k φ_k). Any φ gives ∇·σ = 0, so blending cannot break pointwise equilibrium;
# φ is C¹ across the knot lines (cell edges), hence traction σ·n is continuous there
function airy_sigma_blended(F::AiryField, x, y)
    wx = pu_weights_1d(x, F.xc[1], F.Δ.x, F.nc.x)
    wy = pu_weights_1d(y, F.yc[1], F.Δ.y, F.nc.y)
    return (phi_blended_deriv(F, wx, wy, 0, 2, x, y), phi_blended_deriv(F, wx, wy, 2, 0, x, y), -phi_blended_deriv(F, wx, wy, 1, 1, x, y))
end

# Each φ_k is only defined up to an affine part, and Airy(w_k a_k) ≠ 0: choose the affine parts so that overlapping
# φ_k agree as well as possible (weighted least squares on 5 sample points per cell). Only affects accuracy.
function align_affine!(F::AiryField)
    N  = F.nc.x * F.nc.y
    II, JJ, VV, rhs, r = Int[], Int[], Float64[], Float64[], 0
    for j in 1:F.nc.y, i in 1:F.nc.x, (ox, oy) in ((0, 0), (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
        x, y = F.xc[i] + ox * F.Δ.x, F.yc[j] + oy * F.Δ.y
        act = [(kx + (ky - 1) * F.nc.x, Wx[1] * Wy[1]) for (kx, Wx) in pu_weights_1d(x, F.xc[1], F.Δ.x, F.nc.x)
                                                       for (ky, Wy) in pu_weights_1d(y, F.yc[1], F.Δ.y, F.nc.y)]
        for p in eachindex(act), q in p+1:length(act)
            (k, wk), (l, wl) = act[p], act[q]
            s = sqrt(wk * wl)
            r += 1
            for (m, sg) in ((k, 1.0), (l, -1.0))
                mi, mj = mod1(m, F.nc.x), cld(m, F.nc.x)
                ξ, η = (x - F.xc[mi]) / F.Δ.x, (y - F.yc[mj]) / F.Δ.y
                append!(II, (r, r, r)); append!(JJ, 3m .- (2, 1, 0)); append!(VV, sg * s .* (1.0, ξ, η))
            end
            push!(rhs, -s * (phi_deriv(F, k, 0, 0, x, y; affine=false) - phi_deriv(F, l, 0, 0, x, y; affine=false)))
        end
    end
    M  = sparse(II, JJ, VV, r, 3N)
    Nm = M' * M
    a  = cholesky(Symmetric(Nm + 1e-10 * mean(diag(Nm)) * I)) \ (M' * rhs)   # regularisation fixes the global affine
    F.A .= reshape(a, 3, N)
    return F
end

# Marker stresses from a stress-field closure σf(x, y) -> (σxx, σyy, σxy); P and τ recovered with τzz from bilinear
function markers_from_sigma(σf, xm, ym, τzz_m)
    Nm = length(xm)
    τxx, τyy, τxy, P = zeros(Nm), zeros(Nm), zeros(Nm), zeros(Nm)
    Threads.@threads for m in 1:Nm
        σxx, σyy, σxy = σf(xm[m], ym[m])
        # σzz = τzz - P, P = -(σxx + σyy + σzz)/3
        P[m]   = -(σxx + σyy + τzz_m[m]) / 2
        τxx[m] = σxx + P[m]
        τyy[m] = σyy + P[m]
        τxy[m] = σxy
    end
    return τxx, τyy, τxy, P
end

# Independent pointwise check: central finite-difference divergence of the reconstructed field σf at each marker
# (h ≪ distance of the markers to the cell edges). Returns |∇·σ| per marker and the largest derivative magnitude
function fd_divergence(σf, xm, ym, h)
    div = zeros(length(xm)); dmax = 0.0
    for m in eachindex(xm)
        x, y = xm[m], ym[m]
        sxp, sxm, syp, sym = σf(x + h, y), σf(x - h, y), σf(x, y + h), σf(x, y - h)
        d = ((sxp[1] - sxm[1]), (syp[3] - sym[3]), (sxp[3] - sxm[3]), (syp[2] - sym[2])) ./ (2h)
        div[m] = hypot(d[1] + d[2], d[3] + d[4])
        dmax = max(dmax, maximum(abs, d))
    end
    return div, dmax
end

# Stress jumps across interior cell edges (sampled at 3 points per edge, ±ε on each side):
# traction (normal-normal, normal-tangential) must be continuous for equilibrium, σ_tt may jump
function edge_jumps(σf, X, Δ, nc; ε=1e-6)
    jt, jtt = 0.0, 0.0
    for j in 1:nc.y, i in 2:nc.x, s in (0.25, 0.5, 0.75)        # vertical edges x = X.v.x[i]: traction (σxx, σxy)
        x, y = X.v.x[i], X.v.y[j] + s * Δ.y
        l, r = σf(x - ε * Δ.x, y), σf(x + ε * Δ.x, y)
        jt  = max(jt, abs(l[1] - r[1]), abs(l[3] - r[3])); jtt = max(jtt, abs(l[2] - r[2]))
    end
    for j in 2:nc.y, i in 1:nc.x, s in (0.25, 0.5, 0.75)        # horizontal edges y = X.v.y[j]: traction (σxy, σyy)
        x, y = X.v.x[i] + s * Δ.x, X.v.y[j]
        b, t = σf(x, y - ε * Δ.y), σf(x, y + ε * Δ.y)
        jt  = max(jt, abs(b[2] - t[2]), abs(b[3] - t[3])); jtt = max(jtt, abs(b[1] - t[1]))
    end
    return jt, jtt
end

# Divergence of a linear least-squares fit σ ≈ c₀ + c₁x + c₂y through the markers of each cell (NaN in mixed cells)
function cell_fit_divergence(xm, ym, τxx, τyy, τxy, P, nc, nmark, mixed)
    divx, divy = fill(NaN, nc.x, nc.y), fill(NaN, nc.x, nc.y)
    n = nmark^2
    for j in 1:nc.y, i in 1:nc.x
        mixed[i, j] && continue
        ms = ((j - 1) * nc.x + (i - 1)) * n .+ (1:n)
        A  = [ones(n) xm[ms] ym[ms]]
        cxx, cyy, cxy = A \ (τxx[ms] .- P[ms]), A \ (τyy[ms] .- P[ms]), A \ τxy[ms]
        divx[i, j] = cxx[2] + cxy[3]
        divy[i, j] = cxy[2] + cyy[3]
    end
    return divx, divy
end

@views function marker_stress_comparison(a, nc, Δ, L, rad; nmark=4, w_mixed=1.0, airy_p=5)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    X = a.X

    # Grid stresses (interior centroids / all vertices)
    P_c   = a.Pt[inx_c, iny_c] .- mean(a.Pt[inx_c, iny_c])
    τxx_c = a.τ.xx[inx_c, iny_c]
    τyy_c = a.τ.yy[inx_c, iny_c]
    τxy_v = a.τ.xy[inx_v, iny_v]

    ops = divergence_operators(nc, Δ)
    Rx_grid, Ry_grid = grid_residual(ops, τxx_c, τyy_c, τxy_v, P_c)
    @printf("Grid solution:            max|R| = %.3e  (reference: discrete balance of the solver)\n", max(maximum(abs, Rx_grid), maximum(abs, Ry_grid)))

    # (2) Markers: nmark × nmark per cell (one per sub-cell), randomly perturbed
    Random.seed!(1234)
    xm, ym = Float64[], Float64[]
    for j in 1:nc.y, i in 1:nc.x, jm in 1:nmark, im in 1:nmark
        push!(xm, X.v.x[i] + (im - 0.5 + 0.4 * (rand() - 0.5)) * Δ.x / nmark)
        push!(ym, X.v.y[j] + (jm - 0.5 + 0.4 * (rand() - 0.5)) * Δ.y / nmark)
    end
    Nm    = length(xm)
    phase = [(xm[m]^2 + ym[m]^2 <= rad^2) ? 2 : 1 for m in 1:Nm]

    # (3) Standard bilinear interpolation (centroid fields extended with zero-gradient ghosts)
    ext(F) = (G = zeros(size(F, 1) + 2, size(F, 2) + 2); G[2:end-1, 2:end-1] .= F;
              G[1, :] .= G[2, :]; G[end, :] .= G[end-1, :]; G[:, 1] .= G[:, 2]; G[:, end] .= G[:, end-1]; G)
    τxx_e, τyy_e, P_e = ext(τxx_c), ext(τyy_c), ext(P_c)
    xc0, yc0 = X.c_e.x[1], X.c_e.y[1]
    xv0, yv0 = X.v.x[1], X.v.y[1]
    τxx_b = [bilinear(τxx_e, xc0, yc0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    τyy_b = [bilinear(τyy_e, xc0, yc0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    P_b   = [bilinear(P_e,   xc0, yc0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    τxy_b = [bilinear(τxy_v, xv0, yv0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    s̃     = [τxx_b; τyy_b; τxy_b; P_b]

    # Marker -> grid maps; boundary τxy is set by the free slip condition (τxy = 0), so it has no marker contribution
    inner(nx, ny) = (act = trues(nx, ny); act[1, :] .= false; act[end, :] .= false; act[:, 1] .= false; act[:, end] .= false; act)
    Πc = marker_to_grid_matrix(xm, ym, X.c.x[1], X.c.y[1], Δ.x, Δ.y, nc.x, nc.y, trues(nc.x, nc.y))
    Πv = marker_to_grid_matrix(xm, ym, xv0, yv0, Δ.x, Δ.y, nc.x + 1, nc.y + 1, inner(nc.x + 1, nc.y + 1))
    B  = equilibrium_constraint(ops, Πc, Πv, Δ, Nm)

    # Sub-grid: each marker is the centroid value of its own sub-cell; sub-grid vertices average the surrounding markers
    ns   = (x = nmark * nc.x, y = nmark * nc.y)
    Δs   = (x = Δ.x / nmark, y = Δ.y / nmark)
    LIs  = LinearIndices((ns.x, ns.y))
    sub  = [LIs[clamp(floor(Int, (xm[m] - xv0) / Δs.x) + 1, 1, ns.x), clamp(floor(Int, (ym[m] - yv0) / Δs.y) + 1, 1, ns.y)] for m in 1:Nm]
    Πc_s = sparse(sub, 1:Nm, ones(Nm), ns.x * ns.y, Nm)
    Πv_s = marker_to_grid_matrix(xm, ym, xv0, yv0, Δs.x, Δs.y, ns.x + 1, ns.y + 1, inner(ns.x + 1, ns.y + 1))
    ops_s = divergence_operators(ns, Δs)
    B_s   = equilibrium_constraint(ops_s, Πc_s, Πv_s, Δs, Nm)

    # Weights W: confidence per marker; markers in cells cut by the interface can be made cheaper to modify
    # (w_mixed < 1) so that the correction goes where bilinear interpolation is least reliable
    cell_of(m) = (clamp(floor(Int, (xm[m] - xv0) / Δ.x) + 1, 1, nc.x), clamp(floor(Int, (ym[m] - yv0) / Δ.y) + 1, 1, nc.y))
    mixed = falses(nc.x, nc.y)
    for m in 1:Nm
        i, j = cell_of(m)
        mixed[i, j] |= phase[m] != phase[1 + (m - 1) - mod(m - 1, nmark^2)]   # differs from first marker of the cell
    end
    w    = [mixed[cell_of(m)...] ? w_mixed : 1.0 for m in 1:Nm]
    Winv = spdiagm(repeat(1 ./ w, 4))

    # (4) grid-level projection (no body force here, f = 0)
    s4 = project(B, s̃, Winv, zeros(size(B, 1)))

    # (7) per-cell Airy reconstruction, (8) the same φ_k blended with a partition of unity
    τzz_b = .-(τxx_b .+ τyy_b)
    t_fit   = @elapsed (AF, misfit) = airy_fit(X, Δ, nc, τxx_c, τyy_c, τxy_v, P_c; p=airy_p)
    t_align = @elapsed align_affine!(AF)
    σ_cell(x, y)  = airy_sigma_cell(AF, x, y)
    σ_blend(x, y) = airy_sigma_blended(AF, x, y)
    t7 = @elapsed s7 = markers_from_sigma(σ_cell,  xm, ym, τzz_b)
    t8 = @elapsed s8 = markers_from_sigma(σ_blend, xm, ym, τzz_b)
    @printf("Airy p = %d: fit %.2f s, affine alignment %.2f s, marker evaluation %.2f s (cell) / %.2f s (blended) | fit misfit rms: median %.2e, max %.2e\n",
        airy_p, t_fit, t_align, t7, t8, sort(vec(misfit))[end÷2+1], maximum(misfit))

    # Independent pointwise check: FD divergence of each reconstructed continuous field at the markers
    σ_bilin(x, y) = (bilinear(τxx_e, xc0, yc0, Δ.x, Δ.y, x, y) - bilinear(P_e, xc0, yc0, Δ.x, Δ.y, x, y),
                     bilinear(τyy_e, xc0, yc0, Δ.x, Δ.y, x, y) - bilinear(P_e, xc0, yc0, Δ.x, Δ.y, x, y),
                     bilinear(τxy_v, xv0, yv0, Δ.x, Δ.y, x, y))
    for (name, σf) in (("(3) bilinear", σ_bilin), ("(7) Airy per cell", σ_cell), ("(8) Airy blended", σ_blend))
        dv, dmax = fd_divergence(σf, xm, ym, 1e-4 * Δ.x)
        jt, jtt = edge_jumps(σf, X, Δ, nc)
        @printf("Pointwise FD |∇·σ| at markers, %-18s: median %.2e, max %.2e (max |∂σ| = %.2e) | cell-edge jumps: traction %.2e, σ_tt %.2e\n",
            name, sort(dv)[end÷2+1], maximum(dv), dmax, jt, jtt)
    end

    split4(s) = (s[1:Nm], s[Nm+1:2Nm], s[2Nm+1:3Nm], s[3Nm+1:4Nm])
    methods = (("(3) bilinear", split4(s̃)), ("(4) grid projection", split4(s4)),
               ("(7) Airy p = $airy_p", s7), ("(8) Airy p = $airy_p blended", s8))

    # Diagnostics: grid residual after marker -> grid, sub-grid (marker-level) residual, per-cell fit divergence
    back(τxx, τyy, τxy, P)     = (reshape(Πc * τxx, nc.x, nc.y), reshape(Πc * τyy, nc.x, nc.y), reshape(Πv * τxy, nc.x + 1, nc.y + 1), reshape(Πc * P, nc.x, nc.y))
    back_sub(τxx, τyy, τxy, P) = (reshape(Πc_s * τxx, ns.x, ns.y), reshape(Πc_s * τyy, ns.x, ns.y), reshape(Πv_s * τxy, ns.x + 1, ns.y + 1), reshape(Πc_s * P, ns.x, ns.y))
    diag_out = map(methods) do (name, f)
        R    = grid_residual(ops, back(f...)...)
        Rs   = grid_residual(ops_s, back_sub(f...)...)
        divf = cell_fit_divergence(xm, ym, f..., nc, nmark, mixed)
        dabs = filter(!isnan, sqrt.(divf[1] .^ 2 .+ divf[2] .^ 2))
        @printf("%-24s grid max|R| = %.2e | sub-grid max|R| = %.2e | cell-fit |div σ|: median %.2e, max %.2e | max|Δs| vs (3) = %.2e\n",
            name, max(maximum(abs, R[1]), maximum(abs, R[2])), max(maximum(abs, Rs[1]), maximum(abs, Rs[2])),
            sort(dabs)[end÷2+1], maximum(dabs), maximum(abs, vcat(f...) .- s̃))
        (R=R, Rs=Rs, divf=divf)
    end

    #--------------------------------------------#
    # Figure: columns = grid (FD), (3), (4), (7) Airy per cell, (8) Airy blended; rows = τxx, P, grid residual, sub-grid residual, cell-fit divergence
    Fig = Figure(size=(2500, 2300), fontsize=14)
    θ   = LinRange(0, 2π, 200)
    circle!(ax) = lines!(ax, rad .* cos.(θ), rad .* sin.(θ), color=:black, linewidth=1)
    # Each marker sits in its own sub-cell (nmark × nmark per cell), so marker values are shown as an image, one pixel per marker
    xs  = X.v.x[1] .+ ((1:ns.x) .- 0.5) .* Δs.x
    ys  = X.v.y[1] .+ ((1:ns.y) .- 0.5) .* Δs.y
    to_image(val) = reshape(permutedims(reshape(val, nmark, nmark, nc.x, nc.y), (1, 3, 2, 4)), ns.x, ns.y)
    function panel(pos, title, x, y, val; cmap=:redsblues, crange=(-1, 1) .* maximum(abs, filter(!isnan, val)))
        ax = Axis(Fig[pos...], aspect=DataAspect(), title=title, xlabel="x", ylabel="y")
        hm = heatmap!(ax, x, y, val, colormap=cmap, colorrange=crange, nan_color=:lightgray)
        circle!(ax)
        Colorbar(Fig[pos[1], pos[2]+1], hm)
        return ax
    end
    cr_τ = (-1, 1) .* maximum(abs, τxx_b)
    cr_P = (-1, 1) .* maximum(abs, P_b)
    lR   = (-16, 3)
    lg(R, n) = log10.(abs.(reshape(R, n...)) .+ 1e-16)

    panel((1, 1), "τxx — grid (FD)", X.c.x, X.c.y, τxx_c; crange=cr_τ)
    panel((2, 1), "P — grid (FD)",   X.c.x, X.c.y, P_c;   crange=cr_P)
    panel((3, 1), "log₁₀|Rx| grid — FD", X.v.x[2:end-1], X.c.y, lg(Rx_grid, (nc.x - 1, nc.y)); cmap=:viridis, crange=lR)
    for (k, ((name, f), d)) in enumerate(zip(methods, diag_out))
        col = 2k + 1
        panel((1, col), "τxx — $name", xs, ys, to_image(f[1]); crange=cr_τ)
        panel((2, col), "P — $name",   xs, ys, to_image(f[4]); crange=cr_P)
        panel((3, col), "log₁₀|Rx| grid — $name",     X.v.x[2:end-1], X.c.y, lg(d.R[1], (nc.x - 1, nc.y)); cmap=:viridis, crange=lR)
        panel((4, col), "log₁₀|Rx| sub-grid — $name", xs[1:end-1] .+ Δs.x / 2, ys, lg(d.Rs[1], (ns.x - 1, ns.y)); cmap=:viridis, crange=lR)
        panel((5, col), "log₁₀|div σ|ₓ cell fit — $name", X.c.x, X.c.y, log10.(abs.(d.divf[1]) .+ 1e-16); cmap=:viridis, crange=(-4, 2))
    end

    # Profile across the inclusion along y ≈ 0
    ax = Axis(Fig[4, 1:2], title="σxx = τxx − P along |y| < Δy/4", xlabel="x", ylabel="σxx")
    jc    = argmin(abs.(X.c.y))
    strip = findall(abs.(ym .- X.c.y[jc]) .< Δ.y / 4)
    o     = strip[sortperm(xm[strip])]
    lines!(ax, X.c.x, τxx_c[:, jc] .- P_c[:, jc], color=:black, label="grid (centroids)")
    for ((name, f), c) in zip(methods, (:steelblue, :orangered, :seagreen, :purple))
        scatter!(ax, xm[o], f[1][o] .- f[4][o], markersize=4, color=c, label=name)
    end
    vlines!(ax, [-rad, rad], color=:gray, linestyle=:dash)
    axislegend(ax, position=:rb)

    # Marker change of the blended Airy reconstruction
    panel((5, 1), "τxx: (8) − (3)", xs, ys, to_image(methods[4][2][1] .- τxx_b))

    return Fig
end

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 50, y = 50)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; compressible=false)

    # Parameters
    params_bg = (ρ=1.0, n=1.0, η0=1e0, G=1e6)
    params_in = (ρ=1.0, n=1.0, η0=1e5, G=1e6)

    materials.g .= [0., 0.]
    materials.ρ .= [params_bg.ρ, params_in.ρ]
    materials.n .= [params_bg.n, params_in.n]
    materials.η0 .= [params_bg.η0, params_in.η0]
    materials.G .= [params_bg.G, params_in.G]

    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Solver parameters
    iter_params = IterParams(niter=2, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c]  .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Set material geometry
    rad = 0.1 + 1e-13
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= rad^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= rad^2] .= 2
    Set_PhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    Fig = nothing
    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Post-processing: grid -> marker stress interpolation, bilinear vs. equilibrium projection
        Fig = marker_stress_comparison(a, nc, Δ, L, rad; nmark=4, w_mixed=1.0)
        display(Fig)

    end

    display(to)

    return Fig
end


let
    # # Boundary condition templates
    BCs = [
        :free_slip,
    ]

    # Boundary velocity gradient matrix
    er    = -1
    # ∂𝐕∂𝐱 - velocity gradient tensor
    D_BCs = [
        @SMatrix( [er 0;        #    ∂Vx∂x ∂Vx∂y
                   0 -er] ),    #    ∂Vy∂x ∂Vy∂y  div(V) = 0 = ∂Vx∂x + ∂Vy∂y --> ∂Vy∂y = - ∂Vx∂x
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end
