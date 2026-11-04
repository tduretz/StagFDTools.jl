# Interface geometry for the immersed interface method (IIM).
#
# The interface Γ is the zero level of φ (φ < 0 inside, e.g. the inclusion).
# φ is sampled on the four staggered grids (with ghosts, as in GenerateGrid):
#   φ.x -> Vx nodes (size_x), φ.y -> Vy nodes (size_y), φ.c -> centroids (size_c), φ.v -> vertices (size_v)
#
# Together these grids form a single lattice of spacing Δ/2 (the "fine lattice"):
#   (I, J) odd/odd  -> vertex    v_e[(I+1)/2, (J+1)/2]
#   (I, J) even/even -> centroid c_e[I/2,     J/2    ]
#   (I, J) odd/even -> Vx        vx_e[(I+1)/2, J/2+1 ]
#   (I, J) even/odd -> Vy        vy_e[I/2+1,  (J+1)/2]
# Every interface point needed by the IIM stencils lies on an edge of this lattice.

side(φ) = φ < 0 ? -1 : 1

"""
    LevelSet(φfun, X; tol=1e-10)

Sample the level set function `φfun(x, y)` on the extended grids of `X` (see `GenerateGrid`).
Nodes lying on Γ (|φ| < tol*Δ) are moved to the φ > 0 side, which avoids zero-width slivers.
"""
function LevelSet(φfun, X; tol=1e-10)
    ϵ = tol * min(step(X.v.x), step(X.v.y))
    snap(φ) = abs(φ) < ϵ ? ϵ : φ
    return (
        x = [snap(φfun(x, y)) for x in X.vx_e.x, y in X.vx_e.y],
        y = [snap(φfun(x, y)) for x in X.vy_e.x, y in X.vy_e.y],
        c = [snap(φfun(x, y)) for x in X.c_e.x,  y in X.c_e.y ],
        v = [snap(φfun(x, y)) for x in X.v_e.x,  y in X.v_e.y ],
    )
end

"""
    FineLattice(φ, X, Δ)

Merge the four staggered level set arrays into one array `Φ` of spacing Δ/2, with coordinates `x`, `y`.
"""
function FineLattice(φ, X, Δ)
    nI, nJ = 2 * size(φ.v, 1) - 1, 2 * size(φ.v, 2) - 1
    Φ = zeros(nI, nJ)
    for j in axes(φ.v, 2), i in axes(φ.v, 1)
        Φ[2i-1, 2j-1] = φ.v[i, j]
    end
    for j in axes(φ.c, 2), i in axes(φ.c, 1)
        Φ[2i, 2j] = φ.c[i, j]
    end
    for j in axes(φ.x, 2), i in axes(φ.x, 1)
        J = 2j - 2
        1 <= J <= nJ && (Φ[2i-1, J] = φ.x[i, j])
    end
    for j in axes(φ.y, 2), i in axes(φ.y, 1)
        I = 2i - 2
        1 <= I <= nI && (Φ[I, 2j-1] = φ.y[i, j])
    end
    x = LinRange(X.v_e.x[1], X.v_e.x[end], nI)
    y = LinRange(X.v_e.y[1], X.v_e.y[end], nJ)
    return (Φ=Φ, x=x, y=y, Δ=(x=Δ.x / 2, y=Δ.y / 2))
end

# Lagrange polynomial through (t[k], f[k]) evaluated at s
function lagrange(t, f, s)
    p = zero(s)
    for k in eachindex(t)
        l = one(s)
        for m in eachindex(t)
            m == k || (l *= (s - t[m]) / (t[k] - t[m]))
        end
        p += f[k] * l
    end
    return p
end

"""
    crossing_fraction(Φline, k)

Position t ∈ [0, 1] of the root of φ between samples `k` and `k+1` of the line `Φline`
(sample spacing Δ/2). Uses the cubic through samples k-1:k+2 (quadratic at the ends of the line).
"""
function crossing_fraction(Φline, k)
    ks = max(k - 1, firstindex(Φline)):min(k + 2, lastindex(Φline))
    t  = ks .- k
    f  = Φline[ks]
    lo, hi = 0.0, 1.0
    slo = side(Φline[k])
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        side(lagrange(t, f, mid)) == slo ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

# Gradient of Φ at fine node (I, J): central differences, one-sided on the lattice edges
function ∇Φ(fl, I, J)
    Φ = fl.Φ
    Im, Ip = max(I - 1, 1), min(I + 1, size(Φ, 1))
    Jm, Jp = max(J - 1, 1), min(J + 1, size(Φ, 2))
    return SVector((Φ[Ip, J] - Φ[Im, J]) / ((Ip - Im) * fl.Δ.x),
                   (Φ[I, Jp] - Φ[I, Jm]) / ((Jp - Jm) * fl.Δ.y))
end

# Curvature κ = ∇·(∇φ/|∇φ|) at fine node (I, J) by central differences (κ > 0 for a convex φ < 0 region)
function κΦ(fl, I, J)
    Φ  = fl.Φ
    I  = clamp(I, 2, size(Φ, 1) - 1)
    J  = clamp(J, 2, size(Φ, 2) - 1)
    hx, hy = fl.Δ.x, fl.Δ.y
    φx  = (Φ[I+1, J] - Φ[I-1, J]) / (2hx)
    φy  = (Φ[I, J+1] - Φ[I, J-1]) / (2hy)
    φxx = (Φ[I+1, J] - 2Φ[I, J] + Φ[I-1, J]) / hx^2
    φyy = (Φ[I, J+1] - 2Φ[I, J] + Φ[I, J-1]) / hy^2
    φxy = (Φ[I+1, J+1] - Φ[I+1, J-1] - Φ[I-1, J+1] + Φ[I-1, J-1]) / (4hx * hy)
    return (φxx * φy^2 - 2φx * φy * φxy + φyy * φx^2) / (φx^2 + φy^2)^(3 / 2)
end

"""
    InterfaceCrossings(fl)

Intersections of Γ with the edges of the fine lattice `fl`:
- `tx[I, J]`: fraction along the edge (I, J) → (I+1, J), `NaN` if not crossed
- `ty[I, J]`: fraction along the edge (I, J) → (I, J+1), `NaN` if not crossed
- `points`: list of (x, n, κ, I, J, dir): position, unit normal (pointing to φ > 0), curvature and edge
"""
function InterfaceCrossings(fl)
    Φ = fl.Φ
    nI, nJ = size(Φ)
    tx = fill(NaN, nI - 1, nJ)
    ty = fill(NaN, nI, nJ - 1)
    points = NamedTuple{(:x, :n, :κ, :I, :J, :dir),Tuple{SVector{2,Float64},SVector{2,Float64},Float64,Int,Int,Symbol}}[]
    for J in 1:nJ, I in 1:(nI-1)
        if side(Φ[I, J]) != side(Φ[I+1, J])
            t = crossing_fraction(view(Φ, :, J), I)
            tx[I, J] = t
            xΓ = SVector(fl.x[I] + t * fl.Δ.x, fl.y[J])
            n  = normalize((1 - t) * ∇Φ(fl, I, J) + t * ∇Φ(fl, I + 1, J))
            κ  = (1 - t) * κΦ(fl, I, J) + t * κΦ(fl, I + 1, J)
            push!(points, (x=xΓ, n=n, κ=κ, I=I, J=J, dir=:x))
        end
    end
    for J in 1:(nJ-1), I in 1:nI
        if side(Φ[I, J]) != side(Φ[I, J+1])
            t = crossing_fraction(view(Φ, I, :), J)
            ty[I, J] = t
            xΓ = SVector(fl.x[I], fl.y[J] + t * fl.Δ.y)
            n  = normalize((1 - t) * ∇Φ(fl, I, J) + t * ∇Φ(fl, I, J + 1))
            κ  = (1 - t) * κΦ(fl, I, J) + t * κΦ(fl, I, J + 1)
            push!(points, (x=xΓ, n=n, κ=κ, I=I, J=J, dir=:y))
        end
    end
    return (tx=tx, ty=ty, points=points)
end

"""
    IrregularNodes(φ, nc)

Flag the Vx, Vy and Pt equations whose standard stencil has members on both sides of Γ
(these receive IIM corrections). Index conventions follow `Stokes.jl`:
- Vx[i,j]: Vx[i±1,j], Vx[i,j±1], Pt[i-1:i, j-1], Vy[i:i+1, j-1:j]
- Vy[i,j]: Vy[i±1,j], Vy[i,j±1], Pt[i-1, j-1:j], Vx[i-1:i, j:j+1]
- Pt[i,j]: Vx[i:i+1, j+1], Vy[i+1, j:j+1]
"""
function IrregularNodes(φ, nc)
    inx_Vx, iny_Vx = 2:(nc.x+2), 3:(nc.y+2)
    inx_Vy, iny_Vy = 3:(nc.x+2), 2:(nc.y+2)
    inx_c,  iny_c  = 2:(nc.x+1), 2:(nc.y+1)
    irr = (x = falses(size(φ.x)), y = falses(size(φ.y)), c = falses(size(φ.c)))

    for j in iny_Vx, i in inx_Vx
        s = side(φ.x[i, j])
        irr.x[i, j] = any(side(φ.x[ii, jj]) != s for (ii, jj) in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))) ||
                      any(side(φ.c[ii, j-1]) != s for ii in (i-1):i) ||
                      any(side(φ.y[ii, jj]) != s for ii in i:(i+1), jj in (j-1):j)
    end
    for j in iny_Vy, i in inx_Vy
        s = side(φ.y[i, j])
        irr.y[i, j] = any(side(φ.y[ii, jj]) != s for (ii, jj) in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))) ||
                      any(side(φ.c[i-1, jj]) != s for jj in (j-1):j) ||
                      any(side(φ.x[ii, jj]) != s for ii in (i-1):i, jj in j:(j+1))
    end
    for j in iny_c, i in inx_c
        s = side(φ.c[i, j])
        irr.c[i, j] = any(side(φ.x[ii, j+1]) != s for ii in i:(i+1)) ||
                      any(side(φ.y[i+1, jj]) != s for jj in j:(j+1))
    end
    return irr
end

"""
    MultipleCrossings(Γ)

Count fine-lattice segments of length Δ (two consecutive edges) crossed twice by Γ.
These are features thinner than ~Δ, not handled by the first-order IIM.
"""
function MultipleCrossings(Γ)
    n = 0
    for J in axes(Γ.tx, 2), I in 1:(size(Γ.tx, 1)-1)
        n += !isnan(Γ.tx[I, J]) && !isnan(Γ.tx[I+1, J])
    end
    for J in 1:(size(Γ.ty, 2)-1), I in axes(Γ.ty, 1)
        n += !isnan(Γ.ty[I, J]) && !isnan(Γ.ty[I, J+1])
    end
    return n
end
