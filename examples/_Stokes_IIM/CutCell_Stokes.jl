# STATUS (2026-11-11): work in progress, NOT usable yet. Exact for a straight layer (piecewise
# linear solution), but on curved interfaces the interface velocities v_Γ are under-determined:
# near-null modes made of v_Γ only (σ_min/σ_max ~ 1e-7 at contrast 1e-3, ~1e-19 before merging),
# giving O(1-100) velocity errors at nodes next to Γ (Schmid & Podladchikov test). The pressure
# converges in L1 (~O(h^0.7)) but not in L∞. Too many v_Γ unknowns for what the stiff-side
# pieces see (each P / vertex piece only sees net normal-weighted sums of v_Γ). See the notes at
# the end of the session report; candidate fixes: fewer v_Γ (one per cut pressure cell, linear
# along Γ), or eliminate v_Γ by a local two-sided reconstruction, or a CutFEM/Nitsche variant.
#
# Cut-cell finite volumes on the staggered (MAC) grid, with explicitly doubled unknowns, for
# incompressible Stokes with a piecewise-constant viscosity: ∇·σ = 0, ∇·v = 0, σ = -pI + 2ηε̇,
# [v] = 0, [σ·n] = 0 on Γ = {φ = 0}; Dirichlet velocities on the box.
#
# Every staggered control volume (pressure cells, Vx and Vy volumes, vertex volumes for σxy) that
# is cut by Γ is split into its connected phase pieces, found on one sub-lattice of spacing h/m
# shared by the four grids (so the pieces of all grids are consistent). Unknowns:
#   one vx per Vx piece, one vy per Vy piece, one p per pressure piece, and explicit interface
#   velocities, staggered like the grid: vx_Γ per pair of Vx pieces sharing Γ, vy_Γ per pair of Vy
#   pieces (a single interface value per vertex / pressure piece: no checkerboard along Γ).
# Discretisation (all local, nothing is extrapolated):
#   velocity gradients per piece by Green-Gauss over the piece boundary: face velocities from the
#     same-phase Vx/Vy pieces, v_Γ on Γ, Dirichlet values on walls
#   stresses: σxx, σyy (and p) on pressure pieces, σxy on vertex pieces
#   momentum of each Vx/Vy piece: Σ σ·n over its boundary, Γ included, with its own side's stresses
#   continuity of each pressure piece: ∮ v·n = 0 (Γ included, with v_Γ)
#   traction continuity σ⁻·n = σ⁺·n integrated over each segment (x on Vx segments, y on Vy ones)
# The interface enters through the staircase of the sub-lattice (exact normals in the mean).
using LinearAlgebra, SparseArrays, StaticArrays, Printf

# ---------------------------------------------------------------- linear forms
const LFd = Dict{Int,Float64}
mutable struct Form
    w::LFd
    c::Float64
end
Form() = Form(LFd(), 0.0)
unit(k) = Form(LFd(k => 1.0), 0.0)
function addto!(d::Form, α, s::Form)
    for (k, v) in s.w
        d.w[k] = get(d.w, k, 0.0) + α * v
    end
    d.c += α * s.c
    return d
end
addc!(d::Form, c) = (d.c += c; d)
scale(s::Form, α) = (d = Form(); addto!(d, α, s))

# ---------------------------------------------------------------- sub-lattice and pieces
# label the connected phase components of every block (krange, lrange) of the sub-lattice
function label_blocks(ph, blocks, xs, hs)
    L = zeros(Int32, size(ph))
    phase = Int[]; vol = Float64[]; xsum = SVector{2,Float64}[]
    stack = Tuple{Int,Int}[]
    for (kr, lr) in blocks
        for l in lr, k in kr
            L[k, l] != 0 && continue
            push!(phase, ph[k, l]); push!(vol, 0.0); push!(xsum, SVector(0.0, 0.0))
            id = length(phase)
            L[k, l] = id; push!(stack, (k, l))
            while !isempty(stack)
                (p, q) = pop!(stack)
                vol[id] += hs^2; xsum[id] += SVector(xs[p], xs[q]) * hs^2
                for (dp, dq) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    p2, q2 = p + dp, q + dq
                    (p2 in kr && q2 in lr) || continue
                    (L[p2, q2] == 0 && ph[p2, q2] == ph[k, l]) || continue
                    L[p2, q2] = id; push!(stack, (p2, q2))
                end
            end
        end
    end
    return (L=L, phase=phase, vol=vol, xc=[xsum[i] / vol[i] for i in eachindex(vol)])
end

# Cell merging by relabelling: pieces smaller than θ h² are absorbed into the same-phase
# neighbouring piece (other volume) with which they share the longest boundary. Labels are then
# compacted and the piece data recomputed.
function merge_small(Lb, ph, xs, hs, h; θ=0.5, cvvol=(k, l) -> h^2)
    L = copy(Lb.L)
    M = size(L, 1)
    for pass in 1:6
        n   = maximum(L)
        vol = zeros(n); full = zeros(n)
        for l in 1:M, k in 1:M
            L[k, l] > 0 || continue
            vol[L[k, l]] += hs^2; full[L[k, l]] = cvvol(k, l)
        end
        small = [k for k in 1:n if 0 < vol[k] < θ * full[k]]
        isempty(small) && break
        share = Dict{NTuple{2,Int},Int}()
        for l in 1:M, k in 1:M, (k2, l2) in ((k + 1, l), (k, l + 1))
            (k2 <= M && l2 <= M) || continue
            a, b = L[k, l], L[k2, l2]
            (a > 0 && b > 0 && a != b && ph[k, l] == ph[k2, l2]) || continue
            share[(a, b)] = get(share, (a, b), 0) + 1; share[(b, a)] = get(share, (b, a), 0) + 1
        end
        target = collect(1:n)
        for q in sort(small, by=k -> vol[k])
            best, bs = 0, 0
            for ((a, b), c) in share
                a == q || continue
                target[b] != b && continue           # do not merge into a piece being merged
                if c > bs || (c == bs && vol[b] > vol[best])
                    best = b; bs = c
                end
            end
            best > 0 && target[best] == best && (target[q] = best)
        end
        changed = false
        for I in eachindex(L)
            L[I] > 0 && target[L[I]] != L[I] && (L[I] = target[L[I]]; changed = true)
        end
        changed || break
    end
    # compact
    ids = sort(unique(L[L .> 0]))
    newid = Dict(k => i for (i, k) in enumerate(ids))
    n = length(ids)
    phase = zeros(Int, n); vol = zeros(n); xsum = [SVector(0.0, 0.0) for _ in 1:n]
    for l in 1:M, k in 1:M
        L[k, l] == 0 && continue
        q = newid[L[k, l]]; L[k, l] = q
        phase[q] = ph[k, l]; vol[q] += hs^2; xsum[q] += SVector(xs[k], xs[l]) * hs^2
    end
    return (L=L, phase=phase, vol=vol, xc=[xsum[i] / vol[i] for i in 1:n])
end

"""
    CutCellStokes(φ, η, N, vbc; L=1.0, m=8)

Solve on [-L/2, L/2]² with N×N cells. `φ(x)` level set (φ < 0: phase `neg`), `η = (neg, pos)`,
`vbc(x)` Dirichlet velocity on the walls. Returns pressure at cell centres, velocities at the MAC
nodes, and the pieces.
"""
function CutCellStokes(φ, η, N, vbc; L=1.0, m=8, θ=0.5, vsplit=false, dosolve=true)
    @assert iseven(m)
    h, hs = L / N, L / N / m
    M  = N * m
    xs = [-L / 2 + (k - 0.5) * hs for k in 1:M]
    ph = [φ(SVector(xs[k], xs[l])) < 0 ? -1 : 1 for k in 1:M, l in 1:M]
    ηs(s) = s < 0 ? η.neg : η.pos
    cellr(I) = ((I-1)*m+1):(I*m)
    noder(i) = max(1, (i - 1) * m - m ÷ 2 + 1):min(M, (i - 1) * m + m ÷ 2)
    P  = label_blocks(ph, [(cellr(I), cellr(J)) for J in 1:N for I in 1:N], xs, hs)
    # velocities: one unknown per Vx/Vy volume (continuous velocity) unless vsplit
    phv = vsplit ? ph : ones(Int, size(ph))
    VX = label_blocks(phv, [(noder(i), cellr(J)) for J in 1:N for i in 2:N], xs, hs)
    VY = label_blocks(phv, [(cellr(I), noder(j)) for j in 2:N for I in 1:N], xs, hs)
    V  = label_blocks(ph, [(noder(i), noder(j)) for j in 1:N+1 for i in 1:N+1], xs, hs)
    if θ > 0
        wv(i) = (i == 1 || i == N + 1) ? h / 2 : h                 # clipped vertex volumes at walls
        iv(k) = div(k - 1 + m ÷ 2, m) + 1
        P  = merge_small(P, ph, xs, hs, h; θ)
        vsplit && (VX = merge_small(VX, ph, xs, hs, h; θ); VY = merge_small(VY, ph, xs, hs, h; θ))
        V  = merge_small(V, ph, xs, hs, h; θ, cvvol=(k, l) -> wv(iv(k)) * wv(iv(l)))
    end
    nX, nY, nP = length(VX.phase), length(VY.phase), length(P.phase)
    # interface segments, staggered like the velocities. Every Γ sub-edge is assigned to the Vx
    # volume (and the Vy volume) of its φ < 0 sub-point; a segment is a connected set of Γ
    # sub-edges (sharing corners) assigned to the same volume. vx_Γ lives on Vx segments, vy_Γ on
    # Vy segments; Γ in a wall half volume takes the Dirichlet value.
    cvx(k, l) = (i = div(k - 1 + m ÷ 2, m) + 1; 2 <= i <= N ? (i, div(l - 1, m) + 1) : nothing)
    cvy(k, l) = (j = div(l - 1 + m ÷ 2, m) + 1; 2 <= j <= N ? (div(k - 1, m) + 1, j) : nothing)
    gedges = Tuple{NTuple{2,Int},NTuple{2,Int},Int}[]      # (a, b, dir) with b = a + e_dir
    for l in 1:M, k in 1:M
        k < M && ph[k, l] != ph[k+1, l] && push!(gedges, ((k, l), (k + 1, l), 1))
        l < M && ph[k, l] != ph[k, l+1] && push!(gedges, ((k, l), (k, l + 1), 2))
    end
    corners(e) = (a = e[1]; e[3] == 1 ? ((a[1], a[2] - 1), (a[1], a[2])) : ((a[1] - 1, a[2]), (a[1], a[2])))
    function segments(cv)
        ne = length(gedges)
        parent = collect(1:ne)
        find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
        owner = [(a = e[1]; b = e[2]; cv((ph[a...] < 0 ? a : b)...)) for e in gedges]
        seen = Dict{Tuple{Any,NTuple{2,Int}},Int}()
        for (i, e) in enumerate(gedges)
            owner[i] === nothing && continue
            for c in corners(e)
                j = get!(seen, (owner[i], c), i)
                j != i && (parent[find(i)] = find(j))
            end
        end
        sid = zeros(Int, ne); roots = Dict{Int,Int}()
        for i in 1:ne
            owner[i] === nothing && continue
            sid[i] = get!(roots, find(i), length(roots) + 1)
        end
        return sid, length(roots)
    end
    sidX, nSx = segments(cvx)
    sidY, nSy = segments(cvy)
    eidx = Dict{Tuple{NTuple{2,Int},NTuple{2,Int}},Int}((e[1], e[2]) => i for (i, e) in enumerate(gedges))
    nS = nSx + nSy
    oX, oY, oP, oS = 0, nX, nX + nY, nX + nY + nP
    oSy = oS + nSx
    iλ = oS + nS + 1
    nU = iλ
    midpt(a, b) = SVector((xs[a[1]] + xs[b[1]]) / 2, (xs[a[2]] + xs[b[2]]) / 2)
    function vxΓ(a, b)
        g = sidX[eidx[(a, b)]]
        g == 0 ? Form(LFd(), vbc(midpt(a, b))[1]) : unit(oS + g)
    end
    function vyΓ(a, b)
        g = sidY[eidx[(a, b)]]
        g == 0 ? Form(LFd(), vbc(midpt(a, b))[2]) : unit(oSy + g)
    end
    rowX(a, b) = (g = sidX[eidx[(a, b)]]; g == 0 ? nothing : R[oS+g])
    rowY(a, b) = (g = sidY[eidx[(a, b)]]; g == 0 ? nothing : R[oSy+g])
    # face velocity at a sub-point: the piece of its Vx/Vy volume, or the wall value (half volumes at walls)
    vxat(k, l) = VX.L[k, l] > 0 ? unit(oX + VX.L[k, l]) : Form(LFd(), vbc(SVector(xs[k], xs[l]))[1])
    vyat(k, l) = VY.L[k, l] > 0 ? unit(oY + VY.L[k, l]) : Form(LFd(), vbc(SVector(xs[k], xs[l]))[2])

    # ---- Green-Gauss gradients: pressure pieces ∂x vx, ∂y vy; vertex pieces ∂y vx, ∂x vy
    Dxx = [Form() for _ in 1:nP]; Dyy = [Form() for _ in 1:nP]
    Dyx = [Form() for _ in 1:length(V.phase)]; Dxy = [Form() for _ in 1:length(V.phase)]
    for l in 1:M, k in 1:M
        if k < M          # sub-edge between (k,l) and (k+1,l), normal +x
            a, b = (k, l), (k + 1, l)
            if ph[a...] != ph[b...]
                fx, fy = vxΓ(a, b), vyΓ(a, b)
                addto!(Dxx[P.L[a...]], hs, fx); addto!(Dxx[P.L[b...]], -hs, fx)
                addto!(Dxy[V.L[a...]], hs, fy); addto!(Dxy[V.L[b...]], -hs, fy)
            else
                if P.L[a...] != P.L[b...]      # pressure-cell face: vx node line
                    f = vxat(k, l); addto!(Dxx[P.L[a...]], hs, f); addto!(Dxx[P.L[b...]], -hs, f)
                end
                if V.L[a...] != V.L[b...]      # vertex-volume face: vy node line
                    f = vyat(k, l); addto!(Dxy[V.L[a...]], hs, f); addto!(Dxy[V.L[b...]], -hs, f)
                end
            end
        end
        if l < M          # sub-edge between (k,l) and (k,l+1), normal +y
            a, b = (k, l), (k, l + 1)
            if ph[a...] != ph[b...]
                fx, fy = vxΓ(a, b), vyΓ(a, b)
                addto!(Dyy[P.L[a...]], hs, fy); addto!(Dyy[P.L[b...]], -hs, fy)
                addto!(Dyx[V.L[a...]], hs, fx); addto!(Dyx[V.L[b...]], -hs, fx)
            else
                if P.L[a...] != P.L[b...]
                    f = vyat(k, l); addto!(Dyy[P.L[a...]], hs, f); addto!(Dyy[P.L[b...]], -hs, f)
                end
                if V.L[a...] != V.L[b...]
                    f = vxat(k, l); addto!(Dyx[V.L[a...]], hs, f); addto!(Dyx[V.L[b...]], -hs, f)
                end
            end
        end
    end
    # walls. Tangential wall velocities seen by the vertex pieces follow the interior convention:
    # a Vy (Vx) piece value acts at its node line y = y_v (x = x_v) if it touches it, otherwise
    # at its centroid; the wall value is taken at the same height (abscissa).
    nodeYv = fill(NaN, nY); nodeXv = fill(NaN, nX)
    for l in 1:M, k in 1:M
        l % m == 0 && l < M && VY.L[k, l] > 0 && (nodeYv[VY.L[k, l]] = -L / 2 + (l ÷ m) * h)
        k % m == 0 && k < M && VX.L[k, l] > 0 && (nodeXv[VX.L[k, l]] = -L / 2 + (k ÷ m) * h)
    end
    wally(k, l, xw) = (q = VY.L[k, l]; q == 0 ? vbc(SVector(xw, xs[l]))[2] :
                       vbc(SVector(xw, isnan(nodeYv[q]) ? VY.xc[q][2] : nodeYv[q]))[2])
    wallx(k, l, yw) = (q = VX.L[k, l]; q == 0 ? vbc(SVector(xs[k], yw))[1] :
                       vbc(SVector(isnan(nodeXv[q]) ? VX.xc[q][1] : nodeXv[q], yw))[1])
    for t in 1:M
        xl, xr, yb, yt = SVector(-L / 2, xs[t]), SVector(L / 2, xs[t]), SVector(xs[t], -L / 2), SVector(xs[t], L / 2)
        addc!(Dxx[P.L[1, t]], -hs * vbc(xl)[1]); addc!(Dxx[P.L[M, t]], hs * vbc(xr)[1])
        addc!(Dyy[P.L[t, 1]], -hs * vbc(yb)[2]); addc!(Dyy[P.L[t, M]], hs * vbc(yt)[2])
        addc!(Dxy[V.L[1, t]], -hs * wally(1, t, -L / 2)); addc!(Dxy[V.L[M, t]], hs * wally(M, t, L / 2))
        addc!(Dyx[V.L[t, 1]], -hs * wallx(t, 1, -L / 2)); addc!(Dyx[V.L[t, M]], hs * wallx(t, M, L / 2))
    end
    # ---- stresses
    Sxx = Vector{Form}(undef, nP); Syy = Vector{Form}(undef, nP)
    for k in 1:nP
        e = 2ηs(P.phase[k]) / P.vol[k]
        Sxx[k] = addto!(scale(Dxx[k], e), -1.0, unit(oP + k))
        Syy[k] = addto!(scale(Dyy[k], e), -1.0, unit(oP + k))
    end
    Sxy = [addto!(scale(Dyx[k], ηs(V.phase[k]) / V.vol[k]), ηs(V.phase[k]) / V.vol[k], Dxy[k]) for k in eachindex(V.phase)]
    # ---- equations
    R = [Form() for _ in 1:nU]
    rX(q) = R[oX+q]; rY(q) = R[oY+q]; rP(q) = R[oP+q]
    for l in 1:M, k in 1:M
        if k < M      # normal +x: x-momentum uses σxx (pressure pieces), y-momentum σxy (vertex pieces)
            a, b = (k, l), (k + 1, l)
            Γ = ph[a...] != ph[b...]
            Γv = Γ && vsplit
            qa, qb = VX.L[a...], VX.L[b...]
            if qa != qb || Γv
                qa > 0 && addto!(rX(qa), hs, Sxx[P.L[a...]])
                qb > 0 && addto!(rX(qb), -hs, Sxx[P.L[b...]])
            end
            qa, qb = VY.L[a...], VY.L[b...]
            if qa != qb || Γv
                qa > 0 && addto!(rY(qa), hs, Sxy[V.L[a...]])
                qb > 0 && addto!(rY(qb), -hs, Sxy[V.L[b...]])
            end
            if Γ          # traction continuity: x on the Vx segment, y on the Vy segment
                r = rowX(a, b); r === nothing || (addto!(r, hs, Sxx[P.L[a...]]); addto!(r, -hs, Sxx[P.L[b...]]))
                r = rowY(a, b); r === nothing || (addto!(r, hs, Sxy[V.L[a...]]); addto!(r, -hs, Sxy[V.L[b...]]))
            end
        end
        if l < M      # normal +y
            a, b = (k, l), (k, l + 1)
            Γ = ph[a...] != ph[b...]
            Γv = Γ && vsplit
            qa, qb = VX.L[a...], VX.L[b...]
            if qa != qb || Γv
                qa > 0 && addto!(rX(qa), hs, Sxy[V.L[a...]])
                qb > 0 && addto!(rX(qb), -hs, Sxy[V.L[b...]])
            end
            qa, qb = VY.L[a...], VY.L[b...]
            if qa != qb || Γv
                qa > 0 && addto!(rY(qa), hs, Syy[P.L[a...]])
                qb > 0 && addto!(rY(qb), -hs, Syy[P.L[b...]])
            end
            if Γ
                r = rowX(a, b); r === nothing || (addto!(r, hs, Sxy[V.L[a...]]); addto!(r, -hs, Sxy[V.L[b...]]))
                r = rowY(a, b); r === nothing || (addto!(r, hs, Syy[P.L[a...]]); addto!(r, -hs, Syy[P.L[b...]]))
            end
        end
    end
    for t in 1:M      # Vx/Vy volumes touching the walls (σxy on top/bottom for Vx, left/right for Vy)
        VX.L[t, 1] > 0 && addto!(rX(VX.L[t, 1]), -hs, Sxy[V.L[t, 1]])
        VX.L[t, M] > 0 && addto!(rX(VX.L[t, M]), hs, Sxy[V.L[t, M]])
        VY.L[1, t] > 0 && addto!(rY(VY.L[1, t]), -hs, Sxy[V.L[1, t]])
        VY.L[M, t] > 0 && addto!(rY(VY.L[M, t]), hs, Sxy[V.L[M, t]])
    end
    for k in 1:nP     # continuity (Green-Gauss flux), mean pressure by a Lagrange multiplier
        addto!(rP(k), 1.0, Dxx[k]); addto!(rP(k), 1.0, Dyy[k])
        addto!(rP(k), P.vol[k], unit(iλ))
        addto!(R[iλ], P.vol[k], unit(oP + k))
    end
    # ---- merge velocity slivers: a Vx/Vy piece in a corner of its volume that touches neither
    # centre line of the volume is never used as a face velocity (empty column). Its momentum
    # balance is added to a same-phase neighbouring piece it touches, and its unknown is dropped.
    used = falses(nU)
    for r in R, k in keys(r.w)
        used[k] = true
    end
    nmerged = 0
    for (Lab, off, n) in ((VX.L, oX, nX), (VY.L, oY, nY))
        nbr = [Int[] for _ in 1:n]
        for l in 1:M, k in 1:M, (k2, l2) in ((k + 1, l), (k, l + 1))
            (k2 <= M && l2 <= M) || continue
            a, b = Lab[k, l], Lab[k2, l2]
            (a > 0 && b > 0 && a != b && ph[k, l] == ph[k2, l2]) || continue
            push!(nbr[a], b); push!(nbr[b], a)
        end
        for _ in 1:3, q in 1:n                        # a few passes (chains of slivers)
            used[off+q] && continue
            isempty(R[off+q].w) && R[off+q].c == 0 && continue
            host = findfirst(j -> used[off+j], nbr[q])
            host === nothing && continue
            addto!(R[off+nbr[q][host]], 1.0, R[off+q])
            R[off+q] = unit(off + q)
            used[off+q] = true
            nmerged += 1
        end
        for q in 1:n                                  # isolated slivers (no host): drop
            used[off+q] && continue
            R[off+q] = unit(off + q); used[off+q] = true
        end
    end
    # ---- assemble and solve
    Is, Js, Vs = Int[], Int[], Float64[]
    b = zeros(nU)
    for r in 1:nU
        for (c, v) in R[r].w
            push!(Is, r); push!(Js, c); push!(Vs, v)
        end
        b[r] = -R[r].c
    end
    A = sparse(Is, Js, Vs, nU, nU)
    dosolve || return (A=A, b=b, off=(X=oX, Y=oY, P=oP, S=oS, Sy=oSy), nX=nX, nY=nY, nP=nP, nS=nS, nSx=nSx, gedges=gedges, sidX=sidX, sidY=sidY, P=P, VX=VX, VY=VY, V=V, xs=xs, ph=ph, nmerged=nmerged)
    u = A \ b
    # ---- output on the MAC nodes / cell centres (piece of the sub-point on the node's side)
    pick(Lab, k1, l1, k2, l2, x) = (s = φ(x) < 0 ? -1 : 1; phv[k1, l1] == s || !vsplit ? Lab[k1, l1] : Lab[k2, l2])
    c = m ÷ 2
    pc = [u[oP+P.L[(I-1)*m+c, (J-1)*m+c]] for I in 1:N, J in 1:N]
    vx = zeros(N + 1, N); vy = zeros(N, N + 1)
    for J in 1:N, i in 1:N+1
        x = SVector(-L / 2 + (i - 1) * h, -L / 2 + (J - 0.5) * h)
        if i == 1 || i == N + 1
            vx[i, J] = vbc(x)[1]
        else
            k = (i - 1) * m; l = (J - 1) * m + c
            vx[i, J] = u[oX+pick(VX.L, k, l, k + 1, l, x)]
        end
    end
    for j in 1:N+1, I in 1:N
        x = SVector(-L / 2 + (I - 0.5) * h, -L / 2 + (j - 1) * h)
        if j == 1 || j == N + 1
            vy[I, j] = vbc(x)[2]
        else
            k = (I - 1) * m + c; l = (j - 1) * m
            vy[I, j] = u[oY+pick(VY.L, k, l, k, l + 1, x)]
        end
    end
    xc = [-L / 2 + (I - 0.5) * h for I in 1:N]
    xv = [-L / 2 + (i - 1) * h for i in 1:N+1]
    return (p=pc, vx=vx, vy=vy, xc=xc, xv=xv, u=u, A=A, nP=nP, nX=nX, nY=nY, nS=nS, P=P, VX=VX, VY=VY, V=V,
            off=(X=oX, Y=oY, P=oP, S=oS))
end
