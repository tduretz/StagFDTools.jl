# Staggered FD with TWO pressures in the cells cut by the interface (minimal sharp enrichment).
#
# A cut cell whose faces lie on both sides of Γ carries p⁻ and p⁺. Closure without any viscosity
# (hence no contrast amplification): each sub-volume conserves mass,
#   ∮_{∂V^s} v·n = Σ_partial faces ±ℓ v_n(midpoint) + v(Γ midpoint)·N_Γ^s = 0,     s = −, +
# with velocities interpolated (bilinear) at the midpoints and N_Γ^s = −Σ (outward partial face
# vector areas): exact for linear v and straight Γ.
# Momentum (finite-volume reading): the face of the Vx (Vy) control volume through a split centroid
# is the vertical (horizontal) segment of length Δ; the pressure force on it is ℓ⁻p⁻ + ℓ⁺p⁺ with
# ℓ^s the fractions of that segment. This couples p⁻ and p⁺ across Γ (normal force balance of the
# straddling control volumes); taking only "the pressure of the node's side" leaves the jump
# undetermined (one free constant per phase). Viscous part: effective (averaged) tensor.
include(joinpath(@__DIR__, "FD_Anisotropic.jl"))

# Geometry of the two sub-volumes of the cell at xc: for each side s, the partial faces
# (field, outward sign, length, midpoint) and the vector area of Γ seen from the φ < 0 side, and
# the midpoint of the Γ segment. Exact fluxes for linear v and a straight Γ. `nothing` unless Γ
# crosses the cell boundary exactly twice.
function subcell_geometry(φfun, xc, Δ)
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    SW, SE, NE, NW = xc - ex - ey, xc + ex - ey, xc + ex + ey, xc - ex + ey
    φv(x) = φfun(x[1], x[2])
    terms = Dict(-1 => Tuple{Symbol,Float64,Float64,SVector{2,Float64}}[], 1 => Tuple{Symbol,Float64,Float64,SVector{2,Float64}}[])
    cross = SVector{2,Float64}[]
    for (A, B, f, sg) in ((SE, NE, :x, 1.0), (SW, NW, :x, -1.0), (NW, NE, :y, 1.0), (SW, SE, :y, -1.0))
        sA, sB, L = side(φv(A)), side(φv(B)), norm(B - A)
        if sA == sB
            push!(terms[sA], (f, sg, L, (A + B) / 2))
        else
            a, b = 0.0, 1.0
            for _ in 1:50
                t = (a + b) / 2; side(φv(A + t * (B - A))) == sA ? (a = t) : (b = t)
            end
            t = (a + b) / 2; c = A + t * (B - A); push!(cross, c)
            push!(terms[sA], (f, sg, t * L, (A + c) / 2)); push!(terms[sB], (f, sg, (1 - t) * L, (c + B) / 2))
        end
    end
    length(cross) == 2 || return nothing
    # Σ outward vector areas of the closed sub-volume = 0  ->  Γ vector area seen from the φ<0 side
    Nneg = -sum((f == :x ? SVector(sg * L, 0.0) : SVector(0.0, sg * L)) for (f, sg, L, _) in terms[-1]; init=SVector(0.0, 0.0))
    return (terms=terms, mΓ=(cross[1] + cross[2]) / 2, Nneg=Nneg)
end

# fraction of the segment A→B on the φ < 0 side (at most one crossing)
function segfrac(φfun, A, B)
    sA, sB = side(φfun(A[1], A[2])), side(φfun(B[1], B[2]))
    sA == sB && return sA < 0 ? 1.0 : 0.0
    a, b = 0.0, 1.0
    for _ in 1:50
        t = (a + b) / 2; x = A + t * (B - A); side(φfun(x[1], x[2])) == sA ? (a = t) : (b = t)
    end
    t = (a + b) / 2
    return sA < 0 ? t : 1 - t
end

# bilinear interpolation of a nodal accessor acc(position) on the grid g = (x, y) at x
function interp(acc, g, Δ, x)
    ξ, ζ = (x[1] - g.x[1]) / Δ.x, (x[2] - g.y[1]) / Δ.y
    i, j = floor(Int, ξ), floor(Int, ζ); a, b = ξ - i, ζ - j
    P(k, l) = SVector(g.x[1] + (i + k) * Δ.x, g.y[1] + (j + l) * Δ.y)
    return (1 - a) * (1 - b) * acc(P(0, 0)) + a * (1 - b) * acc(P(1, 0)) + (1 - a) * b * acc(P(0, 1)) + a * b * acc(P(1, 1))
end

function SolveFDTwoP(φfun, X, Δ, η, nc, vbc; avg=:arith, avg_v=avg, m=10, visc=:avg, split=true, θ=0.0)
    size_x, size_y = size(vbc.x), size(vbc.y)
    size_c = (length(X.c_e.x), length(X.c_e.y))
    type   = Fields(fill(:out, size_x...), fill(:out, size_y...), fill(:out, size_c...))
    number = Fields(zeros(Int, size_x...), zeros(Int, size_y...), zeros(Int, size_c...))
    set_boundaries_template!(type, :all_Dirichlet, nc)
    Numbering!(number, type, nc)
    nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
    sd(x) = side(φfun(x[1], x[2]))
    ex, ey = SVector(Δ.x / 2, 0.0), SVector(0.0, Δ.y / 2)
    # split cells: faces on both sides of Γ
    qnum = Dict{CartesianIndex{2},Int}(); geo = Dict{CartesianIndex{2},Any}()
    for I in CartesianIndices(number.Pt)
        (split && number.Pt[I] > 0) || continue
        xc = node_position(Val(:p), I, X)
        G = subcell_geometry(φfun, xc, Δ)
        G === nothing && continue
        lv, lh = segfrac(φfun, xc - ey, xc + ey), segfrac(φfun, xc - ex, xc + ex)
        (θ < lv < 1 - θ || θ < lh < 1 - θ) || continue        # q must enter some momentum equation
        sg = ((1:10) .- 0.5) ./ 10 .- 0.5
        fa = count(φfun(xc[1] + a * Δ.x, xc[2] + c * Δ.y) < 0 for a in sg, c in sg) / 100
        θ < fa < 1 - θ || continue                     # no small sub-volumes (agglomerated instead)
        qnum[I] = length(qnum) + 1; geo[I] = (G..., lv=lv, lh=lh)
    end
    nq   = length(qnum)
    off  = (x=0, y=nVx, p=nVx + nVy, q=nVx + nVy + nPt)
    ndof = nVx + nVy + nPt + nq + 1
    numof(f, I) = f == :q ? qnum[I] : (f == :x ? number.Vx : f == :y ? number.Vy : number.Pt)[I]
    val0 = (x=vbc.x, y=vbc.y)
    key(x) = (round(Int, 2x[1] / Δ.x), round(Int, 2x[2] / Δ.y))
    cc, cv = Dict{NTuple{2,Int},SMatrix{3,3,Float64,9}}(), Dict{NTuple{2,Int},SMatrix{3,3,Float64,9}}()
    Dc(x) = get!(() -> effective_tensor(φfun, x, Δ, η, avg; m), cc, key(x))
    Dv(x) = get!(() -> effective_tensor(φfun, x, Δ, η, avg_v; m), cv, key(x))
    Is, Js, Vs = Int[], Int[], Float64[]
    b = zeros(ndof)
    rows = [(f, I, numof(f, I)) for f in (:x, :y, :p) for I in CartesianIndices(f == :x ? number.Vx : f == :y ? number.Vy : number.Pt) if numof(f, I) > 0]
    append!(rows, [(:q, I, qnum[I]) for I in keys(qnum)])
    for (f0, I0, row) in rows
        x0 = node_position(Val(f0 == :q ? :p : f0), I0, X)
        loc = Dict{Tuple{Symbol,CartesianIndex{2}},Int}(); nodes = Tuple{Symbol,CartesianIndex{2}}[]
        slot(f, I) = get!(loc, (f, I)) do
            push!(nodes, (f, I)); length(nodes)
        end
        function kernel(u)
            vx(x) = u[slot(:x, node_index(:x, x, X, Δ))]
            vy(x) = u[slot(:y, node_index(:y, x, X, Δ))]
            if f0 in (:x, :y)
                s0 = sd(x0)
                function pacc(xc)
                    I = node_index(:p, xc, X, Δ)
                    haskey(qnum, I) || return u[slot(:p, I)]
                    ln = f0 == :x ? geo[I].lv : geo[I].lh            # φ<0 fraction of the CV face
                    pc, pq = u[slot(:p, I)], u[slot(:q, I)]          # centroid side, opposite side
                    pn, pp = sd(xc) < 0 ? (pc, pq) : (pq, pc)
                    return ln * pn + (1 - ln) * pp
                end
                Dcs = visc == :side ?
                      (xc -> haskey(qnum, node_index(:p, xc, X, Δ)) ? 2 * (s0 < 0 ? η.neg : η.pos) * SMatrix{3,3}(1.0I) : Dc(xc)) : Dc
                return resid(Val(f0), vx, vy, pacc, x0, Δ, Dcs, Dv)
            elseif !haskey(qnum, I0)
                return resid(Val(:p), vx, vy, nothing, x0, Δ, Dc, Dv)
            else   # sub-volume continuity, side of the centroid (row :p) or the opposite side (row :q)
                s  = f0 == :p ? sd(x0) : -sd(x0)
                G  = geo[I0]
                vi(f, x) = f == :x ? interp(vx, X.vx_e, Δ, x) : interp(vy, X.vy_e, Δ, x)
                vΓ = SVector(vi(:x, G.mΓ), vi(:y, G.mΓ))
                N  = s < 0 ? G.Nneg : -G.Nneg
                return (sum(sg * L * vi(f, m) for (f, sg, L, m) in G.terms[s]) + dot(vΓ, N)) / (Δ.x * Δ.y)
            end
        end
        kernel(zeros(64))                                   # discover the stencil
        u0 = [(f == :q || numof(f, I) > 0) ? 0.0 : val0[f][I] for (f, I) in nodes]
        g  = ForwardDiff.gradient(kernel, u0)
        r  = off[f0] + row
        b[r] -= kernel(u0)
        for (j, (f, I)) in enumerate(nodes)
            (f == :q || numof(f, I) > 0) && g[j] != 0 && (push!(Is, r); push!(Js, off[f] + numof(f, I)); push!(Vs, g[j]))
        end
        f0 == :p && (push!(Is, r, ndof); push!(Js, ndof, r); push!(Vs, 1.0, 1.0))
    end
    K   = sparse(Is, Js, Vs, ndof, ndof)
    sol = K \ b
    v   = (x=copy(vbc.x), y=copy(vbc.y))
    p   = zeros(size_c); q = fill(NaN, size_c)
    for (f, a) in ((:x, v.x), (:y, v.y), (:p, p)), I in CartesianIndices(a)
        numof(f, I) > 0 && (a[I] = sol[off[f] + numof(f, I)])
    end
    for (I, k) in qnum
        q[I] = sol[off.q + k]
    end
    return (v=v, p=p, q=q, K=K, nq=nq)
end
