# Cut-cell finite volumes with explicitly doubled unknowns for ∇·(β∇u) = f, [u] = 0, [β∂ₙu] = 0.
#
# Cell-centred grid on [-1, 1]², N×N cells. Every cut cell carries one unknown per connected phase
# piece (a cell crossed by a thin layer has three: matrix | layer | matrix). The geometry of the
# pieces (volume, centroid, face apertures, interface measure and normal) is obtained by sub-sampling
# the cut cells with m×m points (4-connected components per phase); uncut cells are a single piece.
#
# Fluxes (two-point, symmetric, M-matrix):
#   same phase across a cell face:  F = β A (u_b - u_a) / d,   A: aperture, d = (x_b - x_a)·e
#   across Γ (pieces a, b on both sides, same or neighbouring cells):
#                                   F = |Γ_ab| (u_b - u_a) / (d_a/β_a + d_b/β_b)
#     d_a, d_b: normal distances from the centroids to Γ. Flux continuity is built in and the series
#     coefficient is dominated by the soft side, so errors are not amplified by the contrast.
#   Dirichlet boundary: F = β A (u_bc - u_a) / d.
# Optional `tcorr`: deferred tangential correction of the interface flux, u_b - u_a is replaced by
# u_b - u_a - ∇u·(x_b - x_a)_t, the tangential gradient taken from a least-squares fit of the piece
# values of the soft side around the crossing (fixed-point iterations on the same SPD operator).
using LinearAlgebra, SparseArrays, StaticArrays, Printf

struct Pieces
    phase::Vector{Int}
    cell::Vector{NTuple{2,Int}}
    vol::Vector{Float64}
    xc::Vector{SVector{2,Float64}}
end

function cutcell_geometry(φ, N; m=16)
    h, hs = 2 / N, 2 / N / m
    sd(x) = φ(x) < 0 ? -1 : 1
    xsub(I, a) = -1 + (I - 1) * h + (a - 0.5) * hs
    phase = Int[]; cell = NTuple{2,Int}[]; vol = Float64[]; xsum = SVector{2,Float64}[]
    pid  = zeros(Int, N, N)                       # uncut cells: their piece
    labs = Dict{NTuple{2,Int},Matrix{Int}}()      # cut cells: piece of each sub-point
    for J in 1:N, I in 1:N
        xc = SVector(-1 + (I - 0.5) * h, -1 + (J - 0.5) * h)
        if abs(φ(xc)) > 1.2 * h
            push!(phase, sd(xc)); push!(cell, (I, J)); push!(vol, h^2); push!(xsum, xc * h^2)
            pid[I, J] = length(phase)
            continue
        end
        ph  = [sd(SVector(xsub(I, a), xsub(J, b))) for a in 1:m, b in 1:m]
        if all(==(ph[1]), ph)
            push!(phase, ph[1]); push!(cell, (I, J)); push!(vol, h^2); push!(xsum, xc * h^2)
            pid[I, J] = length(phase)
            continue
        end
        lab = zeros(Int, m, m)
        for b in 1:m, a in 1:m
            lab[a, b] != 0 && continue
            push!(phase, ph[a, b]); push!(cell, (I, J)); push!(vol, 0.0); push!(xsum, SVector(0.0, 0.0))
            k = length(phase)
            stack = [(a, b)]; lab[a, b] = k
            while !isempty(stack)
                (p, q) = pop!(stack)
                vol[k] += hs^2
                xsum[k] += SVector(xsub(I, p), xsub(J, q)) * hs^2
                for (dp, dq) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    p2, q2 = p + dp, q + dq
                    (1 <= p2 <= m && 1 <= q2 <= m) || continue
                    (lab[p2, q2] == 0 && ph[p2, q2] == ph[a, b]) || continue
                    lab[p2, q2] = k; push!(stack, (p2, q2))
                end
            end
        end
        labs[(I, J)] = lab
    end
    pcs = Pieces(phase, cell, vol, [xsum[k] / vol[k] for k in eachindex(vol)])
    lbl(I, J, a, b) = (L = get(labs, (I, J), nothing); L === nothing ? pid[I, J] : L[a, b])
    # connections from sub-edges
    face  = Dict{NTuple{2,Int},Tuple{Float64,SVector{2,Float64}}}()     # (a, b) → (aperture, e)
    intf  = Dict{NTuple{2,Int},Tuple{SVector{2,Float64},SVector{2,Float64},Float64}}()  # (neg, pos) → (∫n ds, ∫x ds, ∫ds)
    bnd   = [Tuple{Float64,SVector{2,Float64},SVector{2,Float64}}[] for _ in eachindex(phase)]  # (A, x_face, outward e)
    function edge!(ka, kb, e, xm)
        ka == kb && return
        if phase[ka] == phase[kb]
            A, _ = get(face, (ka, kb), (0.0, e))
            face[(ka, kb)] = (A + hs, e)
        else
            kn, kp, n = phase[ka] < 0 ? (ka, kb, e) : (kb, ka, -e)          # n from neg to pos
            N0, X0, L0 = get(intf, (kn, kp), (SVector(0.0, 0.0), SVector(0.0, 0.0), 0.0))
            intf[(kn, kp)] = (N0 + hs * n, X0 + hs * xm, L0 + hs)
        end
    end
    for J in 1:N, I in 1:N
        cut = haskey(labs, (I, J))
        if cut                                                               # internal sub-edges
            for b in 1:m, a in 1:m
                a < m && edge!(lbl(I, J, a, b), lbl(I, J, a + 1, b), SVector(1.0, 0.0), SVector(xsub(I, a) + hs / 2, xsub(J, b)))
                b < m && edge!(lbl(I, J, a, b), lbl(I, J, a, b + 1), SVector(0.0, 1.0), SVector(xsub(I, a), xsub(J, b) + hs / 2))
            end
        end
        if I < N
            if !cut && !haskey(labs, (I + 1, J))
                pid[I, J] != pid[I+1, J] && phase[pid[I, J]] == phase[pid[I+1, J]] && (face[(pid[I, J], pid[I+1, J])] = (h, SVector(1.0, 0.0)))
                phase[pid[I, J]] != phase[pid[I+1, J]] && (intf_fix!(intf, pid[I, J], pid[I+1, J], phase, SVector(1.0, 0.0), SVector(-1 + I * h, -1 + (J - 0.5) * h), h, hs))
            else
                for b in 1:m
                    edge!(lbl(I, J, m, b), lbl(I + 1, J, 1, b), SVector(1.0, 0.0), SVector(-1 + I * h, xsub(J, b)))
                end
            end
        end
        if J < N
            if !cut && !haskey(labs, (I, J + 1))
                pid[I, J] != pid[I, J+1] && phase[pid[I, J]] == phase[pid[I, J+1]] && (face[(pid[I, J], pid[I, J+1])] = (h, SVector(0.0, 1.0)))
                phase[pid[I, J]] != phase[pid[I, J+1]] && (intf_fix!(intf, pid[I, J], pid[I, J+1], phase, SVector(0.0, 1.0), SVector(-1 + (I - 0.5) * h, -1 + J * h), h, hs))
            else
                for a in 1:m
                    edge!(lbl(I, J, a, m), lbl(I, J + 1, a, 1), SVector(0.0, 1.0), SVector(xsub(I, a), -1 + J * h))
                end
            end
        end
        for b in 1:m                                                         # Dirichlet walls
            I == 1 && push!(bnd[lbl(1, J, 1, b)], (hs, SVector(-1.0, xsub(J, b)), SVector(-1.0, 0.0)))
            I == N && push!(bnd[lbl(N, J, m, b)], (hs, SVector(1.0, xsub(J, b)), SVector(1.0, 0.0)))
        end
        for a in 1:m
            J == 1 && push!(bnd[lbl(I, 1, a, 1)], (hs, SVector(xsub(I, a), -1.0), SVector(0.0, -1.0)))
            J == N && push!(bnd[lbl(I, N, a, m)], (hs, SVector(xsub(I, a), 1.0), SVector(0.0, 1.0)))
        end
    end
    return (; N, h, m, pcs, face, intf, bnd, pid, labs)
end

# two uncut neighbouring cells of different phases (Γ passes between their sub-points on the face
# line): the whole face is interface
function intf_fix!(intf, ka, kb, phase, e, xm, h, hs)
    kn, kp, n = phase[ka] < 0 ? (ka, kb, e) : (kb, ka, -e)
    N0, X0, L0 = get(intf, (kn, kp), (SVector(0.0, 0.0), SVector(0.0, 0.0), 0.0))
    intf[(kn, kp)] = (N0 + h * n, X0 + h * xm, L0 + h)
end

function solve_cutcell(G, β, ubc; f=x -> 0.0, tcorr=0, φ=nothing)
    (; pcs, face, intf, bnd, h) = G
    np  = length(pcs.phase)
    βs(s) = s < 0 ? β.neg : β.pos
    Is, Js, Vs = Int[], Int[], Float64[]
    b   = [f(pcs.xc[k]) * pcs.vol[k] for k in 1:np]
    add!(i, j, v) = (push!(Is, i); push!(Js, j); push!(Vs, v))
    couple!(a, c, T) = (add!(a, a, -T); add!(a, c, T); add!(c, c, -T); add!(c, a, T))
    for ((ka, kb), (A, e)) in face
        d = max(dot(pcs.xc[kb] - pcs.xc[ka], e), 0.05h)
        couple!(ka, kb, βs(pcs.phase[ka]) * A / d)
    end
    links = []
    for ((kn, kp), (Nv, Xs, L)) in intf
        Γ  = norm(Nv)
        Γ < 1e-14 && continue
        n  = Nv / Γ
        xΓ = Xs / L
        dn = max(dot(xΓ - pcs.xc[kn], n), 0.05h)
        dp = max(dot(pcs.xc[kp] - xΓ, n), 0.05h)
        T  = Γ / (dn / β.neg + dp / β.pos)
        couple!(kn, kp, T)
        push!(links, (kn, kp, T, n, xΓ))
    end
    for k in 1:np, (A, xf, e) in bnd[k]
        d = max(abs(dot(xf - pcs.xc[k], e)), 0.05h)
        T = βs(pcs.phase[k]) * A / d
        add!(k, k, -T); b[k] -= T * ubc(xf)
    end
    K = sparse(Is, Js, Vs, np, np)
    F = cholesky(Symmetric(-K))
    u = F \ (-b)
    it = 0
    if tcorr > 0
        # deferred tangential correction from least-squares gradients of same-phase neighbours
        nbr = [Int[] for _ in 1:np]
        for (ka, kb) in keys(face); push!(nbr[ka], kb); push!(nbr[kb], ka); end
        for (kn, kp) in keys(intf); push!(nbr[kn], kp); push!(nbr[kp], kn); end
        function grad(k, u)
            M = zeros(2, 2); r = zeros(2)
            for j in nbr[k]
                pcs.phase[j] == pcs.phase[k] || continue
                dx = pcs.xc[j] - pcs.xc[k]; M .+= dx * dx'; r .+= dx * (u[j] - u[k])
            end
            return det(M) > 1e-12 * h^4 ? SVector{2}(M \ r) : SVector(0.0, 0.0)
        end
        soft = β.neg <= β.pos ? -1 : 1
        for k in 1:tcorr
            bc = copy(b)
            for (kn, kp, T, n, xΓ) in links
                ks = soft < 0 ? kn : kp
                g  = grad(ks, u)
                δ  = pcs.xc[kp] - pcs.xc[kn]
                δt = δ - dot(δ, n) * n
                c  = T * dot(g, δt)                   # flux correction, from kn's balance: +T(u_p - u_n - c/T)
                bc[kn] += c; bc[kp] -= c
            end
            un = F \ (-bc)
            dif = norm(un - u, Inf) / max(norm(un, Inf), 1e-300)
            u, it = un, k
            dif < 1e-10 && break
        end
    end
    return (u=u, it=it, np=np, K=K)
end

# value at the cell centre of cell (I, J): the piece containing the centre sub-point
function cell_value(G, u, I, J)
    L = get(G.labs, (I, J), nothing)
    L === nothing && return u[G.pid[I, J]]
    c = G.m ÷ 2 + 1
    return u[L[c, c]]
end
