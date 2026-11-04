# Jump conditions across Γ for z = ηv and p (port of Landry's compute_jumps.cxx, first order).
#
# Conventions: [q] = q₊ - q₋ (φ > 0 side minus φ < 0 side), Δη = η₊ - η₋, n points to φ > 0,
# t = (-n_y, n_x). With v continuous across Γ, incompressibility on both sides and continuous
# traction (no surface tension), all jumps follow from v_Γ and ∂ₜv_Γ = (∇v) t:
#   [z]       =  Δη v_Γ
#   [∂ₜz]     =  Δη ∂ₜv_Γ                    (v continuous along Γ)
#   [∂ₙz_n]   = -Δη t·∂ₜv_Γ                   (∂ₙv_n = -∂ₜv_t on both sides)
#   [∂ₙz_t]   = -Δη n·∂ₜv_Γ                   (η(∂ₙv_t + ∂ₜv_n) continuous)
#   [p]       = -2Δη t·∂ₜv_Γ                  (-p + 2η∂ₙv_n continuous)

"""
    InterfaceJumps(Δη, vΓ, ∂tvΓ, n)

Jumps `[z]` (vector), `[∇z]` (matrix, `[∇z][a, b] = [∂_b z_a]`) and `[p]` at a point of Γ with
normal `n`, from the interface velocity `vΓ` and its tangential derivative `∂tvΓ` (Cartesian
components). Linear in (`vΓ`, `∂tvΓ`).
"""
function InterfaceJumps(Δη, vΓ, ∂tvΓ, n)
    t   = tangent(n)
    ∂tn = dot(n, ∂tvΓ)                 # ∂ₜv_n
    ∂tt = dot(t, ∂tvΓ)                 # ∂ₜv_t
    # [∂_β z_α] in the (n, t) frame: rows α = (n, t), columns β = (n, t)
    Gnt = Δη * @SMatrix [-∂tt  ∂tn;
                         -∂tn  ∂tt]
    R   = hcat(n, t)                   # (n, t) -> (x, y)
    return (z = Δη * vΓ, ∇z = R * Gnt * R', p = -2Δη * ∂tt)
end

# Second-order jumps. Along Γ (arc length s, dt/ds = -κ n, dn/ds = κ t, κ = ∇·n), the first-order
# relations hold at every point, with w = ∂ₛv and w' = ∂ₛₛv:
#   [∇z] = G(s) = Δη ( w tᵀ - ((t·w) n + (n·w) t) nᵀ )
# so their derivative along Γ gives [∂ₜ∂_β z_a] = dG/ds. With n, t frozen at x_Γ:
#   [∂ₙₙz_n] = -[∂ₙₜz_t]                       (∂ₙ of ∇·z = 0)
#   [∂ₜp]    = d[p]/ds,  [p] = -2Δη t·w         (along Γ)
#   [∂ₙₙz_t] = [∂ₜp] - [∂ₜₜz_t]                 (t-momentum: ∇²z = ∇p, no body-force jump)
#   [∂ₙp]    = [∂ₙₙz_n] + [∂ₜₜz_n]              (n-momentum)

"""
    InterfaceJumps2(Δη, vΓ, w, w2, n, κ)

First- and second-order jumps at a point of Γ with normal `n` and curvature `κ = ∇·n`, from the
interface velocity `vΓ` and its first and second derivatives along Γ, `w = ∂ₛv`, `w2 = ∂ₛₛv`
(Cartesian components). Returns `z`, `∇z`, `∇∇z` (`∇∇z[a, b, c] = [∂_b ∂_c z_a]`), `p` and `∇p`.
Assumes no body-force jump. Linear in (`vΓ`, `w`, `w2`).
"""
function InterfaceJumps2(Δη, vΓ, w, w2, n, κ)
    J1 = InterfaceJumps(Δη, vΓ, w, n)
    t  = tangent(n)
    nw, tw, nw2, tw2 = dot(n, w), dot(t, w), dot(n, w2), dot(t, w2)
    dG = Δη * (w2 * t' - κ * w * n' -
               ((tw2 - 2κ * nw) * n + (nw2 + 2κ * tw) * t) * n' -
               κ * (tw * n + nw * t) * t')
    un, ut = dG' * n, dG' * t                    # u_α[β] = [∂ₜ∂_β z_α]
    zn_nt, zn_tt = dot(un, n), dot(un, t)
    zt_nt, zt_tt = dot(ut, n), dot(ut, t)
    zn_nn = -zt_nt
    pt    = 2Δη * (κ * nw - tw2)
    zt_nn = pt - zt_tt
    pn    = zn_nn + zn_tt
    R  = hcat(n, t)
    Hn = R * @SMatrix([zn_nn zn_nt; zn_nt zn_tt]) * R'
    Ht = R * @SMatrix([zt_nn zt_nt; zt_nt zt_tt]) * R'
    Hx, Hy = n[1] * Hn + t[1] * Ht, n[2] * Hn + t[2] * Ht
    ∇∇z = SArray{Tuple{2,2,2}}((Hx[1, 1], Hy[1, 1], Hx[2, 1], Hy[2, 1], Hx[1, 2], Hy[1, 2], Hx[2, 2], Hy[2, 2]))
    return (z=J1.z, ∇z=J1.∇z, ∇∇z=∇∇z, p=J1.p, ∇p=pn * n + pt * t)
end

# Compressible viscous flow (bulk viscosity ξ, p = -ξ∇·v, τ = 2η(ε̇ - ⅓(∇·v)I)). With z = ηv and
# k = η/ξ on each side (k = 0: incompressible), the side equations are
#   ∇²z + ⅓∇(∇·z) - ∇p = 0,   ∇·z + k p = 0   =>   ∇²z = (1 + k/3)∇p.
# ∂ₙzₙ now depends on the normal traction, so the jumps need the traction T = σn on Γ besides v_Γ:
# on each side, everything at x_Γ follows from (w, w2, T, dT = d(σn)/ds) and the jumps are the
# differences of the two sides. In the (n, t) frame, with C = n·T and S = t·T:
#   ∂ₜz = η w,   ∂ₙzₙ = (kC - (1 - ⅔k) η t·w)/(1 + ⁴⁄₃k),   ∂ₙzₜ = S - η n·w
#   p   = -C + ⁴⁄₃∂ₙzₙ - ⅔∂ₜzₜ
# Second order: d/ds of the above gives [∂ₜ∂_β z_α] and ∂ₜp; then
#   ∂ₙp = (∂ₜₜzₙ - ∂ₙₜzₜ)/(1 + ⁴⁄₃k),   ∂ₙₙzₙ = -∂ₙₜzₜ - k∂ₙp,   ∂ₙₙzₜ = (1 + k/3)∂ₜp - ∂ₜₜzₜ
# For k = 0 on both sides, C and S cancel and the jumps reduce to InterfaceJumps2.

# Fields on one side of Γ at x_Γ (see above)
function SideFields(η, k, w, w2, T, dT, n, κ)
    t = tangent(n)
    C, S   = dot(n, T), dot(t, T)
    dC, dS = dot(n, dT) + κ * dot(t, T), dot(t, dT) - κ * dot(n, T)
    nw, tw = dot(n, w), dot(t, w)
    dnw, dtw = κ * tw + dot(n, w2), -κ * nw + dot(t, w2)     # d(n·w)/ds, d(t·w)/ds
    μ  = 1 + 4k / 3
    A  = @SMatrix [(k * C - (1 - 2k / 3) * η * tw) / μ   η * nw;     # A[α, β] = ∂_β z_α, (n, t) frame
                   S - η * nw                             η * tw]
    dA = @SMatrix [(k * dC - (1 - 2k / 3) * η * dtw) / μ   η * dnw;
                   dS - η * dnw                             η * dtw]
    R, dR = hcat(n, t), hcat(κ * t, -κ * n)
    dG = dR * A * R' + R * dA * R' + R * A * dR'                     # dG[a, b] = ∂ₜ∂_b z_a
    p  = -C + 4 / 3 * A[1, 1] - 2 / 3 * A[2, 2]
    pt = -dC + 4 / 3 * dA[1, 1] - 2 / 3 * dA[2, 2]
    un, ut = dG' * n, dG' * t
    zn_nt, zn_tt = dot(un, n), dot(un, t)
    zt_nt, zt_tt = dot(ut, n), dot(ut, t)
    pn    = (zn_tt - zt_nt) / μ
    zn_nn = -zt_nt - k * pn
    zt_nn = (1 + k / 3) * pt - zt_tt
    Hn = R * @SMatrix([zn_nn zn_nt; zn_nt zn_tt]) * R'
    Ht = R * @SMatrix([zt_nn zt_nt; zt_nt zt_tt]) * R'
    Hx, Hy = n[1] * Hn + t[1] * Ht, n[2] * Hn + t[2] * Ht
    ∇∇z = SArray{Tuple{2,2,2}}((Hx[1, 1], Hy[1, 1], Hx[2, 1], Hy[2, 1], Hx[1, 2], Hy[1, 2], Hx[2, 2], Hy[2, 2]))
    return (∇z=R * A * R', ∇∇z=∇∇z, p=p, ∇p=pn * n + pt * t)
end

"""
    InterfaceJumpsC(η, k, vΓ, w, w2, T, dT, n, κ)

Jumps for compressible viscous flow, `k = (neg=η₋/ξ₋, pos=η₊/ξ₊)` (0: incompressible side), from
the interface velocity `vΓ`, its derivatives along Γ `w = ∂ₛv`, `w2 = ∂ₛₛv`, the traction
`T = σn` and its derivative along Γ `dT = d(σn)/ds` (Cartesian components). Same fields as
`InterfaceJumps2`; `∇∇z` and `∇p` need `w2` and `dT`. Assumes no body-force jump. Linear in
(`vΓ`, `w`, `w2`, `T`, `dT`).
"""
function InterfaceJumpsC(η, k, vΓ, w, w2, T, dT, n, κ)
    sm = SideFields(η.neg, k.neg, w, w2, T, dT, n, κ)
    sp = SideFields(η.pos, k.pos, w, w2, T, dT, n, κ)
    return (z=(η.pos - η.neg) * vΓ, ∇z=sp.∇z - sm.∇z, ∇∇z=sp.∇∇z - sm.∇∇z, p=sp.p - sm.p, ∇p=sp.∇p - sm.∇p)
end
