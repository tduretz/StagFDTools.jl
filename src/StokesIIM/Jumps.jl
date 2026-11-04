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
