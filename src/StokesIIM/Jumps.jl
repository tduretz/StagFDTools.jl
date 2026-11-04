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
