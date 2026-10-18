let
    Φ    = 5e-2
    Kf   = 1e9
    KΦ   = 5e9
    Ks   = 5e10
    ηΦ   = 1e220
    λ̇    = 1e-13
    ∂Q∂p = -sind(5)
    Δt   = 1e10
    ΔPf = Kf .* KΦ .* Δt .* ηΦ .* λ̇ .* ∂Q∂p ./ (-Kf .* KΦ .* Δt .* Φ + Kf .* KΦ .* Δt - Kf .* Φ .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ + Ks .* Φ .* ηΦ + KΦ .* Φ .* ηΦ)
    ΔPf_Claude = (Kf .* KΦ) / (Kf*(1-Φ) + Φ*(Ks+KΦ)) * λ̇ .* ∂Q∂p * Δt

    @show (ΔPf - ΔPf_Claude)/ΔPf * 100

    @show ΔPt = KΦ .* Δt .* Φ .* ηΦ .* λ̇ .* ∂Q∂p .* (Kf - Ks) ./ (-Kf .* KΦ .* Δt .* Φ + Kf .* KΦ .* Δt - Kf .* Φ .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ + Ks .* Φ .* ηΦ + KΦ .* Φ .* ηΦ)
    @show ΔPt_Claude =  Φ * (Ks - Kf)/Kf * abs(ΔPf)

    @show (ΔPt - ΔPt_Claude)/ΔPt * 100

end