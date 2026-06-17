invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + 1/2*(-x[1]-x[2])^2 + x[3]^2) 

function StrainRateTrial(τII, Pt, Pf, ηve, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δt)
    ε̇II_trial = τII/2/ηve
    return ε̇II_trial
end

# F(τ, Pt, Pf, Φ, C, cosϕ, sinϕ, λ̇, ηvp, α) = τ - (1-Φ)*C*cosϕ - (Pt - α*Pf)*sinϕ  - λ̇*ηvp 

F(τII, Pt, Pf, Φ, c, cosϕ, sinϕ, λ̇, ηvp, α1) = τII - (Pt - α1*Pf)*sinϕ - c*cosϕ - λ̇*ηvp
