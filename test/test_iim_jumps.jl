using Test, StaticArrays, LinearAlgebra, StagFDTools, StagFDTools.StokesIIM
import ExactFieldSolutions: Stokes2D_Schmid2003_V, Stokes2D_Schmid2003_L, Stokes2D_Schmid2003_p

@testset "IIM jump conditions" begin
    # Exact jumps of the Schmid & Podladchikov (2003) solution (gr ≠ 0 adds simple shear)
    for mc in (1e-3, 7.0, 1e4), gr in (0.0, 0.6)
        params = (mm=1.0, mc=mc, rc=2.0, gr=gr, er=1.0)
        Δη, ϵ = params.mm - params.mc, 1e-7
        err = 0.0
        for θ in range(0, 2π, 37)
            n  = SVector(cos(θ), sin(θ))
            xΓ = params.rc * n
            xp, xm = (1 + ϵ) * xΓ, (1 - ϵ) * xΓ
            Lp, Lm = Stokes2D_Schmid2003_L(xp, params), Stokes2D_Schmid2003_L(xm, params)
            V  = Stokes2D_Schmid2003_V(xΓ, params)
            J  = InterfaceJumps(Δη, V, Lm * tangent(n), n)
            Jz = params.mm * Stokes2D_Schmid2003_V(xp, params) - params.mc * Stokes2D_Schmid2003_V(xm, params)
            JG = params.mm * Lp - params.mc * Lm
            Jp = Stokes2D_Schmid2003_p(xp, params) - Stokes2D_Schmid2003_p(xm, params)
            scale = max(abs(Δη), 1.0)
            err = max(err, norm(J.z - Jz) / scale, norm(J.∇z - JG) / scale, abs(J.p - Jp) / scale)
        end
        @test err < 1e-5
    end
end

using ForwardDiff
@testset "IIM second-order jump conditions" begin
    for mc in (1e-3, 7.0, 1e4), gr in (0.0, 0.6)
        params = (mm=1.0, mc=mc, rc=2.0, gr=gr, er=1.0)
        Δη, ϵ, κ = params.mm - params.mc, 1e-9, 1 / params.rc
        V(x)  = Stokes2D_Schmid2003_V(x, params)
        P(x)  = Stokes2D_Schmid2003_p(x, params)
        ∇V(x) = ForwardDiff.jacobian(V, x)
        H(x, a) = ForwardDiff.hessian(y -> V(y)[a], x)
        err = 0.0
        for θ in range(0, 2π, 37)
            n  = SVector(cos(θ), sin(θ))
            t  = tangent(n)
            xΓ = params.rc * n
            xp, xm = (1 + ϵ) * xΓ, (1 - ϵ) * xΓ
            # inputs from the inclusion side (v, ∂ₛv, ∂ₛₛv are continuous along Γ)
            Lm = ∇V(xm)
            w  = Lm * t
            w2 = SVector(dot(t, H(xm, 1) * t), dot(t, H(xm, 2) * t)) - κ * Lm * n
            J  = InterfaceJumps2(Δη, V(xΓ), w, w2, n, κ)
            scale = max(abs(Δη), 1.0)
            for a in 1:2
                JH = params.mm * H(xp, a) - params.mc * H(xm, a)
                err = max(err, norm(J.∇∇z[a, :, :] - JH) / scale)
            end
            Jgp = ForwardDiff.gradient(P, xp) - ForwardDiff.gradient(P, xm)
            err = max(err, norm(J.∇p - Jgp) / scale)
        end
        @test err < 1e-6
    end
end

import ExactFieldSolutions: Stokes2D_Moutzouris_circle, Stokes2D_Moutzouris_ellipse
@testset "IIM compressible jump conditions" begin
    # k = 0 on both sides: the incompressible jumps, whatever the traction
    n, κ = normalize(SVector(0.3, -1.1)), 0.7
    vΓ, w, w2, T, dT = (SVector(randn(2)...) for _ in 1:5)
    J2, JC = InterfaceJumps2(-4.0, vΓ, w, w2, n, κ), InterfaceJumpsC((neg=5.0, pos=1.0), (neg=0.0, pos=0.0), vΓ, w, w2, T, dT, n, κ)
    @test JC.z ≈ J2.z && JC.∇z ≈ J2.∇z && JC.∇∇z ≈ J2.∇∇z && JC.p ≈ J2.p && JC.∇p ≈ J2.∇p

    # Exact jumps of Moutzouris (2026): uniform inclusion, compressible host and inclusion
    for (f, t) in ((Stokes2D_Moutzouris_circle, 1.0), (Stokes2D_Moutzouris_ellipse, 2.0)),
        (ηi, ξi) in ((1e3, 1e5), (1e-2, 0.5), (7.0, 7.0))
        prm = (ηm=1.0, ηi=ηi, ξm=2.0, ξi=ξi, ri=0.2, t=t, α=π / 6, ε̇=1.0, γ̇=0.3, ζ̇=0.5, ε̇zz=0.0)
        η, k = (neg=ηi, pos=1.0), (neg=ηi / ξi, pos=1.0 / 2.0)
        a, b = t * prm.ri, prm.ri
        S(x) = f(x; params=prm)
        σ(s) = s.τ - s.p * I
        h, err1, err2, mag = 1e-5, 0.0, 0.0, 0.0
        for θ in range(0, 2π, 25)[1:end-1]
            xΓ = SVector(a * cos(θ), b * sin(θ))
            n  = normalize(SVector(cos(θ) / a, sin(θ) / b))
            tg = tangent(n)
            κ  = a * b / (a^2 * sin(θ)^2 + b^2 * cos(θ)^2)^1.5
            sm = S(xΓ - 1e-9 * n)
            # inclusion side is uniform: ∂ₛₛv = -κ (∇v) n, d(σn)/ds = κ σ t
            J  = InterfaceJumpsC(η, k, sm.V, sm.L * tg, -κ * sm.L * n, σ(sm) * n, κ * σ(sm) * tg, n, κ)
            xp = xΓ + 2h * n
            sp = S(xp)
            scale = max(ηi, 1.0)
            sp1 = S(xΓ + 1e-9 * n)
            err1 = max(err1, norm(J.z - (sp1.V - ηi * sm.V)) / scale, norm(J.∇z - (sp1.L - ηi * sm.L)) / scale, abs(J.p - (sp1.p - sm.p)) / scale)
            # matrix second derivatives by central differences 2h off Γ (inclusion: zero)
            e = (SVector(h, 0.0), SVector(0.0, h))
            dL = [(S(xp + e[c]).L - S(xp - e[c]).L) / 2h for c in 1:2]
            for a_ in 1:2, b_ in 1:2, c in 1:2
                err2 = max(err2, abs(J.∇∇z[a_, b_, c] - dL[c][a_, b_]))
                mag  = max(mag, abs(dL[c][a_, b_]))
            end
            ∇p = SVector((S(xp + e[1]).p - S(xp - e[1]).p) / 2h, (S(xp + e[2]).p - S(xp - e[2]).p) / 2h)
            err2 = max(err2, norm(J.∇p - ∇p))
        end
        @test err1 < 1e-5
        @test err2 / mag < 1e-3          # central differences 2h off Γ: O(h) error
    end
end
