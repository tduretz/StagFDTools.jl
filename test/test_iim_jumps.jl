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
