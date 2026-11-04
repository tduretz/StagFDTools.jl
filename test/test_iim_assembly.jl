using Test, StaticArrays, LinearAlgebra, StagFDTools, StagFDTools.StokesIIM
import Statistics: mean

@testset "IIM assembly" begin

    nc = (x=24, y=20)
    L  = (x=2.4, y=2.0)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid((min=0.0, max=L.x), (min=0.0, max=L.y), Δ, nc)
    η  = (neg=50.0, pos=1.0)
    ηof(φ) = side(φ) < 0 ? η.neg : η.pos

    # Simple shear parallel to an inclined interface, with a velocity offset U:
    # v = (U + γ± s) t,  s = n·(x - x0),  η₋γ₋ = η₊γ₊,  p = 0.
    # v is continuous with a kink at Γ, [z] = Δη U t ≠ 0, and the first-order IIM is exact.
    n  = normalize(SVector(0.45, 1.0))
    t  = tangent(n)
    x0 = SVector(1.13, 1.04)
    U, τ = 0.7, 1.3
    vel(x) = (s = dot(n, x - x0); (U + τ * s / (s < 0 ? η.neg : η.pos)) * t)
    φ  = LevelSet((x, y) -> dot(n, SVector(x, y) - x0), X)
    va = (x=[vel(SVector(X.vx_e.x[i], X.vx_e.y[j]))[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
          y=[vel(SVector(X.vy_e.x[i], X.vy_e.y[j]))[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    zbc = (x=ηof.(φ.x) .* va.x, y=ηof.(φ.y) .* va.y)

    @testset "exact for piecewise linear solution with a kink" begin
        sol = SolveIIM(φ, X, Δ, η, nc, zbc)
        ix, jx, iy, jy = 3:(nc.x+1), 3:(nc.y+2), 3:(nc.x+2), 3:(nc.y+1)
        @test maximum(abs.(sol.v.x[ix, jx] .- va.x[ix, jx])) < 1e-9
        @test maximum(abs.(sol.v.y[iy, jy] .- va.y[iy, jy])) < 1e-9
        P = sol.p[2:(nc.x+1), 2:(nc.y+1)]
        @test maximum(abs.(P .- mean(P))) < 1e-8
    end

    @testset "without corrections the same problem is not solved" begin
        sol = SolveIIM(φ, X, Δ, η, nc, zbc; corrections=false)
        @test maximum(abs.(sol.v.x[3:(nc.x+1), 3:(nc.y+2)] .- va.x[3:(nc.x+1), 3:(nc.y+2)])) > 1e-3
    end
end
