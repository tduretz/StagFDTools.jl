using Test, StaticArrays, LinearAlgebra, StagFDTools, StagFDTools.StokesIIM

@testset "IIM interface trace" begin

    nc = (x=16, y=14)
    L  = (x=2.0, y=1.75)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid((min=0.0, max=L.x), (min=0.0, max=L.y), Δ, nc)
    η  = (neg=1e3, pos=2.0)

    # Straight inclined interface through the middle of the domain
    n  = normalize(SVector(1.0, 0.6))
    x0 = SVector(1.03, 0.91)
    φ  = LevelSet((x, y) -> dot(n, SVector(x, y) - x0), X)
    Γ  = InterfaceCrossings(FineLattice(φ, X, Δ))
    tr = InterfaceTraces(Γ, φ, X, Δ, η, nc)
    @test !isempty(tr)

    # Velocity continuous across Γ, linear on each side, with a kink: v₊ - v₋ = c (n·(x - x0))
    a, b, c = SVector(0.3, -0.7), SMatrix{2,2}(1.0, -0.4, 0.8, 0.2), SVector(2.5, -1.5)
    vel(x) = a + b * x + (dot(n, x - x0) > 0) * c * dot(n, x - x0)
    ηof(φ) = side(φ) < 0 ? η.neg : η.pos
    z = (x = [ηof(φ.x[i, j]) * vel(SVector(X.vx_e.x[i], X.vx_e.y[j]))[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
         y = [ηof(φ.y[i, j]) * vel(SVector(X.vy_e.x[i], X.vy_e.y[j]))[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])

    @testset "exact for continuous piecewise linear v" begin
        errv, errt = 0.0, 0.0
        for T in tr
            t = tangent(T.p.n)
            for (k, comp) in enumerate((:x, :y))
                vΓ, ∂tvΓ = apply_trace(T[comp], z[comp])
                errv = max(errv, abs(vΓ - vel(T.p.x)[k]))
                errt = max(errt, abs(∂tvΓ - dot(b[k, :], t)))   # ∂ₜv is the same on both sides
            end
        end
        @test errv < 1e-10
        @test errt < 1e-10
    end

    @testset "weights consistent with z = ηv" begin
        for T in tr, (comp, φc) in ((:x, φ.x), (:y, φ.y))
            ηk = SVector(ntuple(k -> ηof(φc[T[comp].idx[k]]), 6))
            @test sum(T[comp].v .* ηk) ≈ 1           # constant v is reproduced
            @test abs(sum(T[comp].t .* ηk)) < 1e-10  # and has zero tangential derivative
            @test all(side.(φc[T[comp].idx[1:3]]) .== -1) && all(side.(φc[T[comp].idx[4:6]]) .== 1)
        end
    end
end
