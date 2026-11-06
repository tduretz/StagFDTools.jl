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

@testset "IIM interface trace, quadratic fits" begin
    nc = (x=20, y=18)
    L  = (x=2.0, y=1.8)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid((min=0.0, max=L.x), (min=0.0, max=L.y), Δ, nc)
    η  = (neg=1e3, pos=2.0)
    n  = normalize(SVector(1.0, 0.6))
    t  = tangent(n)
    x0 = SVector(1.03, 0.91)
    φ  = LevelSet((x, y) -> dot(n, SVector(x, y) - x0), X)
    tr = InterfaceTraces(InterfaceCrossings(FineLattice(φ, X, Δ)), φ, X, Δ, η, nc; fit=:quadratic)
    # Piecewise quadratic v, continuous across Γ with equal ∂ₛv and ∂ₛₛv: v₊ - v₋ = (n·(x - x0)) ℓ(x)
    q(x)   = SVector(0.3 + x[1] - 0.4x[2] + 0.7x[1]^2 - 0.2x[1] * x[2], -0.5 + 0.2x[1] + 0.9x[2]^2 + 0.3x[1] * x[2])
    ℓ(x)   = SVector(1.5 - 0.8x[1] + 0.4x[2], -0.6 + 0.5x[2])
    vel(x) = q(x) + (dot(n, x - x0) > 0) * dot(n, x - x0) * ℓ(x)
    ηof(φ) = side(φ) < 0 ? η.neg : η.pos
    z = (x = [ηof(φ.x[i, j]) * vel(SVector(X.vx_e.x[i], X.vx_e.y[j]))[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
         y = [ηof(φ.y[i, j]) * vel(SVector(X.vy_e.x[i], X.vy_e.y[j]))[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    ∇q(x) = SMatrix{2,2}(1 + 1.4x[1] - 0.2x[2], 0.2 + 0.3x[2], -0.4 - 0.2x[1], 1.8x[2] + 0.3x[1])  # ∇q[a, b] = ∂_b q_a
    Hq    = (SMatrix{2,2}(1.4, -0.2, -0.2, 0.0), SMatrix{2,2}(0.0, 0.3, 0.3, 1.8))
    errv, errs, errss = 0.0, 0.0, 0.0
    for T in tr, (k, comp) in enumerate((:x, :y))
        vΓ, ∂s, ∂ss = apply_trace(T[comp], z[comp])
        errv  = max(errv,  abs(vΓ - q(T.p.x)[k]))
        errs  = max(errs,  abs(∂s - dot(∇q(T.p.x)[k, :], t)))
        errss = max(errss, abs(∂ss - dot(t, Hq[k] * t)))       # straight Γ: κ = 0
    end
    @test errv < 1e-9 && errs < 1e-8 && errss < 1e-6
end

@testset "IIM one-sided cubic fit" begin
    nc = (x=20, y=18)
    L  = (x=2.0, y=1.8)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid((min=0.0, max=L.x), (min=0.0, max=L.y), Δ, nc)
    n  = normalize(SVector(1.0, 0.6))
    x0 = SVector(1.03, 0.91)
    φ  = LevelSet((x, y) -> dot(n, SVector(x, y) - x0), X)
    # A cubic polynomial on side -1 is reproduced exactly (value, gradient, Hessian at xΓ)
    c(x)  = 0.3 + x[1] - 0.4x[2] + 0.7x[1]^2 - 0.2x[1] * x[2] + 0.5x[1]^3 - 0.3x[1]^2 * x[2] + 0.2x[2]^3
    ∇c(x) = SVector(1 + 1.4x[1] - 0.2x[2] + 1.5x[1]^2 - 0.6x[1] * x[2], -0.4 - 0.2x[1] - 0.3x[1]^2 + 0.6x[2]^2)
    Hc(x) = SMatrix{2,2}(1.4 + 3x[1] - 0.6x[2], -0.2 - 0.6x[1], -0.2 - 0.6x[1], 1.2x[2])
    ranges = (2:(nc.x+2), 3:(nc.y+2))
    xΓ = x0 + 0.1 * tangent(n)
    F  = QuadraticFit(xΓ, xΓ - Δ.x * n, -1, φ.x, X.vx_e, ranges, Δ; npts=24, degree=3)
    f  = [c(SVector(X.vx_e.x[I[1]], X.vx_e.y[I[2]])) for I in F.idx]
    @test all(I -> side(φ.x[I]) == -1, F.idx)
    @test abs(dot(F.λ, f) - c(xΓ)) < 1e-10
    @test norm(F.g * f - ∇c(xΓ)) < 1e-9
    @test norm([dot(F.H[a, b, :], f) for a in 1:2, b in 1:2] - Hc(xΓ)) < 1e-7
end
