using Test, StaticArrays, LinearAlgebra, StagFDTools, StagFDTools.StokesIIM

@testset "IIM geometry" begin

    nc = (x=12, y=10)
    L  = (x=3.0, y=2.5)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid((min=0.0, max=L.x), (min=0.0, max=L.y), Δ, nc)

    @testset "fine lattice merges the staggered grids" begin
        f(x, y) = x + 2y^2
        fl = FineLattice(LevelSet(f, X), X, Δ)
        @test all(isapprox(fl.Φ[I, J], f(fl.x[I], fl.y[J]), atol=1e-9) for I in axes(fl.Φ, 1), J in axes(fl.Φ, 2))
    end

    @testset "crossings and normals exact for a straight interface" begin
        n  = normalize(SVector(1.0, 0.7))
        x0 = SVector(1.37, 1.11)
        fl = FineLattice(LevelSet((x, y) -> dot(n, SVector(x, y) - x0), X), X, Δ)
        Γ  = InterfaceCrossings(fl)
        @test !isempty(Γ.points)
        @test maximum(abs(dot(n, p.x - x0)) for p in Γ.points) < 1e-12
        @test maximum(norm(p.n - n) for p in Γ.points) < 1e-12
        @test MultipleCrossings(Γ) == 0
    end

    @testset "irregular equations for a vertical interface" begin
        # Γ at x = 1.4: Vx nodes at x = 1.25, 1.5 are irregular through ∂xx Vx (neighbours across)
        # plus ∂x P and ∂xy Vy; Pt at x = 1.375 has its east Vx face across Γ
        φ   = LevelSet((x, y) -> x - 1.4, X)
        irr = IrregularNodes(φ, nc)
        cols(A, xs) = sort(unique(xs[I[1]] for I in findall(A)))
        @test cols(irr.x, X.vx_e.x) ≈ [1.25, 1.5]
        @test cols(irr.y, X.vy_e.x) ≈ [1.375, 1.625]
        @test cols(irr.c, X.c_e.x) ≈ [1.375]
    end

    @testset "nodes on Γ are moved to the + side" begin
        φ = LevelSet((x, y) -> x - 1.5, X)   # x = 1.5 is a vertex/Vx line
        @test all(φ.x[i, j] > 0 for i in axes(φ.x, 1), j in axes(φ.x, 2) if abs(X.vx_e.x[i] - 1.5) < 1e-12)
    end
end
