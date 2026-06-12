using Test, StaticArrays, StagFDTools

@testset "Stencil operators" verbose=true begin

    A = @SMatrix [1.0 2.0 3.0;
                  4.0 5.0 6.0;
                  7.0 8.0 9.0]

    @testset "Selection" begin
        @test inn(A)   === @SMatrix [5.0;;]
        @test inn_x(A) === @SMatrix [4.0 5.0 6.0]
        @test inn_y(A) === @SMatrix [2.0; 5.0; 8.0;;]
    end

    @testset "Averaging" begin
        @test av(A)  === @SMatrix [3.0 4.0; 6.0 7.0]
        @test avx(A) === @SMatrix [2.5 3.5 4.5; 5.5 6.5 7.5]
        @test avy(A) === @SMatrix [1.5 2.5; 4.5 5.5; 7.5 8.5]
        # Harmonic average of a constant field is the constant
        C = @SMatrix fill(2.0, 3, 3)
        @test harm(C) ≈ @SMatrix fill(2.0, 2, 2)
        @test harm(A)[1,1] ≈ 4 / (1/1 + 1/4 + 1/2 + 1/5)
    end

    @testset "Differences" begin
        @test ∂x(A)     === @SMatrix [3.0 3.0 3.0; 3.0 3.0 3.0]
        @test ∂y(A)     === @SMatrix [1.0 1.0; 1.0 1.0; 1.0 1.0]
        @test ∂x_inn(A) === @SMatrix [3.0; 3.0;;]
        @test ∂y_inn(A) === @SMatrix [1.0 1.0]
        a = @SMatrix [1.0 2.0]
        b = @SMatrix [3.0; 4.0;;]
        @test ∂kk(a, b) === @SMatrix [6.0;;]
    end

    @testset "MMatrix variants" begin
        M = MMatrix{3,3}(A)
        @test inn(M) == inn(A)
        @test av(M)  == av(A)
        @test ∂x(M)  == ∂x(A)
    end

    @testset "Deviatoric strain rate" begin
        Dxx, Dxy, Dyx, Dyy = 1.0, 2.0, 4.0, -3.0
        ε̇xx, ε̇yy, ε̇xy, ε̇kk = deviatoric_strain_rate(Dxx, Dxy, Dyx, Dyy)
        @test ε̇kk ≈ Dxx + Dyy
        @test ε̇xx ≈ Dxx - ε̇kk/3
        @test ε̇yy ≈ Dyy - ε̇kk/3
        @test ε̇xy ≈ (Dxy + Dyx)/2
        # 3D convention: the in-plane deviator trace equals ε̇kk/3 (zz carries the rest)
        @test ε̇xx + ε̇yy ≈ ε̇kk/3

        v = SVector(Dxx, Dxx)
        ε̇xxv, ε̇yyv, ε̇xyv, ε̇kkv = deviatoric_strain_rate(v, SVector(Dxy, Dxy), SVector(Dyx, Dyx), SVector(Dyy, Dyy))
        @test all(ε̇xxv .≈ ε̇xx) && all(ε̇yyv .≈ ε̇yy) && all(ε̇xyv .≈ ε̇xy) && all(ε̇kkv .≈ ε̇kk)
    end

    @testset "Effective strain rate" begin
        _2GΔt = 0.5
        ε̇xx, ε̇yy, ε̇xy = effective_strain_rate(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, _2GΔt)
        @test ε̇xx ≈ 1.0 + 4.0*_2GΔt
        @test ε̇yy ≈ 2.0 + 5.0*_2GΔt
        @test ε̇xy ≈ 3.0 + 6.0*_2GΔt

        one2 = SVector(1.0, 1.0)
        ε̇xxv, ε̇yyv, ε̇xyv = effective_strain_rate(1.0*one2, 2.0*one2, 3.0*one2, 4.0*one2, 5.0*one2, 6.0*one2, _2GΔt)
        @test all(ε̇xxv .≈ ε̇xx) && all(ε̇yyv .≈ ε̇yy) && all(ε̇xyv .≈ ε̇xy)
    end

end
