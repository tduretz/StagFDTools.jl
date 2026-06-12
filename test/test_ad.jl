using Test, StaticArrays, StagFDTools

@testset "AD wrappers" verbose=true begin

    @testset "Gradients and derivatives" begin
        f(x) = sum(x .^ 2)
        x = SVector(1.0, 2.0, 3.0)
        @test ad_gradient(f, x) ≈ 2x
        v, g = ad_value_and_gradient(f, x)
        @test v ≈ 14.0
        @test g ≈ 2x

        # Extra arguments are passed as AD constants
        fc(x, a) = a * sum(x .^ 2)
        @test ad_gradient(fc, x, 3.0) ≈ 6x

        h(x, a) = a * x^3
        @test ad_derivative(h, 2.0, 3.0) ≈ 36.0
        v, d = ad_value_and_derivative(h, 2.0, 3.0)
        @test v ≈ 24.0
        @test d ≈ 36.0
    end

    @testset "Jacobians" begin
        f(x) = SVector(x[1]^2, x[1]*x[2])
        x = SVector(2.0, 3.0)
        Jref = [4.0 0.0; 3.0 2.0]
        @test ad_jacobian(f, x) ≈ Jref
        v, J = ad_value_and_jacobian(f, x)
        @test v ≈ SVector(4.0, 6.0)
        @test J ≈ Jref

        # *_first variants differentiate only the first output of a tuple-valued function
        ftup(x) = (SVector(x[1]^2, x[2]^2), sum(x))
        v1, J1 = ad_value_and_jacobian_first(ftup, x)
        @test v1 ≈ SVector(4.0, 9.0)
        @test J1 ≈ [4.0 0.0; 0.0 6.0]
        v2, J2 = ad_jacobian_first(ftup, x)
        @test v2 ≈ v1
        @test J2 ≈ J1
    end

    @testset "Partial gradients" begin
        p(x, y) = sum(x .* x .* y)
        x, y = SVector(1.0, 2.0), SVector(3.0, 4.0)
        gx, gy = ad_partial_gradients(p, (x, y))
        @test gx ≈ 2 .* x .* y
        @test gy ≈ x .^ 2
    end

    @testset "Const / Duplicated interface" begin
        r(x, y) = sum(x .* x) + 2*sum(y)
        x, y = SVector(1.0, 2.0), SVector(3.0, 4.0)
        gx = MVector(0.0, 0.0)
        forwarddiff_gradients!(r, Duplicated(x, gx), Const(y))
        @test gx ≈ 2x

        res = forwarddiff_gradient(sum, x)
        @test res[1] ≈ SVector(1.0, 1.0)
    end

end
