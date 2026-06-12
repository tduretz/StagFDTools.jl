using Test, StaticArrays, LinearAlgebra, SparseArrays, ExtendableSparse
import StagFDTools.Poisson
import StagFDTools.Stokes

@testset "Sparsity patterns" verbose=true begin

    @testset "Poisson" begin
        nc = (x = 3, y = 4)

        type = Poisson.Fields( fill(:out, (nc.x+2, nc.y+2)) )
        type.u[2:end-1,2:end-1] .= :in
        type.u[:,1]             .= :Dirichlet
        type.u[:,end]           .= :Neumann

        number = Poisson.Fields( fill(0, (nc.x+2, nc.y+2)) )
        Poisson.Numbering!(number, type, nc)

        nu = maximum(number.u)
        @test nu == nc.x*nc.y

        # 5-point stencil: symmetric pattern with a full diagonal
        pattern = Poisson.Fields( Poisson.Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )
        M = Poisson.Fields( Poisson.Fields( ExtendableSparseMatrix(nu, nu) ))
        Poisson.SparsityPattern!(M, number, pattern, nc)
        A = sparse(M.u.u)
        @test A == A'
        @test all(diag(A) .== 1.0)
        # Interior rows couple to at most 5 unknowns
        @test maximum(sum(A .!= 0, dims=2)) <= 5

        # 9-point stencil strictly contains the 5-point one
        pattern9 = Poisson.Fields( Poisson.Fields( @SMatrix([1 1 1; 1 1 1; 1 1 1]) ) )
        M9 = Poisson.Fields( Poisson.Fields( ExtendableSparseMatrix(nu, nu) ))
        Poisson.SparsityPattern!(M9, number, pattern9, nc)
        A9 = sparse(M9.u.u)
        @test A9 == A9'
        @test nnz(A9) > nnz(A)
        @test all(A9[findall(!iszero, A)] .== 1.0)
    end

    @testset "Stokes" begin
        nc = (x = 4, y = 3)
        (; size_x, size_y, size_c) = Stokes.Ranges(nc)

        type = Stokes.Fields(
            fill(:out, size_x),
            fill(:out, size_y),
            fill(:out, size_c),
        )
        Stokes.set_boundaries_template!(type, :all_Dirichlet, nc)

        pattern = Stokes.Fields(
            Stokes.Fields(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0;  0 1 0])),
            Stokes.Fields(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0])),
            Stokes.Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
        )

        number = Stokes.Fields(
            fill(0, size_x),
            fill(0, size_y),
            fill(0, size_c),
        )
        Stokes.Numbering!(number, type, nc)

        nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
        @test nPt == nc.x*nc.y
        @test nVx > 0 && nVy > 0

        M = Stokes.Fields(
            Stokes.Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)),
            Stokes.Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)),
            Stokes.Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
        )
        Stokes.SparsityPattern!(M, number, pattern, nc)

        # Velocity block pattern is symmetric
        K = [sparse(M.Vx.Vx) sparse(M.Vx.Vy); sparse(M.Vy.Vx) sparse(M.Vy.Vy)]
        @test K == K'
        @test all(diag(K) .== 1.0)

        # Gradient and divergence patterns are transposes of each other
        Q  = [sparse(M.Vx.Pt); sparse(M.Vy.Pt)]
        Qᵀ = [sparse(M.Pt.Vx) sparse(M.Pt.Vy)]
        @test Q' == Qᵀ
    end

end
