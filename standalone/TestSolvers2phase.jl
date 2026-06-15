using LinearAlgebra, JLD2, IterativeSolvers, Preconditioners, SparseArrays

function test_solvers_2phase( n )
    @load "test_matrix_2phase_$(n).jld2" 𝑀 𝑀1 r

    D = diag(𝑀)
    Λ = spdiagm( 1 ./ sqrt.(D) )
    @show typeof(Λ)

    𝑀1_sym = 1/2 .* (𝑀1 .+ 𝑀1')
    𝑀1_sym = Λ * 𝑀1_sym * Λ
    𝑀      = Λ * 𝑀 * Λ
    𝑀1     = Λ * 𝑀1 * Λ

    x = zero(r)

    f = r - 𝑀*x 
    @show norm(f)

    @info "backslash"
    @time 𝑀\r

    # @info "ILU"
    # @time pc = CholeskyPreconditioner(𝑀1_sym, 16)
    # @time gmres!(x, 𝑀, r; Pl=pc)

    @info "AMG RS"
    x .= 0.0
    @time pc = AMGPreconditioner{RugeStuben}(𝑀1)
    @time gmres!(x, 𝑀, r; Pl=pc, abstol=1e-5, reltol=1e-5)

    @info "AMG SA"
    x .= 0.0
    @time pc = AMGPreconditioner{SmoothedAggregation}(𝑀1)
    @time gmres!(x, 𝑀, r; Pl=pc, abstol=1e-6, reltol=1e-6)

    @info "AMG SA"
    x .= 0.0
    @time pc = AMGPreconditioner{SmoothedAggregation}(𝑀1)
    @time bicgstabl!(x, 𝑀, r, 2; Pl=pc, abstol=1e-6, reltol=1e-6)

    @info "AMG SA"
    x .= 0.0
    @time pc = AMGPreconditioner{SmoothedAggregation}(𝑀1)
    @time idrs!(x, 𝑀, r; s=16, Pl=pc, abstol=1e-6, reltol=1e-6)
    
    # # for iter=1:10

    #     # f = r - 𝑀*x 
    #     # @show norm(f)
    # #     x .+=  0.01 .* (M_chol\f)

    # # end

    # f = r - 𝑀*x 
    # @show norm(f)

end

test_solvers_2phase( 400 )

# let 
#     # System matrix 
#     # Non symmetric for whatever reason
#     M = [1.0  0.0  0.0  0.0 0.0;
#          0.0  2.6 -1.0  0.0 0.0;
#          0.0 -0.5  6.1 -1.9 0.0;
#          0.0  0.0 -1.3  7.0 0.0;
#          0.0  0.0  0.0  0.0 1.0]

#     # Preconditioner
#     # Symmetric postive definite (e.g.: Picard preconditioner for a Newton solve)
#     N = [1.0  0.0  0.0  0.0 0.0;
#          0.0  2.0 -1.0  0.0 0.0;
#          0.0 -1.0  2.0 -1.0 0.0;
#          0.0  0.0 -1.0  2.0 0.0;
#          0.0  0.0  0.0  0.0 1.0]
#     N  = sparse(N)

#     # Right-hand side
#     b = [1.0; 2.0; 2.5; 1.5; 4.0]

#     # Cholesky factors
#     N_chol = cholesky(N)
#     # N_chol = lu(M) # This leads to the correct solution in 1 GMRES iteration but it's more expensive than cholesky(N)

#     # Allocate solution arrays
#     x = zeros(size(b))
    
#     # Call GMRES with cholesky factors as preconditionner
#     @show typeof(N_chol)
#     gmres!(x, M, b; Pl=N_chol, verbose=true) # internally uses ldiv!(M_chol, b)

#     # Check that's it's similar to M⁻¹b 
#     @show x .- M\b
# end

# main()
