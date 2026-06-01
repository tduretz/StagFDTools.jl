using JLD2

@views function KSP_GCR_Stokes!( x::Vector{Float64}, M::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, eps::Float64, noisy::Bool, Kuu::SparseMatrixCSC{Float64, Int64}, Kup::SparseMatrixCSC{Float64, Int64}, Kpu::SparseMatrixCSC{Float64, Int64}, Kpp::SparseMatrixCSC{Float64, Int64} )
    # KSP GCR solver
    norm_r, norm0 = 0.0, 0.0
    N         = length(x)
    restart   = 25
    maxit     = 1000
    ncyc, its = 0, 0
    i1, i2, success=0,0,0
    # Arrays for coupled problem
    f      = zeros(Float64, N)
    v      = zeros(Float64, N)
    s      = zeros(Float64, N)
    val    = zeros(Float64, restart)
    VV     = zeros(Float64, (restart,N))
    SS     = zeros(Float64, (restart,N))
    # Coupled
    # Initial residual
    f      = b - M*x 
    norm_r = norm(f)
    norm0  = norm_r;
    #
    ndofu = size(Kup,1)
    ndofp = size(Kup,2)

    Kppi = spdiagm( 1 ./ diag(Kpp) )

    Kuusc = Kuu - Kup*(Kppi*Kpu) # OK
    PC    =  0.5*(Kuusc + Kuusc') 
    t = @elapsed Kuuf    = cholesky(Hermitian(PC),check = false)
    # Kppf = cholesky(Hermitian(Kpp),check = false)
    # @printf("Cholesky took = %02.2e s\n", t)
    # Arrays for decoupled problem
    su    = zeros(Float64, ndofu)
    fusc  = zeros(Float64, ndofu)
    sp    = zeros(Float64, ndofp)
    fu    = zeros(Float64, ndofu)
    fp    = zeros(Float64, ndofp)
    fu     .= f[1:ndofu]
    fp     .= f[ndofu+1:end]
    if (noisy) @printf("       %1.4d KSP GCR Residual %1.12e %1.12e\n", 0, norm_r, norm_r/norm0); end
    # Solving procedure
     while ( success == 0 && its<maxit ) 
        for i1=1:restart
            # Apply preconditioner, s = PC^{-1} f
            
            # Exact
            # s = M\f

            # Not bad
            # s[1:ndofu]     .= Kuu \ (fu )
            # s[ndofu+1:end] .= Kpp \ (fp - Kpu*su)

            # Schur complement
            fusc           .= fu  - Kup*(Kppi * fp + sp)
            s[1:ndofu]     .= Kuuf \ fusc
            s[ndofu+1:end] .= Kppi * (fp - Kpu*su)

            # Action of Jacobian on s: v = J*s
            v .= M*s
            # -------------------------------
            # Orthogonalisation (modified Gram-Schmidt)
            # -------------------------------
            for i2 = 1:i1-1
                β = dot(VV[i2,:], v) / dot(VV[i2,:], VV[i2,:])
                v .-= β .* VV[i2,:]
                s .-= β .* SS[i2,:]
            end
            # -------------------------------
            # GCR optimal step length
            # -------------------------------
            den = dot(v, v)
            α   = dot(f, v) / den
            # -------------------------------
            # Update solution and residual
            # -------------------------------
            x .+= α .* s
            f .-= α .* v
            # -----------------
            norm_r  = norm(f) 
             fu     .= f[1:ndofu]
             fp     .= f[ndofu+1:end]
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", its, norm(fu)/sqrt(length(fu)), norm(fp)/sqrt(length(fp)))
            if norm(fu)/sqrt(length(fu)) < 1e-10 && norm(fp)/sqrt(length(fp)) < 1e-10                
                success = 1
                println("converged")
                break
            end
            # Store 
             VV[i1,:] .= v
             SS[i1,:] .= s
            its              += 1
        end
        its  += 1
        ncyc += 1
    end
    if (noisy>1) @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
    return its
end

function main()
    @load "data/matrix_2phases_r50.jld2"

    # Sparse monolithic direct 
    @time x = 𝑀 \ r 

    # Try monolithic GCR
    ϵ     = 1e-8
    noisy = true

    # Block matrices
    #################
    Kuu  = [M.Vx.Vx M.Vx.Vy;
            M.Vy.Vx M.Vy.Vy]
    Kup  = [M.Vx.Pt;
            M.Vy.Pt]
    Kuq = [M.Vx.Pf;
            M.Vy.Pf]
    #################
    Kpu  = [M.Pt.Vx M.Pt.Vy]
    Kpp  = M.Pt.Pt
    Kpq  = M.Pt.Pf
     #################
    Kqu  = [M.Pf.Vx M.Pf.Vy]
    Kqp  = M.Pf.Pt
    Kqq  = M.Pf.Pf

    # Dofs
    ndofu = size(Kup,1)
    ndofp = size(Kup,2)

    # Blocks residuals
    ru  = r[1:ndofu]
    rp  = r[ndofu+1:ndofu+ndofp]
    rq  = r[ndofu+ndofp+1:end]

    # # Reduction of the block matrix
    # Kppi = spdiagm( 1.0 ./ diag(Kpp))
    # K̂uu  = Kuu - Kup*Kppi*Kpu
    # K̂uq  = Kuq - Kup*Kppi*Kpq
    # K̂qu  = Kqu - Kqp*Kppi*Kpu
    # K̂qq  = Kqq - Kqp*Kppi*Kpq

    # # Reduction of the block residuals
    # r̂u   = ru - Kup * Kppi * rp
    # r̂q   = rq - Kqp * Kppi * rp

    Kppf = cholesky(Symmetric(Kpp))

    applyKppinv(A) = Kppf \ copy(A)
    applyKppinv_vec(v) = Kppf \ v
    K̂uu = Kuu - Kup * applyKppinv(Kpu)
    K̂uq = Kuq - Kup * applyKppinv(Kpq)
    
    K̂qu = Kqu - Kqp * applyKppinv(Kpu)
    K̂qq = Kqq - Kqp * applyKppinv(Kpq)
    
    r̂u = ru - Kup * applyKppinv_vec(rp)
    r̂q = rq - Kqp * applyKppinv_vec(rp)

    # Monolith matrix / RHS for GCR solver
    K̂ = [K̂uu  K̂uq; K̂qu K̂qq]
    r̂ = [r̂u; r̂q]
    x̂ = zero(r̂)

    # Solve
    x̂ = K̂ \ r̂
    # KSP_GCR_Stokes!( x̂, K̂, r̂, ϵ, noisy, K̂uu, K̂uq, K̂qu, K̂qq)

    # Recover solutions
    u = x̂[1:ndofu]
    q = x̂[ndofu+1:end]
    p = applyKppinv( (rp − Kpu * u − Kpq * q ) )

    x1 = [u; p; q]

    # Check residual
    f = 𝑀*x1 - r
    @show norm(f)

    # Check solutions
    @show norm(x1 - x)

    # # Check u from direct solve and GCR
    # u_direct = x[1:ndofu]
    # @show norm(u - u_direct)

    # # Check p from direct solve and GCR
    # p_direct = x[ndofu+1:ndofu+ndofp]
    # @show norm(p - p_direct)

    # # Check q from direct solve and GCR
    # q_direct = x[ndofu+ndofp+1:end]
    # @show norm(q - q_direct)

    # u_direct = x[1:ndofu]
    # p_direct = x[ndofu+1:ndofu+ndofp]
    # q_direct = x[ndofu+ndofp+1:end]

    # x̂_direct = [u_direct; q_direct]
    # @show norm(K̂*x̂_direct - r̂)

    # #############################
    # p_reconstructed = Kppi * (rp - Kpu*u_direct - Kpq*q_direct)

    # @show norm(p_reconstructed - p_direct)

    # res_p =
    # Kpu*u_direct +
    # Kpp*p_direct +
    # Kpq*q_direct -
    # rp

    # @show norm(res_p)

    # Itest = Kppi * Kpp
    # @show norm(Itest - I, Inf)

    # @show norm(Kpp - Diagonal(diag(Kpp)), Inf)
    # nnz_offdiag = nnz(Kpp - spdiagm(diag(Kpp)))
    # @show nnz_offdiag

    # M.Pt.Pt
end

main()