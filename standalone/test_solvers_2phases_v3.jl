using JLD2, Printf, ExtendableSparse, SparseArrays, LinearAlgebra, IncompleteLU
using IncompleteLU

@views function KSP_GCR_TwoPhases_no_opt!( x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, eps::Float64, noisy::Bool, M )
    # KSP GCR solver
    norm_r, norm0 = 0.0, 0.0
    N         = length(x)
    restart   = 25
    maxit     = 2000
    ncyc, its = 0, 0
    i1, i2, success=0,0,0
    # Arrays for coupled problem
    f      = zeros(Float64, N)
    v      = zeros(Float64, N)
    s      = zeros(Float64, N)
    VV     = zeros(Float64, (restart,N))
    SS     = zeros(Float64, (restart,N))
    # Initial couples residual
    f      = b - 𝑀*x 
    norm_r = norm(f)
    norm0  = norm_r;
    #
    𝑀 = [
        M.Vx.Vx M.Vx.Vy M.Vx.Pt M.Vx.Pf;
        M.Vy.Vx M.Vy.Vy M.Vy.Pt M.Vy.Pf;
        M.Pt.Vx M.Pt.Vy M.Pt.Pt M.Pt.Pf;
        M.Pf.Vx M.Pf.Vy M.Pf.Pt M.Pf.Pf;
    ]

    Juu = [M.Vx.Vx M.Vx.Vy;
            M.Vy.Vx M.Vy.Vy]
    Jup = [M.Vx.Pt;
            M.Vy.Pt]
    Juq = [M.Vx.Pf;
            M.Vy.Pf]
    Jpu = [M.Pt.Vx M.Pt.Vy]
    Jpp = M.Pt.Pt
    Jpq = M.Pt.Pf
    Jqu = [M.Pf.Vx M.Pf.Vy]
    Jqp = M.Pf.Pt
    Jqq = M.Pf.Pf

    ndofu = size(Jup,1)
    ndorp = size(Jup,2)

    J̃pu    = Jpu  #- Jpq*spdiagm(1 ./ diag(Jqq)) * Jqu#  - Jpu*spdiagm(1 ./ diag(Juu)) * Jup 
    J̃pp    = Jpp  - Jpq*spdiagm(1 ./ diag(Jqq)) * Jqp # - Jpu*spdiagm(1 ./ diag(Juu)) * Jup 
    J̃uu    = Juu  - Jup*spdiagm(1 ./ diag(Jpp)) * Jpu # - Juq*spdiagm(1 ./ diag(Jqq)) * Jqu 
    Jqq_f  = cholesky(Hermitian(SparseMatrixCSC(Jqq)), check = false)        # Cholesky factors
    Juu_f  = cholesky(Hermitian(SparseMatrixCSC(J̃uu)), check = false)        # Cholesky factors
    Jpp_f  = cholesky(Hermitian(SparseMatrixCSC(J̃pp)), check = false)        # Cholesky factors
  
    # Arrays for decoupled problem    
    ru    = zeros(Float64, ndofu)
    rp   = zeros(Float64, ndorp)
    rq   = zeros(Float64, ndorp)
    du   = zeros(Float64, ndofu)
    dp   = zeros(Float64, ndorp)
    dq   = zeros(Float64, ndorp)
    r̃u   = zeros(Float64, ndofu)
    r̃p   = zeros(Float64, ndorp)
    r̃q   = zeros(Float64, ndorp)

    # indices
    iu = 1:ndofu
    ip = ndofu+1:ndofu+ndorp
    iq = ndofu+ndorp+1:length(f)

    ru  .= f[iu]
    rp  .= f[ip]
    rq  .= f[iq]
    if (noisy) @printf("       %1.4d KSP GCR Residual %1.12e %1.12e\n", 0, norm_r, norm_r/norm0); end
    
    # Solving procedure
    while ( success == 0 && its<maxit ) 
        for i1=1:restart
            # Apply preconditioner, s = PC^{-1} f
            # s .= 𝑀 \ f
            r̃p    .= rp .- Jpq*(Jqq_f \ rq)# .- Jpu*(Juu_f \ ru)
            r̃u    .= ru .- Jup*(Jpp_f \ r̃p)# .- Juq*(Jqq_f \ rq)
            du    .= Juu_f \  r̃u
            dp    .= Jpp_f \ (r̃p .- J̃pu*du)
            dq    .= Jqq_f \ (rq .- Jqp*dp .- Jqu*du)
            s[iu] .= du
            s[ip] .= dp
            s[iq] .= dq
            # Action of Jacobian on s: v = J*s
            v .= 𝑀*s
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
            ru   .= f[1:ndofu]
            rp  .= f[ndofu+1:ndofu+ndorp]
            rq  .= f[ndofu+ndorp+1:end]
            noisy && @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity 1 res. = %2.2e\n Continuity 2 res. = %2.2e\n", its, norm(ru)/sqrt(length(ru)), norm(rp)/sqrt(length(rp)), norm(rq)/sqrt(length(rq)))
            if norm(ru)/(length(ru)) < 1e-10 && norm(rp)/(length(rp)) < 1e-10 && norm(rq)/(length(rq)) < 1e-10 #(norm_r < eps * norm0 )
                success = 1
                println("converged")
                break
            end
            # Store 
            VV[i1,:] .= v
            SS[i1,:] .= s
            its      += 1
        end
        its  += 1
        ncyc += 1
    end
    if (noisy>1) @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
    @printf("Final residual = %.3e after %d iterations\n", norm_r, its)

    return its
end

function main()

    # Load test matrix and vectors (assuming we are in the standalone/ directory or project root)
    filepath = joinpath(@__DIR__, "../data/matrix_2phases_r300.jld2")
    r = load(filepath, "r")
    𝑀 = load(filepath, "𝑀")
 
    # Load test preconditioner
    filepath = joinpath(@__DIR__, "../data/matrix_2phases_r300_ncons.jld2")
    M = load(filepath, "M")

    # Sparse monolithic direct 
    @info "Direct solve"
    @time x = 𝑀 \ r 

    # Try monolithic GCR
    @info "Direct-iterative solve"
    x    .= 0.0 
    ϵ     = 1e-6
    noisy = false
    @time KSP_GCR_TwoPhases_no_opt!( x, 𝑀, r, ϵ, noisy, M )

end

main()