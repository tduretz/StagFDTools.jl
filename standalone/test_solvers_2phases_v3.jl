using JLD2, Printf, ExtendableSparse, SparseArrays, LinearAlgebra

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

    Jvv  = [M.Vx.Vx M.Vx.Vy;
            M.Vy.Vx M.Vy.Vy]
    Jvp  = [M.Vx.Pt;
            M.Vy.Pt]
    Juq  = [M.Vx.Pf;
            M.Vy.Pf]
    Jpv  = [M.Pt.Vx M.Pt.Vy]
    Jpp  = M.Pt.Pt
    Jpq = M.Pt.Pf
    Jqu = [M.Pf.Vx M.Pf.Vy]
    Jpq = M.Pf.Pt
    Jqq  = M.Pf.Pf

    ndofu = size(Jvp,1)
    ndofp = size(Jvp,2)

    J̃pv    = Jpv  - Jpq*spdiagm(1 ./ diag(Jqq)) * Jqu  
    J̃pp    = Jpp  - Jpq*spdiagm(1 ./ diag(Jqq)) * Jpq 
    J̃vv    = Jvv  - Jvp*spdiagm(1 ./ diag(J̃pp)) * Jpv 
    Jqq_f  = cholesky(Hermitian(SparseMatrixCSC(Jqq)), check = false  )        # Cholesky factors
    Juu_f  = cholesky(Hermitian(SparseMatrixCSC(J̃vv)), check = false)        # Cholesky factors
    Jpp_f  = cholesky(Hermitian(SparseMatrixCSC(J̃pp)), check = false)        # Cholesky factors
  
    # Arrays for decoupled problem    
    fu    = zeros(Float64, ndofu)
    fp   = zeros(Float64, ndofp)
    fq   = zeros(Float64, ndofp)
    du   = zeros(Float64, ndofu)
    dp   = zeros(Float64, ndofp)
    dq   = zeros(Float64, ndofp)
    r̃u   = zeros(Float64, ndofu)
    r̃p   = zeros(Float64, ndofp)

    # indices
    iu = 1:ndofu
    ip = ndofu+1:ndofu+ndofp
    iq = ndofu+ndofp+1:length(f)

    fu  .= f[iu]
    fp  .= f[ip]
    fq  .= f[iq]
    if (noisy) @printf("       %1.4d KSP GCR Residual %1.12e %1.12e\n", 0, norm_r, norm_r/norm0); end
    
    # Solving procedure
    while ( success == 0 && its<maxit ) 
        for i1=1:restart
            # Apply preconditioner, s = PC^{-1} f
            # s .= 𝑀 \ f
            r̃p    .= fp .- Jpq*(Jqq_f \ fq)
            r̃u    .= fu .- Jvp*(Jpp_f \ r̃p)
            du    .= Juu_f \  r̃u
            dp    .= Jpp_f \ (r̃p .- J̃pv*du)
            dq    .= Jqq_f \ (fq .- Jpq*dp .- Jqu*du)
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
            fu   .= f[1:ndofu]
            fp  .= f[ndofu+1:ndofu+ndofp]
            fq  .= f[ndofu+ndofp+1:end]
            noisy && @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity 1 res. = %2.2e\n Continuity 2 res. = %2.2e\n", its, norm(fu)/sqrt(length(fu)), norm(fp)/sqrt(length(fp)), norm(fq)/sqrt(length(fq)))
            if norm(fu)/(length(fu)) < 1e-10 && norm(fp)/(length(fp)) < 1e-10 && norm(fq)/(length(fq)) < 1e-10 #(norm_r < eps * norm0 )
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
    
    @load "data/matrix_2phases_r300_ncons.jld2"

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