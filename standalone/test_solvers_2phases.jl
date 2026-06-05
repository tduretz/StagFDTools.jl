using JLD2, Printf, SparseArrays
import LinearAlgebra: norm

@views function KSP_GCR_TwoPhases!( x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, eps::Float64, noisy::Bool, M )
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
    # Coupled
    # Initial residual
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
    # Kuu  = [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
    # Kup  = [M.Vx.Pt; M.Vy.Pt]
    # Kpu  = [M.Pt.Vx M.Pt.Vy]
    # Kpp  =  M.Pt.Pt
    # Pinv = nnz(Kpp) == 0 ? fill(ηb, ndofp) : 1.0 ./ diag(Kpp)
    # Kppi = spdiagm(Pinv)
    # ndofu = size(Kup,1)
    # ndofp = size(Kup,2)
    # @show size(Kup)
    # @show size(Pinv)
    # @show size(Kpu)
    # Kuusc = Kuu - Kup * Kppi * Kpu 
    # PC    =  0.5*(Kuusc + Kuusc') 
    # t = @elapsed Kuuf    = cholesky(Hermitian(PC), check = false)
    # @printf("Cholesky 1 took = %02.2e s\n", t)
    # t = @elapsed Kppf    = cholesky(Hermitian(M.Pf.Pf), check = false)
    # @printf("Cholesky 2 took = %02.2e s\n", t)


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

    # Kuu  = [M.Vx.Vx M.Vx.Vy;
    #         M.Vy.Vx M.Vy.Vy]
    # Kup  = [M.Vx.Pt;
    #       M.Vy.Pt]
    # Kuq  = [M.Vx.Pf;
    #          M.Vy.Pf]
    # Kpu  = [M.Pt.Vx M.Pt.Vy]
    # Kpp  = M.Pt.Pt
    # Kpq = M.Pt.Pf
    # Kqu = [M.Pf.Vx M.Pf.Vy]
    # Kqp = M.Pf.Pt
    # Kqq  = M.Pf.Pf
    # Kvv  = Jvv



    ndofu = size(Jvp,1)
    ndofp = size(Jvp,2)


    J̃pv  = Jpv  - Jpq*spdiagm(1 ./ diag(Jqq)) * Jqu  
    J̃pp  = Jpp  - Jpq*spdiagm(1 ./ diag(Jqq)) * Jpq 
    J̃vv  = Jvv  - Jvp*spdiagm(1 ./ diag(J̃pp)) * Jpv 
    Jqq_f  = cholesky(Hermitian(SparseMatrixCSC(Jqq)), check = false  )        # Cholesky factors
    Juu_f  = cholesky(Hermitian(SparseMatrixCSC(J̃vv)), check = false)        # Cholesky factors
    Jpp_f  = cholesky(Hermitian(SparseMatrixCSC(J̃pp)), check = false)        # Cholesky factors
  
    # Arrays for decoupled problem
    su    = zeros(Float64, ndofu)
    spt   = zeros(Float64, ndofp)
    spf   = zeros(Float64, ndofp)
    
    fu    = zeros(Float64, ndofu)
    fp   = zeros(Float64, ndofp)
    fq   = zeros(Float64, ndofp)
    du    = zeros(Float64, ndofu)
    dp   = zeros(Float64, ndofp)
    dq   = zeros(Float64, ndofp)

    fu  .= f[1:ndofu]
    fp  .= f[ndofu+1:ndofu+ndofp]
    fq  .= f[ndofu+ndofp+1:end]
    if (noisy) @printf("       %1.4d KSP GCR Residual %1.12e %1.12e\n", 0, norm_r, norm_r/norm0); end
    # Solving procedure
     while ( success == 0 && its<maxit ) 
        for i1=1:restart
            # Apply preconditioner, s = PC^{-1} f
            # s = PC\f

            r̃p     = fp -  Jpq*(Jqq_f \ fq)
            s1     = Jpp_f \ r̃p
            r̃u     = fu - Jvp*s1
            du     = Juu_f \ r̃u
            s1     = r̃p - J̃pv*du
            dp     = Jpp_f  \ s1
            s1     = fq - Jpq*dp - Jqu*du
            dq     = Jqq_f \ s1

            # dq  =  (Jqq_f \  (fq - Jqu*du  - Jpq*dp ))
            # dp  =  (Jpp_f \ (fp - Jpv *du  - Jpq*dq ))
            # du  =  (Juu_f \ (fu  - Jvp *dp            ))


            # s[1:ndofu]             .= su
            # s[ndofu+1:ndofu+ndofp] .= spt
            # s[ndofu+ndofp+1:end]   .= spf

            # du = Kuu \ fu
            # dp = Kpp \ (fp - Kpu*du)
            # dq = Kqq \ (fq - Kqu*du - Kqp*dp)

            s[1:ndofu]             .= du
            s[ndofu+1:ndofu+ndofp] .= dp
            s[ndofu+ndofp+1:end]   .= dq


            # s .= 𝑀 \ f

            # Action of Jacobian on s: v = J*s
            # JacobianAction!(v, 𝑀, s; r,kv,T,fc,TW,TE,dx,n)
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
            its              += 1
        end
        its  += 1
        ncyc += 1
    end
    if (noisy>1) @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
    return its
end

function main()
@load "data/matrix_2phases_r200.jld2"

# Sparse monolithic direct 
@time x = 𝑀 \ r 

# Try monolithic GCR
x    .= 0.0 
ϵ     = 1e-6
noisy = false
@time KSP_GCR_TwoPhases!( x, 𝑀, -r, ϵ, noisy, M )

# Kuu  = [M.Vx.Vx M.Vx.Vy;
#             M.Vy.Vx M.Vy.Vy]
#     Kup  = [M.Vx.Pt;
#           M.Vy.Pt]
#     Kuq = [M.Vx.Pf;
#              M.Vy.Pf]
#     Kpu  = [M.Pt.Vx M.Pt.Vy]
#     Kpp  = M.Pt.Pt
#     Kpq = M.Pt.Pf
#     Kqu = [M.Pf.Vx M.Pf.Vy]
#     Kqp = M.Pf.Pt
#     Kqq  = M.Pf.Pf

#     @show size(R.x)

#     ndofu = size(Kup,1)
#     ndofp = size(Kup,2)
#     ru  = r[1:ndofu]
#     rp  = r[ndofu+1:ndofu+ndofp]
#     rq  = r[ndofu+ndofp+1:end]
    

#     Kppi = spdiagm( 1.0 ./ diag(Kpp))
#     K̂uu  = Kuu - Kup*Kppi*Kpu
#     K̂uq  = Kuq - Kup*Kppi*Kpq
#     K̂qu  = Kqu - Kqp*Kppi*Kpu
#     K̂qq  = Kqq - Kqp*Kppi*Kpq

#     @show size(ru)
#     @show size(Kup * Kppi * rp)
#     r̂u   = ru - Kup * Kppi * rp
#     r̂q   = rq - Kqp * Kppi * rp


#     KK = [K̂uu  K̂uq; K̂qu K̂qq]

    # lu(KK)

    # dp = Kppi * (rp − Kpu * du − Kpq*dq )

end

main()