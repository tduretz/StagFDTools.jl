using StagFDTools, JLD2
using Printf, ExtendableSparse, SparseArrays, LinearAlgebra

@views function KSP_GCR_TwoPhases_setup(𝑀::SparseMatrixCSC{Float64, Int64}, M; restart::Int=25, maxit::Int=2000)
    VxVx = sparse(M.Vx.Vx); VxVy = sparse(M.Vx.Vy); VxPt = sparse(M.Vx.Pt)
    VyVx = sparse(M.Vy.Vx); VyVy = sparse(M.Vy.Vy); VyPt = sparse(M.Vy.Pt)
    PtVx = sparse(M.Pt.Vx); PtVy = sparse(M.Pt.Vy); PtPt = sparse(M.Pt.Pt)
    PfVx = sparse(M.Pf.Vx); PfVy = sparse(M.Pf.Vy); PfPt = sparse(M.Pf.Pt); PfPf = sparse(M.Pf.Pf)

    Jvv = [VxVx VxVy; VyVx VyVy]
    Jvp = [VxPt; VyPt]
    Jpv = [PtVx PtVy]
    Jpp = PtPt
    Jpq = PfPt
    Jqu = [PfVx PfVy]
    Jqq = PfPf

    ndofu = size(Jvp, 1)
    ndofp = size(Jvp, 2)
    N = size(𝑀, 1)

    Dqq = spdiagm(0 => 1.0 ./ diag(Jqq))
    J̃pv = Jpv - Jpq * Dqq * Jqu
    J̃pp = Jpp - Jpq * Dqq * Jpq
    Dpp = spdiagm(0 => 1.0 ./ diag(J̃pp))
    J̃vv = Jvv - Jvp * Dpp * Jpv

    Jqq_f = cholesky(Hermitian(SparseMatrixCSC(Jqq)), check=false)
    Juu_f = cholesky(Hermitian(SparseMatrixCSC(J̃vv)), check=false)
    Jpp_f = cholesky(Hermitian(SparseMatrixCSC(J̃pp)), check=false)

    f = zeros(Float64, N)
    v = zeros(Float64, N)
    s = zeros(Float64, N)
    VV = zeros(Float64, N, restart)
    SS = zeros(Float64, N, restart)

    iu = 1:ndofu
    ip = (ndofu + 1):(ndofu + ndofp)
    iq = (ndofu + ndofp + 1):N

    return (;
        A=𝑀, Jvp, Jpq, Jqu, J̃pv, Jqq_f, Juu_f, Jpp_f,
        ndofu, ndofp, restart, maxit,
        f, v, s, fu=view(f, iu), fp=view(f, ip), fq=view(f, iq),
        su=view(s, iu), sp=view(s, ip), sq=view(s, iq),
        VV, SS, VVcols=[view(VV, :, i) for i in 1:restart], SScols=[view(SS, :, i) for i in 1:restart],
        Vnorm2=zeros(Float64, restart),
        du=zeros(Float64, ndofu), dp=zeros(Float64, ndofp), dq=zeros(Float64, ndofp),
        r̃u=zeros(Float64, ndofu), r̃p=zeros(Float64, ndofp),
        tmpp=zeros(Float64, ndofp), tmpq=zeros(Float64, ndofp), tmpq2=zeros(Float64, ndofp),
    )
end

@views function KSP_GCR_TwoPhases_opt!(
    x::Vector{Float64}, b::Vector{Float64}, eps::Float64, noisy::Bool, cache
)
    (; A, Jvp, Jpq, Jqu, J̃pv, Jqq_f, Juu_f, Jpp_f,
        ndofu, ndofp, restart, maxit, f, v, s, fu, fp, fq, su, sp, sq, VVcols, SScols, Vnorm2,
        du, dp, dq, r̃u, r̃p, tmpp, tmpq, tmpq2) = cache

    mul!(f, A, x)
    @. f = b - f
    norm_r = norm(f)
    norm0 = norm_r
    noisy && @printf("       %1.4d KSP GCR Residual %1.12e %1.12e\n", 0, norm_r, norm_r / norm0)

    its = 0
    ncyc = 0

    while its < maxit
        for k in 1:restart
            # Apply block triangular preconditioner, s = PC^{-1} f.
            ldiv!(tmpq, Jqq_f, fq)
            mul!(r̃p, Jpq, tmpq)
            @. r̃p = fp - r̃p

            ldiv!(tmpp, Jpp_f, r̃p)
            mul!(r̃u, Jvp, tmpp)
            @. r̃u = fu - r̃u

            ldiv!(du, Juu_f, r̃u)

            mul!(tmpp, J̃pv, du)
            @. tmpp = r̃p - tmpp
            ldiv!(dp, Jpp_f, tmpp)

            mul!(tmpq, Jpq, dp)
            mul!(tmpq2, Jqu, du)
            @. tmpq = fq - tmpq - tmpq2
            ldiv!(dq, Jqq_f, tmpq)

            copyto!(su, du)
            copyto!(sp, dp)
            copyto!(sq, dq)

            mul!(v, A, s)

            for j in 1:(k - 1)
                Vj = VVcols[j]
                Sj = SScols[j]
                β = dot(Vj, v) / Vnorm2[j]
                BLAS.axpy!(-β, Vj, v)
                BLAS.axpy!(-β, Sj, s)
            end

            den = dot(v, v)
            α = dot(f, v) / den

            BLAS.axpy!(α, s, x)
            BLAS.axpy!(-α, v, f)

            norm_r = norm(f)
            noisy && @printf(
                "  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity 1 res. = %2.2e\n Continuity 2 res. = %2.2e\n",
                its, norm(fu) / sqrt(length(fu)), norm(fp) / sqrt(length(fp)), norm(fq) / sqrt(length(fq))
            )

            if norm(fu) / length(fu) < 1e-10 && norm(fp) / length(fp) < 1e-10 && norm(fq) / length(fq) < 1e-10
                noisy && println("converged")
                @printf("Final residual = %.3e after %d iterations\n", norm_r, its)
                return its
            end

            copyto!(VVcols[k], v)
            copyto!(SScols[k], s)
            Vnorm2[k] = den
            its += 1
        end
        its += 1
        ncyc += 1
    end

    noisy && noisy > 1 && @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r / norm0)
    @printf("Final residual = %.3e after %d iterations\n", norm_r, its)
    return its
end

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
    Jqq_f  = cholesky(Hermitian(SparseMatrixCSC(Jqq)), check = false)        # Cholesky factors
    Juu_f  = cholesky(Hermitian(SparseMatrixCSC(J̃vv)), check = false)        # Cholesky factors
    Jpp_f  = cholesky(Hermitian(SparseMatrixCSC(J̃pp)), check = false)        # Cholesky factors
  
    # Arrays for decoupled problem    
    fu   = zeros(Float64, ndofu)
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

function KSP_GCR_TwoPhases_cached!(x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, eps::Float64, noisy::Bool, M)
    cache = KSP_GCR_TwoPhases_setup(𝑀, M)
    return KSP_GCR_TwoPhases_opt!(x, b, eps, noisy, cache)
end

function main()
    
    r = load(joinpath(@__DIR__, "data/matrix_2phases_r50.jld2"),"r")
    𝑀 = load(joinpath(@__DIR__, "data/matrix_2phases_r50.jld2"),"𝑀")
    M = load(joinpath(@__DIR__, "data/matrix_2phases_r50.jld2"),"M")
   
    # Sparse monolithic direct 
    @info "Direct solve"
    @time x = 𝑀 \ r 

    # Try monolithic GCR
    @info "Direct-iterative solve"
    x    .= 0.0 
    ϵ     = 1e-6
    noisy = false
    @time KSP_GCR_TwoPhases_no_opt!( x, 𝑀, r, ϵ, noisy, M )

    @info "Optimized GCR setup"
    @time cache = KSP_GCR_TwoPhases_setup(𝑀, M)

    @info "Optimized GCR solve"
    x .= 0.0
    @time KSP_GCR_TwoPhases_opt!(x, r, ϵ, noisy, cache)

end

main()
