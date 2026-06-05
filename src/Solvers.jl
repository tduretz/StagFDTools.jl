using SparseArrays

convergence_tolerance(norm0::Float64; reltol::Float64=KSP_RELTOL, abstol::Float64=KSP_ABSTOL) =
    max(abstol, reltol * norm0)

has_converged(norm_r::Float64, norm0::Float64; reltol::Float64=KSP_RELTOL, abstol::Float64=KSP_ABSTOL) =
    norm_r <= convergence_tolerance(norm0; reltol, abstol)

function linear_tol(r, r0, iter; α=9)
    # Inexact Newton-Raphson: Botti paper
    if iter == 1
        return r / 10
    else
        η = r0 / (r0 + α * (r0 - r))
        return η * r
    end
end

function mechanical_solver!(dx, M, r, 𝐊, 𝐐, 𝐐ᵀ, 𝐏, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC;
    solver=:PH, ηb=1e5, ϵ_l=1e-9, niter_l=10, restart=20, noisy=true
)
    if solver == :PH
        # Decoupled Powell & Hestenes using LU as PC
        fu = @views -r[1:size(𝐊, 1)]
        fp = @views -r[size(𝐊, 1)+1:end]
        @time u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu, ηb=1e5, niter_l=10, ϵ_l=ϵ_l, noisy=true)
        @views dx[1:size(𝐊, 1)] .= u
        @views dx[size(𝐊, 1)+1:end] .= p
    elseif solver == :GCR
        # Coupled GCR with Cholesky as PC
        𝐌 = [M.Vx.Vx M.Vx.Vy M.Vx.Pt; M.Vy.Vx M.Vy.Vy M.Vy.Pt; M.Pt.Vx M.Pt.Vy M.Pt.Pt]
        KSP_GCR_Stokes!(dx, 𝐌, .-r, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC, ηb=1e5, ϵ_l=ϵ_l, restart=20)
    end
end

function KSP_GCR_Stokes!(
    x, M, b, Kuu, Kup, Kpu, Kpp;
    ηb=1e3, ϵ_l=1e-9, restart=25, maxit=1000, noisy=true
)

    @views begin

        Kuu = sparse(Kuu)
        Kup = sparse(Kup)
        Kpu = sparse(Kpu)
        Kpp = sparse(Kpp)
        M = sparse(M)

        ndofu = size(Kup, 1)
        ndofp = size(Kup, 2)
        N = length(x)

        Pinv = nnz(Kpp) == 0 ? fill(ηb, ndofp) : 1.0 ./ diag(Kpp)

        Kuusc = Kuu - Kup * spdiagm(Pinv) * Kpu

        Kf = cholesky(Hermitian(Kuusc), check=false)

        f = similar(x)
        s = similar(x)
        v = similar(x)

        mul!(f, M, x)
        @. f = b - f

        norm0 = norm(f)

        fu = f[1:ndofu]
        fp = f[ndofu+1:end]

        su = s[1:ndofu]
        sp = s[ndofu+1:end]

        VV = zeros(eltype(x), N, restart)
        SS = zeros(eltype(x), N, restart)

        tmpu = zeros(eltype(x), ndofu)
        tmpp = zeros(eltype(x), ndofp)
        fusc = zeros(eltype(x), ndofu)

        its = 0

        while its < maxit

            for k = 1:restart

                its += 1

                fill!(s, 0.0)
                @. tmpp = Pinv * fp

                mul!(tmpu, Kup, tmpp)
                @. fusc = fu - tmpu

                ldiv!(su, Kf, fusc)

                mul!(tmpp, Kpu, su)

                @. sp += Pinv * (fp - tmpp)

                mul!(v, M, s)

                for j = 1:k-1
                    hj = dot(v, VV[:, j])
                    BLAS.axpy!(-hj, VV[:, j], v)
                    BLAS.axpy!(-hj, SS[:, j], s)
                end

                nrm = norm(v)

                @. v /= nrm
                @. s /= nrm

                α = dot(f, v)

                BLAS.axpy!(α, s, x)
                BLAS.axpy!(-α, v, f)

                if norm(fu) / sqrt(ndofu) < ϵ_l &&
                   norm(fp) / sqrt(ndofp) < ϵ_l

                    noisy && println("KSP converged in $its iterations")
                    return its
                end

                copyto!(VV[:, k], v)
                copyto!(SS[:, k], s)

            end
        end

        noisy && println("KSP failed after $its iterations")

        return its
    end
end

function DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol, ηb=1e3, niter_l=10, ϵ_l=1e-11, 𝐊_PC=I(size(𝐊, 1)), noisy=true)

    if nnz(𝐏) == 0 # incompressible limit
        𝐏inv = ηb .* I(size(𝐏, 1))
    else # compressible case
        𝐏inv = spdiagm(1.0 ./ diag(𝐏))
    end
    𝐊sc = 𝐊 .- 𝐐 * (𝐏inv * 𝐐ᵀ)
    𝐊sc_PC = 𝐊_PC .- 𝐐 * (𝐏inv * 𝐐ᵀ)

    if fact == :chol
        L_PC = I(size(𝐊sc, 1))
        𝐊fact = cholesky(Hermitian(L_PC * 𝐊sc), check=false)
    elseif fact == :symchol
        L_PC = 𝐊sc'
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
        @time Ksym = L_PC * 𝐊sc
        @time 𝐊fact = cholesky(Hermitian(Ksym), check=false)
    elseif fact == :PCchol
        L_PC = I(size(𝐊sc, 1))
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
    elseif fact == :lu
        L_PC = I(size(𝐊sc, 1))
        @time 𝐊fact = lu(L_PC * 𝐊sc)
    end
    ru = zeros(size(𝐊, 1))
    u = zeros(size(𝐊, 1))
    ru = zeros(size(𝐊, 1))
    fusc = zeros(size(𝐊, 1))
    p = zeros(size(𝐐, 2))
    rp = zeros(size(𝐐, 2))
    # Iterations
    for rit = 1:niter_l
        ru .= fu .- 𝐊 * u .- 𝐐 * p
        rp .= fp .- 𝐐ᵀ * u .- 𝐏 * p
        nrmu, nrmp = norm(ru), norm(rp)
        noisy && @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu / sqrt(length(ru)), nrmp / sqrt(length(rp)))
        if nrmu / sqrt(length(ru)) < ϵ_l && nrmp / sqrt(length(rp)) < ϵ_l
            break
        end
        fusc .= fu .- 𝐐 * (𝐏inv * fp .+ p)
        u .= 𝐊fact \ (L_PC * fusc)

        # # Iterative refinement
        # ϵ_ref = 1e-7
        # for iter_ref=1:10
        #     ru .= 𝐊sc*u .- fusc
        #     @printf("  --> Iterative refinement %02d\n res.   = %2.2e\n", iter_ref, norm(ru)/sqrt(length(ru)))
        #     norm(ru)/sqrt(length(ru)) < ϵ_ref && break
        #     du  = 𝐊fact\(L_PC*ru)
        #     u  .-= du
        # end

        p .+= 𝐏inv * (fp .- 𝐐ᵀ * u .- 𝐏 * p)
    end
    return u, p
end

# ==============================================================================
# Original Custom GCR Solver (from test_solvers_2phases_v3_opt.jl)
# ==============================================================================
@views function KSP_GCR_TwoPhases_setup( M; restart::Int=25, maxit::Int=2000)

    # Construct PC
    VxVx = sparse(M.Vx.Vx); VxVy = sparse(M.Vx.Vy); VxPt = sparse(M.Vx.Pt)
    VyVx = sparse(M.Vy.Vx); VyVy = sparse(M.Vy.Vy); VyPt = sparse(M.Vy.Pt)
    PtVx = sparse(M.Pt.Vx); PtVy = sparse(M.Pt.Vy); PtPt = sparse(M.Pt.Pt); PtPf = sparse(M.Pt.Pf)
    PfVx = sparse(M.Pf.Vx); PfVy = sparse(M.Pf.Vy); PfPt = sparse(M.Pf.Pt); PfPf = sparse(M.Pf.Pf)
    
    Jvv = [VxVx VxVy; VyVx VyVy]
    Jvp = [VxPt; VyPt]
    Jpv = [PtVx PtVy]
    Jpp = PtPt
    Jpq = PtPf
    Jqp = PfPt # added
    Jqu = [PfVx PfVy]
    Jqq = PfPf

    Dqq = spdiagm(0 => 1.0 ./ diag(Jqq))
    # Jpv = Jpv # - Jpq * Dqq * Jqu # not needed
    J̃pp = Jpp - Jpq * Dqq * Jqp
    Dpp = spdiagm(0 => 1.0 ./ diag(J̃pp))
    J̃vv = Jvv - Jvp * Dpp * Jpv

    Jqq_f_sym = cholesky(Hermitian(SparseMatrixCSC(Jqq)), check=false)
    Juu_f_sym = cholesky(Hermitian(SparseMatrixCSC(J̃vv)), check=false)

    ndofu = size(Jvp, 1)
    ndofp = size(Jvp, 2)
    N     = ndofu + ndofp + ndofp

    f = zeros(Float64, N)
    v = zeros(Float64, N)
    s = zeros(Float64, N)
    VV = zeros(Float64, N, restart)
    SS = zeros(Float64, N, restart)

    iu = 1:ndofu
    ip = (ndofu + 1):(ndofu + ndofp)
    iq = (ndofu + ndofp + 1):N

    return (;
        Jqq_f_sym, Juu_f_sym,
        restart, maxit,
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
    x::Vector{Float64}, A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, noisy::Bool, M, cache;
    abstol::Float64=KSP_ABSTOL, reltol=1e-6
)
    (;  Jqq_f_sym, Juu_f_sym, restart, maxit, f, v, s, fu, fp, fq, su, sp, sq, VVcols, SScols, Vnorm2,
        du, dp, dq, r̃u, r̃p, tmpp, tmpq, tmpq2) = cache

    # Construct PC
    VxVx = sparse(M.Vx.Vx); VxVy = sparse(M.Vx.Vy); VxPt = sparse(M.Vx.Pt)
    VyVx = sparse(M.Vy.Vx); VyVy = sparse(M.Vy.Vy); VyPt = sparse(M.Vy.Pt)
    PtVx = sparse(M.Pt.Vx); PtVy = sparse(M.Pt.Vy); PtPt = sparse(M.Pt.Pt); PtPf = sparse(M.Pt.Pf)
    PfVx = sparse(M.Pf.Vx); PfVy = sparse(M.Pf.Vy); PfPt = sparse(M.Pf.Pt); PfPf = sparse(M.Pf.Pf)
    
    Jvv = [VxVx VxVy; VyVx VyVy]
    Jvp = [VxPt; VyPt]
    Jpv = [PtVx PtVy]
    Jpp = PtPt
    Jpq = PtPf
    Jqp = PfPt # added
    Jqu = [PfVx PfVy]
    Jqq = PfPf

    Dqq = spdiagm(0 => 1.0 ./ diag(Jqq))
    # Jpv = Jpv # - Jpq * Dqq * Jqu # not needed
    J̃pp = Jpp - Jpq * Dqq * Jqp
    Dpp = spdiagm(0 => 1.0 ./ diag(J̃pp))
    J̃vv = Jvv - Jvp * Dpp * Jpv

    Jqq_f = cholesky!(Jqq_f_sym, Hermitian(SparseMatrixCSC(Jqq)), check=false)
    Jvv_f = cholesky!(Juu_f_sym, Hermitian(SparseMatrixCSC(J̃vv)), check=false)
    Jpp_f = Dpp #cholesky(Hermitian(SparseMatrixCSC(J̃pp)), check=false) # PC is diagonal !
    #-----------------------------------

    # Initial resiual
    mul!(f, A, x)
    @. f = b - f
    norm_r = norm(f)
    norm0 = norm_r
    tol = convergence_tolerance(norm0; reltol, abstol)
    noisy && @printf("       %1.4d KSP GCR Residual %1.12e %1.12e\n", 0, norm_r, norm_r / norm0)

    if has_converged(norm_r, norm0; reltol, abstol)
        @printf("Final residual = %.3e after %d iterations (tol %.3e)\n", norm_r, 0, tol)
        return 0
    end

    its = 0
    ncyc = 0

    while its < maxit
        for k in 1:restart
            # Apply block triangular preconditioner, s = PC^{-1} f.
            ldiv!(tmpq, Jqq_f, fq)
            mul!(r̃p, Jpq, tmpq)
            @. r̃p = fp - r̃p

            # ldiv!(tmpp, Jpp_f, r̃p) 
            mul!(tmpp, Jpp_f, r̃p) # PC is diagonal !
            mul!(r̃u, Jvp, tmpp)
            @. r̃u = fu - r̃u

            ldiv!(du, Jvv_f, r̃u)

            mul!(tmpp, Jpv, du)
            @. tmpp = r̃p - tmpp
            # ldiv!(dp, Jpp_f, tmpp)
            mul!(dp, Jpp_f, tmpp) # PC is diagonal !

            mul!(tmpq, Jqp, dp)
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

            if has_converged(norm_r, norm0; reltol, abstol)
                noisy && println("converged")
                @printf("Final residual = %.3e after %d iterations (tol %.3e)\n", norm_r, its, tol)
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
    @printf("Final residual = %.3e after %d iterations (tol %.3e)\n", norm_r, its, tol)
    return its
end

function two_phases_mechanical_solver!(dx, M, r, M_PC;
    solver=:PH, solver_cache=0, ηb=1e5, ϵ_l=1e-9, niter_l=10, restart=20, noisy=true
)
    # Two-phases operator as block matrix
    𝑀 = [
        M.Vx.Vx M.Vx.Vy M.Vx.Pt M.Vx.Pf;
        M.Vy.Vx M.Vy.Vy M.Vy.Pt M.Vy.Pf;
        M.Pt.Vx M.Pt.Vy M.Pt.Pt M.Pt.Pf;
        M.Pf.Vx M.Pf.Vy M.Pf.Pt M.Pf.Pf;
    ]
    if solver == :LU
        # Backslash 
        dx .= -𝑀 \ r    
    elseif solver == :GCR
        # Coupled GCR with fancy PC from Raess et al., 2017
        KSP_GCR_TwoPhases_opt!(dx, 𝑀, .-r, noisy, M_PC, solver_cache; reltol=ϵ_l, abstol=ϵ_l )
    end
end