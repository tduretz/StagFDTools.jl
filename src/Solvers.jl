using SparseArrays

function linear_tol(r, r0, iter; α=9)
    # Inexact Newton-Raphson: Botti paper
    if iter==1
        return r/10
    else
        η = r0 / (r0 + α*(r0 - r))
        return η * r
    end
end

function mechanical_solver!( dx, M, r, 𝐊, 𝐐, 𝐐ᵀ, 𝐏, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC; 
    solver=:PH, ηb=1e5, ϵ_l=1e-9, niter_l=10, restart=20, noisy=true
    ) 
    if solver == :PH
        # Decoupled Powell & Hestenes using LU as PC
        fu   = @views -r[1:size(𝐊,1)]
        fp   = @views -r[size(𝐊,1)+1:end]
        @time u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu, ηb=1e5, niter_l=10, ϵ_l=ϵ_l, noisy=true)
        @views dx[1:size(𝐊,1)]     .= u
        @views dx[size(𝐊,1)+1:end] .= p
    elseif solver == :GCR
        # Coupled GCR with Cholesky as PC
        𝐌 = [M.Vx.Vx M.Vx.Vy M.Vx.Pt; M.Vy.Vx M.Vy.Vy M.Vy.Pt; M.Pt.Vx M.Pt.Vy  M.Pt.Pt]            
        KSP_GCR_Stokes!( dx, 𝐌, .-r, 𝐊_PC, 𝐐_PC, 𝐐ᵀ_PC, 𝐏_PC, ηb=1e5, ϵ_l=ϵ_l, restart=20 )
    end
end

function KSP_GCR_Stokes!(
    x, M, b, Kuu, Kup, Kpu, Kpp;
    ηb = 1e3, ϵ_l = 1e-9, restart = 25, maxit = 1000, noisy=true
)

    @views begin

        Kuu = sparse(Kuu)
        Kup = sparse(Kup)
        Kpu = sparse(Kpu)
        Kpp = sparse(Kpp)
        M   = sparse(M)

        ndofu = size(Kup,1)
        ndofp = size(Kup,2)
        N     = length(x)

        Pinv = nnz(Kpp) == 0 ? fill(ηb, ndofp) : 1.0 ./ diag(Kpp)

        Kuusc = Kuu - Kup * spdiagm(Pinv) * Kpu
        Kf    = cholesky(Hermitian(Kuusc), check=false)

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

        VV  = zeros(eltype(x), N, restart)
        SS  = zeros(eltype(x), N, restart)

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
                    hj = dot(v, VV[:,j])
                    BLAS.axpy!(-hj, VV[:,j], v)
                    BLAS.axpy!(-hj, SS[:,j], s)
                end

                nrm = norm(v)

                @. v /= nrm
                @. s /= nrm

                α = dot(f, v)

                BLAS.axpy!( α, s, x)
                BLAS.axpy!(-α, v, f)

                if norm(fu)/sqrt(ndofu) < ϵ_l &&
                   norm(fp)/sqrt(ndofp) < ϵ_l

                    noisy && println("KSP converged in $its iterations")
                    return its
                end

                copyto!(VV[:,k], v)
                copyto!(SS[:,k], s)

            end
        end

        noisy && println("KSP failed after $its iterations")

        return its
    end
end

function DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e3, niter_l=10, ϵ_l=1e-11, 𝐊_PC=I(size(𝐊,1)), noisy=true)
    
    if nnz(𝐏) == 0 # incompressible limit
        𝐏inv  = ηb .* I(size(𝐏,1))
    else # compressible case
        𝐏inv  = spdiagm(1.0 ./diag(𝐏))
    end
    𝐊sc      = 𝐊    .- 𝐐*(𝐏inv*𝐐ᵀ)
    𝐊sc_PC   = 𝐊_PC .- 𝐐*(𝐏inv*𝐐ᵀ)

    if fact == :chol
        L_PC  = I(size(𝐊sc,1))
        𝐊fact = cholesky(Hermitian(L_PC*𝐊sc), check=false)
    elseif fact == :symchol
        L_PC  = 𝐊sc'
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
        @time Ksym = L_PC*𝐊sc
        @time 𝐊fact = cholesky(Hermitian(Ksym), check=false)
    elseif fact == :PCchol
        L_PC  = I(size(𝐊sc,1))
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
    elseif fact == :lu
        L_PC  = I(size(𝐊sc,1))
        @time 𝐊fact = lu(L_PC*𝐊sc)
    end
    ru    = zeros(size(𝐊,1))
    u     = zeros(size(𝐊,1))
    ru    = zeros(size(𝐊,1))
    fusc  = zeros(size(𝐊,1))
    p     = zeros(size(𝐐,2))
    rp    = zeros(size(𝐐,2))
    # Iterations
    for rit=1:niter_l           
        ru   .= fu .- 𝐊*u  .- 𝐐*p
        rp   .= fp .- 𝐐ᵀ*u .- 𝐏*p
        nrmu, nrmp = norm(ru), norm(rp)
        noisy && @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
        if nrmu/sqrt(length(ru)) < ϵ_l && nrmp/sqrt(length(rp)) < ϵ_l
            break
        end
        fusc .= fu  .- 𝐐*(𝐏inv*fp .+ p)
        u    .= 𝐊fact\(L_PC*fusc)

        # # Iterative refinement
        # ϵ_ref = 1e-7
        # for iter_ref=1:10
        #     ru .= 𝐊sc*u .- fusc
        #     @printf("  --> Iterative refinement %02d\n res.   = %2.2e\n", iter_ref, norm(ru)/sqrt(length(ru)))
        #     norm(ru)/sqrt(length(ru)) < ϵ_ref && break
        #     du  = 𝐊fact\(L_PC*ru)
        #     u  .-= du
        # end
   
        p   .+= 𝐏inv*(fp .- 𝐐ᵀ*u .- 𝐏*p)
    end
    return u, p
end