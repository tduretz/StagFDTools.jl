using StagFDTools, JLD2
using Printf, ExtendableSparse, SparseArrays, LinearAlgebra
using IterativeSolvers

const KSP_RESTART = 25
const KSP_MAXIT = 2000
const KSP_RELTOL = 1e-10
const KSP_ABSTOL = 1e-10
const KSP_IDRS_S = 4
const KSP_BICGSTABL_L = 4

convergence_tolerance(norm0::Float64; reltol::Float64=KSP_RELTOL, abstol::Float64=KSP_ABSTOL) =
    max(abstol, reltol * norm0)

has_converged(norm_r::Float64, norm0::Float64; reltol::Float64=KSP_RELTOL, abstol::Float64=KSP_ABSTOL) =
    norm_r <= convergence_tolerance(norm0; reltol, abstol)

# ==============================================================================
# Custom Optimized Preconditioner for Two-Phase Flow Solver
# ==============================================================================
struct TwoPhasesPreconditioner{FQ, FU, FP, MJvp, MJpq, MJqu, MJpv, IndU, IndP, IndQ, VdU, VdP, VdQ, VrU, VrP, Vtmpp, Vtmpq, Vtmpq2}
    Jqq_f::FQ
    Juu_f::FU
    Jpp_f::FP
    Jvp::MJvp
    Jpq::MJpq
    Jqu::MJqu
    J̃pv::MJpv
    iu::IndU
    ip::IndP
    iq::IndQ
    du::VdU
    dp::VdP
    dq::VdQ
    r̃u::VrU
    r̃p::VrP
    tmpp::Vtmpp
    tmpq::Vtmpq
    tmpq2::Vtmpq2
end

function TwoPhasesPreconditioner(𝑀::SparseMatrixCSC{Float64, Int64}, M)
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

    iu = 1:ndofu
    ip = (ndofu + 1):(ndofu + ndofp)
    iq = (ndofu + ndofp + 1):N

    du = zeros(Float64, ndofu)
    dp = zeros(Float64, ndofp)
    dq = zeros(Float64, ndofp)
    r̃u = zeros(Float64, ndofu)
    r̃p = zeros(Float64, ndofp)
    tmpp = zeros(Float64, ndofp)
    tmpq = zeros(Float64, ndofp)
    tmpq2 = zeros(Float64, ndofp)

    return TwoPhasesPreconditioner(
        Jqq_f, Juu_f, Jpp_f, Jvp, Jpq, Jqu, J̃pv,
        iu, ip, iq, du, dp, dq, r̃u, r̃p, tmpp, tmpq, tmpq2
    )
end

import LinearAlgebra: ldiv!, \

@views function LinearAlgebra.ldiv!(y::AbstractVector{Float64}, P::TwoPhasesPreconditioner, x::AbstractVector{Float64})
    xu = view(x, P.iu)
    xp = view(x, P.ip)
    xq = view(x, P.iq)

    yu = view(y, P.iu)
    yp = view(y, P.ip)
    yq = view(y, P.iq)

    ldiv!(P.tmpq, P.Jqq_f, xq)
    mul!(P.r̃p, P.Jpq, P.tmpq)
    @. P.r̃p = xp - P.r̃p

    ldiv!(P.tmpp, P.Jpp_f, P.r̃p)
    mul!(P.r̃u, P.Jvp, P.tmpp)
    @. P.r̃u = xu - P.r̃u

    ldiv!(P.du, P.Juu_f, P.r̃u)

    mul!(P.tmpp, P.J̃pv, P.du)
    @. P.tmpp = P.r̃p - P.tmpp
    ldiv!(P.dp, P.Jpp_f, P.tmpp)

    mul!(P.tmpq, P.Jpq, P.dp)
    mul!(P.tmpq2, P.Jqu, P.du)
    @. P.tmpq = xq - P.tmpq - P.tmpq2
    ldiv!(P.dq, P.Jqq_f, P.tmpq)

    copyto!(yu, P.du)
    copyto!(yp, P.dp)
    copyto!(yq, P.dq)

    return y
end

function LinearAlgebra.:\(P::TwoPhasesPreconditioner, x::AbstractVector{Float64})
    y = similar(x)
    ldiv!(y, P, x)
    return y
end

function LinearAlgebra.ldiv!(P::TwoPhasesPreconditioner, x::AbstractVector{Float64})
    ldiv!(x, P, x)
    return x
end

# ==============================================================================
# Original Custom GCR Solver (from test_solvers_2phases_v3_opt.jl)
# ==============================================================================
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
    x::Vector{Float64}, b::Vector{Float64}, reltol::Float64, noisy::Bool, cache;
    abstol::Float64=KSP_ABSTOL
)
    (; A, Jvp, Jpq, Jqu, J̃pv, Jqq_f, Juu_f, Jpp_f,
        ndofu, ndofp, restart, maxit, f, v, s, fu, fp, fq, su, sp, sq, VVcols, SScols, Vnorm2,
        du, dp, dq, r̃u, r̃p, tmpp, tmpq, tmpq2) = cache

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

function KSP_GMRES_TwoPhases_iterativesolvers!(
    x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, P::TwoPhasesPreconditioner;
    restart::Int=KSP_RESTART, maxiter::Int=KSP_MAXIT, reltol::Float64=KSP_RELTOL, abstol::Float64=KSP_ABSTOL,
    initially_zero::Bool=false, verbose::Bool=false
)
    residual = similar(b)
    mul!(residual, 𝑀, x)
    @. residual = b - residual
    norm_r = norm(residual)
    norm0 = norm_r
    tol = convergence_tolerance(norm0; reltol, abstol)
    residuals = Float64[norm_r]
    total_iters = 0
    total_mvps = 1
    converged = has_converged(norm_r, norm0; reltol, abstol)

    while !converged && total_iters < maxiter
        iters_this_cycle = min(restart, maxiter - total_iters)
        _, history = gmres!(
            x, 𝑀, b;
            Pl=P,
            restart=restart,
            maxiter=iters_this_cycle,
            reltol=0.0,
            abstol=0.0,
            initially_zero=false,
            log=true,
            verbose=verbose,
        )

        total_iters += history.iters
        total_mvps += history.mvps
        mul!(residual, 𝑀, x)
        @. residual = b - residual
        norm_r = norm(residual)
        push!(residuals, norm_r)
        converged = has_converged(norm_r, norm0; reltol, abstol)

        history.iters == 0 && break
    end

    return x, (; isconverged=converged, iters=total_iters, mvps=total_mvps, residuals, norm0, tol)
end

function KSP_GMRES_TwoPhases_iterativesolvers!(
    x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, M;
    kwargs...
)
    P = TwoPhasesPreconditioner(𝑀, M)
    return KSP_GMRES_TwoPhases_iterativesolvers!(x, 𝑀, b, P; kwargs...)
end

function KSP_IDRS_TwoPhases_iterativesolvers!(
    x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, P::TwoPhasesPreconditioner;
    s::Int=KSP_IDRS_S, maxiter::Int=KSP_MAXIT, reltol::Float64=KSP_RELTOL, abstol::Float64=KSP_ABSTOL,
    verbose::Bool=false
)
    initial_residual = b - 𝑀 * x
    norm0 = norm(initial_residual)
    tol = convergence_tolerance(norm0; reltol, abstol)

    x, history = idrs!(
        x, 𝑀, b;
        Pl=P,
        s=s,
        maxiter=maxiter,
        reltol=reltol,
        abstol=abstol,
        log=true,
        verbose=verbose,
    )

    residual = b - 𝑀 * x
    norm_r = norm(residual)

    return x, (; isconverged=history.isconverged, iters=history.iters, mvps=history.mvps,
        residuals=history[:resnorm], true_residual=norm_r, norm0, tol, s)
end

function KSP_IDRS_TwoPhases_iterativesolvers!(
    x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, M;
    kwargs...
)
    P = TwoPhasesPreconditioner(𝑀, M)
    return KSP_IDRS_TwoPhases_iterativesolvers!(x, 𝑀, b, P; kwargs...)
end

function KSP_BICGSTABL_TwoPhases_iterativesolvers!(
    x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, P::TwoPhasesPreconditioner;
    l::Int=KSP_BICGSTABL_L, max_mv_products::Int=KSP_MAXIT, reltol::Float64=KSP_RELTOL,
    abstol::Float64=KSP_ABSTOL, verbose::Bool=false
)
    initial_residual = b - 𝑀 * x
    norm0 = norm(initial_residual)
    tol = convergence_tolerance(norm0; reltol, abstol)

    residuals = Float64[norm0]
    total_iters = 0
    total_mvps = 1
    true_converged = has_converged(norm0, norm0; reltol, abstol)

    while !true_converged && total_mvps < max_mv_products
        mvps_this_cycle = min(2 * l, max_mv_products - total_mvps)
        _, history = bicgstabl!(
            x, 𝑀, b, l;
            Pl=P,
            max_mv_products=mvps_this_cycle,
            reltol=0.0,
            abstol=0.0,
            log=true,
            verbose=verbose,
        )

        total_iters += history.iters
        total_mvps += history.mvps

        residual = b - 𝑀 * x
        norm_r = norm(residual)
        push!(residuals, norm_r)
        true_converged = has_converged(norm_r, norm0; reltol, abstol)

        history.iters == 0 && break
    end

    residual = b - 𝑀 * x
    norm_r = norm(residual)

    return x, (; isconverged=true_converged, iters=total_iters, mvps=total_mvps,
        residuals, true_residual=norm_r, norm0, tol, l)
end

function KSP_BICGSTABL_TwoPhases_iterativesolvers!(
    x::Vector{Float64}, 𝑀::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, M;
    kwargs...
)
    P = TwoPhasesPreconditioner(𝑀, M)
    return KSP_BICGSTABL_TwoPhases_iterativesolvers!(x, 𝑀, b, P; kwargs...)
end

# ==============================================================================
# Main Demonstration and Verification
# ==============================================================================
function main()
    # Load test matrix and vectors (assuming we are in the standalone/ directory or project root)
    filepath = joinpath(@__DIR__, "matrix_2phases_r50.jld2")
    if !isfile(filepath)
        filepath = joinpath(@__DIR__, "data/matrix_2phases_r50.jld2")
    end
    
    @info "Loading data from $filepath"
    r = load(filepath, "r")
    𝑀 = load(filepath, "𝑀")
    M = load(filepath, "M")

    # Sparse monolithic direct solve
    @info "--- 1. Direct Solve ---"
    x_direct = zero(r)
    @time x_direct .= 𝑀 \ r 
    @printf("Direct true residual: %.3e\n", norm(r .- 𝑀 * x_direct))

    # Original Custom GCR Setup and Solve (optimized)
    @info "--- 2. Optimized Custom GCR Solver ---"
    @time cache = KSP_GCR_TwoPhases_setup(𝑀, M)
    x_gcr = zero(r)
    reltol = KSP_RELTOL
    abstol = KSP_ABSTOL
    noisy = false
    @printf("Convergence criterion: norm(b - A*x) <= max(%.1e, %.1e * norm0)\n", abstol, reltol)
    @time KSP_GCR_TwoPhases_opt!(x_gcr, r, reltol, noisy, cache; abstol)
    
    # Verify GCR difference from direct solve
    @printf("GCR error vs Direct solve: %.3e\n", norm(x_gcr .- x_direct))

    # IterativeSolvers.jl GMRES Solve using our custom Optimized Preconditioner
    @info "--- 3. IterativeSolvers.jl GMRES with Custom Preconditioner ---"
    @time precond = TwoPhasesPreconditioner(𝑀, M)
    x_gmres = zero(r)
    
    # Warm up / run GMRES
    @time begin
        # GMRES is advanced one restart cycle at a time so it uses the same true-residual
        # stopping criterion as the custom GCR solver.
        x_gmres, history = KSP_GMRES_TwoPhases_iterativesolvers!(
            x_gmres, 𝑀, r, precond;
            restart=KSP_RESTART,
            maxiter=KSP_MAXIT,
            reltol,
            abstol,
            initially_zero=true,
        )
    end
    
    @printf("GMRES converged: %s\n", history.isconverged)
    @printf("GMRES iterations: %d, matrix-vector products: %d\n", history.iters, history.mvps)
    @printf("GMRES true residual: %.3e (tol %.3e)\n", history.residuals[end], history.tol)
    @printf("GMRES error vs Direct solve: %.3e\n", norm(x_gmres .- x_direct))
    @printf("GMRES error vs GCR solve:    %.3e\n", norm(x_gmres .- x_gcr))

    # IterativeSolvers.jl IDR(s) Solve using our custom Optimized Preconditioner
    @info "--- 4. IterativeSolvers.jl IDR(s) with Custom Preconditioner ---"
    x_idrs = zero(r)

    @time begin
        x_idrs, history_idrs = KSP_IDRS_TwoPhases_iterativesolvers!(
            x_idrs, 𝑀, r, precond;
            s=KSP_IDRS_S,
            maxiter=KSP_MAXIT,
            reltol,
            abstol,
        )
    end

    @printf("IDR(%d) converged: %s\n", history_idrs.s, history_idrs.isconverged)
    @printf("IDR(%d) iterations: %d, matrix-vector products: %d\n", history_idrs.s, history_idrs.iters, history_idrs.mvps)
    @printf("IDR(%d) true residual: %.3e (tol %.3e)\n", history_idrs.s, history_idrs.true_residual, history_idrs.tol)
    @printf("IDR(%d) error vs Direct solve: %.3e\n", history_idrs.s, norm(x_idrs .- x_direct))
    @printf("IDR(%d) error vs GCR solve:    %.3e\n", history_idrs.s, norm(x_idrs .- x_gcr))

    # IterativeSolvers.jl BiCGStab(l) Solve using our custom Optimized Preconditioner
    @info "--- 5. IterativeSolvers.jl BiCGStab(l) with Custom Preconditioner ---"
    x_bicg = zero(r)

    @time begin
        x_bicg, history_bicg = KSP_BICGSTABL_TwoPhases_iterativesolvers!(
            x_bicg, 𝑀, r, precond;
            l=KSP_BICGSTABL_L,
            max_mv_products=KSP_MAXIT,
            reltol,
            abstol,
        )
    end

    @printf("BiCGStab(%d) converged: %s\n", history_bicg.l, history_bicg.isconverged)
    @printf("BiCGStab(%d) iterations: %d, matrix-vector products: %d\n",
        history_bicg.l, history_bicg.iters, history_bicg.mvps)
    @printf("BiCGStab(%d) true residual: %.3e (tol %.3e)\n",
        history_bicg.l, history_bicg.true_residual, history_bicg.tol)
    @printf("BiCGStab(%d) error vs Direct solve: %.3e\n",
        history_bicg.l, norm(x_bicg .- x_direct))
    @printf("BiCGStab(%d) error vs GCR solve:    %.3e\n",
        history_bicg.l, norm(x_bicg .- x_gcr))
end

main()
