# Stress-form IIM (second order) for the viscous inclusion of Schmid & Podladchikov (2003),
# solved matrix free with dynamic relaxation (DYREL) and Powell-Hestenes/Anderson pressure
# updates, in vectorised Julia (same structure as Stokes2D_DYREL_Decoupled_ViscousInclusion_Anderson.jl).
#
# The setup (level set, interface crossings, one-sided quadratic fits, jump formulas) reuses
# StagFDTools.StokesIIM; see docs/IIM_notes.pdf, steps 5-7.
#
# Two extended fields: V⁻, P⁻ equal V, P except at the +side nodes of a band around Γ, where
# they hold the extension of the -side solution (and likewise V⁺, P⁺):
#   v^s_k = (η_k v_k + s([z] + [∇z]·δ + ½δᵀ[∇∇z]δ)) / η_s,   p^s_k = p_k + s([p] + [∇p]·δ)
# with the jumps at the crossing nearest to node k (δ = x_k - x_Γ), computed from the traces of
# the current iterate. The standard stencils are evaluated on both fields with the sharp
# viscosity of each side, and each equation takes the residual of its own side.
# Pseudo-time stepping is done on z = ηv (rows scaled by 1/η of their side): the extension
# couples a soft row to stiff nodes with weights η₋/η₊, which would dominate the Gershgorin
# bounds in v. In z, the bounds are those of a unit-viscosity problem.
using CairoMakie, Printf, Statistics, LinearAlgebra, SparseArrays, StaticArrays
using StagFDTools, StagFDTools.StokesIIM
import ExactFieldSolutions: Stokes2D_Schmid2003_V, Stokes2D_Schmid2003_p

function Gershgorin_Stokes2D_SC(ηc, ηv, γ, Δx, Δy, ncx, ncy)
    ebE, ebW, ebN, ebS = γ, γ, γ, γ
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    Cxx   = ones(ncx-1, ncy)
    Cxy   = ones(ncx-1, ncy)
    @. Cxx = abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    @. Cxy = abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))
    cVxC  = ones(ncx-1, ncy)
    @. cVxC .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx
    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    Cyy  = ones(ncx, ncy-1)
    Cyx  = ones(ncx, ncy-1)
    @. Cyy = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(ebN ./ Δy .^ 2 + (4 // 3) * ηN ./ Δy .^ 2) + abs.(ebS ./ Δy .^ 2 + (4 // 3) * ηS ./ Δy .^ 2) + abs.((ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    @. Cyx = abs.(ebN ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ebN ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy) + ηW ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy) + ηW ./ (Δx .* Δy))
    cVyC  = ones(ncx, ncy-1)
    @. cVyC .= (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx
    λmaxVx = 1.0./cVxC .* (Cxx .+ Cxy)
    λmaxVy = 1.0./cVyC .* (Cyx .+ Cyy)
    dτVx =  2.0./sqrt.(λmaxVx)*0.98
    dτVy =  2.0./sqrt.(λmaxVy)*0.98
    return dτVx, dτVy, cVxC, cVyC
end

# Interface data: trace operator M (26 quantities per crossing, linear in u = [Vx; Vy; Pt]),
# and for every field and target side the band nodes to extend, with their crossing and δ
function InterfaceSetup(Vx, Vy, Pt, η, ηs, Lx, Ly, ncx, ncy, rc)
    Δ  = (x=Lx/ncx, y=Ly/ncy)
    X  = GenerateGrid((min=-Lx/2, max=Lx/2), (min=-Ly/2, max=Ly/2), Δ, (x=ncx, y=ncy))
    φe = LevelSet((x, y) -> sqrt(x^2 + y^2) - rc, X)
    Γ  = InterfaceCrossings(FineLattice(φe, X, Δ))
    trc = InterfaceTraces(Γ, φe, X, Δ, η, (x=ncx, y=ncy); fit=:quadratic, pressure=true)
    # standalone arrays = StagFDTools extended arrays without their outer layer
    inner(A) = A[2:end-1, 2:end-1]
    φ   = (x=inner(φe.x), y=inner(φe.y), p=inner(φe.c))
    pos = (x=(inner(X.vx_e.x .+ 0 .* X.vx_e.y'), inner(0 .* X.vx_e.x .+ X.vx_e.y')),
           y=(inner(X.vy_e.x .+ 0 .* X.vy_e.y'), inner(0 .* X.vy_e.x .+ X.vy_e.y')),
           p=(inner(X.c_e.x  .+ 0 .* X.c_e.y'),  inner(0 .* X.c_e.x  .+ X.c_e.y')))
    nVx, nVy, nP = length(Vx), length(Vy), length(Pt)
    off = (x=0, y=nVx, p=nVx + nVy)
    sz  = (x=size(Vx), y=size(Vy), p=size(Pt))
    lin(f, I) = off[f] + LinearIndices(sz[f])[I[1]-1, I[2]-1]
    Nc  = length(trc)
    Is, Js, Vs = Int[], Int[], Float64[]
    add!(q, c, j, v) = (push!(Is, (q-1)*Nc + c); push!(Js, j); push!(Vs, v))
    for (c, T) in enumerate(trc)
        t = tangent(T.p.n)
        for (a, f, Tr) in ((1, :x, T.x), (2, :y, T.y)), k in eachindex(Tr.idx)
            j, ηk, is = lin(f, Tr.idx[k]), ηs(Tr.s[k]), Tr.s[k] < 0 ? 1 : 2
            add!(a, c, j, ηk*Tr.v[k]); add!(2+a, c, j, ηk*Tr.t[k]); add!(4+a, c, j, ηk*Tr.tt[k])   # v_Γ, ∂ₛv, ∂ₛₛv
            for b in 1:2
                add!( 6 + (is-1)*4 + (a-1)*2 + b, c, j, ηk*Tr.g[b,k])                              # ∇z on side is
                add!(14 + (is-1)*4 + (a-1)*2 + b, c, j, ηk*dot(Tr.H[b,:,k], t))                    # ∂ₜ∇z on side is
            end
        end
        for k in eachindex(T.pt.idx)
            j, is = lin(:p, T.pt.idx[k]), T.pt.s[k] < 0 ? 1 : 2
            add!(22+is, c, j, T.pt.λ[k]); add!(24+is, c, j, dot(T.pt.g[:,k], t))                  # p, ∂ₜp on side is
        end
    end
    M  = sparse(Is, Js, Vs, 26Nc, nVx + nVy + nP)
    xΓ = [T.p.x for T in trc]
    # band nodes: side -s, within 2.5 cells of Γ, extended to side s
    band = Dict{Tuple{Symbol,Int},Any}()
    for f in (:x, :y, :p), s in (-1, 1)
        ids = findall((side.(φ[f]) .== -s) .& (abs.(φ[f]) .< 2.5Δ.x))
        xk  = [SVector(pos[f][1][I], pos[f][2][I]) for I in ids]
        c   = [argmin(norm.(xΓ .- Ref(x))) for x in xk]
        band[(f, s)] = (ids=LinearIndices(sz[f])[ids], c=c, δ=xk .- xΓ[c])
    end
    return (M=M, trc=trc, Nc=Nc, band=band, side=map(A -> side.(A), φ))
end

# Jumps at all crossings from the traces of the current iterate
function InterfaceJumpsAll(u, IF, η)
    q = reshape(IF.M*u, IF.Nc, 26)
    return map(1:IF.Nc) do c
        n, κ = IF.trc[c].p.n, IF.trc[c].p.κ
        t    = tangent(n)
        G(o, s) = SMatrix{2,2}(q[c,o+(s-1)*4+1], q[c,o+(s-1)*4+3], q[c,o+(s-1)*4+2], q[c,o+(s-1)*4+4])
        σ(G, p) = -p*I + G + G' - 2/3*tr(G)*I
        Tn  = sum(σ(G( 6,s), q[c,22+s])*n                              for s in 1:2)/2
        dTn = sum(σ(G(14,s), q[c,24+s])*n + κ*σ(G(6,s), q[c,22+s])*t   for s in 1:2)/2
        InterfaceJumpsC(η, (neg=0.0, pos=0.0), SVector(q[c,1], q[c,2]), SVector(q[c,3], q[c,4]),
                        SVector(q[c,5], q[c,6]), Tn, dTn, n, κ)
    end
end

# Extension of the side-s solution to the band nodes of the other side
zext(J, c, δ, a) = J[c].z[a] + dot(J[c].∇z[a,:], δ) + dot(δ, J[c].∇∇z[a,:,:]*δ)/2
pext(J, c, δ)    = J[c].p + dot(J[c].∇p, δ)
function extend!(Vxs, Vys, Pts, Vx, Vy, Pt, J, IF, s, ηs)
    Vxs .= Vx; Vys .= Vy; Pts .= Pt
    b = IF.band[(:x, s)]; Vxs[b.ids] .= (ηs(-s) .* Vx[b.ids] .+ s .* zext.(Ref(J), b.c, b.δ, 1)) ./ ηs(s)
    b = IF.band[(:y, s)]; Vys[b.ids] .= (ηs(-s) .* Vy[b.ids] .+ s .* zext.(Ref(J), b.c, b.δ, 2)) ./ ηs(s)
    b = IF.band[(:p, s)]; Pts[b.ids] .= Pt[b.ids] .+ s .* pext.(Ref(J), b.c, b.δ)
end

# Divergence of a velocity field
@views divergence!(∇V, Vx, Vy, Δx, Δy) = (∇V .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy)

# Standard stencils with a uniform viscosity η. The Powell-Hestenes penalty is not added to τ:
# it enters through the pressure P̃ = P - γ_c ∇V_c (see Stokes2D_IIM), so that it uses the
# one-sided divergence of the continuity residual and the penalty of each centroid's side.
@views function residuals!(Rx, Ry, ∇V, Exx, Eyy, Exy, Txx, Tyy, Txy, Vx, Vy, Pt, η, Δx, Δy)
    divergence!(∇V, Vx, Vy, Δx, Δy)
    Exx .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
    Eyy .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
    Exy .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
    Txx .= 2.0.*η.*Exx
    Tyy .= 2.0.*η.*Eyy
    Txy .= 2.0.*η.*Exy
    Rx  .= .-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy
    Ry  .= .-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx
end

@views function Stokes2D_IIM(ncx; ηr=1e3, m_anderson=0, γ_fac=3.0, ϵ=1e-6, ϵ_vel_fac=1e-2, nout=round(Int, 0.4*ncx), iterMax=1e5, verbose=false)
    # Physics
    Lx, Ly  = 10.0, 10.0
    params  = (mm=1.0, mc=ηr, rc=2.0, gr=0.0, er=1.0)
    η       = (neg=params.mc, pos=params.mm)         # inclusion (φ < 0), matrix
    ηs(s)   = s < 0 ? η.neg : η.pos
    # Numerics
    ncy     = ncx
    Δx, Δy  = Lx/ncx, Ly/ncy
    # Arrays (layout of the standalone codes)
    Pt  = zeros(ncx  ,ncy  ); Vx  = zeros(ncx+1,ncy+2); Vy  = zeros(ncx+2,ncy+1)
    ∇V  = zeros(ncx  ,ncy  ); ∇Vpen = zeros(ncx, ncy); P̃ = zeros(ncx, ncy); Exx = zeros(ncx  ,ncy  ); Eyy = zeros(ncx  ,ncy  ); Exy = zeros(ncx+1,ncy+1)
    Txx = zeros(ncx  ,ncy  ); Tyy = zeros(ncx  ,ncy  ); Txy = zeros(ncx+1,ncy+1)
    Rx  = zeros(ncx-1,ncy  ); Ry  = zeros(ncx  ,ncy-1); Rx0 = zeros(ncx-1,ncy  ); Ry0 = zeros(ncx  ,ncy-1)
    dVx = zeros(ncx-1,ncy  ); dVy = zeros(ncx  ,ncy-1); dVxdt = zeros(ncx-1,ncy); dVydt = zeros(ncx,ncy-1)
    βVx = zeros(ncx-1,ncy  ); βVy = zeros(ncx  ,ncy-1); αVx = zeros(ncx-1,ncy  ); αVy = zeros(ncx  ,ncy-1)
    cVx = zeros(ncx-1,ncy  ); cVy = zeros(ncx  ,ncy-1)
    # one field per side: extended values, residuals
    F   = [(Vx=similar(Vx), Vy=similar(Vy), Pt=similar(Pt), Rx=similar(Rx), Ry=similar(Ry), ∇V=similar(∇V)) for _ in 1:2]
    # Coordinates, exact Dirichlet velocities on boundary and ghost nodes
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    Vx  .= [Stokes2D_Schmid2003_V(SVector(x, y), params)[1] for x in xv,  y in yce]
    Vy  .= [Stokes2D_Schmid2003_V(SVector(x, y), params)[2] for x in xce, y in yv ]
    Pa   = [Stokes2D_Schmid2003_p(SVector(x, y), params)    for x in xc,  y in yc ]
    Vx[2:end-1,2:end-1] .= 0; Vy[2:end-1,2:end-1] .= 0
    # Interface
    IF   = InterfaceSetup(Vx, Vy, Pt, η, ηs, Lx, Ly, ncx, ncy, params.rc)
    sVx, sVy, sP = IF.side.x[2:end-1,2:end-1], IF.side.y[2:end-1,2:end-1], IF.side.p
    ηVx, ηVy, ηP = ηs.(sVx), ηs.(sVy), ηs.(sP)
    u    = zeros(length(Vx) + length(Vy) + length(Pt))
    pack!(u) = (u .= vcat(vec(Vx), vec(Vy), vec(Pt)))
    function residuals_IIM!()
        J = InterfaceJumpsAll(pack!(u), IF, η)
        for (i, s) in enumerate((-1, 1))
            Fs = F[i]
            extend!(Fs.Vx, Fs.Vy, Fs.Pt, Vx, Vy, Pt, J, IF, s, ηs)
            divergence!(Fs.∇V, Fs.Vx, Fs.Vy, Δx, Δy)
        end
        ∇V .= ifelse.(sP .< 0, F[1].∇V, F[2].∇V)          # one-sided divergence (continuity residual)
        # augmented-Lagrangian pressure P̃ = P - γ_c ∇V_c (γ_c = γ_fac η of the centroid's side),
        # extended to the other side like P; the Powell-Hestenes update below is P <- P̃
        P̃ .= Pt .- γ_fac .* ηP .* ∇V
        for (i, s) in enumerate((-1, 1))
            Fs = F[i]
            extend!(Fs.Vx, Fs.Vy, Fs.Pt, Vx, Vy, P̃, J, IF, s, ηs)
            residuals!(Fs.Rx, Fs.Ry, ∇Vpen, Exx, Eyy, Exy, Txx, Tyy, Txy, Fs.Vx, Fs.Vy, Fs.Pt, ηs(s), Δx, Δy)
        end
        # each equation takes its own side, scaled to z = ηv
        Rx .= ifelse.(sVx .< 0, F[1].Rx, F[2].Rx) ./ ηVx
        Ry .= ifelse.(sVy .< 0, F[1].Ry, F[2].Ry) ./ ηVy
    end
    # Pseudo-time steps of a unit-viscosity problem (unknown z)
    dtVx, dtVy, cVxC, cVyC = Gershgorin_Stokes2D_SC(ones(ncx,ncy), ones(ncx+1,ncy+1), γ_fac, Δx, Δy, ncx, ncy)
    βVx .= 2 .* dtVx ./ (2 .+ cVx.*dtVx); βVy .= 2 .* dtVy ./ (2 .+ cVy.*dtVy)
    αVx .= (2 .- cVx.*dtVx) ./ (2 .+ cVx.*dtVx); αVy .= (2 .- cVy.*dtVy) ./ (2 .+ cVy.*dtVy)
    # Anderson acceleration
    nP = length(Pt); ΔP = zeros(nP, m_anderson); ΔF = zeros(nP, m_anderson)
    f  = zeros(nP); P_prev = zeros(nP); f_prev = zeros(nP); k_AA = 0; err_prev = Inf
    errVx0 = errVy0 = errPt0 = 1.0
    iter = 1; err = 2ϵ; nPH = 0
    err_evo_V = Float64[]; err_evo_P = Float64[]; err_evo_it = Int[]
    for itPH = 1:100
        nPH += 1
        residuals_IIM!()
        errVx = norm(Rx); errVy = norm(Ry); errPt = norm(∇V)
        if itPH==1 errVx0=errVx; errVy0=errVy; errPt0=errPt; end
        err = maximum([errVx/errVx0, errVy/errVy0, errPt/errPt0])
        verbose && @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, ∇V=%1.3e]\n", itPH, iter, iter/ncx, err, errVx/errVx0, errVy/errVy0, errPt/errPt0)
        push!(err_evo_P, errPt/errPt0); push!(err_evo_V, max(errVx/errVx0, errVy/errVy0)); push!(err_evo_it, iter)
        (err < ϵ) && break
        k_AA = err > err_prev ? 1 : k_AA + 1
        err_prev = err
        ϵ_vel = err*ϵ_vel_fac
        itPT = 0
        while (err > ϵ_vel && itPT <= iterMax)
            itPT += 1
            Rx0 .= Rx; Ry0 .= Ry
            residuals_IIM!()
            Rx ./= cVxC; Ry ./= cVyC
            dVxdt .= αVx.*dVxdt .+ βVx.*Rx
            dVydt .= αVy.*dVydt .+ βVy.*Ry
            Vx[2:end-1,2:end-1] .+= dVxdt.*dtVx ./ ηVx          # z update, back to v
            Vy[2:end-1,2:end-1] .+= dVydt.*dtVy ./ ηVy
            if mod(iter, nout) == 0
                errVx = norm(cVxC.*Rx); errVy = norm(cVyC.*Ry)
                err   = maximum([errVx/errVx0, errVy/errVy0])
                dVx  .= dVxdt.*dtVx; dVy .= dVydt.*dtVy
                λminV = (abs(sum(dVx.*(Rx .- Rx0))) + abs(sum(dVy.*(Ry .- Ry0)))) / (sum(dVx.*dVx) + sum(dVy.*dVy))
                cVx  .= 2*sqrt(λminV)*0.55; cVy .= 2*sqrt(λminV)*0.55
                βVx  .= 2 .* dtVx ./ (2 .+ cVx.*dtVx); βVy .= 2 .* dtVy ./ (2 .+ cVy.*dtVy)
                αVx  .= (2 .- cVx.*dtVx) ./ (2 .+ cVx.*dtVx); αVy .= (2 .- cVy.*dtVy) ./ (2 .+ cVy.*dtVy)
            end
            iter += 1
        end
        # Powell-Hestenes step P <- P̃ = P - γ_c ∇V_c (one-sided divergence), with Anderson
        residuals_IIM!()
        f .= vec(γ_fac .* ηP .* .-∇V)
        if k_AA > 1 && m_anderson > 0
            slot = (k_AA - 2) % m_anderson + 1
            ΔP[:, slot] .= vec(Pt) .- P_prev
            ΔF[:, slot] .= f .- f_prev
        end
        P_prev .= vec(Pt); f_prev .= f
        Pt .+= reshape(f, size(Pt))
        n_AA = min(k_AA - 1, m_anderson)
        if n_AA > 0
            θ = ΔF[:, 1:n_AA] \ f
            all(isfinite, θ) || error("Anderson acceleration is non-finite: θ = $θ")
            Pt .-= reshape((ΔP[:, 1:n_AA] .+ ΔF[:, 1:n_AA]) * θ, size(Pt))
        end
    end
    P  = Pt .- mean(Pt)
    Pa = Pa .- mean(Pa)
    return (; P, Pa, xc, yc, nPH, nDR=iter-1, err, ϵP=maximum(abs.(P .- Pa)), err_evo_V, err_evo_P, err_evo_it)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Convergence with resolution against the exact solution and the assembled (direct) IIM solvers
    include(joinpath(@__DIR__, "..", "..", "examples", "_Stokes_IIM", "IIM_Schmid2003.jl"))
    ncs = [40, 80, 160]
    res = Dict()
    for ηr in (10.0, 1e3), n in ncs
        p = (mm=1.0, mc=ηr, rc=2.0, gr=0.0, er=1.0)
        t = @elapsed r = Stokes2D_IIM(n; ηr)
        dz = IIM_Schmid2003(n, p; order=2, fit=:quadratic)
        ds = IIM_Schmid2003(n, p; order=2, fit=:quadratic, form=:stress)
        res[(ηr, n)] = (r=r, dz=dz.ϵP.Linf, ds=ds.ϵP.Linf, Pds=ds.P)
        @printf("ηr = %-6g nc = %3d: DYREL L∞(P) = %.4e (direct z-form %.4e, stress form %.4e), |P - P_direct,stress| = %.2e, iter/nx = %4.0f, PH = %2d (%.0f s)\n",
                ηr, n, r.ϵP, dz.ϵP.Linf, ds.ϵP.Linf, maximum(abs.(r.P .- ds.P)), r.nDR/n, r.nPH, t)
    end
    h   = 10.0 ./ ncs
    fig = Figure(size=(1500, 820), fontsize=15)
    ax  = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="h", ylabel="max|P - P_ex|", title="Pressure error, η_inc/η_mat = 10³")
    scatterlines!(ax, h, [res[(1e3, n)].r.ϵP for n in ncs], label="DYREL, matrix free")
    scatterlines!(ax, h, [res[(1e3, n)].ds for n in ncs], linestyle=:dash, label="direct, stress form")
    scatterlines!(ax, h, [res[(1e3, n)].dz for n in ncs], linestyle=:dot, label="direct, z-form")
    lines!(ax, h, 0.3 .* (h ./ h[1]).^2, color=:black, linestyle=:dashdot, label="O(h²)")
    axislegend(ax, position=:rb, framevisible=false)
    ax  = Axis(fig[1, 2], xlabel="nc", ylabel="iterations / nc", title="DYREL iterations (tolerance 10⁻⁶)")
    for ηr in (10.0, 1e3)
        scatterlines!(ax, ncs, [res[(ηr, n)].r.nDR / n for n in ncs], label="η_inc/η_mat = $(ηr)")
    end
    axislegend(ax, position=:lt, framevisible=false)
    r = res[(1e3, ncs[end])].r
    ax  = Axis(fig[1, 3], xlabel="iterations", ylabel="log₁₀ relative residual", title="nc = $(ncs[end]), η_inc/η_mat = 10³")
    lines!(ax, r.err_evo_it, log10.(r.err_evo_V), label="momentum")
    lines!(ax, r.err_evo_it, log10.(r.err_evo_P), label="continuity")
    axislegend(ax, framevisible=false)
    θ    = range(0, 2π, 200)
    pmax = maximum(abs, r.Pa)
    for (j, (name, P)) in enumerate((("DYREL, matrix free", r.P), ("exact", r.Pa)))
        ax = Axis(fig[2, j], aspect=DataAspect(), title="P, $name")
        hm = heatmap!(ax, r.xc, r.yc, P, colormap=:vik, colorrange=(-pmax, pmax))
        lines!(ax, 2cos.(θ), 2sin.(θ), color=:black, linewidth=0.8); limits!(ax, -3.5, 3.5, -3.5, 3.5)
        j == 2 && Colorbar(fig[2, 4], hm)
    end
    ax = Axis(fig[2, 3], aspect=DataAspect(), title="log₁₀ |P_DYREL - P_direct, stress|")
    hm = heatmap!(ax, r.xc, r.yc, log10.(abs.(r.P .- res[(1e3, ncs[end])].Pds) .+ 1e-12), colormap=:viridis, colorrange=(-6, 0))
    lines!(ax, 2cos.(θ), 2sin.(θ), color=:white, linewidth=0.8); limits!(ax, -3.5, 3.5, -3.5, 3.5)
    Colorbar(fig[2, 5], hm)
    save(joinpath(@__DIR__, "Stokes2D_DYREL_IIM_Schmid.png"), fig, px_per_unit=2)
end
