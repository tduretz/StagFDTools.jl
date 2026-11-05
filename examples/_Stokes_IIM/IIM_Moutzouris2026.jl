using StagFDTools, StagFDTools.StokesIIM, LinearAlgebra, StaticArrays
import Statistics: mean
import ExactFieldSolutions: Stokes2D_Moutzouris_circle, Stokes2D_Moutzouris_ellipse

# IIM solution of the compressible viscous inclusion of Moutzouris (2026) (circle, or ellipse
# with semi-axes (t ri, ri)), with exact Dirichlet velocities. The pressure is determined
# (compressible), so errors are absolute.
# - order, fit: IIM variant (see AssembleIIM)
# - exact_jumps: diagnostic, exact jumps on the interface instead of reconstructed ones
# - form: :z (Landry, AssembleIIM) or :stress (StagFDTools-style stress form, AssembleIIMStress)
# - extend: extension used by the stress form (:z or :v)

Moutzouris(shape) = shape == :circle ? Stokes2D_Moutzouris_circle : Stokes2D_Moutzouris_ellipse
semi_axes(shape, params) = shape == :circle ? (params.ri, params.ri) : (params.t * params.ri, params.ri)

# Closest point of the ellipse (a cos θ, b sin θ) to x, outward normal and curvature
function ellipse_projection(x, a, b)
    θ = atan(x[2] / b, x[1] / a)
    for _ in 1:20
        e  = SVector(a * cos(θ), b * sin(θ)) - x
        d  = SVector(-a * sin(θ), b * cos(θ))
        dd = SVector(-a * cos(θ), -b * sin(θ))
        θ -= dot(e, d) / (dot(d, d) + dot(e, dd))
    end
    n = normalize(SVector(cos(θ) / a, sin(θ) / b))
    return SVector(a * cos(θ), b * sin(θ)), n, a * b / (a^2 * sin(θ)^2 + b^2 * cos(θ)^2)^1.5
end

function IIM_Moutzouris2026(n, params; shape=:circle, order=1, fit=:linear, exact_jumps=false, L=1.0, form=:z, extend=:z)
    nc = (x=n, y=n)
    Δ  = (x=L / n, y=L / n)
    X  = GenerateGrid((min=-L / 2, max=L / 2), (min=-L / 2, max=L / 2), Δ, nc)
    a, b = semi_axes(shape, params)
    φ  = LevelSet((x, y) -> b * (sqrt((x / a)^2 + (y / b)^2) - 1), X)
    η  = (neg=params.ηi, pos=params.ηm)
    ξ  = (neg=params.ξi, pos=params.ξm)
    S(x) = Moutzouris(shape)(x; params)
    ηof(f) = side(f) < 0 ? η.neg : η.pos
    Va  = (x=[S(SVector(X.vx_e.x[i], X.vx_e.y[j])).V[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
           y=[S(SVector(X.vy_e.x[i], X.vy_e.y[j])).V[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    zbc = (x=ηof.(φ.x) .* Va.x, y=ηof.(φ.y) .* Va.y)

    # Diagnostic: exact jumps at the projection on Γ (the inclusion is uniform:
    # ∂ₛₛv = -κ (∇v) n, d(σn)/ds = κ σ t)
    function Je(p)
        xΓ, ne, κ = ellipse_projection(p.x, a, b)
        s  = S(xΓ - 1e-10 * ne)
        σ  = s.τ - s.p * I
        te = tangent(ne)
        k  = (neg=η.neg / ξ.neg, pos=η.pos / ξ.pos)
        Jz = InterfaceJumpsC(η, k, s.V, s.L * te, -κ * s.L * ne, σ * ne, κ * σ * te, ne, κ)
        form == :z ? Jz : merge(Jz, InterfaceJumpsV(η, k, s.L * te, -κ * s.L * ne, σ * ne, κ * σ * te, ne, κ))
    end
    sol = form == :z ? SolveIIM(φ, X, Δ, η, nc, zbc; ξ, order, fit, Jexact=exact_jumps ? Je : nothing) :
                       SolveIIMStress(φ, X, Δ, η, nc, Va; ξ, order, fit, extend, Jexact=exact_jumps ? Je : nothing)

    ic = 2:(n+1)
    P  = sol.p[ic, ic]
    Pa = [S(SVector(X.c_e.x[i], X.c_e.y[j])).p for i in ic, j in ic]
    ix, jx = 3:(n+1), 3:(n+2)
    ϵV = maximum(abs.(sol.v.x[ix, jx] .- Va.x[ix, jx]))
    return (P=P, P_ana=Pa, X=X, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))), ϵV=ϵV, sol=sol, φ=φ)
end
