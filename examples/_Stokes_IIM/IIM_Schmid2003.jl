using StagFDTools, StagFDTools.StokesIIM, LinearAlgebra, StaticArrays
import Statistics: mean
import ExactFieldSolutions: Stokes2D_Schmid2003_V, Stokes2D_Schmid2003_p, Stokes2D_Schmid2003_L

# IIM solution of the viscous inclusion of Schmid & Podladchikov (2003) with exact Dirichlet
# velocities. Pressure errors are computed after removing the mean of both pressures.
# - order, fit: IIM variant (see AssembleIIM)
# - exact_jumps: diagnostic, exact jumps on the circle instead of reconstructed ones
# - form: :z (Landry, AssembleIIM) or :stress (StagFDTools-style stress form, AssembleIIMStress)
# - extend: extension used by the stress form (:z or :v)

function IIM_Schmid2003(n, params; order=1, fit=:linear, exact_jumps=false, L=10.0, form=:z, extend=:z)
    nc = (x=n, y=n)
    Δ  = (x=L / n, y=L / n)
    X  = GenerateGrid((min=-L / 2, max=L / 2), (min=-L / 2, max=L / 2), Δ, nc)
    φ  = LevelSet((x, y) -> sqrt(x^2 + y^2) - params.rc, X)
    η  = (neg=params.mc, pos=params.mm)
    ηof(f) = side(f) < 0 ? η.neg : η.pos
    Va  = (x=[Stokes2D_Schmid2003_V(SVector(X.vx_e.x[i], X.vx_e.y[j]), params)[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
           y=[Stokes2D_Schmid2003_V(SVector(X.vy_e.x[i], X.vy_e.y[j]), params)[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    zbc = (x=ηof.(φ.x) .* Va.x, y=ηof.(φ.y) .* Va.y)

    # Diagnostic: exact jumps on the circle (the inclusion velocity is linear: ∂ₛₛv = -κ (∇v) n)
    Lin, κ = Stokes2D_Schmid2003_L(SVector(0.0, 0.0), params), 1 / params.rc
    Je(p) = (ne = normalize(p.x);
             InterfaceJumps2(η.pos - η.neg, Stokes2D_Schmid2003_V(params.rc * ne, params), Lin * tangent(ne), -κ * Lin * ne, ne, κ))
    if form == :z
        sol = SolveIIM(φ, X, Δ, η, nc, zbc; order, fit, Jexact=exact_jumps ? Je : nothing)
    else
        function Jv(p)                       # same, for the stress form (needs σn on Γ)
            ne, te = normalize(p.x), tangent(normalize(p.x))
            σ = -Stokes2D_Schmid2003_p(SVector(0.0, 0.0), params) * I + params.mc * (Lin + Lin')
            k = (neg=0.0, pos=0.0)
            merge(InterfaceJumpsC(η, k, Stokes2D_Schmid2003_V(params.rc * ne, params), Lin * te, -κ * Lin * ne, σ * ne, κ * σ * te, ne, κ),
                  InterfaceJumpsV(η, k, Lin * te, -κ * Lin * ne, σ * ne, κ * σ * te, ne, κ))
        end
        sol = SolveIIMStress(φ, X, Δ, η, nc, Va; order, fit, extend, Jexact=exact_jumps ? Jv : nothing)
    end

    ic = 2:(n+1)
    P  = sol.p[ic, ic] .- mean(sol.p[ic, ic])
    Pa = [Stokes2D_Schmid2003_p(SVector(X.c_e.x[i], X.c_e.y[j]), params) for i in ic, j in ic]
    Pa = Pa .- mean(Pa)
    return (P=P, P_ana=Pa, X=X, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))), sol=sol, φ=φ)
end
