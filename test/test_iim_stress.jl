using Test, StaticArrays, LinearAlgebra, StagFDTools, StagFDTools.StokesIIM
import Statistics: mean
import ExactFieldSolutions: Stokes2D_Moutzouris_circle

# Stress form (AssembleIIMStress): StagFDTools-style kernels (Vx↔Vy interpolation, stresses at
# centroids and vertices) with one-sided stencils
@testset "IIM stress form, exact piecewise linear solutions" begin
    nc = (x=24, y=20)
    L  = (x=2.4, y=2.0)
    Δ  = (x=L.x / nc.x, y=L.y / nc.y)
    X  = GenerateGrid((min=0.0, max=L.x), (min=0.0, max=L.y), Δ, nc)
    η  = (neg=50.0, pos=1.0)
    n  = normalize(SVector(0.45, 1.0))
    t  = tangent(n)
    x0 = SVector(1.13, 1.04)
    φ  = LevelSet((x, y) -> dot(n, SVector(x, y) - x0), X)
    ix, jx = 3:(nc.x+1), 3:(nc.y+2)
    grid(vel) = (x=[vel(SVector(X.vx_e.x[i], X.vx_e.y[j]))[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
                 y=[vel(SVector(X.vy_e.x[i], X.vy_e.y[j]))[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    P(sol) = sol.p[2:(nc.x+1), 2:(nc.y+1)]

    # incompressible simple shear parallel to Γ, with a kink
    va = grid(x -> (s = dot(n, x - x0); (0.7 + 1.3 * s / (s < 0 ? η.neg : η.pos)) * t))
    for (fit, order) in ((:linear, 1), (:quadratic, 1), (:quadratic, 2)), extend in (:z, :v)
        sol = SolveIIMStress(φ, X, Δ, η, nc, va; fit, order, extend)
        @test maximum(abs.(sol.v.x[ix, jx] .- va.x[ix, jx])) < 1e-9
        @test maximum(abs.(P(sol) .- mean(P(sol)))) < 1e-8
    end
    @test maximum(abs.(SolveIIMStress(φ, X, Δ, η, nc, va; corrections=false).v.x[ix, jx] .- va.x[ix, jx])) > 1e-3

    # compressible uniform strain on each side (see test_iim_assembly.jl)
    ξ  = (neg=200.0, pos=3.0)
    σ(L, η, ξ) = (ξ - 2η / 3) * tr(L) * I + η * (L + L')
    Lm = @SMatrix [0.4 -0.9; 0.3 0.6]
    M  = hcat((σ(e * n', η.pos, ξ.pos) * n for e in (SVector(1.0, 0.0), SVector(0.0, 1.0)))...)
    Lp = Lm + (M \ (σ(Lm, η.neg, ξ.neg) * n - σ(Lm, η.pos, ξ.pos) * n)) * n'
    sd(x) = dot(n, x - x0) < 0
    vc = grid(x -> SVector(0.7, -0.2) + (sd(x) ? Lm : Lp) * (x - x0))
    Pa = [sd(SVector(X.c_e.x[i], X.c_e.y[j])) ? -ξ.neg * tr(Lm) : -ξ.pos * tr(Lp) for i in 2:(nc.x+1), j in 2:(nc.y+1)]
    for (fit, order) in ((:linear, 1), (:quadratic, 2))
        sol = SolveIIMStress(φ, X, Δ, η, nc, vc; ξ, fit, order)
        @test maximum(abs.(sol.v.x[ix, jx] .- vc.x[ix, jx])) < 1e-9
        @test maximum(abs.(P(sol) .- Pa)) < 1e-8
        # vertex pressure (pressure column of 𝐷): one-sided average exact, plain average not
        Pex = [sd(SVector(x, y)) ? -ξ.neg * tr(Lm) : -ξ.pos * tr(Lp) for x in X.v_e.x, y in X.v_e.y]
        ok  = .!isnan.(VertexPressure(sol.S, sol.v, sol.p, φ, X))
        @test maximum(abs.(VertexPressure(sol.S, sol.v, sol.p, φ, X; order)[ok] .- Pex[ok])) < 1e-8
        @test maximum(abs.(VertexPressure(sol.S, sol.v, sol.p, φ, X; onesided=false)[ok] .- Pex[ok])) > 1e-1
    end
end

@testset "IIM stress form, compressible inclusion" begin
    # Moutzouris (2026) circle, η contrast 10³: the stress form with z-extension matches the
    # z-form; the v-extension amplifies the Taylor remainder by the contrast
    prm = (ηm=1.0, ηi=1e3, ξm=2.0, ξi=1e5, ri=0.2, t=1.0, α=0.0, ε̇=1.0, γ̇=0.0, ζ̇=0.5, ε̇zz=0.0)
    S(x) = Stokes2D_Moutzouris_circle(x; params=prm)
    n  = 40
    nc = (x=n, y=n)
    Δ  = (x=1 / n, y=1 / n)
    X  = GenerateGrid((min=-0.5, max=0.5), (min=-0.5, max=0.5), Δ, nc)
    φ  = LevelSet((x, y) -> sqrt(x^2 + y^2) - prm.ri, X)
    η, ξ = (neg=prm.ηi, pos=prm.ηm), (neg=prm.ξi, pos=prm.ξm)
    va = (x=[S(SVector(X.vx_e.x[i], X.vx_e.y[j])).V[1] for i in axes(φ.x, 1), j in axes(φ.x, 2)],
          y=[S(SVector(X.vy_e.x[i], X.vy_e.y[j])).V[2] for i in axes(φ.y, 1), j in axes(φ.y, 2)])
    Pa = [S(SVector(X.c_e.x[i], X.c_e.y[j])).p for i in 2:(n+1), j in 2:(n+1)]
    err(sol) = maximum(abs.(sol.p[2:(n+1), 2:(n+1)] .- Pa))
    @test err(SolveIIMStress(φ, X, Δ, η, nc, va; ξ, order=2, fit=:quadratic)) < 0.3                # 0.27
    @test err(SolveIIMStress(φ, X, Δ, η, nc, va; ξ, order=2, fit=:quadratic, extend=:v)) > 1.0     # 2.0
end
