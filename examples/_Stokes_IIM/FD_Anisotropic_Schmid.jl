# Schmid & Podladchikov (2003) inclusion with FD and effective viscosities in the cut cells:
# binary phases, isotropic arithmetic / harmonic averages of the fractions, transversely isotropic
# tensor (FD_Anisotropic.jl). Pressure errors on all centroids, as in xGFM_Schmid2003.jl.
include(joinpath(@__DIR__, "FD_Anisotropic.jl"))
include(joinpath(@__DIR__, "IIM_Schmid2003.jl"))
import Statistics: mean

function FDTensor_Schmid2003(n, params; avg=:aniso, avg_v=avg, L=10.0)
    nc = (x=n, y=n)
    Δ  = (x=L / n, y=L / n)
    X  = GenerateGrid((min=-L / 2, max=L / 2), (min=-L / 2, max=L / 2), Δ, nc)
    φf(x, y) = sqrt(x^2 + y^2) - params.rc
    Va = (x=[Stokes2D_Schmid2003_V(SVector(x, y), params)[1] for x in X.vx_e.x, y in X.vx_e.y],
          y=[Stokes2D_Schmid2003_V(SVector(x, y), params)[2] for x in X.vy_e.x, y in X.vy_e.y])
    sol = SolveFDTensor(φf, X, Δ, (neg=params.mc, pos=params.mm), nc, Va; avg, avg_v)
    ic = 2:(n+1)
    P  = sol.p[ic, ic] .- mean(sol.p[ic, ic])
    Pa = [Stokes2D_Schmid2003_p(SVector(X.c_e.x[i], X.c_e.y[j]), params) for i in ic, j in ic]
    Pa = Pa .- mean(Pa)
    Va_ = Va
    ev = max(maximum(abs, sol.v.x .- Va_.x), maximum(abs, sol.v.y .- Va_.y))
    return (P=P, P_ana=Pa, X=X, ϵP=(L1=mean(abs.(P .- Pa)), Linf=maximum(abs.(P .- Pa))), ev=ev, sol=sol)
end

const VARIANTS = Dict("binary" => (:binary, :binary), "arith" => (:arith, :arith), "harm" => (:harm, :harm),
    "geom" => (:geom, :geom), "aniso" => (:aniso, :aniso), "anisodiag" => (:anisodiag, :anisodiag),
    "aniso_v" => (:arith, :aniso), "aniso_c" => (:aniso, :arith), "harm_v" => (:arith, :harm))
function main_aniso(; ns=(20, 40, 80, 160), avgs=("binary", "arith", "harm", "aniso"))
    rows = []
    for mc in (1e3, 1e-3), n in ns, avg in avgs
        ps = (mm=1.0, mc=mc, rc=2.0, gr=0.0, er=1.0)
        ac, av = VARIANTS[avg]
        t  = @elapsed r = FDTensor_Schmid2003(n, ps; avg=ac, avg_v=av)
        push!(rows, (mc=mc, n=n, method=avg, L1=r.ϵP.L1, Linf=r.ϵP.Linf, Linfv=r.ev))
        @printf("mc=%-6g n=%4d %-9s L1(P) %.3e  L∞(P) %.3e  L∞(v) %.3e  %.1fs\n", mc, n, avg, r.ϵP.L1, r.ϵP.Linf, r.ev, t)
        flush(stdout)
    end
    open(joinpath(@__DIR__, "FD_Anisotropic_Schmid.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_aniso()
end
