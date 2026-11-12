# Poisson analogue of the thinning-neck test (examples/_Stokes_IIM/NeckThinning.jl): conductive
# layer (β = βl) with a Gaussian neck in a matrix (β = 1), u = x on the boundary of [-1, 1]², so
# the flux along the layer has to squeeze through the neck. Compares, on N = 40 and 80:
#   FD with face coefficients from layer fractions (arithmetic / harmonic, as markers would give),
#   xGFM, IIM u-form (orders 1, 2), IIM z-form (Landry weights, orders 1, 2), hybrid IIMh2,
#   cut cell (CutCell.jl).
# Reference: cut cell at N = 1280. Metrics: potential drop across the neck (its resistance), and
# L∞ error of u at points farther than 1.5h from Γ. Writes CutCell_Neck_beta*.csv / .png in results/.
include(joinpath(@__DIR__, "Poisson_Interface.jl"))
include(joinpath(@__DIR__, "CutCell.jl"))
import CairoMakie as cm

const PNECK = (H=0.4, σ=0.2, x0=0.0086, y0=0.0142)
const OUT = mkpath(joinpath(@__DIR__, "results"))

function neck_phi(w; H=PNECK.H, σ=PNECK.σ, x0=PNECK.x0, y0=PNECK.y0)
    A = (H - w) / 2
    return x -> begin
        g  = exp(-((x[1] - x0) / σ)^2)
        hh = H / 2 - A * g
        dh = A * g * 2 * (x[1] - x0) / σ^2
        (abs(x[2] - y0) - hh) / sqrt(1 + dh^2)
    end
end

# problem data in the form of `derived` (Poisson_Interface.jl): no jumps, no source, u = x on ∂Ω
function neck_problem(w, βl)
    φ = neck_phi(w)
    z(x) = 0.0
    nrm(x) = (g = FD.gradient(φ, x); g / norm(g))
    tng(n) = SVector(-n[2], n[1])
    κ(x) = tr(FD.jacobian(nrm, x))
    β = (neg=βl, pos=1.0)
    return (φ=φ, β=β, βs=s -> s < 0 ? β.neg : β.pos, u=(x, s) -> x[1], ∇u=(x, s) -> SVector(1.0, 0.0),
            f=(x, s) -> 0.0, nrm=nrm, tng=tng, κ=κ, a=z, b=z, as=z, ass=z, bs=z, jf=z)
end

# setup() of Poisson_Interface.jl with the problem data given directly
function setupP(P, N)
    h  = 2 / N
    xs = range(-1, 1, length=N + 1)
    X(i, j) = SVector(xs[i], xs[j])
    S  = [side(P.φ(X(i, j))) for i in 1:N+1, j in 1:N+1]
    LI = LinearIndices(S)
    cross = Dict{NTuple{3,Int},Any}()
    for j in 1:N+1, i in 1:N+1, (dir, di, dj) in ((1, 1, 0), (2, 0, 1))
        (i + di <= N + 1 && j + dj <= N + 1) || continue
        S[i, j] == S[i+di, j+dj] && continue
        x0, x1 = X(i, j), X(i + di, j + dj)
        lo, hi = 0.0, 1.0
        f0 = P.φ(x0)
        for _ in 1:60
            m = (lo + hi) / 2
            sign(P.φ(x0 + m * (x1 - x0))) == sign(f0) ? (lo = m) : (hi = m)
        end
        θ  = (lo + hi) / 2
        xΓ = x0 + θ * (x1 - x0)
        n  = P.nrm(xΓ)
        cross[(i, j, dir)] = (x=xΓ, θ=θ, n=n, t=P.tng(n), κ=P.κ(xΓ))
    end
    return (; P, N, h, xs, X, S, LI, cross)
end

# node-centred FD, face coefficient from the layer fraction of the h×h box around the face midpoint
function solve_fdfrac(G, avg; nsub=10)
    (; P, N, h, X, LI) = G
    nI = (N - 1)^2
    id(i, j) = (j - 2) * (N - 1) + (i - 1)
    inner(i, j) = 2 <= i <= N && 2 <= j <= N
    s = ((1:nsub) .- 0.5) ./ nsub .- 0.5
    function βface(xm)
        f = count(P.φ(xm + h * SVector(a, b)) < 0 for a in s, b in s) / nsub^2
        avg == :arithmetic ? f * P.β.neg + (1 - f) * P.β.pos : 1 / (f / P.β.neg + (1 - f) / P.β.pos)
    end
    Is, Js, Vs = Int[], Int[], Float64[]
    b = zeros(nI)
    for j in 2:N, i in 2:N, (di, dj) in NBR
        r  = id(i, j)
        βf = βface((X(i, j) + X(i + di, j + dj)) / 2)
        push!(Is, r); push!(Js, r); push!(Vs, -βf)
        inner(i + di, j + dj) ? (push!(Is, r); push!(Js, id(i + di, j + dj)); push!(Vs, βf)) : (b[r] -= βf * X(i + di, j + dj)[1])
    end
    u = sparse(Is, Js, Vs, nI, nI) \ b
    U = [X(i, j)[1] for i in 1:N+1, j in 1:N+1]
    for j in 2:N, i in 2:N; U[i, j] = u[id(i, j)]; end
    return U
end

# node values of a node-centred solution as an (N+1)×(N+1) array
nodal(G, sol) = [sol.U[G.LI[i, j]] for i in 1:G.N+1, j in 1:G.N+1]

# reference: cut-cell values at cell centres + coordinates, and bilinear interpolation
function reference(w, βl, N)
    P = neck_problem(w, βl)
    G = cutcell_geometry(P.φ, N; m=8)
    s = solve_cutcell(G, P.β, x -> x[1])
    xc = [-1 + (I - 0.5) * G.h for I in 1:N]
    return (U=[cell_value(G, s.u, I, J) for I in 1:N, J in 1:N], xs=xc, P=P)
end
interp(R, x) = (xs = R.xs; h = xs[2] - xs[1];
    i = clamp(floor(Int, (x[1] - xs[1]) / h) + 1, 1, length(xs) - 1); j = clamp(floor(Int, (x[2] - xs[1]) / h) + 1, 1, length(xs) - 1);
    tx = (x[1] - xs[i]) / h; ty = (x[2] - xs[j]) / h;
    (1 - tx) * (1 - ty) * R.U[i, j] + tx * (1 - ty) * R.U[i+1, j] + (1 - tx) * ty * R.U[i, j+1] + tx * ty * R.U[i+1, j+1])

# metrics from point values (xs, U) on a tensor grid
function metrics(xs, U, R, φ, h)
    itp(x) = (hh = xs[2] - xs[1]; i = clamp(floor(Int, (x[1] - xs[1]) / hh) + 1, 1, length(xs) - 1);
              j = clamp(floor(Int, (x[2] - xs[1]) / hh) + 1, 1, length(xs) - 1); tx = (x[1] - xs[i]) / hh; ty = (x[2] - xs[j]) / hh;
              (1 - tx) * (1 - ty) * U[i, j] + tx * (1 - ty) * U[i+1, j] + (1 - tx) * ty * U[i, j+1] + tx * ty * U[i+1, j+1])
    xa, xb = SVector(PNECK.x0 - 0.6, PNECK.y0), SVector(PNECK.x0 + 0.6, PNECK.y0)
    drop = itp(xb) - itp(xa)
    e = 0.0
    for j in eachindex(xs), i in eachindex(xs)
        x = SVector(xs[i], xs[j])
        (abs(φ(x)) > 1.5h && maximum(abs.(x)) < 1 - 1e-9) || continue
        e = max(e, abs(U[i, j] - interp(R, x)))
    end
    return (drop=drop, eu=e)
end

const PMETHODS = [:fd_arith, :fd_harm, :xGFM, :IIM1, :IIM2, :IIMz1, :IIMz2, :IIMh2, :cutcell]

function run_method(m, w, βl, N)
    P = neck_problem(w, βl)
    if m == :cutcell
        G = cutcell_geometry(P.φ, N)
        s = solve_cutcell(G, P.β, x -> x[1])
        xs = [-1 + (I - 0.5) * G.h for I in 1:N]
        return xs, [cell_value(G, s.u, I, J) for I in 1:N, J in 1:N], G.h, P.φ
    end
    G  = setupP(P, N)
    U  = m == :fd_arith ? solve_fdfrac(G, :arithmetic) : m == :fd_harm ? solve_fdfrac(G, :harmonic) : nodal(G, solve(G, m))
    return collect(G.xs), U, G.h, P.φ
end

function main_poisson(; βl=100.0, Nref=1280, Ns=(40, 80), wratios=(8, 6, 4, 2, 1, 0.5))
    rows = []
    h40 = 2 / first(Ns)
    for wr in wratios
        w = min(wr * h40, PNECK.H)
        t = @elapsed R = reference(w, βl, Nref)
        dref = interp(R, SVector(PNECK.x0 + 0.6, PNECK.y0)) - interp(R, SVector(PNECK.x0 - 0.6, PNECK.y0))
        @printf("βl=%g w/h40=%-4g reference N=%d (%.0fs): drop %.5f\n", βl, wr, Nref, t, dref)
        push!(rows, (βl=βl, wr=wr, w=w, N=Nref, method="reference", drop=dref, eu=0.0, ok=true))
        for N in Ns, m in PMETHODS
            local res
            ok = true
            t = @elapsed try
                xs, U, h, φ = run_method(m, w, βl, N)
                res = metrics(xs, U, R, φ, h)
            catch e
                ok = false
                @printf("    N=%3d %-9s FAILED: %s\n", N, m, sprint(showerror, e)[1:min(end, 100)])
            end
            ok || (push!(rows, (βl=βl, wr=wr, w=w, N=N, method=string(m), drop=NaN, eu=NaN, ok=false)); continue)
            push!(rows, (βl=βl, wr=wr, w=w, N=N, method=string(m), drop=res.drop, eu=res.eu, ok=true))
            @printf("    N=%3d %-9s w/h %5.2f  drop/ref %7.3f  L∞u (|φ|>1.5h) %.2e  %.1fs\n", N, m, w * N / 2, res.drop / dref, res.eu, t)
        end
        flush(stdout)
    end
    tag = @sprintf("_beta%g", βl)
    open(joinpath(OUT, "CutCell_Neck$tag.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end
    plot_poisson(rows, Ns, tag)
    return rows
end

const PCOLS = Dict("fd_arith" => :gray, "fd_harm" => :black, "xGFM" => :darkorange, "IIM1" => :lightblue, "IIM2" => :steelblue,
                   "IIMz1" => :plum, "IIMz2" => :purple, "IIMh2" => :teal, "cutcell" => :red)
function plot_poisson(rows, Ns, tag)
    cm.set_theme!(fontsize=14)
    fig = cm.Figure(size=(1300, 520))
    βl = rows[1].βl
    for (k, key) in enumerate((:drop, :eu))
        ax = cm.Axis(fig[1, k], xscale=log2, yscale=key == :eu ? log10 : identity, xreversed=true, xticks=[0.5, 1, 2, 4, 8, 16],
                     xlabel="w / h (minimum neck width in cells)",
                     title=key == :drop ? "potential drop across the neck / reference" : "L∞(u) error at |φ| > 1.5h (clipped at 1; u ranges over [-1, 1])")
        for m in PMETHODS, (s, N) in enumerate(Ns)
            sel = filter(r -> r.method == string(m) && r.N == N && r.ok && r.w < PNECK.H, rows)
            isempty(sel) && continue
            x = [r.w * N / 2 for r in sel]
            y = key == :drop ? [r.drop / only(filter(q -> q.method == "reference" && q.wr == r.wr, rows)).drop for r in sel] : [clamp(r.eu, 1e-8, 1.0) for r in sel]
            cm.scatterlines!(ax, x, y, color=PCOLS[string(m)], marker=s == 1 ? :circle : :utriangle, linestyle=s == 1 ? :solid : :dash,
                             linewidth=m == :cutcell ? 3 : 1.5, label="$m, N = $N")
        end
        key == :drop ? (cm.hlines!(ax, [1.0], color=:black, linestyle=:dot); cm.ylims!(ax, 0.85, 1.25)) : cm.ylims!(ax, 3e-4, 1.2)
        k == 1 && cm.Legend(fig[1, 3], ax, framevisible=false, labelsize=11, nbanks=1)
    end
    cm.Label(fig[0, 1:3], "Poisson neck: layer β = $βl, matrix β = 1, u = x on ∂Ω; reference cut cell N = 1280")
    cm.save(joinpath(OUT, "CutCell_Neck$tag.png"), fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_poisson(βl=100.0)
    main_poisson(βl=10.0)
end
