# Free surface with the immersed interface method, no sticky air: gravity-driven relaxation of
# a Gaussian topography. Setup of FullStaggeredGrid/scripts_2D_PT/Stokes2D_FSG_PT_v6.jl:
# incompressible, η = ρ = 1, g = -1, box x ∈ [-3, 3], y ≥ -5, free slip or no slip on the sides
# and the bottom, free surface y = h(x) = exp(-x²/σ²) (σ = 0.5) cutting the grid.
#
# Unknowns in the fluid, φ = y - h(x) ≤ 0, on a regular MAC grid that extends above the
# surface, plus (faces = :adjacent) the velocity faces next to a fluid centroid, so that the
# surface pressure enters every fluid column and row. Standard stencils,
# σ = -p I + 2η(ε̇ - ⅓(∇·v) I). Every other stencil value in the void is a ghost:
#   velocity  Taylor extension of the fluid solution from the closest point x_Γ of the surface,
#             v_k = v_Γ + ∇v·δ + ½δᵀ∇∇v δ (δ = x_k - x_Γ);
#   pressure  linear along the grid line through the fluid neighbour and p_Γ at the crossing
#             (pghost = :linear), or p_k = p_Γ + ∇p·δ (pghost = :taylor).
# v_Γ, ∂ₛv and ∂ₛₛv come from one-sided quadratic least-squares fits of the fluid nodes
# (QuadraticFit, npts nodes; 10 nodes give nearly degenerate fits along a 45° surface, 16 are
# robust). The normal derivatives, p_Γ and ∇p_Γ follow from the traction σn on Γ (zero by
# default) and the momentum and continuity equations there (SideFields with the body force).
# All values are linear forms in the unknowns, so the residuals are assembled directly.
# Also used for the inviscid inclusion of Schmid & Podladchikov (2003), Circle geometry and
# Dirichlet velocities (IIM_FreeSurface_Schmid.jl).
using CairoMakie, Printf, LinearAlgebra, SparseArrays, StaticArrays
using StagFDTools.StokesIIM
import StagFDTools.StokesIIM: SideFields

# Linear form Σ w[k] u[i[k]] + c in the unknowns u
struct Lin
    i::Vector{Int}
    w::Vector{Float64}
    c::Float64
end
Lin(c::Real=0.0) = Lin(Int[], Float64[], c)
Base.:+(a::Lin, b::Lin) = Lin(vcat(a.i, b.i), vcat(a.w, b.w), a.c + b.c)
Base.:+(a::Lin, c::Real) = Lin(a.i, a.w, a.c + c)
Base.:*(s::Real, a::Lin) = Lin(a.i, s .* a.w, s * a.c)
Base.:-(a::Lin, b::Lin) = a + (-1.0) * b
Base.:/(a::Lin, s::Real) = (1 / s) * a
evaluate(L::Lin, u) = L.c + sum(L.w[k] * u[L.i[k]] for k in eachindex(L.i); init=0.0)

# Surface y = h(x) and its derivatives
struct Topography{F,G,H}
    h::F; dh::G; d2h::H
end
Gaussian(A, σ, y0) = Topography(x -> y0 + A * exp(-x^2 / σ^2), x -> -2x / σ^2 * A * exp(-x^2 / σ^2),
                                x -> (4x^2 / σ^4 - 2 / σ^2) * A * exp(-x^2 / σ^2))

# Closest point of the surface to xk: x_Γ, normal n (to the void), curvature κ = ∇·n
function closest_point(topo, xk, Δ)
    (; h, dh, d2h) = topo
    xs = range(xk[1] - 4Δ, xk[1] + 4Δ, length=161)
    x  = xs[argmin(@. (xs - xk[1])^2 + (h(xs) - xk[2])^2)]
    for _ in 1:30
        r  = h(x) - xk[2]
        dx = ((x - xk[1]) + r * dh(x)) / (1 + dh(x)^2 + r * d2h(x))
        x -= dx
        abs(dx) < 1e-14 && break
    end
    s = sqrt(1 + dh(x)^2)
    return SVector(x, h(x)), SVector(-dh(x), 1.0) / s, -d2h(x) / s^3
end

# Geometry interface: level set (fluid where ≤ 0) and closest surface point (x_Γ, n, κ)
levelset(t::Topography, x) = x[2] - t.h(x[1])
closest(t::Topography, xk, Δ) = closest_point(t, xk, Δ)
# Circular void of radius r centred at c, fluid outside (inviscid inclusion)
struct Circle
    c::SVector{2,Float64}
    r::Float64
end
levelset(c::Circle, x) = c.r - norm(x - c.c)
closest(c::Circle, xk, Δ) = (e = normalize(xk - c.c); (c.c + c.r * e, -e, -1 / c.r))

# topo: Topography or Circle.
# bc = :free_slip or :no_slip on the sides and the bottom, or :dirichlet with velocities vbc(x).
# kernel = :compact       σxx, σyy from ∂x vx, ∂y vy at centroids, τxy from ∂y vx + ∂x vy at vertices
#          :interpolated  StagFDTools stress form: Vx averaged to the Vy nodes and Vy to the Vx nodes
#                         (4-point averages), all of (∂x vx, ∂y vy, ∂y vx, ∂x vy) at centroids and
#                         vertices, τ = 𝐷 ε̇ with 𝐷 = 2η I (3×3), as needed for anisotropy/plasticity
function FreeSurface(ncx; topo=Gaussian(1.0, 0.5, 0.0), η=1.0, ρ=1.0, g=-1.0, xmin=-3.0, xmax=3.0, ymin=-5.0, ytop=1.5,
                     bc=:free_slip, vbc=nothing, faces=:adjacent, kernel=:compact, force=nothing, traction=nothing, θ=0.0, pghost=:linear, npts=24, degree=3, order=2, pΓexact=nothing, vexact=nothing)
    Δ   = (xmax - xmin) / ncx
    ncy = round(Int, (ytop - ymin) / Δ)
    xv  = collect(range(xmin, xmax, length=ncx + 1)); xc = xv[1:end-1] .+ Δ / 2
    yv  = collect(range(ymin, ymin + ncy * Δ, length=ncy + 1)); yc = yv[1:end-1] .+ Δ / 2
    X   = (x=(x=xv, y=yc), y=(x=xc, y=yv), p=(x=xc, y=yc))
    pos(f, i, j) = SVector(xmin + (i - (f == :x ? 1 : 0.5)) * Δ, ymin + (j - (f == :y ? 1 : 0.5)) * Δ)  # also outside
    lset(f) = [levelset(topo, SVector(x, y)) for x in X[f].x, y in X[f].y]
    φ   = (x=lset(:x), y=lset(:y), p=lset(:p))
    # active nodes: φ ≤ -θΔ (nodes on the surface are fluid, so that a momentum equation always
    # reaches the void centroids above; θ > 0 also makes the nodes within θΔ of the surface ghosts)
    φa  = map(a -> a .+ (θ - 1e-10) * Δ, φ)
    onwall(f, i, j) = (f == :x && (i == 1 || i == ncx + 1)) || (f == :y && (j == 1 || j == ncy + 1))
    # faces = :adjacent: the velocity faces next to a fluid centroid are unknowns as well, with the
    # momentum equation and the void-centroid pressure as ghost (through p_Γ), so that the surface
    # pressure reaches every fluid column and row (else not where Γ is nearly parallel to a grid line)
    if faces == :adjacent
        fc = φa.p .< 0
        φa = (x=[φa.x[i, j] < 0 || (!onwall(:x, i, j) && (fc[i-1, j] || fc[i, j])) ? -1.0 : φa.x[i, j] for i in axes(φa.x, 1), j in axes(φa.x, 2)],
              y=[φa.y[i, j] < 0 || (!onwall(:y, i, j) && (fc[i, j-1] || fc[i, j])) ? -1.0 : φa.y[i, j] for i in axes(φa.y, 1), j in axes(φa.y, 2)],
              p=φa.p)
    end
    rng = (x=(1:(ncx + 1), 1:ncy), y=(1:ncx, 1:(ncy + 1)), p=(1:ncx, 1:ncy))
    # normal velocity given on the sides and the bottom, where they are in the fluid (else ghosts)
    fixed(f, i, j) = onwall(f, i, j) && !(f == :y && j == ncy + 1) && φ[f][i, j] <= 0
    wallv(f, x) = bc == :dirichlet ? vbc(x)[f == :x ? 1 : 2] : 0.0

    # Numbering of the fluid unknowns
    num = map(a -> zeros(Int, size(a)), φ)
    ndof = 0
    for f in (:x, :y, :p), I in CartesianIndices(φ[f])
        φa[f][I] < 0 && !fixed(f, I[1], I[2]) && (num[f][I] = (ndof += 1))
    end

    # Ghost values: Taylor extension from the closest surface point
    # body force f(x) and surface traction (T, dT/ds)(x_Γ, n, κ): gravity and zero traction by
    # default, other choices for manufactured solutions
    fbody(x) = isnothing(force) ? SVector(0.0, ρ * g) : force(x)
    Tsurf(xΓ, n, κ) = isnothing(traction) ? (Z, Z) : traction(xΓ, n, κ)
    Z     = zero(SVector{2,Float64})
    ghosts = Dict{Tuple{Symbol,Int,Int},Lin}()
    function trace(f, xΓ, n, κ)          # v_Γ, ∂ₛv, ∂ₛₛv of one component, from fluid nodes only
        t  = tangent(n)
        xo = xΓ - Δ / max(abs(n[1]), abs(n[2])) * n
        F  = QuadraticFit(xΓ, xo, -1, φa[f], X[f], rng[f], (x=Δ, y=Δ); npts, degree)
        vk = [fluid(f, I[1], I[2]) for I in F.idx]
        lin(c) = reduce(+, (c[k] * vk[k] for k in eachindex(vk)); init=Lin())
        return lin(F.λ), lin([dot(F.g[:, k], t) for k in eachindex(vk)]),
               lin([dot(t, F.H[:, :, k] * t) - κ * dot(n, F.g[:, k]) for k in eachindex(vk)])
    end
    function ghost(f, i, j)
        get!(ghosts, (f, i, j)) do
            xk = pos(f, i, j)
            isnothing(vexact) || f == :p || return Lin(vexact(xk)[f == :x ? 1 : 2])   # diagnostic: exact extension
            xΓ, n, κ = closest(topo, xk, Δ)
            δ  = xk - xΓ
            # extension (velocity components, pressure) as affine function of (∂ₛvx, ∂ₛvy, ∂ₛₛvx, ∂ₛₛvy)
            function ext(w, w2, fb, T=Z, dT=Z)
                S = SideFields(η, 0.0, w, w2, T, dT, n, κ; f=fb)
                c2 = order == 2 ? 0.5 : 0.0
                SVector((dot(S.∇z[1, :], δ) + c2 * dot(δ, S.∇∇z[1, :, :] * δ)) / η,
                        (dot(S.∇z[2, :], δ) + c2 * dot(δ, S.∇∇z[2, :, :] * δ)) / η,
                        S.p + (order == 2 ? dot(S.∇p, δ) : 0.0))
            end
            e0 = ext(Z, Z, fbody(xΓ), Tsurf(xΓ, n, κ)...)
            E  = (ext(SVector(1.0, 0.0), Z, Z), ext(SVector(0.0, 1.0), Z, Z),
                  ext(Z, SVector(1.0, 0.0), Z), ext(Z, SVector(0.0, 1.0), Z))
            Tx, Ty = trace(:x, xΓ, n, κ), trace(:y, xΓ, n, κ)
            data   = (Tx[2], Ty[2], Tx[3], Ty[3])
            a      = f == :x ? 1 : f == :y ? 2 : 3
            L      = reduce(+, (E[m][a] * data[m] for m in 1:4); init=Lin(e0[a]))
            a < 3 ? L + (a == 1 ? Tx : Ty)[1] : L
        end
    end
    fluid(f, i, j) = fixed(f, i, j) ? Lin(wallv(f, pos(f, i, j))) : Lin([num[f][i, j]], [1.0], 0.0)
    # ghost of the tangential velocity across a wall at xw, from the nodes 1 and 2 cells inside (ik, jk
    # and the next one): even (free slip), odd about the wall value (no slip), or quadratic through
    # the wall value (Dirichlet: (8v_w - 6v₁ + v₂)/3; the linear mirror gives first-order pressure
    # errors at the corners). Where the wall point is in the void, an extension as any other.
    function mirror(f, i, j, xw, ik, jk)
        levelset(topo, xw) > 0 && return ghost(f, i, j)
        bc == :free_slip && return val(f, ik, jk)
        bc == :no_slip   && return (-1.0) * val(f, ik, jk)
        return (-2.0) * val(f, ik, jk) + (1 / 3) * val(f, 2ik - i, 2jk - j) + 8 / 3 * wallv(f, xw)
    end
    function val(f, i, j)
        f == :y && i == 0       && return mirror(:y, i, j, SVector(xmin, X.y.y[j]), 1, j)
        f == :y && i == ncx + 1 && return mirror(:y, i, j, SVector(xmax, X.y.y[j]), ncx, j)
        f == :x && j == 0       && return mirror(:x, i, j, SVector(X.x.x[i], ymin), i, 1)
        f == :x && i == 0       && return 2 * val(:x, 1, j) - val(:x, 2, j)   # normal velocity: linear
        f == :x && i == ncx + 2 && return 2 * val(:x, ncx + 1, j) - val(:x, ncx, j)
        return φa[f][i, j] < 0 || fixed(f, i, j) ? fluid(f, i, j) : ghost(f, i, j)
    end
    vx(i, j) = val(:x, i, j)
    vy(i, j) = val(:y, i, j)
    pp(i, j) = val(:p, i, j)
    # Pressure ghosts for pghost = :linear (Gibou et al. 2002 for Dirichlet conditions): along the
    # grid line from the fluid centroid k to the void centroid g, linear through p_k and p_Γ at the
    # crossing (fraction s of the segment): p_g = p_k + (p_Γ - p_k)/s. Only p_Γ = -C + ⁴⁄₃∂ₙzₙ - ⅔∂ₜzₜ
    # (first derivatives of the fits) is needed, and p_g is tied to the fluid pressure.
    function surface_pressure(xΓ, n, κ)
        isnothing(pΓexact) || return Lin(pΓexact(xΓ))     # diagnostic: exact surface pressure
        T, dT = Tsurf(xΓ, n, κ)
        pΓ(w) = SideFields(η, 0.0, w, Z, T, dT, n, κ; f=fbody(xΓ)).p
        p0 = pΓ(Z)
        Tx, Ty = trace(:x, xΓ, n, κ), trace(:y, xΓ, n, κ)
        return (pΓ(SVector(1.0, 0.0)) - p0) * Tx[2] + (pΓ(SVector(0.0, 1.0)) - p0) * Ty[2] + p0
    end
    pghosts = Dict{NTuple{4,Int},Lin}()
    function pres(i, j, ik, jk)          # pressure at centroid (i, j) seen from the stencil partner (ik, jk)
        (φa.p[i, j] < 0 || pghost == :taylor || φa.p[ik, jk] >= 0) && return pp(i, j)
        get!(pghosts, (i, j, ik, jk)) do
            xk, xg = pos(:p, ik, jk), pos(:p, i, j)
            F(t) = levelset(topo, xk + t * (xg - xk))
            F(1.0) <= 0 && return pp(i, j)
            lo, hi = 0.0, 1.0                       # (b is the right-hand side vector below)
            for _ in 1:60
                mid = (lo + hi) / 2; F(mid) < 0 ? (lo = mid) : (hi = mid)
            end
            sfrac = max((lo + hi) / 2, 1e-3)
            pk = pp(ik, jk)
            pk + (surface_pressure(closest(topo, xk + sfrac * (xg - xk), Δ)...) - pk) / sfrac
        end
    end

    # Stencils
    𝐷 = 2η * SMatrix{3,3}(1.0I)
    function stress(Dxx, Dyy, Dxy, Dyx)             # (τxx, τyy, τxy) = 𝐷 ε̇ (deviatoric)
        θ = Dxx + Dyy
        ε = (Dxx - θ / 3, Dyy - θ / 3, (Dxy + Dyx) / 2)
        return ntuple(a -> reduce(+, (𝐷[a, b] * ε[b] for b in 1:3); init=Lin()), 3)
    end
    V̄x(i, j) = (vx(i, j - 1) + vx(i + 1, j - 1) + vx(i, j) + vx(i + 1, j)) / 4   # at Vy node (i, j)
    V̄y(i, j) = (vy(i - 1, j) + vy(i, j) + vy(i - 1, j + 1) + vy(i, j + 1)) / 4   # at Vx node (i, j)
    function σ(i, j, a, ik, jk)         # σxx (a = 1) or σyy (a = 2) at centroid (i, j), partner (ik, jk)
        Dxx = (vx(i + 1, j) - vx(i, j)) / Δ
        Dyy = (vy(i, j + 1) - vy(i, j)) / Δ
        kernel == :compact && return stress(Dxx, Dyy, Lin(), Lin())[a] - pres(i, j, ik, jk)
        Dxy = (V̄x(i, j + 1) - V̄x(i, j)) / Δ
        Dyx = (V̄y(i + 1, j) - V̄y(i, j)) / Δ
        return stress(Dxx, Dyy, Dxy, Dyx)[a] - pres(i, j, ik, jk)
    end
    function τxy(i, j)                  # at vertex (i, j); walls: τxy = 0 (free slip) or mirrored ghosts
        wall = i == 1 || i == ncx + 1 || j == 1
        wall && bc == :free_slip && return Lin()
        ∂y = vx(i, j) - vx(i, j - 1)
        ∂x = vy(i, j) - vy(i - 1, j)
        (kernel == :compact || wall) && return stress(Lin(), Lin(), ∂y / Δ, ∂x / Δ)[3]
        Dxx = (V̄x(i, j) - V̄x(i - 1, j)) / Δ
        Dyy = (V̄y(i, j) - V̄y(i, j - 1)) / Δ
        return stress(Dxx, Dyy, ∂y / Δ, ∂x / Δ)[3]
    end
    residual(::Val{:x}, i, j) = (σ(i, j, 1, i - 1, j) - σ(i - 1, j, 1, i, j)) / Δ + (τxy(i, j + 1) - τxy(i, j)) / Δ + fbody(pos(:x, i, j))[1]
    residual(::Val{:y}, i, j) = (τxy(i + 1, j) - τxy(i, j)) / Δ + (σ(i, j, 2, i, j - 1) - σ(i, j - 1, 2, i, j)) / Δ + fbody(pos(:y, i, j))[2]
    residual(::Val{:p}, i, j) = (vx(i + 1, j) - vx(i, j)) / Δ + (vy(i, j + 1) - vy(i, j)) / Δ

    Is, Js, Vs, b = Int[], Int[], Float64[], zeros(ndof)
    for f in (:x, :y, :p), I in CartesianIndices(num[f])
        (r = num[f][I]) > 0 || continue
        R = residual(Val(f), I[1], I[2])
        append!(Is, fill(r, length(R.i))); append!(Js, R.i); append!(Vs, R.w)
        b[r] = -R.c
    end
    K = sparse(Is, Js, Vs, ndof, ndof)
    u = K \ b

    # Fields on the full arrays (ghost values in the void band, NaN beyond)
    field(f) = [num[f][i, j] > 0 || fixed(f, i, j) || haskey(ghosts, (f, i, j)) ? evaluate(val(f, i, j), u) : NaN
                for i in axes(φ[f], 1), j in axes(φ[f], 2)]
    V = (x=field(:x), y=field(:y), p=field(:p))
    value(f, i, j) = evaluate(val(f, i, j), u)     # any node: fluid value or ghost (extension)
    # v_Γ and p_Γ (= 2η ∂ₙvₙ) at the surface point closest to x (topography: above abscissa x)
    surface(x::Real) = surface(SVector(x, topo.h(x)))
    function surface(x::SVector)
        xΓ, n, κ = closest(topo, x, Δ)
        Tx, Ty = trace(:x, xΓ, n, κ), trace(:y, xΓ, n, κ)
        w, w2  = SVector(evaluate(Tx[2], u), evaluate(Ty[2], u)), SVector(evaluate(Tx[3], u), evaluate(Ty[3], u))
        S = SideFields(η, 0.0, w, w2, Tsurf(xΓ, n, κ)..., n, κ; f=fbody(xΓ))
        return (v=SVector(evaluate(Tx[1], u), evaluate(Ty[1], u)), p=S.p, ∇p=S.∇p)
    end
    vtop = topo isa Topography ? surface(0.0).v : nothing   # surface velocity at the top of the bump
    return (; V, φ, X, Δ, ncx, ncy, ndof, K, num, vtop, topo, value, surface, nghost=length(ghosts))
end

# Cell-centred values in the fluid (NaN in the void) for plotting
function centroid_fields(r)
    (; V, φ) = r
    vxc = (V.x[1:end-1, :] .+ V.x[2:end, :]) ./ 2
    vyc = (V.y[:, 1:end-1] .+ V.y[:, 2:end]) ./ 2
    m   = φ.p .< 0
    mask(a) = ifelse.(m, a, NaN)
    return (x=mask(vxc), y=mask(vyc), p=mask(V.p))
end

function main()
    # Hydrostatic check: flat surface between grid lines, exact solution v = 0, p = ρ|g|(y0 - y)
    r0  = FreeSurface(48; topo=Topography(x -> 0.0371, x -> 0.0, x -> 0.0))
    pex = [0.0371 - y for x in r0.X.p.x, y in r0.X.p.y]
    ok  = r0.φ.p .< 0
    @printf("hydrostatic: max|v| = %.2e, max|p - pex| = %.2e\n",
            max(maximum(abs, filter(!isnan, r0.V.x)), maximum(abs, filter(!isnan, r0.V.y))), maximum(abs.(r0.V.p[ok] .- pex[ok])))

    # Stress-form kernel with Vx ↔ Vy interpolation: identical for isotropic viscosity
    rc, ri = FreeSurface(96; bc=:no_slip), FreeSurface(96; bc=:no_slip, kernel=:interpolated)
    @printf("interpolated vs compact kernel: max difference %.2e\n",
            maximum(q -> maximum(abs, filter(!isnan, rc.V[q] .- ri.V[q])), (:x, :y, :p)))

    runs = Dict{Tuple{Symbol,Int},Any}()
    for bc in (:no_slip, :free_slip), ncx in (48, 96, 192, 384)
        t = @elapsed runs[(bc, ncx)] = FreeSurface(ncx; bc)
        r = runs[(bc, ncx)]; C = centroid_fields(r)
        @printf("%-9s ncx = %3d  ndof = %6d  Vy(top) = %.5f  max|Vx| = %.5f  max Vy = %.5f  max p = %.4f  (%.1f s)\n",
                bc, ncx, r.ndof, r.vtop[2], maximum(abs, filter(!isnan, C.x)), maximum(filter(!isnan, C.y)),
                maximum(filter(!isnan, C.p)), t)
    end
    # Self-convergence: max difference with the 4-cell average of the next finer grid, over the
    # cells of the coarse grid whose 4 children are in the fluid, and away from the surface (y < -0.5)
    for bc in (:no_slip, :free_slip), (nc, nf) in ((48, 96), (96, 192), (192, 384))
        c, f, y = centroid_fields(runs[(bc, nc)]), centroid_fields(runs[(bc, nf)]), runs[(bc, nc)].X.p.y
        e = map((:x, :y, :p)) do q
            F = f[q]; Fc = (F[1:2:end, 1:2:end] .+ F[2:2:end, 1:2:end] .+ F[1:2:end, 2:2:end] .+ F[2:2:end, 2:2:end]) ./ 4
            d = abs.(c[q] .- Fc)
            maximum(filter(!isnan, d)), maximum(filter(!isnan, d[:, y .< -0.5]))
        end
        @printf("%-9s %3d/%3d  all cells: ΔVx %.2e ΔVy %.2e Δp %.2e | y < -0.5: ΔVx %.2e ΔVy %.2e Δp %.2e\n",
                bc, nc, nf, first.(e)..., last.(e)...)
    end

    # Figure: IIM fields (ncx = 192), colour ranges of the reference figures
    xs  = range(-3, 3, length=400)
    fig = Figure(size=(1500, 900), fontsize=16)
    for (row, bc) in enumerate((:no_slip, :free_slip))
        r = runs[(bc, 192)]; C = centroid_fields(r)
        for (k, (q, lab, cr)) in enumerate(((:x, "Vx", (-0.055, 0.055)), (:y, "Vy", (-0.2, 0.1)), (:p, "P", (0.0, 5.0))))
            ax = Axis(fig[row, 2k-1], aspect=DataAspect(), title="$lab, $(replace(string(bc), "_" => " ")) (ncx = $(r.ncx))", limits=(-3.2, 3.2, -5.2, 1.3))
            hm = heatmap!(ax, r.X.p.x, r.X.p.y, C[q], colormap=:turbo, colorrange=cr)
            lines!(ax, xs, r.topo.h.(xs), color=:black)
            Colorbar(fig[row, 2k], hm)
        end
    end
    save(joinpath(@__DIR__, "IIM_FreeSurface.png"), fig)
    return runs
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
