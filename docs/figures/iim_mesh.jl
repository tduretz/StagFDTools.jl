# Sketch of the discretisation in wlandry/discontinuous_stokes (IIM on a MAC grid)
using CairoMakie, LinearAlgebra

const h  = 1.0
const nc = 8                                # cells per direction in the sketch
const xc, yc, R = -1.0, -1.5, 7.2           # interface = circle, inside = high η
φ(x, y) = hypot(x - xc, y - yc) - R
sgn(p)  = sign(φ(p...))
nrm(p)  = normalize([p[1] - xc, p[2] - yc])

# ---------------- MAC nodes ----------------
P  = [(i + 0.5, j + 0.5) for i in 0:nc-1 for j in 0:nc-1]
VX = [(1.0i, j + 0.5)    for i in 0:nc   for j in 0:nc-1]
VY = [(i + 0.5, 1.0j)    for i in 0:nc-1 for j in 0:nc  ]

# Stencil of each equation (members whose sign is compared with the centre)
stencil_vx(x, y) = [(x - h, y), (x + h, y), (x, y - h), (x, y + h),          # zx (2zx_xx + zx_yy)
                    (x - h/2, y), (x + h/2, y),                              # p   (p_x)
                    (x - h/2, y - h/2), (x + h/2, y - h/2),                  # zy  (zy_xy)
                    (x - h/2, y + h/2), (x + h/2, y + h/2)]
stencil_vy(x, y) = [(y2, x2) for (x2, y2) in stencil_vx(y, x)]
stencil_p(x, y)  = [(x - h/2, y), (x + h/2, y), (x, y - h/2), (x, y + h/2)]  # div z
inbox(p) = -1e-9 <= p[1] <= nc + 1e-9 && -1e-9 <= p[2] <= nc + 1e-9
irregular(p, st) = any(q -> inbox(q) && sgn(q) != sgn(p), st(p...))

# Interface ∩ half grid lines (x = k h/2 or y = k h/2): where jumps are evaluated
function crossings()
    pts = Tuple{Float64,Float64}[]
    for k in 0:2nc, s in (-1, 1)
        c = k * h / 2
        d = R^2 - (c - xc)^2                     # vertical line x = c
        d >= 0 && (y = yc + s * sqrt(d); 0 <= y <= nc && push!(pts, (c, y)))
        d = R^2 - (c - yc)^2                     # horizontal line y = c
        d >= 0 && (x = xc + s * sqrt(d); 0 <= x <= nc && push!(pts, (x, c)))
    end
    pts
end
# Intersection on segment a–b (exact for the circle; code uses quadratic interpolation of φ)
function xing(a, b)
    f(t) = φ((a .+ t .* (b .- a))...)
    lo, hi = 0.0, 1.0
    for _ in 1:60
        m = (lo + hi) / 2
        sign(f(m)) == sign(f(lo)) ? (lo = m) : (hi = m)
    end
    a .+ lo .* (b .- a)
end

# ------------- interface value v_Γ: one-sided linear fit (compute_dv_dtt.cxx) -------------
# For each side ±: centre at x_Γ ± L n (L = h/max|n_i|), take the 3 closest same-side
# nodes of that velocity component (not all on one grid line), fit a + b·dx, extrapolate to x_Γ.
function fit_cloud(xg, nodes, side)
    n = nrm(xg); L = h / maximum(abs.(n))
    xo = xg .+ side * L .* n
    cand = sort(filter(q -> sgn(q) == side, nodes), by = q -> norm(q .- xo))
    sel = cand[1:2]
    for q in cand[3:end]
        if (q[1] != sel[1][1] || q[1] != sel[2][1]) && (q[2] != sel[1][2] || q[2] != sel[2][2])
            push!(sel, q); break
        end
    end
    xo, sel
end

# ---------------- figure ----------------
cin, cout = RGBf(0.85, 0.33, 0.10), RGBf(0.00, 0.45, 0.70)
phasecol(p) = sgn(p) < 0 ? cin : cout
θ = range(0, 2π, 400)
circ = Point2f.(xc .+ R .* cos.(θ), yc .+ R .* sin.(θ))

fig = Figure(size = (1500, 640), fontsize = 15)

function base!(ax, lims)
    poly!(ax, circ, color = (cin, 0.10), strokewidth = 0)
    for k in 0:nc
        lines!(ax, [k, k], [0, nc], color = :gray70, linewidth = 1)
        lines!(ax, [0, nc], [k, k], color = :gray70, linewidth = 1)
    end
    lines!(ax, circ, color = :black, linewidth = 2.5)
    limits!(ax, lims...)
    hidedecorations!(ax); hidespines!(ax)
end

# ---- (a) whole grid ----
ax1 = Axis(fig[1, 1], aspect = DataAspect(),
           title = L"\text{(a) standard MAC unknowns only } (z = \eta v,\ p)\text{: no extra DOFs}")
base!(ax1, (-0.3, nc + 0.3, -0.3, nc + 0.3))
for (nodes, st, mk, rot) in ((P, stencil_p, :circle, 0.0),
                             (VX, stencil_vx, :rtriangle, 0.0),
                             (VY, stencil_vy, :utriangle, 0.0))
    reg = filter(p -> !irregular(p, st), nodes)
    irr = filter(p -> irregular(p, st), nodes)
    scatter!(ax1, Point2f.(reg), marker = mk, markersize = 13, color = phasecol.(reg))
    scatter!(ax1, Point2f.(irr), marker = mk, markersize = 17, color = phasecol.(irr),
             strokecolor = :black, strokewidth = 2)
end
V = [(1.0i, 1.0j) for i in 0:nc for j in 0:nc]
scatter!(ax1, Point2f.(V), marker = :xcross, markersize = 7, color = :gray50)
scatter!(ax1, Point2f.(crossings()), marker = :diamond, markersize = 10, color = :gold,
         strokecolor = :black, strokewidth = 1)

# ---- (b) zoom on one irregular x-momentum equation ----
c0, nb = (5.0, 3.5), (4.0, 3.5)          # vx node outside; left neighbour inside
@assert sgn(c0) > 0 && sgn(nb) < 0
ax2 = Axis(fig[1, 2], aspect = DataAspect(),
           title = L"\text{(b) x momentum at an irregular } z_x \text{ node: corrected stencil}")
base!(ax2, (2.2, 7.6, 1.0, 6.4))

st = stencil_vx(c0...)
szx = [c0; st[1:4]]; sp = st[5:6]; szy = st[7:10]

# interpolation clouds for the interface point on the x-line (dd_pos in x)
xg = xing(c0, nb)
for (side, col) in ((1, cout), (-1, cin))
    for (nodes, mk) in ((VX, :rtriangle), (VY, :utriangle))
        xo, sel = fit_cloud(xg, nodes, side)
        poly!(ax2, Point2f.(sel), color = (col, 0.12), strokecolor = (col, 0.8),
              strokewidth = 1.2, linestyle = mk == :rtriangle ? :solid : :dash)
        lines!(ax2, Point2f[xg, xo], color = :black, linewidth = 1)
        scatter!(ax2, Point2f(xo), marker = :cross, markersize = 14, color = :black)
    end
end
n = nrm(xg)
arrows2d!(ax2, [Point2f(xg)], [Vec2f(0.8n...)], color = :black, shaftwidth = 2, tipwidth = 9)
text!(ax2, Point2f(xg .+ 0.85 .* n .+ (0.05, 0.0)), text = "n", fontsize = 17)

# all grid nodes faint
scatter!(ax2, Point2f.(VX), marker = :rtriangle, markersize = 11, color = (:gray60, 0.5))
scatter!(ax2, Point2f.(VY), marker = :utriangle, markersize = 11, color = (:gray60, 0.5))
scatter!(ax2, Point2f.(P),  marker = :circle,    markersize = 10, color = (:gray60, 0.5))
# stencil
scatter!(ax2, Point2f.(szx), marker = :rtriangle, markersize = 22, color = phasecol.(szx),
         strokecolor = :black, strokewidth = 1.5)
scatter!(ax2, Point2f.(szy), marker = :utriangle, markersize = 22, color = phasecol.(szy),
         strokecolor = :black, strokewidth = 1.5)
scatter!(ax2, Point2f.(sp), marker = :circle, markersize = 20, color = phasecol.(sp),
         strokecolor = :black, strokewidth = 1.5)
scatter!(ax2, Point2f(c0), marker = :rect, markersize = 34, color = :transparent,
         strokecolor = :black, strokewidth = 2.5)

# interface points this equation uses (Interface.hxx)
dd  = [xing(c0, q) for q in st[1:4]  if sgn(q) != sgn(c0)]
dp  = [xing(c0, q) for q in sp       if sgn(q) != sgn(c0)]
cor = [xing(q, (q[1], c0[2])) for q in szy if sgn(q) != sgn(c0)]   # on cell edge
scatter!(ax2, Point2f.(dd),  marker = :star5,   markersize = 22, color = :gold,
         strokecolor = :black, strokewidth = 1)
scatter!(ax2, Point2f.(dp),  marker = :diamond, markersize = 17, color = :gold,
         strokecolor = :black, strokewidth = 1)
scatter!(ax2, Point2f.(cor), marker = :hexagon, markersize = 17, color = :gold,
         strokecolor = :black, strokewidth = 1)
text!(ax2, Point2f(xg .+ (-0.45, 0.12)), text = L"x_\Gamma", fontsize = 20)

# ---- legend ----
elems = [MarkerElement(marker = :circle,    color = :gray40, markersize = 13),
         MarkerElement(marker = :rtriangle, color = :gray40, markersize = 13),
         MarkerElement(marker = :utriangle, color = :gray40, markersize = 13),
         MarkerElement(marker = :xcross,    color = :gray50, markersize = 9),
         PolyElement(color = (cin, 0.6)), PolyElement(color = (cout, 0.6)),
         MarkerElement(marker = :rtriangle, color = :white, strokecolor = :black,
                       strokewidth = 2, markersize = 15),
         MarkerElement(marker = :diamond, color = :gold, strokecolor = :black,
                       strokewidth = 1, markersize = 11),
         MarkerElement(marker = :star5, color = :gold, strokecolor = :black,
                       strokewidth = 1, markersize = 16),
         MarkerElement(marker = :diamond, color = :gold, strokecolor = :black,
                       strokewidth = 1, markersize = 14),
         MarkerElement(marker = :hexagon, color = :gold, strokecolor = :black,
                       strokewidth = 1, markersize = 14),
         MarkerElement(marker = :cross, color = :black, markersize = 13),
         PolyElement(color = (:gray, 0.15), strokecolor = :gray30, strokewidth = 1.2)]
labs = ["p (cell centre)", L"z_x = \eta v_x \text{ (x face)}", L"z_y = \eta v_y \text{ (y face)}",
        "vertex: level set only (dist_edge)",
        L"\text{phase } -\text{: inclusion } (\eta_-)", L"\text{phase } +\text{: matrix } (\eta_+)",
        L"\text{irregular node: stencil crosses } \Gamma \rightarrow + C",
        L"\text{(a) } \Gamma \cap \text{ half grid lines: jump evaluation points}",
        L"x_\Gamma \text{ for } \partial_{xx} z_x,\ \partial_{yy} z_x \text{ (dd\_pos)}",
        L"x_\Gamma \text{ for } \partial_x p \text{ (dp\_pos)}",
        L"x_\Gamma \text{ for } \partial_{xy} z_y \text{, cut corner (corner\_pos)}",
        L"x_\Gamma \pm L\,n\text{: one sided fit centres}",
        L"\text{three node linear fit per side/component} \rightarrow v_\Gamma,\ \partial_t v_\Gamma"]
Legend(fig[1, 3], elems, labs, framevisible = false, rowgap = 6)
colsize!(fig.layout, 1, Relative(0.38)); colsize!(fig.layout, 2, Relative(0.38))

out = joinpath(@__DIR__, "iim_mesh.png")
save(out, fig, px_per_unit = 2)
println(out)
