# Figures of the free-surface test in the style of the FEM (FCFV_Geodynamics.jl) and
# deformed-mesh FD (FullStaggeredGrid) figures: one field per figure, cells drawn as polygons,
# cut cells clipped at the surface. Values: cell averages of Vx, Vy and the centroid pressure;
# in cut cells with a void centroid, the extension of the fluid solution (ghost values).
# No slip on the sides and the bottom, as in both references.
using CairoMakie
import CairoMakie.GeometryBasics: Polygon
include(joinpath(@__DIR__, "IIM_FreeSurface.jl"))

function cell_polygons(r; nsub=8)
    (; X, Δ, topo) = r
    polys, cells = Polygon[], CartesianIndex{2}[]
    for j in eachindex(X.p.y), i in eachindex(X.p.x)
        x0, x1, y0 = X.x.x[i], X.x.x[i+1], X.y.y[j]
        y1 = y0 + Δ
        xs = range(x1, x0, length=nsub + 1)
        top = clamp.(topo.h.(xs), y0, y1)
        all(top .<= y0 + 1e-12) && continue                  # cell entirely in the void
        push!(polys, Polygon(Point2f[(x0, y0), (x1, y0), (Point2f(x, y) for (x, y) in zip(xs, top))...]))
        push!(cells, CartesianIndex(i, j))
    end
    return polys, cells
end

function plot_field(polys, vals, file; colorrange)
    f = Figure(size=(1200, 1000))
    Axis(f[1, 1], aspect=DataAspect())
    poly!(polys, color=vals, colormap=:turbo, colorrange=colorrange, strokewidth=0.5, strokecolor=:white)
    Colorbar(f[1, 2], colormap=:turbo, limits=colorrange, flipaxis=true, size=25)
    save(file, f; px_per_unit=1)
end

r = FreeSurface(96; bc=:no_slip)
polys, cells = cell_polygons(r)
vals = (Vx=[(r.value(:x, I[1], I[2]) + r.value(:x, I[1] + 1, I[2])) / 2 for I in cells],
        Vy=[(r.value(:y, I[1], I[2]) + r.value(:y, I[1], I[2] + 1)) / 2 for I in cells],
        P =[r.value(:p, I[1], I[2]) for I in cells])
out = joinpath(@__DIR__, "..", "..", "docs", "IIM_notes", "figures")
for (q, cr) in ((:Vx, (-0.055, 0.055)), (:Vy, (-0.2, 0.1)), (:P, (0.0, 5.0)))
    plot_field(polys, vals[q], joinpath(out, "FreeSurfaceTest_IIM_$q.png"); colorrange=cr)
end
# Reference figures, copied next to the IIM ones
for (src, tag) in (("/Users/tduretz/REPO/FCFV_Geodynamics.jl/images", "FEM"), ("/Users/tduretz/REPO/FullStagerredGrid/images", "FDM")), q in ("Vx", "Vy", "P")
    f = joinpath(src, "FreeSurfaceTest_$q.png")
    isfile(f) && cp(f, joinpath(out, "FreeSurfaceTest_$(tag)_$q.png"); force=true)
end

# Surface pressure p_Γ = 2η∂ₙvₙ and surface velocity, 10- and 16-point quadratic fits (no slip)
let xs = range(-2.95, 2.95, length=800)
    f = Figure(size=(1500, 430), fontsize=16)
    axs = (Axis(f[1, 1], xlabel="x", ylabel="p_Γ", title="surface pressure, 10-point quadratic fits", limits=(nothing, (0, 0.36))),
           Axis(f[1, 2], xlabel="x", ylabel="p_Γ", title="surface pressure, 16-point quadratic fits", limits=(nothing, (0, 0.36))),
           Axis(f[1, 3], xlabel="x", ylabel="v_Γ", title="surface velocity (vx solid, vy dashed), 16 points"))
    for (n, c) in ((96, :steelblue), (192, :orange), (384, :black))
        for (k, np) in enumerate((10, 16))
            r = FreeSurface(n; bc=:no_slip, npts=np, degree=2)
            sv = r.surface.(xs)
            lines!(axs[k], xs, getfield.(sv, :p), color=c, label="ncx = $n")
            if np == 16
                lines!(axs[3], xs, [s.v[1] for s in sv], color=c)
                lines!(axs[3], xs, [s.v[2] for s in sv], color=c, linestyle=:dash)
            end
        end
    end
    axislegend(axs[1], position=:ct)
    save(joinpath(out, "IIM_FreeSurface_SurfacePressure.png"), f)
end
