# Export the plastic strain rate of the whole gouge strip (band_analysis.jl fatbox_region: all gouge cells up to the contacts, central 80%
# of its length; 0 elsewhere) of the HR runs at θ = 30-60°, step 500, for fault extraction with Fatbox (fatbox_extract.py).
# Files: Tet<θ>_ldot.bin (Float64, column-major nx × ny, [1/s]) and Tet<θ>_meta.txt (nx ny x1 y1 Δ [mm]).
# Usage: julia --project=<repo> fatbox_export.jl
using JLD2, Printf
load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(@__DIR__, "..", "band_analysis.jl"))
root = joinpath(@__DIR__, "..", "..", "_HR_hardening_2")
for θ in (30, 50, 60)
    d = load_step(joinpath(root, @sprintf("_ShearApparatus_v9_Tet%02d_H03", θ), "step_0500.jld2"))
    r = fatbox_region(d)
    F = r.λ̇ .* r.R
    write(joinpath(@__DIR__, "Tet$(θ)_ldot.bin"), F)
    open(joinpath(@__DIR__, "Tet$(θ)_meta.txt"), "w") do io
        println(io, size(F, 1), " ", size(F, 2), " ", r.x[1], " ", r.y[1], " ", r.x[2] - r.x[1])
    end
    println("θ = $θ: ", size(F), ", ", count(r.R), " cells")
end
