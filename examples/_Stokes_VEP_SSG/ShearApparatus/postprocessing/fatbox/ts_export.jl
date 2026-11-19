# Time series for Fatbox: export the plastic strain rate of the analysed gouge (as fatbox_export.jl) of the HR runs at θ = 30-60°
# for every saved step (10, 20, …, 500) into timeseries/Tet<θ>_s<step>_ldot.bin and _meta.txt.
# Then: <Fatbox>/.venv/bin/python fatbox_extract.py 0.03 1.0 timeseries ; julia ts_analyse.jl
using JLD2, Printf
load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(@__DIR__, "..", "band_analysis.jl"))
root = joinpath(@__DIR__, "..", "..", "_HR_hardening_2")
out  = mkpath(joinpath(@__DIR__, "timeseries"))
for θ in (30, 50, 60), step in 10:10:500
    f = joinpath(root, @sprintf("_ShearApparatus_v9_Tet%02d_H03", θ), @sprintf("step_%04d.jld2", step))
    isfile(f) || (println("missing $f"); continue)
    r = fatbox_region(load_step(f))
    write(joinpath(out, @sprintf("Tet%d_s%04d_ldot.bin", θ, step)), r.λ̇ .* r.R)
    open(joinpath(out, @sprintf("Tet%d_s%04d_meta.txt", θ, step)), "w") do io
        println(io, size(r.λ̇, 1), " ", size(r.λ̇, 2), " ", r.x[1], " ", r.y[1], " ", r.x[2] - r.x[1])
    end
end
println("exported to $out")
