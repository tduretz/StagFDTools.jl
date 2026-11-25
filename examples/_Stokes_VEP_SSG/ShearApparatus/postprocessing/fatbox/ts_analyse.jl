# Time series of the shear band families (Fatbox segments classified by the nearest expected plane: R1, R2 at the Arthur angle, P, Y;
# band_analysis.jl, fatbox_bands with planes = true) in the HR runs at θ = 15-85°, every saved step (10, 20, …, 500).
# Weighting: plastic strain rate λ̇ (τII is only stored at the checkpoints); at step 500 the λ̇-weighted shares are compared with the
# work-rate (λ̇ τII) shares from the checkpoint.
# Output (timeseries/): bands_timeseries.jld2 (named tuples, see below), families.csv, segments.csv.
#   families.csv: theta, step, delta_mm, mu_app, family (R1, R2, P, Y, none, outside), n_segments, length_mm, share (of Σλ̇ in the analysed
#                 gouge), angle (length-weighted axial mean of the segment orientations, from the vertical), spread (length-weighted
#                 rms deviation), expected (expected plane angle, from the vertical)
#   segments.csv: theta, step, delta_mm, family, angle (from the vertical), length_mm, share
# Usage: julia --project=<repo> -t 4 ts_analyse.jl
using JLD2, Printf, Statistics
load_step(file) = (; (Symbol(k) => v for (k, v) in load(file))...)
include(joinpath(@__DIR__, "..", "band_analysis.jl"))
root = joinpath(@__DIR__, "..", "..", "_HR_hardening_2")
ts   = joinpath(@__DIR__, "timeseries")
# length-weighted axial mean and rms deviation of line angles (degrees)
function axmean(α, w)
    z = sum(w .* cis.(2deg2rad.(α))); m = mod(rad2deg(angle(z)) / 2, 180)
    m, sqrt(sum(w .* adiff.(α, m) .^ 2) / sum(w))
end
fam_names = (:R1, :R2, :P, :Y)
rows_f = String[]; rows_s = String[]; data = []
for θ in (15, 20, 30, 40, 50, 60, 70, 80, 85), step in 10:10:500
    folder = joinpath(root, @sprintf("_ShearApparatus_v9_Tet%02d_H03", θ))
    prefix = @sprintf("Tet%d_s%04d", θ, step)
    isfile(joinpath(ts, prefix * "_nodes.csv")) || continue
    d  = load_step(joinpath(folder, @sprintf("step_%04d.jld2", step)))
    r  = fatbox_region(d)
    δ  = d.probes.t[d.it] * d.ε̇yy * d.Ly * d.sc.L * 1e3; μ = d.probes.app_fric_lab[d.it]
    if sum(r.λ̇[r.R]) == 0 || filesize(joinpath(ts, prefix * "_nodes.csv")) < 30   # no plastic flow / no segment yet
        push!(data, (; θ, step, δ, μ, fams = NamedTuple[], segs = NamedTuple[])); continue
    end
    rb = fatbox_bands(r, ones(size(r.λ̇)), θ; planes=true, dir=ts, prefix)   # τII = 1: shares of the plastic strain rate
    exp_ = Dict(:R1 => r.exp.R1A, :R2 => r.exp.R2A, :P => r.exp.P, :Y => r.exp.Y)
    segs = [(; α = fromvert(s.α), L = s.L, fam = rb.segfam[q] == 0 ? :none : rb.fams[rb.segfam[q]].name, share = rb.segshare[q])
            for (q, s) in enumerate(rb.segs)]
    fams = map(fam_names) do nm
        ss = filter(s -> s.fam == nm, segs)
        k  = findfirst(f -> f.name == nm, rb.fams)
        m, sd = isempty(ss) ? (NaN, NaN) : axmean([s.α for s in ss], [s.L for s in ss])
        (; name = nm, n = length(ss), L = sum((s.L for s in ss); init=0.0), share = k === nothing ? 0.0 : rb.fams[k].diss,
           angle = isnan(m) ? NaN : fromvert(m + 90), spread = sd, expected = fromvert(exp_[nm]))
    end
    nn = filter(s -> s.fam == :none, segs)
    fams = [fams..., (; name = :none, n = length(nn), L = sum((s.L for s in nn); init=0.0), share = rb.other, angle = NaN, spread = NaN, expected = NaN),
                     (; name = :outside, n = 0, L = 0.0, share = rb.diffuse.diss, angle = NaN, spread = NaN, expected = NaN)]
    push!(data, (; θ, step, δ, μ, σ1 = fromvert(r.exp.σ1), fams, segs))
    for f in fams
        push!(rows_f, @sprintf("%d,%d,%.4f,%.4f,%s,%d,%.3f,%.5f,%.2f,%.2f,%.2f", θ, step, δ, μ, f.name, f.n, f.L, f.share, f.angle, f.spread, f.expected))
    end
    for s in segs
        push!(rows_s, @sprintf("%d,%d,%.4f,%s,%.2f,%.3f,%.5f", θ, step, δ, s.fam, s.α, s.L, s.share))
    end
    step % 100 == 0 && @printf("θ = %d, step %d (δ = %.2f mm): %s\n", θ, step, δ,
        join([@sprintf("%s %.0f%% (%d, %+.0f°)", f.name, 100f.share, f.n, f.angle) for f in fams if f.n > 0 || f.name == :outside], ", "))
end
open(joinpath(ts, "families.csv"), "w") do io
    println(io, "theta,step,delta_mm,mu_app,family,n_segments,length_mm,share,angle,spread,expected"); foreach(l -> println(io, l), rows_f)
end
open(joinpath(ts, "segments.csv"), "w") do io
    println(io, "theta,step,delta_mm,family,angle,length_mm,share"); foreach(l -> println(io, l), rows_s)
end
jldsave(joinpath(ts, "bands_timeseries.jld2"); data)

# Check of the weighting at step 500: λ̇ shares vs work-rate shares (checkpoint τII)
for θ in (30, 50, 60)
    folder = joinpath(root, @sprintf("_ShearApparatus_v9_Tet%02d_H03", θ))
    d   = load_step(joinpath(folder, "step_0500.jld2")); r = fatbox_region(d)
    τII = load(joinpath(folder, "checkpoint_0500.jld2"), "state").grid.τ.II[2:end-1, 2:end-1] .* d.sc.σ
    a = fatbox_bands(r, ones(size(r.λ̇)), θ; planes=true, dir=ts, prefix=@sprintf("Tet%d_s0500", θ))
    b = fatbox_bands(r, τII, θ; planes=true, dir=ts, prefix=@sprintf("Tet%d_s0500", θ))
    @printf("θ = %d, step 500, share λ̇ / λ̇τII: %s\n", θ, join([@sprintf("%s %.1f/%.1f%%", fa.name, 100fa.diss, 100fb.diss) for (fa, fb) in zip(a.fams, b.fams)], ", "))
end
