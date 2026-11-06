# For each run folder given as argument, find the most recent checkpoint that can be read completely.
# Prints one line per folder on stdout:  <folder> <status> <step> <time [s]>
#   status: ok (restart from step) | done (time_max reached, nothing to do) | none (no readable checkpoint)
# Unreadable (e.g. truncated by a job killed while writing) checkpoints are skipped, and renamed *.jld2.corrupt with --rename
using JLD2, Printf

# Unscaled end time and time scale, read from the folder's copy of the script
function run_params(dir)
    src = read(joinpath(dir, "ShearApparatus_v9.jl"), String)
    tmax = parse(Float64, match(r"^\s*time_max\s*=\s*([0-9.eE+-]+)"m, src)[1])
    st   = parse(Float64, match(r"sc\s*=\s*\(.*t\s*=\s*([0-9.eE+-]+)\)", src)[1])
    return tmax, st
end

# Reads every dataset of the checkpoint (a truncated file throws here)
function read_checkpoint(file, step)
    jldopen(file, "r") do f
        it, time, Δt, state = f["it"], f["time"], f["Δt"], f["state"]
        it == step || error("checkpoint contains step $it")
        isfinite(time) && isfinite(Δt) && Δt > 0 || error("time = $time, Δt = $Δt")
        nbad = count(x -> !isfinite(x), Iterators.flatten(vec(a) for a in leaves(state.grid)))
        nbad > 0 && @warn "$file: $nbad non-finite values in the grid fields"
        return time
    end
end
leaves(x::AbstractArray{<:Number}) = (x,)
leaves(x::Union{NamedTuple, Tuple}) = Iterators.flatten(map(leaves, values(x)))
leaves(x) = ()

const RENAME = "--rename" in ARGS
for dir in filter(!=("--rename"), ARGS)
    dir  = rstrip(dir, '/')
    outs = filter(d -> startswith(d, "_ShearApparatus_v9_") && isdir(joinpath(dir, d)), readdir(dir))
    if length(outs) != 1
        @warn "$dir: expected one output folder, found $(length(outs))"
        println("$dir none 0 0"); continue
    end
    out   = joinpath(dir, only(outs))
    files = filter(s -> occursin(r"^checkpoint_\d+\.jld2$", s), readdir(out))
    steps = sort(parse.(Int, match.(r"\d+", files) .|> m -> m.match); rev=true)
    tmax, st = run_params(dir)
    found = false
    for step in steps
        file = joinpath(out, @sprintf("checkpoint_%04d.jld2", step))
        time = try
            read_checkpoint(file, step)
        catch e
            @warn "$file is not readable" * (RENAME ? ", renamed to *.corrupt" : "") exception=e
            RENAME && mv(file, file * ".corrupt"; force=true)
            continue
        end
        status = time * st >= tmax * (1 - 1e-10) ? "done" : "ok"
        @printf("%s %s %d %.1f\n", dir, status, step, time * st)
        found = true
        break
    end
    found || println("$dir none 0 0")
end
