# Make one movie per ShearApparatus*_Tet*_H* run folder (next to postprocessing/) from its figure_XXXX.png frames.
# H.264/MP4 (yuv420p) plays everywhere (QuickTime, PowerPoint, browsers).
# 2000x2000 frames downscaled to 1600 px, crf 20: ~2-3 MB for 100 frames, text stays sharp.
# Requires ffmpeg on the PATH (e.g. `brew install ffmpeg`).

function make_movie(folder; fps=10, width=1600, crf=20, pattern="figure_*.png")
    frames = filter(f -> occursin(r"^figure_\d+\.png$", f), readdir(folder))
    isempty(frames) && (@warn "No frames in $folder"; return nothing)
    out = joinpath(folder, basename(folder) * ".mp4")
    cmd = `ffmpeg -hide_banner -loglevel error -y -framerate $fps -pattern_type glob -i $(joinpath(folder, pattern))
           -vf "scale=$width:-2:flags=lanczos,format=yuv420p" -c:v libx264 -preset slow -crf $crf -tune animation
           -movflags +faststart $out`
    run(cmd)
    @info "$(basename(out)): $(length(frames)) frames, $(round(filesize(out)/1e6, digits=1)) MB"
    return out
end

let
    root    = dirname(@__DIR__) # run folders are next to postprocessing/
    folders = filter(d -> isdir(d) && occursin(r"^ShearApparatus.*_Tet\d+_H\d+$", basename(d)), readdir(root, join=true))
    for folder in folders
        make_movie(folder; fps=10, width=1600, crf=20)
    end
end
