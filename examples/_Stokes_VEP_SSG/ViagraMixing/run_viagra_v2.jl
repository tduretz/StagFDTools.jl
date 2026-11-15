# Viagra benchmark: grid averaging (iso-strain = voigt, iso-stress = reuss, iso-work = isowork) vs. marker averaging
# (markers = iso-strain, markers_stress, markers_work), 51×51, 200 steps. Wall time per scheme in ViagraMixing_timing_v2.txt
include(joinpath(@__DIR__, "ViagraTest_mixing.jl"))
schemes = length(ARGS) > 0 ? Symbol.(ARGS) : [:voigt, :reuss, :isowork, :markers, :markers_stress, :markers_work]
for s in schemes
    t = @elapsed main(s; nc=(x=51, y=51), nt=200, tag="_v2")
    open(joinpath(@__DIR__, "ViagraMixing_timing_v2.txt"), "a") do io
        println(io, "$s $t")
    end
    println("==== $s done in $(round(t, digits=1)) s")
end
