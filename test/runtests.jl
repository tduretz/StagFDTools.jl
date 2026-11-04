using Test

@testset "StagFDTools.jl" verbose=true begin
    include("test_operators.jl")
    include("test_ad.jl")
    include("test_sparsity.jl")
    include("test_xy.jl")
    include("test_iim_geometry.jl")
    include("test_iim_trace.jl")
    include("test_iim_jumps.jl")
    include("test_iim_assembly.jl")
end
