using Test

@testset "StagFDTools.jl" verbose=true begin
    include("test_operators.jl")
    include("test_ad.jl")
    include("test_sparsity.jl")
    include("test_xy.jl")
end
