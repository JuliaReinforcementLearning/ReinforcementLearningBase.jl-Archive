using ReinforcementLearningBase
using Test
import ReinforcementLearningBase: CRL
import ReinforcementLearningBase.CRL

@testset "ReinforcementLearningBase" begin
    include("examples/examples.jl")
    include("common_rl_env.jl")
end
