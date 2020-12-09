module ReinforcementLearningBase

const RLBase = ReinforcementLearningBase
export RLBase

include("inline_export.jl")
include("interface.jl")
include("CommonRLInterface.jl")
include("base.jl")
include("examples/examples.jl")

end # module
