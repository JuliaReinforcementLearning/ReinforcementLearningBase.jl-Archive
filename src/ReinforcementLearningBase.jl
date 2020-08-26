module ReinforcementLearningBase

const RLBase = ReinforcementLearningBase
export RLBase

include("inline_export.jl")
include("interface.jl")
include("base.jl")
include("implementations/implementations.jl")
include("CommonRLInterface.jl")

end # module
