struct Nought end
const NOUGHT = Nought()
struct Cross end
const CROSS = Cross()

mutable struct TicTacToeEnv <: AbstractEnv
    board::Matrix{Union{Nothing, Nought, Cross}}
    player::Union{Nought, Cross}
    TicTacToeEnv() = new(fill(nothing, 3, 3), NOUGHT)
end

NumAgentStyle(::TicTacToeEnv) = MultiAgent(2)
DynamicStyle(::TicTacToeEnv) = SEQUENTIAL
ActionStyle(::TicTacToeEnv) = FULL_ACTION_SET
InformationStyle(::TicTacToeEnv) = PERFECT_INFORMATION
StateStyle(::TicTacToeEnv) = (Observation{Int}(), Observation{Matrix{Int}}())
RewardStyle(::TicTacToeEnv) = TERMINAL_REWARD
UtilityStyle(::TicTacToeEnv) = ZERO_SUM
ChanceStyle(::TicTacToeEnv) = DETERMINISTIC
