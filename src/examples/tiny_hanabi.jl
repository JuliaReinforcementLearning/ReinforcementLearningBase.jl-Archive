struct TinyHanabiEnv <: AbstractEnv
end

NumAgentStyle(::TinyHanabiEnv) = MultiAgent(2)
DynamicStyle(::TinyHanabiEnv) = SEQUENTIAL
ActionStyle(::TinyHanabiEnv) = MINIMAL_ACTION_SET
InformationStyle(::TinyHanabiEnv) = IMPERFECT_INFORMATION
StateStyle(::TinyHanabiEnv) = (Information{Tuple{Vararg{Int}}}(),)
RewardStyle(::TinyHanabiEnv) = TERMINAL_REWARD
UtilityStyle(::TinyHanabiEnv) = IDENTICAL_UTILITY
ChanceStyle(::TinyHanabiEnv) = DETERMINISTIC