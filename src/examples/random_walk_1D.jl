struct RandomWalk1D <: AbstractEnv
    reward_table::Vector{Float64}
    start_pos::Int
    pos::Int
end

"""
    RandomWalk1D(;reward_table=[-1., zeros(5)..., 1.0], start_pos=4)

An agent is placed at the `start_pos` and can either move `:left` or `:right`.
After each movement, the agent get a reward based on its position in
`reward_table`. The game terminates when the agent reaches either end.

Compared to the [`MultiArmBanditsEnv`](@ref), the state space is more
complicated. (Well, not that complicated though.)
"""
function RandomWalk1D(;reward_table=[-1., zeros(5)..., 1.0], start_pos=4)
    RandomWalk1D(reward_table, start_pos, start_pos)
end

action_space(env::RandomWalk1D) = (:left, :right)

function (env::RandomWalk1D)(action)
    if action == :left
        env.pos = max(env.pos-1, 1)
    elseif action == :right
        env.pos = min(env.pos+1, length(env.reward_table))
    else
        @error "invalid action: $action"
    end
end

state(env::RandomWalk1D) = env.pos
state_space(env::RandomWalk1D) = Base.OneTo(length(env.reward_table))
reward(env::RandomWalk1D) = env.reward_table[env.pos]
is_terminated(env::RandomWalk1D) = env.pos == 1 || env.pos == length(env.reward_table)
reset!(env::RandomWalk1D) = env.pos = env.start_pos

NumAgentStyle(::RandomWalk1D) = SINGLE_AGENT
DynamicStyle(::RandomWalk1D) = SEQUENTIAL
ActionStyle(::RandomWalk1D) = MINIMAL_ACTION_SET
InformationStyle(::RandomWalk1D) = PERFECT_INFORMATION
StateStyle(::RandomWalk1D) = (Observation{Int}(),)
RewardStyle(::RandomWalk1D) = STEP_REWARD
UtilityStyle(::RandomWalk1D) = GENERAL_SUM
ChanceStyle(::RandomWalk1D) = DETERMINISTIC