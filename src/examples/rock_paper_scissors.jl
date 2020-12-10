"""
    RockPaperScissorsEnv()

[Rock Paper Scissors](https://en.wikipedia.org/wiki/Rock_paper_scissors) is a
simultaneous, zero sum game.
"""
mutable struct RockPaperScissorsEnv <: AbstractEnv
    reward::Tuple{Int, Int}
    is_done::Bool
end

NumAgentStyle(::RockPaperScissorsEnv) = MultiAgent(2)
DynamicStyle(::RockPaperScissorsEnv) = SIMULTANEOUS
UtilityStyle(::RockPaperScissorsEnv) = ZERO_SUM
players(::RockPaperScissorsEnv) = (1,2)
current_player(::RockPaperScissorsEnv) = SIMULTANEOUS_PLAYER

action_space(::RockPaperScissorsEnv, ::Int) = ('💎', '📃', '✂')
action_space(::RockPaperScissorsEnv, ::SimultaneousPlayer) = Tuple((i,j) for i in ('💎', '📃', '✂') for j in ('💎', '📃', '✂'))
action_space(env::RockPaperScissorsEnv) = action_space(env, SIMULTANEOUS_PLAYER)

"Since it's a one-shot game, the state space doesn't have much meaning."
state_space(::RockPaperScissorsEnv) = (1,)
state(::RockPaperScissorsEnv) = 1

reward(env::RockPaperScissorsEnv) = env.is_done ? env.reward : (0, 0)
is_terminated(::RockPaperScissorsEnv) = env.is_done
reset!(env::RockPaperScissorsEnv) = env.is_done = false

function (env::RockPaperScissorsEnv)((x,y))
    if x == y
        env.reward = (0,0)
    elseif x == '💎' && y == '✂' || x == '✂' && y == '📃' || x == '📃' && y == '💎'
        env.reward = (1,-1)
    else
        env.reward = (-1, 1)
    end
    env.is_done = true
end

NumAgentStyle(::RockPaperScissorsEnv) = MultiAgent(2)
DynamicStyle(::RockPaperScissorsEnv) = SIMULTANEOUS
ActionStyle(::RockPaperScissorsEnv) = MINIMAL_ACTION_SET
InformationStyle(::RockPaperScissorsEnv) = IMPERFECT_INFORMATION
StateStyle(::RockPaperScissorsEnv) = (Observation{Int}(),)
RewardStyle(::RockPaperScissorsEnv) = TERMINAL_REWARD
UtilityStyle(::RockPaperScissorsEnv) = ZERO_SUM
ChanceStyle(::RockPaperScissorsEnv) = DETERMINISTIC
