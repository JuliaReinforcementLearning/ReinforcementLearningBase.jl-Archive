export LotteryEnv, WrappedEnv, MultiThreadEnv

using MacroTools: @forward
using Random

import Base.Threads.@spawn

#####
# LotteryEnv
#####

"""
    LotteryEnv()

Here we use an example introduced in [Monte Carlo Tree Search: A Tutorial](https://www.informs-sim.org/wsc18papers/includes/files/021.pdf) to demenstrate how to write an environment.

Assume you have \$10 in your pocket, and you are faced with the following three choices:

1. buy a PowerRich lottery ticket (win \$100M w.p. 0.01; nothing otherwise);
2. buy a MegaHaul lottery ticket (win \$1M w.p. 0.05; nothing otherwise);
3. do not buy a lottery ticket.
"""
mutable struct LotteryEnv <: AbstractEnv
    reward::Int
    is_done::Bool
    rng::MersenneTwister
end

LotteryEnv(;seed=nothing) = LotteryEnv(0, false, MersenneTwister(seed))

get_action_space(env::LotteryEnv) = DiscreteSpace((:PowerRich, :MegaHaul, nothing))

function (env::LotteryEnv)(action::Union{Symbol,Nothing})
    if action == :PowerRich
        env.reward = rand(env.rng) < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        env.reward = rand(env.rng) < 0.05 ? 1_000_000 : -10
    else
        env.reward = 0
    end
    env.is_done = true
end

observe(env::LotteryEnv) = (reward=env.reward, terminal=env.is_done)

reset!(env::LotteryEnv) = env.is_done = false

#####
# WrappedEnv
#####

"""
    WrappedEnv(;preprocessor, env, postprocessor=identity)

The observation of the inner `env` is first transformed by the `preprocessor`.
And the action is transformed by `postprocessor` and then send to the inner `env`.
"""
Base.@kwdef struct WrappedEnv{P,E<:AbstractEnv,T} <: AbstractEnv
    preprocessor::P
    env::E
    postprocessor::T = identity
end

"TODO: Deprecate"
WrappedEnv(p, env) = WrappedEnv(preprocessor = p, env = env)

(env::WrappedEnv)(args...) = env.env(env.postprocessor(args)...)

@forward WrappedEnv.env DynamicStyle,
ChanceStyle,
InformationStyle,
RewardStyle,
UtilityStyle,
ActionStyle,
get_action_space,
get_observation_space,
get_current_player,
get_player_id,
get_num_players,
get_history,
render,
reset!,
Random.seed!,
Base.copy

observe(env::WrappedEnv, player) = env.preprocessor(observe(env.env, player))
observe(env::WrappedEnv) = env.preprocessor(observe(env.env))

#####
# MultiThreadEnv
#####

"""
    MultiThreadEnv(envs::Vector{<:AbstractEnv})

Wrap multiple environments in one environment.
Each environment will run in parallel by leveraging `Threads.@spawn`.
"""
struct MultiThreadEnv{O,E} <: AbstractEnv
    obs::BatchObs{O}
    envs::Vector{E}
end

MultiThreadEnv(envs) = MultiThreadEnv(BatchObs([observe(env) for env in envs]), envs)

RLBase.get_action_space(env::MultiThreadEnv) = get_action_space(env.envs[1])
RLBase.get_observation_space(env::MultiThreadEnv) = get_observation_space(env.envs[1])

function (env::MultiThreadEnv)(actions)
    @sync for i in 1:length(env)
        @spawn begin
            env[i](actions[i])
            env.obs[i] = observe(env.envs[i])
        end
    end
end

RLBase.observe(env::MultiThreadEnv) = env.obs

function RLBase.reset!(env::MultiThreadEnv; is_force = false)
    if is_force
        for i in 1:length(env)
            reset!(env.envs[i])
        end
    else
        @sync for i in 1:length(env)
            if get_terminal(env.obs[i])
                @spawn begin
                    reset!(env.envs[i])
                    env.obs[i] = observe(env.envs[i])
                end
            end
        end
    end
end

@forward MultiThreadEnv.envs Base.getindex, Base.length, Base.setindex!

# TODO general APIs for MultiThreadEnv are missing