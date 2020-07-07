export WrappedEnv, SubjectiveEnv, MultiThreadEnv,
    AbstractPreprocessor, CloneStatePreprocessor, ComposedPreprocessor

using MacroTools: @forward
using Random

import Base.Threads.@spawn

#####
# SubjectiveEnv
#####

Base.@kwdef struct SubjectiveEnv{E<:AbstractEnv,P}
    env::E
    player::P
end

for f in ENV_API
    @eval $f(x::SubjectiveEnv, args...;kwargs...) = $f(x.env, args...;kwargs...)
end

for f in MULTI_AGENT_ENV_API
    @eval $f(x::SubjectiveEnv) = $f(x.env, x.player)
end

#####
# WrappedEnv
#####

"""
    WrappedEnv(;preprocessor=identity, env, postprocessor=identity)

The observation of the inner `env` is first transformed by the `preprocessor`.
And the action is transformed by `postprocessor` and then send to the inner `env`.
"""
Base.@kwdef struct WrappedEnv{P,E<:AbstractEnv,T} <: AbstractEnv
    preprocessor::P = identity
    env::E
    postprocessor::T = identity
end

(env::WrappedEnv)(args...) = env.env(env.postprocessor(args)...)

for f in vcat(ENV_API, MULTI_AGENT_ENV_API)
    if f != :observe
        @eval $f(x::WrappedEnv, args...;kwargs...) = $f(x.env, args...;kwargs...)
    end
end

observe(env::WrappedEnv, player) = env.preprocessor(observe(env.env, player))
observe(env::WrappedEnv) = env.preprocessor(observe(env.env))

#####
## Preprocessors
#####

abstract type AbstractPreprocessor end

"""
    (p::AbstractPreprocessor)(obs)

By default a [`StateOverriddenObs`](@ref) is returned to avoid modifying original observation.
"""
(p::AbstractPreprocessor)(obs) = StateOverriddenObs(obs = obs, state = p(get_state(obs)))

"""
    ComposedPreprocessor(p::AbstractPreprocessor...)

Compose multiple preprocessors.
"""
struct ComposedPreprocessor{T} <: AbstractPreprocessor
    preprocessors::T
end

ComposedPreprocessor(p...) = ComposedPreprocessor(p)
(p::ComposedPreprocessor)(obs) = reduce((x, f) -> f(x), p.preprocessors, init = obs)

#####
# CloneStatePreprocessor
#####

"""
    CloneStatePreprocessor()

Do `deepcopy` for the state in an observation.
"""
struct CloneStatePreprocessor <: AbstractPreprocessor end

(p::CloneStatePreprocessor)(obs) = StateOverriddenObs(obs, deepcopy(get_state(obs)))

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

function (env::MultiThreadEnv)(actions)
    @sync for i in 1:length(env)
        @spawn begin
            env[i](actions[i])
            env.obs[i] = observe(env.envs[i])
        end
    end
end

observe(env::MultiThreadEnv) = env.obs

function reset!(env::MultiThreadEnv; is_force = false)
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

# !!! some might not be meaningful, use with caution.
for f in vcat(ENV_API, MULTI_AGENT_ENV_API)
    if f != :reset
        @eval $f(x::MultiThreadEnv, args...;kwargs...) = $f(x.envs[1], args...;kwargs...)
    end
end
