@doc """
[ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl) (**RLBase**)
provides some common constants, traits, abstractions and interfaces in developing reinforcement learning algorithms in 
Julia. From the concept level, they can be organized in the following parts:

- [Policy](@ref)
- [EnvironmentModel](@ref)
- [Environment](@ref)
  - [Traits for Environment](@ref)
  - [Observation of Environment](@ref)
  - [Observation Space and Action Space](@ref)
""" RLBase

import Base: copy, copy!, length, in, eltype
import Random: seed!, rand, AbstractRNG
import AbstractTrees: children, has_children
import Markdown

#####
# Policy
#####

"""
    (π::AbstractPolicy)(obs) -> action

Policy is the most basic concept in reinforcement learning. A policy is a functional object which takes in an observation and generate an action.
"""
@api abstract type AbstractPolicy end
@api (π::AbstractPolicy)(obs)


"""
    update!(π::AbstractPolicy, experience)

Update the policy `π` with online/offline experience.
"""
@api update!(π::AbstractPolicy, experience) = nothing

"""
    get_prob(π::AbstractPolicy, obs)

Get the probability distribution of actions based on policy `π` given an observation `obs`. 
"""
@api get_prob(π::AbstractPolicy, obs)

"""
    get_prob(π::AbstractPolicy, obs, action)

Only valid for environments with discrete action space.
"""
@api get_prob(π::AbstractPolicy, obs, action)

"""
    get_priority(π::AbstractPolicy, experience)

Usually used in offline policies.
"""
@api get_priority(π::AbstractPolicy, experience)

#####
# Environment
#####

"""
    (env::AbstractEnv)(action) = env(action, get_current_player(env)) -> env
    (env::AbstractEnv)(action, player) -> env

Super type of all reinforcement learning environments.
"""
@api abstract type AbstractEnv end

#####
## Traits for Environment
## mostly borrowed from https://github.com/deepmind/open_spiel/blob/master/open_spiel/spiel.h
#####

#####
### NumAgentStyle
#####

abstract type AbstractNumAgentStyle end

@api struct SingleAgent <: AbstractNumAgentStyle end
@api const SINGLE_AGENT = SingleAgent()

@api struct MultiAgent{N} <: AbstractNumAgentStyle end

function MultiAgent(n::Int)
    if n < 0
        throw(ArgumentError("number of agents must be > 1, get $n"))
    elseif n == 1
        throw(ArgumentError("do you want mean SINGLE_AGENT?"))
    else
        MultiAgent{n}()
    end
end

@api const TWO_AGENT = MultiAgent(2)

"""
    NumAgentStyle(env)
"""
@env_api NumAgentStyle(env::AbstractEnv) = SINGLE_AGENT

#####
### DynamicStyle
#####

abstract type AbstractDynamicStyle end

@api struct Sequential <: AbstractDynamicStyle end
@api struct Simultaneous <: AbstractDynamicStyle end

"Environment with the [`DynamicStyle`](@ref) of `SEQUENTIAL` must takes actions from different players one-by-one."
@api const SEQUENTIAL = Sequential()

"Environment with the [`DynamicStyle`](@ref) of `SIMULTANEOUS` must take in actions from some (or all) players at one time"
@api const SIMULTANEOUS = Simultaneous()

"""
    DynamicStyle(env::AbstractEnv) = SEQUENTIAL

Determine whether the players can play simultaneously or not. Default value is [`SEQUENTIAL`](@ref)
"""
@env_api DynamicStyle(env::AbstractEnv) = SEQUENTIAL

#####
### InformationStyle
#####

abstract type AbstractInformationStyle end

@api struct PerfectInformation <: AbstractInformationStyle end
@api struct ImperfectInformation <: AbstractInformationStyle end

"All players observe the same state"
@api const PERFECT_INFORMATION = PerfectInformation()

"The inner state of some players' observations may be different"
@api const IMPERFECT_INFORMATION = ImperfectInformation()

"""
    InformationStyle(env) = PERFECT_INFORMATION

Specify whether the `env` is [PERFECT_INFORMATION](@ref) or [IMPERFECT_INFORMATION](@ref).
Return [PERFECT_INFORMATION](@ref) by default.
"""
@env_api InformationStyle(env::AbstractEnv) = PERFECT_INFORMATION

#####
### ChanceStyle
#####

abstract type AbstractChanceStyle end
abstract type AbstractStochasticChanceStyle end

@api struct Deterministic <: AbstractChanceStyle end
@api struct Stochastic <: AbstractChanceStyle end
@api struct ExplicitStochastic <: AbstractStochasticChanceStyle end
@api struct SampledStochastic <: AbstractStochasticChanceStyle end

"No chance player in the environment. And the game is deterministic."
@api const DETERMINISTIC = Deterministic()

"No chance player in the environment. And the game is stochastic."
@api const STOCHASTIC = Stochastic()

"Environment contains chance player and the probability is known."
@api const EXPLICIT_STOCHASTIC = ExplicitStochastic()

"""
Environment contains chance player and the probability is unknown.
Usually only a dummy action is allowed in this case.
"""
@api const SAMPLED_STOCHASTIC = SampledStochastic()

"""
    ChanceStyle(env) = DETERMINISTIC
"""
@env_api ChanceStyle(env::AbstractEnv) = DETERMINISTIC

#####
### RewardStyle
#####

abstract type AbstractRewardStyle end

@api struct StepReward <: AbstractRewardStyle end
@api struct TerminalReward <: AbstractRewardStyle end

"We can get reward after each step"
@api const STEP_REWARD = StepReward()

"Only get reward at the end of environment"
@api const TERMINAL_REWARD = TerminalReward()

"Specify whether we can get reward after each step or only at the end of an game. Possible values are [STEP_REWARD](@ref) or [TERMINAL_REWARD](@ref)"
@env_api RewardStyle(env::AbstractEnv) = STEP_REWARD

#####
### UtilityStyle
#####

abstract type AbstractUtilityStyle end

@api struct ZeroSum <: AbstractUtilityStyle end
@api struct ConstantSum <: AbstractUtilityStyle end
@api struct GeneralSum <: AbstractUtilityStyle end
@api struct IdenticalUtility <: AbstractUtilityStyle end

"Rewards of all players sum to 0"
@api const ZERO_SUM = ZeroSum()

"Rewards of all players sum to a constant"
@api const CONSTANT_SUM = ConstantSum()

"Total rewards of all players may be different in each step"
@api const GENERAL_SUM = GeneralSum()

"Every player gets the same reward"
@api const IDENTICAL_REWARD = IdenticalUtility()

"""
    UtilityStyle(env::AbstractEnv)

Specify the utility style in multi-agent environments.
Possible values are:

- [ZERO_SUM](@ref)
- [CONSTANT_SUM](@ref)
- [GENERAL_SUM](@ref)
- [IDENTICAL_REWARD](@ref)
"""
@env_api UtilityStyle(env::AbstractEnv) = GENERAL_SUM

#####
### ActionStyle
#####

abstract type AbstractActionStyle end
@api struct FullActionSet <: AbstractActionStyle end

"The action space of the environment may contains illegal actions"
@api const FULL_ACTION_SET = FullActionSet()

@api struct MinimalActionSet <: AbstractActionStyle end

"All actions in the action space of the environment are legal"
@api const MINIMAL_ACTION_SET = MinimalActionSet()

"""
    ActionStyle(env::AbstractEnv)

Specify whether the observation contains a full action set or a minimal action set.
By default the [`MINIMAL_ACTION_SET`](@ref) is returned.
"""
@env_api ActionStyle(obs) = MINIMAL_ACTION_SET

# !!! TODO: Deprecate
ActionStyle(obs::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) = FULL_ACTION_SET
ActionStyle(obs::NamedTuple{(:reward, :terminal, :state, :legal_actions_mask)}) =
    FULL_ACTION_SET
ActionStyle(
    obs::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)},
) = FULL_ACTION_SET

function print_traits(io::IO, env::AbstractEnv)
    s = """
    ## Traits

    | Trait Type | Value |
    |:---------- | ----- |
    | NumAgentStyle | $(NumAgentStyle(env)) |
    | ActionStyle | $(ActionStyle(env)) |
    | DynamicStyle | $(DynamicStyle(env)) |
    | ChanceStyle | $(ChanceStyle(env)) |
    | InformationStyle | $(InformationStyle(env)) |
    | RewardStyle | $(RewardStyle(env)) |
    | UtilityStyle | $(UtilityStyle(env)) |
    """
    print(io, Markdown.parse(s))
end

#####
# General
#####

const DEFAULT_PLAYER = :DEFAULT_PLAYER

@api get_name(env::AbstractEnv) = typeof(env).name

@api (env::AbstractEnv)(action, player=get_current_player(env))

"Make an independent copy of `env`"
@api copy(env::AbstractEnv)
@api copyto!(dest::AbstractEnv, src::AbstractEnv)

"""
    get_actions(env, player=get_current_player(env))

Get all available actions from environment.
See also: [`get_legal_actions`](@ref)
"""
@env_api get_actions(env::AbstractEnv, player=get_current_player(env))

"""
    get_legal_actions(env, player=get_current_player(env))

Only valid for observations of [`FULL_ACTION_SET`](@ref).
"""
@env_api get_legal_actions(env::AbstractEnv, player=get_current_player(env))

"""
    get_legal_actions_mask(env) -> AbstractArray{Bool}

Required for environments of [`FULL_ACTION_SET`](@ref).
"""
@env_api get_legal_actions_mask(env::AbstractEnv, player=get_current_player(env))

"""
    get_state(env) -> state

The state can be of any type. Usually it's an `AbstractArray`.
See also https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/issues/48
If `state` is not an `AbstractArray`, to use it in neural network algorithms,
a `convert(AbstractArray, state)` should be provided.
To get the string representation, a `convert(String, state)` should also be provided.
"""
@env_api get_state(env::AbstractEnv, player=get_current_player(env)) = env.state

"""
    get_current_player(env)

Return the next player to take action.
For [Extensive Form Games](https://en.wikipedia.org/wiki/Extensive-form_game),
A *chance player* may be returned. (See also [`get_chance_player`](@ref))
For [SIMULTANEOUS](@ref) environments, a *simultaneous player* may be returned.
(See also [`get_simultaneouse_player`](@ref)).
"""
@env_api get_current_player(env::AbstractEnv) = DEFAULT_PLAYER

"""
    get_chance_player(env)

Only valid for environments with a chance player.
"""
@env_api get_chance_player(env::AbstractEnv)

"""
    get_simultaneouse_player(env)

Only valid for environments of [`SIMULTANEOUS`](@ref) style.
"""
@env_api get_simultaneouse_player(env)

"""
    get_spectator_player(env)

Used in imperfect multi-agent environments.
"""
@env_api get_spectator_player(env)

"""
    get_players(player, env::AbstractEnv) -> Int

Get the index of current player. Result should be an Int and starts from 1.
Usually used in multi-agent environments.
"""
@env_api get_players(player, env::AbstractEnv) = (DEFAULT_PLAYER,)

@env_api get_num_players(env::AbstractEnv) = get_num_players(NumAgentStyle(env))

get_num_players(::SingleAgent) = 1
get_num_players(::MultiAgent{N}) where N = N

"Reset the internal state of an environment"
@env_api reset!(env::AbstractEnv)

"Set the seed of internal rng"
@env_api seed!(env::AbstractEnv, seed)

"Get all actions in each ply"
@env_api get_history(env::AbstractEnv)

"""
    get_terminal(env)
"""
@env_api get_terminal(env) = env.terminal

"""
    get_reward(env)
"""
@env_api get_reward(env) = env.reward

"""
    get_prob(env, player)

Only valid for environments of [`EXPLICIT_STOCHASTIC`](@ref) style.
Here `player` must be a chance player.
"""
@env_api get_prob(env::AbstractEnv, player)

@env_api function child(env::AbstractEnv, action)
    new_env = copy(env)
    new_env(action)
    new_env
end

@env_api has_children(env::AbstractEnv) = !get_terminal(env)

@env_api children(env::AbstractEnv) = (child(env, action) for action in get_legal_actions(env))

function Base.show(io::IO, env::AbstractEnv)
    println(io, Markdown.parse("""
    # $(get_name(env))
    """))
    print_traits(io, env)
    println()
    println(io, Markdown.parse("""
    ## Current state
    $(get_state(env))
    """))
end

#####
## Space
#####

"""
Describe the span of observations and actions.
Usually the following methods are implemented:

- `Base.length`
- `Base.in`
- `Random.rand`
- `Base.eltype`
"""
@api abstract type AbstractSpace end

@api length(::AbstractSpace)
@api in(x, s::AbstractSpace)
@api rand(rng::AbstractRNG, s::AbstractSpace)
@api eltype(s::AbstractSpace)

#####
# Observation
#####

struct Observation{E<:AbstractEnv,P}
    env::E
    player::P
end

@env_api function observe(env::AbstractEnv, player=get_current_player(env))
    if NumAgentStyle(env) === SINGLE_AGENT
        env
    else
        Observation(env, player)
    end
end

#####
# EnvironmentModel
#####

"""
Describe how to model a reinforcement learning environment.
TODO: need more investigation
Ref: https://bair.berkeley.edu/blog/2019/12/12/mbpo/
- Analytic gradient computation
- Sampling-based planning
- Model-based data generation
- Value-equivalence prediction
"""
@api abstract type AbstractEnvironmentModel end
