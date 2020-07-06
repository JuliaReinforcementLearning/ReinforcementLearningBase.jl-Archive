export RandomPolicy

using Random

"""
    RandomPolicy(action_space, rng)

Construct a random policy with actions in `action_space`.
"""
struct RandomPolicy{S,R<:AbstractRNG} <: AbstractPolicy
    action_space::S
    rng::R
end

Base.show(io::IO, p::RandomPolicy) = print(io, "RandomPolicy($(p.action_space))")

Random.seed!(p::RandomPolicy, seed) = Random.seed!(p.rng, seed)

"""
    RandomPolicy(;seed=nothing)

Randomly choose an action from `get_legal_actions(obs)` at runtime.
"""
RandomPolicy(; seed = nothing) = RandomPolicy(nothing;seed=seed)

"""
    RandomPolicy(action_space; seed=nothing)

Randomly select an action from `action_space` everytime.
"""
RandomPolicy(s; seed = nothing) = RandomPolicy(s, MersenneTwister(seed))

"""
    RandomPolicy(env::AbstractEnv; seed=nothing)

If `env` is of [`FULL_ACTION_SET`](@ref), then action is randomly chosen at runtime
in `get_actions(env)`. Otherwise, the `env` is supposed to be of [`MINIMAL_ACTION_SET`](@ref).
The `get_actions(env)` is supposed to be static and will only be used to initialize
the random policy for once.
"""
function RandomPolicy(env::AbstractEnv; seed = nothing)
    if ActionStyle(env) === MINIMAL_ACTION_SET
        RandomPolicy(get_actions(env); seed=seed)
    elseif ActionStyle(env) === FULL_ACTION_SET
        RandomPolicy(nothing; seed=seed)
    end
end

(p::RandomPolicy{Nothing})(obs) = rand(p.rng, get_legal_actions(obs))
(p::RandomPolicy)(obs) = rand(p.rng, p.action_space)
(p::RandomPolicy)(obs::BatchObs) = [p(x) for x in obs]

# TODO: TBD
# Ideally we should return a Categorical distribution.
# But this means we need to introduce an extra dependency of Distributions
# watch https://github.com/JuliaStats/Distributions.jl/issues/1139
get_prob(p::RandomPolicy, obs) = fill(1 / length(p.action_space), length(p.action_space))

function get_prob(p::RandomPolicy{Nothing}, obs)
    mask = get_legal_actions_mask(obs)
    n = sum(mask)
    prob = zeros(length(mask))
    prob[mask] .= 1 / n
    prob
end

get_prob(p::RandomPolicy, obs, a) = 1 / length(p.action_space)

function get_prob(p::RandomPolicy{Nothing}, obs, a)
    legal_actions = get_legal_actions(obs)
    if a in legal_actions
        1 / length(legal_actions)
    else
        0
    end
end
