mutable struct MultiArmBanditsEnv <: AbstractEnv
    true_reward::Float64
    true_values::Vector{Float64}
    rng::AbstractRNG
    # cache
    reward::Float64
    is_terminated::Bool
end

"""
    MultiArmBanditsEnv(;true_reward=0., k = 10,rng=Random.GLOBAL_RNG)

`true_reward` is the expected reward. `k` is the number of arms. See
[multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) for more
detailed explanation.

This is a **one-shot** game. The environment terminates immediately after taking
in an action. Here we use it to demonstrate how to write a customized
environment with only minimal interfaces defined.
"""
function MultiArmBanditsEnv(;true_reward=0., k = 10, rng=Random.GLOBAL_RNG)
    true_values = true_reward .+ randn(k)
    MultiArmBanditsEnv(
        true_reward,
        true_values,
        rng,
        0.,
        false
    )
end

"""
First we need to define the action space. In the [`MultiArmBanditsEnv`](@ref)
environment, the possible actions are `1` to `k` (which equals to
`length(env.true_values)`).

!!! note
    Although we decide to return an action space of `Base.OneTo`  here, it is
    not a hard requirement. You can return anything else (`Tuple`,
    `Distribution`, etc) that is more suitable to describe your problem and
    handle it correctly in the `you_env(action)` function. Some algorithms may
    require that the action space must be of `Base.OneTo`. However, it's the
    algorithm designer's job to do the checking and conversion.
"""
action_space(env::MultiArmBanditsEnv) = Base.OneTo(length(env.true_values))

"""
In our design, the return of taking an action in `env` is **undefined**. This is
the main difference compared to the interfaces defined in
[OpenAI/Gym](https://github.com/openai/gym). We find that the async manner is
more suitable to describe many complicated environments. However, one of the
inconveniences is that we have to cache some intermediate data for future queries.
"""
function (env::MultiArmBanditsEnv)(action)
    env.reward = randn(env.rng) + env.true_values[action]
    env.is_terminated = true
end

is_terminated(env::MultiArmBanditsEnv) = env.is_terminated

"""
Note that if the `env` is not started yet, the returned value is meaningless.
"""
reward(env::MultiArmBanditsEnv) = env.reward

"""
Since `MultiArmBanditsEnv` is just a one-shot game, it doesn't matter what the
state is after each action. So here we can simply set it to a constant `1`.
"""
state(env::MultiArmBanditsEnv) = 1

state_space(env::MultiArmBanditsEnv) = Base.OneTo(1)

function reset!(env::MultiArmBanditsEnv)
    env.is_terminated = false
end