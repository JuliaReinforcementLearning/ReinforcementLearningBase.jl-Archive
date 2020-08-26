export TabularRandomPolicy

using Random

"""
    TabularRandomPolicy(;table=Dict(), rng=Random.GLOBAL_RNG)

Use a table to store action probabilities.
"""
struct TabularRandomPolicy{S,T,R<:AbstractRNG} <: AbstractPolicy
    table::Dict{S, Vector{T}}
    rng::R
end

Random.seed!(p::TabularRandomPolicy, seed) = Random.seed!(p.rng, seed)

TabularRandomPolicy(;rng=Random.GLOBAL_RNG) = TabularRandomPolicy{Int, Float32}(;rng=rng)
TabularRandomPolicy{S}(;rng=Random.GLOBAL_RNG) where {S} = TabularRandomPolicy{S, Float32}(;rng=rng)
TabularRandomPolicy{S, T}(;rng=Random.GLOBAL_RNG) where {S,T} = TabularRandomPolicy(Dict{S,Vector{T}}(), rng)

(p::TabularRandomPolicy)(env::AbstractEnv) = weighted_sample(p.rng, get_prob(p, env))

function get_prob(p::TabularRandomPolicy, env::AbstractEnv)
    s = get_state(env)
    if haskey(p.table, s)
        p.table[s]
    elseif ActionStyle(env) === FULL_ACTION_SET
        mask = get_legal_actions_mask(env)
        prob = mask ./ sum(mask)
        p.table[s] = prob
        prob
    elseif ActionStyle(env) === MINIMAL_ACTION_SET
        n = length(get_actions(env))
        prob = fill(1 / n, n)
        p.table[s] = prob
        prob
    end
end

update!(p::TabularRandomPolicy, experience::Pair{Int, <:AbstractVector}) = p.table[first(experience)] = last(experience)
