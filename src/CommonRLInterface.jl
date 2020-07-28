import CommonRLInterface

const CRL = CommonRLInterface

#####
# CommonRLEnv
#####

struct CommonRLEnv{T<:AbstractEnv} <: CRL.AbstractEnv
    env::T
end

Base.convert(::Type{CRL.AbstractEnv}, env::AbstractEnv) = convert(CommonRLEnv, env)

function Base.convert(::Type{CommonRLEnv}, env::AbstractEnv)
    new_env = CommonRLEnv(env)
    if ActionStyle(env) === FullActionSet()
        @eval CRL.@provide CRL.valid_actions(x::typeof($new_env)) = get_legal_actions(x.env)
        @eval CRL.@provide CRL.valid_action_mask(env::typeof($new_env)) = get_legal_actions_mask(env.env)
    end
    new_env
end

CRL.@provide CRL.reset!(env::CommonRLEnv) = reset!(env.env)
CRL.@provide CRL.actions(env::CommonRLEnv) = get_actions(env.env)
CRL.@provide CRL.observe(env::CommonRLEnv) = get_state(env.env)
CRL.@provide CRL.state(env::CommonRLEnv) = get_state(env.env)
CRL.@provide CRL.terminated(env::CommonRLEnv) = get_terminal(env.env)
CRL.@provide CRL.player(env::CommonRLEnv) = get_current_player(env.env)

CRL.@provide function CRL.act!(env::CommonRLEnv, a)
    env.env(a)
    get_reward(env.env)
end

CRL.clone(env::CommonRLEnv) = CommonRLEnv(copy(env.env))

#####
# RLBaseEnv
#####

mutable struct RLBaseEnv{T<:CRL.AbstractEnv, R} <: AbstractEnv
    env::T
    r::R
end

Base.convert(::Type{AbstractEnv}, env::CRL.AbstractEnv) = convert(RLBaseEnv, env)
Base.convert(::Type{RLBaseEnv}, env::CRL.AbstractEnv) = RLBaseEnv(env, 0.f0)  # can not determine reward ahead. Assume `Float32`.

get_state(env::RLBaseEnv) = CRL.observe(env.env)
get_actions(env::RLBaseEnv) = CRL.actions(env.env)
get_reward(env::RLBaseEnv) = env.r
get_terminal(env::RLBaseEnv) = CRL.terminated(env.env)
get_legal_actions(env::RLBaseEnv) = CRL.valid_actions(env.env)
get_legal_actions_mask(env::RLBaseEnv) = CRL.valid_action_mask(env.env)
reset!(env::RLBaseEnv) = CRL.reset!(env.env)

(env::RLBaseEnv)(a) = env.r = CRL.act!(env.env, a)
Base.copy(env::CommonRLEnv) = RLBaseEnv(CRL.clone(env.env), env.r)

ActionStyle(env::RLBaseEnv) = CRL.provided(CRL.valid_actions, env.env) ? FullActionSet() : MinimalActionSet()