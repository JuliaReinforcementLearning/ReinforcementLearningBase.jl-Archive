@testset "base" begin
    env = LotteryEnv(;seed=222)
    action_space = get_action_space(env)
    policy = RandomPolicy(env;seed=123)
    rewards = []
    for _ in 1:1000
        reset!(env)
        run(policy, env)
        push!(rewards, get_reward(observe(env)))
    end
end