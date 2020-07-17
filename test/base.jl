@testset "base" begin
    env = LotteryEnv(; seed = 222)
    policy = RandomPolicy(env; seed = 123)
    reset!(env)
    run(policy, env)
    @test get_terminal(env)
end
