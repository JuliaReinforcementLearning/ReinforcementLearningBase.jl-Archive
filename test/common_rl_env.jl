@testset "Test convert between CommonRLEnv and RLBase" begin

    x = LotteryEnv()
    y = convert(CRL.AbstractEnv, x) |> CRL.clone
    z = convert(AbstractEnv, y) |> copy

    @test ActionStyle(z) === ActionStyle(x)
    @test CRL.provided(CRL.valid_actions, y) == true
    @test CRL.provided(CRL.valid_action_mask, y) == true

    @test get_state(x) == CRL.observe(y) == get_state(z)
    @test get_actions(x) == CRL.actions(y) == get_actions(z)
    @test get_legal_action_space(x) == CRL.valid_actions(y) == get_legal_action_space(z)
    @test get_legal_action_space_mask(x) == CRL.valid_action_mask(y) == get_legal_action_space_mask(z)
    @test get_current_player(x) == CRL.player(y) == get_current_player(z)
    @test is_terminated(x) == CRL.terminated(y) == is_terminated(z)

    a = rand(get_actions(x))
    x(a)
    r = CRL.act!(y, a)
    z(a)

    # @test get_reward(x) == r == get_reward(z)

    @test get_state(x) == CRL.observe(y) == get_state(z)
    @test get_actions(x) == CRL.actions(y) == get_actions(z)
    @test get_legal_action_space(x) == CRL.valid_actions(y) == get_legal_action_space(z)
    @test get_legal_action_space_mask(x) == CRL.valid_action_mask(y) == get_legal_action_space_mask(z)
    @test get_current_player(x) == CRL.player(y) == get_current_player(z)
    @test is_terminated(x) == CRL.terminated(y) == is_terminated(z)

    reset!(x)
    CRL.reset!(y)
    reset!(z)

    @test get_state(x) == CRL.observe(y) == get_state(z)
    @test get_actions(x) == CRL.actions(y) == get_actions(z)
    @test get_legal_action_space(x) == CRL.valid_actions(y) == get_legal_action_space(z)
    @test get_legal_action_space_mask(x) == CRL.valid_action_mask(y) == get_legal_action_space_mask(z)
    @test get_current_player(x) == CRL.player(y) == get_current_player(z)
    @test is_terminated(x) == CRL.terminated(y) == is_terminated(z)

end
