"""
    run(π, env::AbstractEnv)

Run the policy `π` in `env` until the end.
"""
Base.run(π, env::AbstractEnv) = run(π, env, DynamicStyle(env), NumAgentStyle(env))

function Base.run(π, env::AbstractEnv, ::Sequential, ::SingleAgent)
    while !get_terminal(env)
        action = π(env)
        env(action)
    end
end

function Base.run(Π, env::AbstractEnv, ::Sequential, ::MultiAgent)
    is_terminal = false
    while !is_terminal
        for π in Π
            if get_terminal(env)
                is_terminal = true
                break
            end
            action = π(env)
            env(action)
        end
    end
end

#####
# printing
#####

function Base.convert(::Type{Markdown.MD}, env::AbstractEnv)
    Markdown.parse("""
    # $(get_name(env))
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
    ## Current state
    $(get_state(env))
    """)
end

Base.show(io::IO, env::AbstractEnv) = show(io, convert(Markdown.MD, env))
Base.show(io::IO, t::MIME"text/plain", env::AbstractEnv) = show(io, t, convert(Markdown.MD, env))

Base.summary(io::IO, env::AbstractEnv) = print(io, "$(get_name(env)): $(NumAgentStyle(env)) $(ActionStyle(env)) $(DynamicStyle(env)) $(ChanceStyle(env)) $(InformationStyle(env)) $(RewardStyle(env)) $(UtilityStyle(env))")
