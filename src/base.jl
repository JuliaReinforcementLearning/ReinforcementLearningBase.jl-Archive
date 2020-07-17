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
