"""
    run(π, env::AbstractEnv)

Run the policy `π` in `env` until the end of an episode. This function is mainly
for test/verification only.
"""
function Base.run(π, env::AbstractEnv)
    while !is_terminated(env)
        action = π(env)
        env(action)
    end
end

#####
# printing
#####

function env_traits()
    [eval(x) for x in RLBase.ENV_API if endswith(String(x), "Style")]
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractEnv) = show(io, MIME"text/markdown"(), env)

function Base.show(io::IO, t::MIME"text/markdown", env::AbstractEnv)
    show(io, t, Markdown.parse("""
    # $(nameof(env))

    ## Traits
    | Trait Type | Value |
    |:---------- | ----- |
    $(join(["|$(string(f))|$(f(env))|" for f in env_traits()], "\n"))

    ## Action Space
    `$(action_space(env))`

    ## State Space
    `$(state_space(env))`

    """))

    if NumAgentStyle(env) !== SINGLE_AGENT
        show(io, t, Markdown.parse("""
            ## Players
            $(join(["- `$p`" for p in players(env)], "\n"))

            ## Current Player
            `$(current_player(env))`
            """))
    end

    show(io, t, Markdown.parse("""
        ## Is Environment Terminated?
        $(is_terminated(env) ? "Yes" : "No")

        ## Current State

        ```
        $(state(env))
        ```
        """))
end
