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

Base.show(io::IO, ::MIME"text/plain", env::AbstractEnv) = print(io, nameof(env))

function Base.show(io::IO, t::MIME"text/markdown", env::AbstractEnv)
    show(io, t, Markdown.parse("""
    # $(name(env))

    ## Traits
    | Trait Type | Value |
    |:---------- | ----- |
    $(join(["|$(string(f))|$(f(env))|" for f in env_traits()], "\n"))

    ## Actions
    $(actions(env))

    ## Players
    $(join(["- `$p`" for p in players(env)], "\n"))

    ## Current Player
    `$(current_player(env))`

    ## Is Environment Terminated?
    $(terminal(env) ? "Yes" : "No")
    """))
end
