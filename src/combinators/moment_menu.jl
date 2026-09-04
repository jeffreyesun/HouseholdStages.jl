# Moments attached to a chain #
#=============================#
# Named moments live in a mutable dict on the chain's spec.

"Attach a moment named `name` to the chain, in place."
function define_moment!(chain::ChainStage, name::Symbol, spec::MomentSpec;
                        overwrite_existing_moment_definitions::Bool = false)
    m = chain.spec.moments
    if haskey(m, name) && !overwrite_existing_moment_definitions
        error("define_moment!: moment :$name is already defined on this chain. " *
              "Pass overwrite_existing_moment_definitions = true to overwrite, " *
              "or rebuild the chain.")
    end
    m[name] = spec
    return chain
end

"Batch `define_moment!`: each kwarg is `name = MomentSpec(...)`."
function define_moments!(chain::ChainStage;
                         overwrite_existing_moment_definitions::Bool = false,
                         kwargs...)
    for (name, spec) in kwargs
        define_moment!(chain, name, spec;
                       overwrite_existing_moment_definitions)
    end
    return chain
end

# A single stage is promoted to a singleton chain to receive moments.
define_moment!(stage::AbstractStage, name::Symbol, spec::MomentSpec; kwargs...) =
    define_moment!(ChainStage((stage,)), name, spec; kwargs...)
define_moments!(stage::AbstractStage; kwargs...) =
    define_moments!(ChainStage((stage,)); kwargs...)

"Evaluate every moment attached to the chain against `Λ` and `env`, keyed by moment name."
function compute_moments(spec::ChainStageSpec, end_layout::GriddedLayout, Λ, env)
    moments = spec.moments
    @assert !isempty(moments) "compute_moments: ChainStageSpec has no moments attached; call define_moment! first."
    out = Dict{Symbol, Any}()
    for (name, mspec) in moments
        out[name] = compute_moment(end_layout, mspec, Λ, env)
    end
    return NamedTuple{Tuple(keys(out))}(Tuple(values(out)))
end

compute_moments(spec::ChainStageSpec, Λ, env) =
    error("compute_moments(spec, Λ, env): a moment integrates against Λ, so it needs the END layout. Call `compute_moments(stage, Λ, env)` or `compute_moments(spec, end_layout, Λ, env)`.")

compute_moments(chain::ChainStage, Λ, env) =
    compute_moments(chain.spec, end_layout(chain), Λ, env)
