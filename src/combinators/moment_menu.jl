# Moment menu — the combinator-layer attachment over a ChainStage (end-goal §7) #
#==============================================================================#
# The chain-attachment convenience for moments: it stores a moment menu on a `ChainStage`'s spec and
# fans out to the PURE `compute_moment` (moments/moments.jl). This is the only part that knows about
# `ChainStage`/`AbstractStage`; the computation half does not — which is why it lives here, in the
# combinators layer, loaded after the stage algebra.

"""
Attach a moment named `name` to the chain (stored in a mutable dict on the Spec, so the menu
extends without rebuilding). Redefining errors unless
`overwrite_existing_moment_definitions = true`.
"""
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

# A single stage can be promoted to a singleton chain to receive moments.
define_moment!(stage::AbstractStage, name::Symbol, spec::MomentSpec; kwargs...) =
    define_moment!(ChainStage((stage,)), name, spec; kwargs...)
define_moments!(stage::AbstractStage; kwargs...) =
    define_moments!(ChainStage((stage,)); kwargs...)

"""
Evaluate every attached moment against `Λ` and `env`, returning a NamedTuple keyed by moment
name. Each scalar moment is `compute_moment(layout, spec, Λ, env)`. Errors if no moments are
attached.
"""
function compute_moments(spec::ChainStageSpec, layout::GriddedLayout, Λ, env)
    moments = spec.moments
    @assert !isempty(moments) "compute_moments: ChainStageSpec has no moments attached; call define_moment! first."
    out = Dict{Symbol, Any}()
    for (name, mspec) in moments
        out[name] = compute_moment(layout, mspec, Λ, env)
    end
    return NamedTuple{Tuple(keys(out))}(Tuple(values(out)))
end

# Compat: spec-only call uses the spec's terminal output layout (chain walks components).
compute_moments(spec::ChainStageSpec, Λ, env) =
    error("compute_moments(spec, Λ, env): need a layout. Call `compute_moments(stage, Λ, env)` or `compute_moments(spec, layout, Λ, env)`.")

compute_moments(chain::ChainStage, Λ, env) =
    compute_moments(chain.spec, chain.buffer.output_layout, Λ, env)
