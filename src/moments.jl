"""
One moment to emit from a chain: a per-cell `(cell; env)` `integrand`, mass-
weighted by Λ and collapsed by `reduce` (`sum`, `mean`, …). Construct via `at_end`.
"""
#TODO (phase2/P15) recast `integrand` as an axis/env kwarg-closure on the `closures.jl`
# backbone (a moment = ⟨closure, population⟩, dispatched per population rep) — see
# `notes/moments_as_closures.txt`. Deferred: it changes the user-facing moment API.
struct MomentSpec{F, R}
    integrand :: F
    reduce    :: R
end

"""
A moment anchored at the end of the chain. A `Symbol` `integrand` is sugar for
the cell-field accessor (`:wealth` → `(cell; env) -> cell.wealth`).
"""
at_end(; integrand, reduce) = MomentSpec(_wrap_integrand(integrand), reduce)

# Integrand may be a closure or a `Symbol` cell-field shortcut.
_wrap_integrand(f) = f
_wrap_integrand(s::Symbol) = (cell; env) -> getproperty(cell, s)

"""
Attach a moment named `name` to the chain (stored in a mutable dict on the Spec,
so the menu extends without rebuilding). Redefining errors unless
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

# Moment computation #
#--------------------#

"""
Evaluate every attached moment against `Λ` and `env`, returning a NamedTuple
keyed by moment name. Each scalar moment is `spec.reduce(integrand .* Λ)`. Errors
if no moments are attached.
"""
function compute_moments(spec::ChainStageSpec, layout::GriddedLayout, Λ, env)
    moments = spec.moments
    @assert !isempty(moments) "compute_moments: ChainStageSpec has no moments attached; call define_moment! first."
    out = Dict{Symbol, Any}()
    for (name, mspec) in moments
        out[name] = _eval_spec(mspec, layout, Λ, env)
    end
    return NamedTuple{Tuple(keys(out))}(Tuple(values(out)))
end

# Compat: spec-only call uses the spec's terminal output layout (chain walks components).
compute_moments(spec::ChainStageSpec, Λ, env) =
    error("compute_moments(spec, Λ, env): need a layout. Call `compute_moments(stage, Λ, env)` or `compute_moments(spec, layout, Λ, env)`.")

compute_moments(chain::ChainStage, Λ, env) =
    compute_moments(chain.spec, chain.buffer.output_layout, Λ, env)

function _eval_spec(spec::MomentSpec, layout::GriddedLayout, Λ, env)
    cells_arr = cell_array(layout)
    return spec.reduce(spec.integrand.(cells_arr; env) .* Λ)
end
