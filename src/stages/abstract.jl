###################
# Stage protocol #
###################
# Stage supertypes, traits, the `@definestage` macro, and the sugar and dependency machinery.

"Pure stage configuration. Layout-free; one struct per stage class."
abstract type AbstractStageSpec end

"A composite's built state: its components, the layouts they span, the per-component construction data (`interiors`), and any tensors it fuses."
abstract type AbstractStageBuffer end

"The user-facing layer: a spec bundled with the buffers built between its two boundary layouts."
abstract type AbstractStage end

abstract type AbstractPrimitiveStage <: AbstractStage end
abstract type AbstractCompositeStage <: AbstractStage end

"Default buffer eltype. Override for specs that infer T from a parameter or array field."
default_eltype(::AbstractStageSpec) = Float64

"The axis a primitive stage's transition acts along, or `nothing` for a spec that acts on no axis."
function operative_axis end

"The `AbstractStageSpec` type a primitive stage bundles; `@definestage` emits the method."
function spec_type end

"""
How exact a spec's derivatives are at a `Dual` eltype: `:exact` for a smooth map, `:exact_ae` for
kinks on a measure-zero set, `:wrong_object` for a hard integer policy. A composite takes the worst.
"""
tangent_grade(::AbstractStageSpec) = :exact
tangent_grade(stage::AbstractStage) = tangent_grade(stage.spec)
_worst_grade(gs) = :wrong_object in gs ? :wrong_object : (:exact_ae in gs ? :exact_ae : :exact)

"Refuse at construction to rebuild a `:wrong_object` spec at a tangent-carrying eltype."
function _assert_tangent_seatable(spec, ::Type{T}) where {T}
    carries_tangents(T) && tangent_grade(spec) === :wrong_object && error(
        "$(typeof(spec)): a Dual-eltype rebuild would silently seat zero-tangent integer kernels. " *
        "Use LogitChoiceStage (ε > 0), the continuous sibling (ContinuousArgmaxStage / " *
        "DeterministicContinuousStage), or an explicit lottery.")
    return nothing
end

"Check the single-axis regrid invariant: the two boundary layouts agree on every axis but the spec's operative one."
function assert_regrid_invariant(spec, start_layout, end_layout)
    @assert axisnames(start_layout) == axisnames(end_layout) "$(typeof(spec)): the two layouts must " *
        "name the same axes in the same order; got $(axisnames(start_layout)) and $(axisnames(end_layout))."
    ax = operative_axis(spec)
    if ax === nothing
        @assert start_layout == end_layout "$(typeof(spec)) has no operative axis, so its two layouts must be equal."
    else
        @assert drop_axis(start_layout, ax) == drop_axis(end_layout, ax) "$(typeof(spec)) may regrid " *
            "only its operative axis `$ax`; the two layouts differ on some other axis."
    end
    return nothing
end

# Stage-keyed delegates #
#-----------------------#

"Composite sugar: run the components, threading `env_changed` down to each."
backward!(stage::AbstractCompositeStage, V_end, env; env_changed::Bool = true) =
    backward!(stage.buffer, stage.spec, V_end, env; env_changed)

"Composite sugar: push `Λ_start` through the components in order."
forward!(stage::AbstractCompositeStage, Λ_start::AbstractArray) =
    forward!(stage.buffer, stage.spec, Λ_start)

# Public accessors #
#------------------#

"The layout the stage's `V_start`/`Λ_start` live on."
start_layout(stage::AbstractPrimitiveStage) = stage.start_layout
start_layout(stage::AbstractCompositeStage) = start_layout(stage.buffer)

"The layout the stage's `V_end`/`Λ_end` live on."
end_layout(stage::AbstractPrimitiveStage) = stage.end_layout
end_layout(stage::AbstractCompositeStage) = end_layout(stage.buffer)

"One entry per component — the construction data it needs beyond the two layouts it spans."
interiors(stage::AbstractCompositeStage) = stage.buffer.interiors

"The stage's single layout, legal only when it does not regrid."
function layout(stage::AbstractStage)
    @assert start_layout(stage) == end_layout(stage) "$(typeof(stage)) regrids, so it has no single " *
        "layout; ask for `start_layout` or `end_layout`."
    return start_layout(stage)
end

"Read the layout-shaped output buffers."
V_start_buffer(stage::AbstractPrimitiveStage) = stage.scratch.V_start
Λ_end_buffer(stage::AbstractPrimitiveStage)   = stage.scratch.Λ_end

# Generic `bundle` fallback; `@definestage` emits the per-spec method.
function bundle(spec::AbstractStageSpec, ::AbstractLayout, ::AbstractLayout)
    error("bundle not implemented for $(typeof(spec))")
end

# Primitive stage protocol — `(spec, start_layout, end_layout)` dispatch #
#-----------------------------------------------------------------------#

"Stamp out a primitive stage's wrapper struct, its constructors, its `spec_type` trait, and its `bundle` methods."
macro definestage(stage_name, spec_name)
    spec_e     = esc(spec_name)
    stg_e      = esc(stage_name)
    bundle_e   = esc(:bundle)
    spectype_e = esc(:spec_type)
    return quote
        mutable struct $stg_e{Spec<:$spec_e, L1<:AbstractLayout, L2<:AbstractLayout, K, Sc, C} <: AbstractPrimitiveStage
            spec         :: Spec
            start_layout :: L1
            end_layout   :: L2
            kernel       :: K
            scratch      :: Sc
            cache        :: C

            function $stg_e{Spec, L1, L2, K, Sc, C}(spec, start_layout, end_layout,
                                                    kernel, scratch, cache) where {Spec<:$spec_e,
                                                    L1<:AbstractLayout, L2<:AbstractLayout, K, Sc, C}
                assert_regrid_invariant(spec, start_layout, end_layout)
                return new{Spec, L1, L2, K, Sc, C}(spec, start_layout, end_layout,
                                                   kernel, scratch, cache)
            end
        end
        function $stg_e(spec::$spec_e, start_layout::AbstractLayout, end_layout::AbstractLayout,
                        ::Type{T}=default_eltype(spec)) where {T}
            _assert_tangent_seatable(spec, T)
            kernel  = allocate_kernel(spec, T, start_layout, end_layout)
            scratch = merge(allocate_scratch(spec, T, start_layout, end_layout),
                            (kernel_scratch = kernel_scratch(kernel, start_layout, end_layout, T),))
            return $stg_e(spec, start_layout, end_layout, kernel, scratch,
                          allocate_cache(spec, T, start_layout, end_layout))
        end
        $spectype_e(::Type{<:$stg_e}) = $spec_e
        $bundle_e(spec::$spec_e, start_layout::AbstractLayout, end_layout::AbstractLayout) =
            $stg_e(spec, start_layout, end_layout)
        $bundle_e(spec::$spec_e, start_layout::AbstractLayout, end_layout::AbstractLayout, ::Type{T}) where {T} =
            $stg_e(spec, start_layout, end_layout, T)
        # The exported name is the bundled stage; carry the spec's docstring to it so `@doc <X>Stage`
        # resolves. The spec is documented above its own definition, before this macro is called.
        @doc (@doc $spec_e) $stg_e
    end
end

# Positional and kwarg sugar, defined once over the supertype.

"Six-field form that infers the type parameters and hands them to the inner constructor."
(::Type{S})(spec, start_layout, end_layout, kernel, scratch, cache) where {S<:AbstractPrimitiveStage} =
    S{typeof(spec), typeof(start_layout), typeof(end_layout), typeof(kernel), typeof(scratch),
      typeof(cache)}(spec, start_layout, end_layout, kernel, scratch, cache)

"Build a primitive stage between two layouts from its spec's keyword arguments."
(::Type{S})(start_layout::AbstractLayout, end_layout::AbstractLayout; kwargs...) where {S<:AbstractPrimitiveStage} =
    S(spec_type(S)(; kwargs...), start_layout, end_layout)

"Build a primitive stage that does not regrid: one layout is sugar for the equal pair."
(::Type{S})(layout::AbstractLayout; kwargs...) where {S<:AbstractPrimitiveStage} =
    S(layout, layout; kwargs...)

"""
Refuse to run a stage on a boundary field whose float precision disagrees with the stage's own — the
signature of feeding a `Float32` field into a block built at `Float64` (the kwarg sugar's default),
which would otherwise run the field silently at the stage's precision. Build the block at the intended
eltype through the explicit path (`Stage(spec, start, end, T)` / `bundle(spec, start, end, T)`).
Integer and other non-float boundaries are left to promote as before.
"""
@inline function _assert_boundary_eltype(stage, boundary)
    Sb = primal_eltype(eltype(V_start_buffer(stage)))
    Bd = primal_eltype(eltype(boundary))
    (Sb <: AbstractFloat && Bd <: AbstractFloat && Sb !== Bd) && error(
        "HouseholdStages: eltype mismatch — a $(eltype(boundary)) field was passed into a " *
        "$(eltype(V_start_buffer(stage)))-built $(nameof(typeof(stage))). These must agree. Build the " *
        "block at the intended precision through the explicit-eltype path — " *
        "`Stage(spec, start, end, $(Bd))` or `bundle(spec, start, end, $(Bd))` — not the kwarg sugar, " *
        "which defaults to Float64 and would run the mismatched field silently at the stage's precision.")
    return nothing
end

"Primitive stateful sugar: run the functional core with the stage's own `kernel`, `scratch` and `cache`, store the kernel it returns back on the stage, and return just `V_start`."
function backward!(stage::AbstractPrimitiveStage, V_end, env; env_changed::Bool = true)
    _assert_boundary_eltype(stage, V_end)
    _, kernel = backward!(stage.scratch.V_start, stage.spec, stage.start_layout, stage.end_layout, V_end;
                          env, kernel=stage.kernel,
                          scratch=stage.scratch, cache=stage.cache, env_changed)
    stage.kernel = kernel
    return stage.scratch.V_start
end

"Primitive stateful sugar: apply the kernel the last `backward!` built to `Λ_start`."
function forward!(stage::AbstractPrimitiveStage, Λ_start::AbstractArray)
    _assert_boundary_eltype(stage, Λ_start)
    return forward!(stage.scratch.Λ_end, stage.spec, stage.start_layout, stage.end_layout, Λ_start;
                    kernel=stage.kernel, scratch=stage.scratch)
end

"Default forward pass: apply the kernel the last `backward!` built."
forward!(Λ_end, ::AbstractStageSpec, ::GriddedLayout, ::GriddedLayout, Λ_start; kernel, scratch) =
    (forward!(Λ_end, kernel, Λ_start; scratch = scratch.kernel_scratch); Λ_end)

# Dependency machinery #
#----------------------#

"The `env` fields the Spec type itself reads. Concrete specs override."
static_env_deps(::Type{<:AbstractStageSpec}) = NamedTuple()

"Names of `env` fields read by this stage — static deps ∪ env-resolved spec fields."
function effective_env_slice(spec::AbstractStageSpec)
    static = keys(static_env_deps(typeof(spec)))
    return Tuple(unique((static..., _env_field_names(spec)...)))
end
effective_env_slice(stage::AbstractStage) = effective_env_slice(stage.spec)

"The deduplicated union of `effective_env_slice` over sub-specs."
_union_env_slices(specs) = Tuple(unique(Iterators.flatten(effective_env_slice(s) for s in specs)))

"Marker wrapping a Symbol that names an `env` field; a spec field holds either a literal value or a `FromEnv(:key)`."
struct FromEnv
    key :: Symbol
end

"Resolve a stage-parameter field: pass through if literal, look up in `env` if `FromEnv`."
resolve(val, env)            = val
resolve(fe::FromEnv, env)    = env[fe.key]
# A `FromEnv` marker survives a resolve with no env.
resolve(fe::FromEnv, ::Nothing) = fe

"Names of the `env` fields the spec currently reads, one per `FromEnv` marker it holds."
function _env_field_names(spec::AbstractStageSpec)
    fields = (getfield(spec, fn) for fn in fieldnames(typeof(spec)))
    return Tuple(f.key for f in fields if f isa FromEnv)
end

"Check that `env` provides every field in `effective_env_slice(spec)`."
function validate_env(spec::AbstractStageSpec, env)
    needed = effective_env_slice(spec)
    missing_keys = Symbol[k for k in needed if !haskey(env, k)]
    isempty(missing_keys) ||
        error("env is missing required fields: $missing_keys; provided: $(keys(env))")
    return nothing
end
validate_env(stage::AbstractStage, env) = validate_env(stage.spec, env)

"Prototype NamedTuple whose names are `effective_env_slice(spec)`; values `nothing`."
function env_schema(spec::AbstractStageSpec)
    names = effective_env_slice(spec)
    return NamedTuple{names}(ntuple(_ -> nothing, length(names)))
end
env_schema(stage::AbstractStage) = env_schema(stage.spec)

"Construct an env NamedTuple from kwargs, validated against `effective_env_slice(spec)`."
function make_env(spec::AbstractStageSpec; kwargs...)
    needed   = effective_env_slice(spec)
    provided = keys(kwargs)
    missing_keys = Symbol[k for k in needed if !(k in provided)]
    isempty(missing_keys) ||
        error("make_env: missing required env fields: $missing_keys; provided: $(collect(provided))")
    return NamedTuple(kwargs)
end
make_env(stage::AbstractStage; kwargs...) = make_env(stage.spec; kwargs...)

# Push a population through a stage: unwrap to the raw masses, apply `forward!`, rewrap.
forward!(stage::AbstractStage, Λ::GriddedPopulation) =
    GriddedPopulation(forward!(stage, masses(Λ)))
