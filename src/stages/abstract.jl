###################
# Stage protocol #
###################
#
# A `Spec` is pure, immutable, layout-free stage configuration. A modern stage
# (`@definestage`) bundles a spec, a layout, and the inferred kernel/scratch/cache;
# it implements the functional core `backward!(spec, layout, V_end; …)` and
# `forward!(spec, layout, Λ; …)`, which the stateful sugar below wraps. The legacy
# `(buffer, spec)` half remains only for the `ChainStage`/`ProductStage` combinators.

"Pure stage configuration. Layout-free; one struct per stage class."
abstract type AbstractStageSpec end

"Per-call buffer: kernel + scratch + V/Λ outputs + layouts + cache."
abstract type AbstractStageBuffer end

"Bundle of one Spec and one Buffer; the user-facing layer."
abstract type AbstractStage end

# Two supertypes split the protocol. `AbstractLegacyStage` — the `(buffer, spec)`
# bundled stages (CacheState + the legacy `backward!`/`forward!` sugar); now only the
# `ChainStage`/`ProductStage` combinators. `AbstractModernStage` — the
# `(spec, layout)`-dispatch stages (stage = spec/layout/kernel/scratch/cache, the
# functional-core/stateful-shell `backward!`/`forward!`); all production primitives.
# Both stay `<: AbstractStage` so representation-agnostic dep machinery (env slices,
# validation) serves both; sugar + buffer accessors dispatch per supertype.
abstract type AbstractLegacyStage <: AbstractStage end
abstract type AbstractModernStage <: AbstractStage end

# Cache fingerprint #
#-------------------#

"""
Records the `(V_end, env)` a `backward!` last consumed, so a later
`forward!` can check its kernel is still valid for the pair it's given.
"""
mutable struct CacheState
    last_V_hash  :: UInt
    last_env     :: Any
    kernel_valid :: Bool
end
CacheState() = CacheState(zero(UInt), nothing, false)

"""
Cheap, GPU-safe staleness fingerprint of `V_end`. Host `Array`s use the full
content `hash`; device arrays fingerprint via on-device reductions `(size, sum,
sum-of-squares)`, since `hash` scalar-indexes a `CuArray` and errors.
"""
_v_fingerprint(V_end::Array) = hash(V_end)
function _v_fingerprint(V_end::AbstractArray)
    s  = sum(V_end)
    s2 = sum(abs2, V_end)
    return hash((size(V_end), length(V_end), s, s2))
end
_v_fingerprint(V_end) = hash(V_end)   # scalars, NamedTuples, …

"""
Stamp the buffer's cache with the `(V_end, env)` `backward!` just
consumed. Call at the end of every concrete `backward!`.
"""
function _seat_cache!(buffer::AbstractStageBuffer, V_end, env)
    c = buffer.cache
    c.last_V_hash  = _v_fingerprint(V_end)
    c.last_env     = env
    c.kernel_valid = true
    return buffer
end

"Mark the cache stale; the next cache-checking `forward!` will refuse to run trusted."
function invalidate!(buffer::AbstractStageBuffer)
    buffer.cache.kernel_valid = false
    return buffer
end
invalidate!(stage::AbstractLegacyStage) = (invalidate!(stage.buffer); stage)

# Allocation protocol #
#---------------------#
# The layout-keyed allocators (`allocate_kernel`/`allocate_scratch`/`allocate_cache`,
# `allocate_buffer`/`io_scratch`, the `input_layout`/`output_layout` defaults) live in
# the layout files (`layouts/gridded_layout.jl`) — allocation is representation-specific.
# A stage overrides the one(s) it needs there. Only the eltype default is a pure stage
# concern:

"Default buffer eltype. Override for specs that infer T from a parameter or array field."
default_eltype(::AbstractStageSpec) = Float64

# Stage-keyed delegates — buffer-first dispatch on the spec-keyed methods #
#------------------------------------------------------------------------#

"`backward!(stage, V_end, env) -> V_start`. Legacy bundled-stage delegate."
backward!(stage::AbstractLegacyStage, V_end, env) =
    backward!(stage.buffer, stage.spec, V_end, env)

"`forward!(stage, Λ_start) -> Λ_end`. Legacy bundled-stage delegate. (A wrapped
`AbstractPopulation` routes to the population seam in population.jl instead.)"
forward!(stage::AbstractLegacyStage, Λ_start::AbstractArray) =
    forward!(stage.buffer, stage.spec, Λ_start)

"""
Cache-checking `forward!`: refuses to run unless the cache agrees with `(V_end, env)`.
`reseat_if_stale` re-runs `backward!` instead of erroring; `check=false` skips the check.
"""
function forward!(stage::AbstractLegacyStage, Λ_start, V_end, env;
                  reseat_if_stale::Bool=false, check::Bool=true)
    if check
        c = stage.buffer.cache
        fresh = c.kernel_valid && _v_fingerprint(V_end) == c.last_V_hash &&
                isequal(env, c.last_env)
        if !fresh
            reseat_if_stale || error("forward!: cached kernel is stale for this (V_end, env). " *
                                     "Pass reseat_if_stale=true or call backward! first.")
            backward!(stage, V_end, env)
        end
    end
    return forward!(stage, Λ_start)
end

# Public accessors #
#------------------#

"`input_layout(stage)` — the layout the stage's buffer was allocated against."
input_layout(stage::AbstractLegacyStage)  = stage.buffer.input_layout
output_layout(stage::AbstractLegacyStage) = stage.buffer.output_layout

"Read the layout-shaped output buffers."
V_start_buffer(stage::AbstractLegacyStage) = stage.buffer.V_start
Λ_end_buffer(stage::AbstractLegacyStage)   = stage.buffer.Λ_end

# `bundle(spec, layout)` — generic fallback raises until @definestage emits the method.
function bundle(spec::AbstractStageSpec, ::GriddedLayout)
    error("bundle not implemented for $(typeof(spec))")
end

# Modern stage protocol — `(spec, layout)` dispatch (phase 2) #
#-----------------------------------------------------------#
# A modern stage is `(spec, layout, kernel, scratch, cache)`. The kernel IS the
# transition's data (the linear operator — a dense self-describing array or a structured
# kernel struct); `cache` is persistent (spec,layout)-level build storage; `scratch` is
# transient compute storage (incl. the resolved gather perm/work-buffers) that ALSO holds the
# layout-shaped output buffers (`V_start`, `Λ_end`) the core writes into and returns —
# reused across VFI/Λ iterations, not reallocated (PHASE2_PLAN §6, §11). Each stage
# implements the functional core `backward!(spec, layout, V_end; …) -> (V_start, kernel)`
# and `forward!(spec, layout, Λ; …) -> Λ_end`; the stateful sugar below wraps them.


"""
Stamp out a modern stage's wrapper struct, constructors, and `bundle`. The struct is
mutable so the stateful sugar can store a freshly-returned kernel (a no-op for the
in-place stages, where the returned kernel IS `stage.kernel`). The three allocators
run at construction; their return types are inferred, not declared.

```julia
@definestage IdentityStage IdentityStageSpec
@definestage MarkovStage   MarkovStageSpec
```
"""
macro definestage(stage_name, spec_name)
    spec_e   = esc(spec_name)
    stg_e    = esc(stage_name)
    bundle_e = esc(:bundle)
    return quote
        mutable struct $stg_e{Spec<:$spec_e, Layout<:AbstractLayout, K, Sc, C} <: AbstractModernStage
            spec    :: Spec
            layout  :: Layout
            kernel  :: K
            scratch :: Sc
            cache   :: C
        end
        function $stg_e(spec::$spec_e, layout::AbstractLayout,
                        ::Type{T}=default_eltype(spec)) where {T}
            kernel  = allocate_kernel(spec, T, layout)
            # The kernel's own gather scratch is merged in here, so a stage with a dense kernel
            # needs no `allocate_scratch` of its own (the default `io_scratch` suffices). Kernels
            # with no gather (single-destination/identity/discount) contribute `kernel_scratch = ()`.
            scratch = merge(allocate_scratch(spec, T, layout),
                            (kernel_scratch = kernel_scratch(kernel, layout, T),))
            return $stg_e(spec, layout, kernel, scratch, allocate_cache(spec, T, layout))
        end
        $stg_e(layout::AbstractLayout; kwargs...) =
            $stg_e($spec_e(; kwargs...), layout)
        $bundle_e(spec::$spec_e, layout::AbstractLayout) = $stg_e(spec, layout)
        $bundle_e(spec::$spec_e, layout::AbstractLayout, ::Type{T}) where {T} =
            $stg_e(spec, layout, T)
    end
end

"""
Modern stateful sugar: run the functional core with the stage's `kernel/scratch/cache`,
store the returned kernel back on the stage, and return just `V_start`. Call signature
is unchanged from the legacy sugar — zero call-site churn.
"""
function backward!(stage::AbstractModernStage, V_end, env)
    _, kernel = backward!(stage.scratch.V_start, stage.spec, stage.layout, V_end;
                          env, kernel=stage.kernel,
                          scratch=stage.scratch, cache=stage.cache)
    stage.kernel = kernel
    return stage.scratch.V_start
end

"Modern stateful sugar: apply the seated kernel (the linear operator) to `Λ_start`.
(A wrapped `AbstractPopulation` routes to the population seam in population.jl instead.)"
forward!(stage::AbstractModernStage, Λ_start::AbstractArray) =
    forward!(stage.scratch.Λ_end, stage.spec, stage.layout, Λ_start;
             kernel=stage.kernel, scratch=stage.scratch)

"""
Default modern primal forward: the forward pass does no seating, so it is always just the
seated kernel's forward verb `K·` with the kernel's plan. A stage whose forward is exactly one
kernel application needs no `forward!` of its own; only a bespoke push (e.g. `SearchMatchingStage`)
overrides on its spec type.
"""
forward!(Λ_end, ::AbstractStageSpec, ::GriddedLayout, Λ_start; kernel, scratch) =
    (forward!(Λ_end, kernel, Λ_start; scratch = scratch.kernel_scratch); Λ_end)

# Bridge accessors — let the (still-legacy) outer loop drive a modern stage during
# migration. Dropped in P18 when the outer loop threads V_start/Λ_end explicitly.
input_layout(stage::AbstractModernStage)  = stage.layout
output_layout(stage::AbstractModernStage) = output_layout(stage.spec, stage.layout)
V_start_buffer(stage::AbstractModernStage) = stage.scratch.V_start
Λ_end_buffer(stage::AbstractModernStage)   = stage.scratch.Λ_end

# No staleness cache in the modern model (freshness is the backward-then-forward
# convention) — `invalidate!` is a no-op so a legacy chain can walk mixed components.
invalidate!(stage::AbstractModernStage) = stage

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

"""
Marker wrapping a Symbol that names an `env` field. Spec fields hold
either a literal value or a `FromEnv(:key)`; `resolve` dispatches on
the value's type.
"""
struct FromEnv
    key :: Symbol
end

"Resolve a stage-parameter field: pass through if literal, look up in `env` if `FromEnv`."
resolve(val, env)            = val
resolve(fe::FromEnv, env)    = env[fe.key]
# Pass-through when env is absent: a FromEnv marker survives a resolve with no env.
resolve(fe::FromEnv, ::Nothing) = fe

"Names of `env` fields the spec currently reads — the `FromEnv` markers held in any field."
function _env_field_names(spec::AbstractStageSpec)
    syms = Symbol[]
    for fn in fieldnames(typeof(spec))
        v = getfield(spec, fn)
        v isa FromEnv && push!(syms, v.key)
    end
    return Tuple(syms)
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

# Numerical helpers #
#-------------------#

"""
In-place numerically-stable softmax of `U / ε` along its trailing
axis. Overwrites `U` with the choice probabilities and writes the
per-row `ε · log Σ exp(U[i,:]/ε)` into `lse` (shaped like `U`'s
leading axes). At least one entry per row must be finite.
"""
function _softmax_and_lse_along_last!(lse::AbstractArray{T},
                                      U::AbstractArray, ε::Real) where {T}
    Uflat   = reshape(U, :, last(size(U)))
    lseflat = vec(lse)
    for (s, u_row) in enumerate(eachrow(Uflat))
        m = maximum(u_row)
        @assert isfinite(m)
        @. u_row = exp((u_row - m) / ε)
        denom = sum(u_row)
        u_row ./= denom
        lseflat[s] = m + ε * log(denom)
    end
    return lse
end
