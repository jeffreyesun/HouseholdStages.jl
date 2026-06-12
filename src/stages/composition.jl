"""
Composition of stages. Pure data: a flat tuple of component Specs
and a mutable moments dict. Layout flows in at allocate time and
chains through the components.

The `moments` slot is the only mutable field on any Spec —
`define_moment!(chain, name, spec)` extends it.
"""
struct ChainStageSpec{Stages<:Tuple} <: AbstractStageSpec
    stages  :: Stages
    moments :: Dict{Symbol, Any}
end

function ChainStageSpec(stages::Tuple;
                        moments::Dict{Symbol, Any}=Dict{Symbol, Any}())
    @assert !isempty(stages)
    flat = _flatten_chain_specs(stages)
    return ChainStageSpec{typeof(flat)}(flat, moments)
end

"""
Flatten any nested `ChainStageSpec` components in a tuple. Single-level
walk; relies on `ChainStageSpec` already being flat.
"""
function _flatten_chain_specs(stages::Tuple)
    out = AbstractStageSpec[]
    for s in stages
        @assert s isa AbstractStageSpec
        s isa ChainStageSpec ? append!(out, s.stages) : push!(out, s)
    end
    return Tuple(out)
end

"""
Chain buffer: a tuple of bundled per-component sub-STAGES (each legacy or modern),
driven through the per-stage sugar so the chain is agnostic to which protocol a
component uses (phase-2 coexistence). The chain's `input_layout` mirrors the first
component, `output_layout` the last.
"""
struct ChainStageBuffer{Stages<:Tuple, LIn, LOut} <: AbstractStageBuffer
    stages        :: Stages
    input_layout  :: LIn
    output_layout :: LOut
    cache         :: CacheState
end

"Chain stage: a tuple of bundled sub-stages composed via `∘`."
struct ChainStage{Spec<:ChainStageSpec, Buffer<:ChainStageBuffer} <: AbstractLegacyStage
    spec   :: Spec
    buffer :: Buffer
end

function ChainStage(stages::Tuple)
    @assert !isempty(stages)
    if all(s -> s isa AbstractStage, stages)
        sub_stages = _flatten_chain_stages(stages)
        spec = ChainStageSpec(map(s -> s.spec, sub_stages))
        buf  = ChainStageBuffer(sub_stages,
                                input_layout(sub_stages[1]),
                                output_layout(sub_stages[end]),
                                CacheState())
        return ChainStage{typeof(spec), typeof(buf)}(spec, buf)
    elseif all(s -> s isa AbstractStageSpec, stages)
        return ChainStage(ChainStageSpec(stages))
    else
        error("ChainStage components must be uniformly AbstractStage or AbstractStageSpec")
    end
end

"""
Flatten a tuple of bundled stages into one flat tuple of leaf sub-stages, unpacking
any nested `ChainStage` (whose `buffer.stages` are themselves leaf sub-stages).
Preserves the original leaf stages — no reallocation.
"""
function _flatten_chain_stages(stages::Tuple)
    out = AbstractStage[]
    for s in stages
        s isa ChainStage ? append!(out, s.buffer.stages) : push!(out, s)
    end
    return Tuple(out)
end

ChainStage(spec::ChainStageSpec) = error("ChainStage(spec) needs a layout; call ChainStage(spec, layout) or compose bundled stages with `∘`.")
ChainStage(spec::ChainStageSpec, layout::GriddedLayout) =
    ChainStage(spec, allocate(spec, layout))

bundle(spec::ChainStageSpec, layout::GriddedLayout) = ChainStage(spec, layout)
bundle(spec::ChainStageSpec, layout::GriddedLayout, ::Type{T}) where {T} =
    ChainStage(spec, allocate(spec, layout, T))

# Allocate — bundle components, chaining layouts #
#------------------------------------------------#

function allocate(spec::ChainStageSpec, layout::GriddedLayout,
                  ::Type{T}=Float64) where {T}
    sub_stages, out_layout = _allocate_chain_stages(spec.stages, layout, T)
    return ChainStageBuffer(sub_stages, layout, out_layout, CacheState())
end

"""
Sequentially bundle component stages, threading each component's `output_layout`
into the next component's input (decision 7: a component may change levels, e.g.
ForgetfulSum). Returns the tuple of sub-stages and the chain's terminal output layout.
"""
function _allocate_chain_stages(specs::Tuple, layout::GriddedLayout, T::Type)
    cur    = layout
    stages = AbstractStage[]
    for s in specs
        stg = bundle(s, cur, T)
        push!(stages, stg)
        cur = output_layout(stg)
    end
    return Tuple(stages), cur
end

# Env slice — union over components #
#-----------------------------------#

"The set of `env` fields the chain's backward/forward needs (union over components)."
function chain_env_names(spec::ChainStageSpec)
    names = Symbol[]
    for s in spec.stages
        for k in effective_env_slice(s)
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end
chain_env_names(chain::ChainStage) = chain_env_names(chain.spec)

effective_env_slice(spec::ChainStageSpec) = chain_env_names(spec)

function validate_env(spec::ChainStageSpec, env)
    needed = chain_env_names(spec)
    missing_keys = Symbol[k for k in needed if !haskey(env, k)]
    isempty(missing_keys) ||
        error("env is missing required fields: $missing_keys; provided: $(keys(env))")
    return nothing
end

# Endpoint accessors #
#--------------------#

V_start_buffer(stage::ChainStage) = V_start_buffer(stage.buffer.stages[1])
Λ_end_buffer(stage::ChainStage)   = Λ_end_buffer(stage.buffer.stages[end])

# Backward sweep — type-stable via @generated #
#--------------------------------------------#
# Heterogeneous component tuple — a runtime `n:-1:1` loop on the tuple is
# type-unstable. `@generated` unrolls. Each component is driven through the
# per-stage sugar `backward!(stage, V, env)`, uniform across legacy/modern.

@generated function backward!(buffer::ChainStageBuffer{Stages},
                              spec::ChainStageSpec, V_end, env) where {Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(V = backward!(buffer.stages[$i], V, env)) for i in N:-1:1]
    return quote
        V = V_end
        $(calls...)
        _seat_cache!(buffer, V_end, env)
        return V
    end
end

# Forward sweep #
#---------------#

@generated function forward!(buffer::ChainStageBuffer{Stages},
                             spec::ChainStageSpec, Λ_start) where {Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(Λ = forward!(buffer.stages[$i], Λ)) for i in 1:N]
    return quote
        Λ = Λ_start
        $(calls...)
        return Λ
    end
end

# Cache invalidation walks components #
#-------------------------------------#

function invalidate!(buffer::ChainStageBuffer)
    buffer.cache.kernel_valid = false
    for s in buffer.stages
        invalidate!(s)
    end
    return buffer
end

# Composition operators #
#-----------------------#

"""
Left-to-right (time-ordered) stage composition: `s1` runs first — the **opposite**
of Julia's `∘` on functions. Auto-flattens nested chains; refuses to compose
chains that already carry moments.
"""
Base.:∘(a::AbstractStageSpec, b::AbstractStageSpec) =
    ChainStageSpec(_compose_spec_tuples(a, b))

Base.:∘(a::AbstractStage, b::AbstractStage) = ChainStage((a, b))

function _compose_spec_tuples(a::AbstractStageSpec, b::AbstractStageSpec)
    if a isa ChainStageSpec
        _assert_no_moments(a, "left")
        if b isa ChainStageSpec
            _assert_no_moments(b, "right")
            return (a.stages..., b.stages...)
        else
            return (a.stages..., b)
        end
    elseif b isa ChainStageSpec
        _assert_no_moments(b, "right")
        return (a, b.stages...)
    else
        return (a, b)
    end
end

_assert_no_moments(spec::ChainStageSpec, side::AbstractString) =
    @assert isempty(spec.moments) "∘: cannot compose a ChainStageSpec that already has moments on its $side; call define_moment! last."
