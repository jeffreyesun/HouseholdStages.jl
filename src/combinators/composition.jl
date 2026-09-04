"Specification of a time-ordered composition: the component specs, flattened, plus attached moments."
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

"Flatten nested `ChainStageSpec` components into one tuple of leaf specs."
function _flatten_chain_specs(stages::Tuple)
    out = AbstractStageSpec[]
    for s in stages
        @assert s isa AbstractStageSpec
        s isa ChainStageSpec ? append!(out, s.stages) : push!(out, s)
    end
    return Tuple(out)
end

# Construction #
#--------------#

"One `nothing` per component."
_no_interiors(components::Tuple) = ntuple(_ -> nothing, length(components))

"What a built stage needs to be rebuilt beyond its two layouts. A primitive needs nothing."
_component_interior(::AbstractPrimitiveStage) = nothing

"Build one component between the two layouts it spans, from its spec and its `interior`."
_bundle_between(spec::AbstractStageSpec, l_start, l_end, ::Nothing, ::Type{T}) where {T} =
    bundle(spec, l_start, l_end, T)

"""
The built form of a chain: its sub-stages, the `n+1` layouts between them (component `k` runs from
`boundaries[k]` to `boundaries[k+1]`), and each component's interior.
"""
struct ChainStageBuffer{Stages<:Tuple, B<:Tuple, I<:Tuple} <: AbstractStageBuffer
    stages     :: Stages
    boundaries :: B
    interiors  :: I
end

ChainStageBuffer(stages::Tuple, boundaries::Tuple) =
    ChainStageBuffer(stages, boundaries, map(_component_interior, stages))

start_layout(buffer::ChainStageBuffer) = first(buffer.boundaries)
end_layout(buffer::ChainStageBuffer)   = last(buffer.boundaries)

"Stages run one after another in time, as built by `∘`."
struct ChainStage{Spec<:ChainStageSpec, Buffer<:ChainStageBuffer} <: AbstractCompositeStage
    spec   :: Spec
    buffer :: Buffer
end

"The `n+1` layouts the chain's components sit between."
boundaries(chain::ChainStage) = chain.buffer.boundaries

_component_interior(chain::ChainStage) =
    (boundaries = boundaries(chain), interiors = interiors(chain))

_bundle_between(spec::ChainStageSpec, l_start, l_end, interior::NamedTuple, ::Type{T}) where {T} =
    ChainStage(spec, (l_start, interior.boundaries[2:end-1]..., l_end), interior.interiors, T)

function ChainStage(stages::Tuple{AbstractStage, Vararg{AbstractStage}})
    sub_stages = _flatten_chain_stages(stages)
    spec = ChainStageSpec(map(s -> s.spec, sub_stages))
    buf  = ChainStageBuffer(sub_stages, _component_boundaries(sub_stages))
    return ChainStage{typeof(spec), typeof(buf)}(spec, buf)
end

ChainStage(stages::Tuple{AbstractStageSpec, Vararg{AbstractStageSpec}}) =
    ChainStage(ChainStageSpec(stages))

ChainStage(::Tuple{}) = throw(AssertionError("ChainStage: a chain needs at least one component."))
ChainStage(::Tuple)   = error("ChainStage components must be uniformly AbstractStage or AbstractStageSpec")

"Flatten a tuple of bundled stages into one tuple of leaf sub-stages, reused as they stand."
function _flatten_chain_stages(stages::Tuple)
    out = AbstractStage[]
    for s in stages
        s isa ChainStage ? append!(out, s.buffer.stages) : push!(out, s)
    end
    return Tuple(out)
end

"The `n+1` boundaries of built components, asserting adjacency: `k`'s end layout is `k+1`'s start."
function _component_boundaries(stages::Tuple)
    bounds = (map(start_layout, stages)..., end_layout(last(stages)))
    for (k, s) in enumerate(stages)
        @assert end_layout(s) == bounds[k + 1] "ChainStage: component $k ($(typeof(s).name.name)) ends on a layout its successor does not start from."
    end
    return bounds
end

ChainStage(spec::ChainStageSpec) = error("ChainStage(spec) needs layouts; call ChainStage(spec, boundaries) or compose bundled stages with `∘`.")

"Build each component between its own pair of adjacent boundaries."
function ChainStage(spec::ChainStageSpec, boundaries::Tuple,
                    interiors::Tuple=_no_interiors(spec.stages), ::Type{T}=Float64) where {T}
    n = length(spec.stages)
    @assert length(boundaries) == n + 1 "ChainStage: a $n-component chain takes $(n + 1) boundary layouts; got $(length(boundaries))."
    @assert length(interiors) == n "ChainStage: one interior per component ($n); got $(length(interiors))."
    stages = ntuple(k -> _bundle_between(spec.stages[k], boundaries[k], boundaries[k + 1],
                                         interiors[k], T), n)
    return ChainStage(spec, ChainStageBuffer(stages, boundaries))
end

"Build a chain from its two end layouts, which must be equal; every boundary is then that layout."
function bundle(spec::ChainStageSpec, l_start::AbstractLayout, l_end::AbstractLayout,
                ::Type{T}=Float64) where {T}
    n = length(spec.stages)
    @assert l_start == l_end "ChainStage: a chain that regrids is not determined by its two ends — pass the full $(n + 1)-long boundary sequence."
    return ChainStage(spec, ntuple(_ -> l_start, n + 1), _no_interiors(spec.stages), T)
end

# Env slice — union over components #
#-----------------------------------#

effective_env_slice(spec::ChainStageSpec) = _union_env_slices(spec.stages)
tangent_grade(spec::ChainStageSpec)        = _worst_grade(map(tangent_grade, spec.stages))

"The `env` fields a chain's sweeps read."
chain_env_names(chain::Union{ChainStage, ChainStageSpec}) = effective_env_slice(chain)

# Endpoint accessors #
#--------------------#

V_start_buffer(stage::ChainStage) = V_start_buffer(stage.buffer.stages[1])
Λ_end_buffer(stage::ChainStage)   = Λ_end_buffer(stage.buffer.stages[end])

"The policy of the chain's one policy-bearing leaf component; errors unless there is exactly one."
function policy(chain::ChainStage)
    leaves = filter(s -> !(s isa ChainStage) && hasmethod(policy, Tuple{typeof(s)}),
                    collect(chain.buffer.stages))
    length(leaves) == 1 ||
        error("policy(::ChainStage): expected exactly one policy-bearing leaf, found $(length(leaves)).")
    return policy(leaves[1])
end

# Sweeps — a fold over the components #
#-------------------------------------#

backward!(buffer::ChainStageBuffer, ::ChainStageSpec, V_end, env; env_changed::Bool = true) =
    foldr((s, V) -> backward!(s, V, env; env_changed), buffer.stages; init = V_end)

forward!(buffer::ChainStageBuffer, ::ChainStageSpec, Λ_start) =
    foldl((Λ, s) -> forward!(s, Λ), buffer.stages; init = Λ_start)

# Composition operators #
#-----------------------#

"""
Time-ordered composition: `a` runs first, the opposite of `Base.∘` on functions. Nested chains
flatten, and a chain that already carries moments cannot be composed.
"""
Base.:∘(a::AbstractStageSpec, b::AbstractStageSpec) =
    ChainStageSpec(_compose_spec_tuples(a, b))

Base.:∘(a::AbstractStage, b::AbstractStage) = ChainStage((a, b))

"What an operand of `∘` contributes to the flat tuple: a chain its components, a leaf itself."
_spec_operand(spec::ChainStageSpec, side)    = (_assert_no_moments(spec, side); spec.stages)
_spec_operand(spec::AbstractStageSpec, side) = (spec,)

_compose_spec_tuples(a::AbstractStageSpec, b::AbstractStageSpec) =
    (_spec_operand(a, "left")..., _spec_operand(b, "right")...)

_assert_no_moments(spec::ChainStageSpec, side::AbstractString) =
    @assert isempty(spec.moments) "∘: cannot compose a ChainStageSpec that already has moments on its $side; call define_moment! last."
