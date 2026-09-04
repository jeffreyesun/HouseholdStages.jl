"""
Specification of `n` stages running in parallel, one per level of `axis` — the block-diagonal direct
sum `⊕ᵢ Kᵢ`. Every factor must carry `axis` at size 1 in both of its layouts and span the same two
layouts as the others; their specs may differ freely.
"""
struct ProductStageSpec{Specs<:Tuple} <: AbstractStageSpec
    components :: Specs
    axis       :: Symbol
end

function ProductStageSpec(components::Tuple; axis::Symbol=:group)
    @assert !isempty(components)
    return ProductStageSpec{typeof(components)}(components, axis)
end

"""
The built form of a product: the per-factor sub-stages, the fused `V` and `Λ` tensors in which each
factor owns one product-axis slice, the product's own two layouts, and each factor's interior.
"""
struct ProductStageBuffer{Stages<:Tuple, Av<:AbstractArray, Al<:AbstractArray, L1, L2, I<:Tuple} <: AbstractStageBuffer
    components   :: Stages
    V_fused      :: Av
    Λ_fused      :: Al
    start_layout :: L1
    end_layout   :: L2
    interiors    :: I
end

start_layout(buffer::ProductStageBuffer) = buffer.start_layout
end_layout(buffer::ProductStageBuffer)   = buffer.end_layout

"Stages running in parallel, one per level of the product axis."
struct ProductStage{Spec<:ProductStageSpec, Buffer<:ProductStageBuffer} <: AbstractCompositeStage
    spec   :: Spec
    buffer :: Buffer
end

_component_interior(stage::ProductStage) = interiors(stage)

_bundle_between(spec::ProductStageSpec, l_start, l_end, interior::Tuple, ::Type{T}) where {T} =
    ProductStage(spec, l_start, l_end, interior, T)

"""
Join built stages into a `ProductStage` along `axis`, growing it from 1 to the number of factors.
Each factor must carry `axis` at size 1 and span the same two layouts as the first.
"""
function product(stages::AbstractStage...; axis::Symbol=:group)
    @assert !isempty(stages)
    n = length(stages)
    l_start, l_end = start_layout(first(stages)), end_layout(first(stages))
    for (i, s) in enumerate(stages)
        _assert_axis_size(start_layout(s), end_layout(s), axis, 1, "product: factor $i",
                          "§1.2, the fixed-layout invariant: `⊕` resizes `$axis` 1 → $n and never " *
                          "introduces it, so declare `$axis` in the block layout and never `⊕` twice on it")
        @assert start_layout(s) == l_start && end_layout(s) == l_end "product: factor $i does not " *
            "sit in parallel between the same two boundaries as factor 1."
    end
    spec = ProductStageSpec(map(s -> s.spec, stages); axis)
    return ProductStage(spec, grow_axis(l_start, axis, n), grow_axis(l_end, axis, n),
                        map(_component_interior, stages))
end

"Assert `axis` has size `n` in both layouts; `why` names the invariant, for the error message."
function _assert_axis_size(l_start, l_end, axis::Symbol, n::Int,
                           who::AbstractString, why::AbstractString)
    for (side, l) in (("start", l_start), ("end", l_end))
        got = axis in axisnames(l) ? _axis_size(l, axis) : "absent"
        @assert got == n "$who: the `$axis` axis is size $got in the $side layout, but must be $n — $why."
    end
    return nothing
end

ProductStage(spec::ProductStageSpec, start_layout::GriddedLayout, end_layout::GriddedLayout,
             interiors::Tuple=_no_interiors(spec.components), ::Type{T}=Float64) where {T} =
    ProductStage(spec, allocate(spec, start_layout, end_layout, interiors, T))

bundle(spec::ProductStageSpec, start_layout::GriddedLayout, end_layout::GriddedLayout,
       ::Type{T}=Float64) where {T} =
    ProductStage(spec, start_layout, end_layout, _no_interiors(spec.components), T)

# Allocate — build the factors, then the fused tensors they slice #
#-----------------------------------------------------------------#

"Build each factor with the product axis shrunk to size 1, and allocate the fused `V`/`Λ` tensors."
function allocate(spec::ProductStageSpec, start_layout::GriddedLayout, end_layout::GriddedLayout,
                  interiors::Tuple, ::Type{T}) where {T}
    n = length(spec.components)
    @assert length(interiors) == n "ProductStage: one interior per component ($n); got $(length(interiors))."
    _assert_axis_size(start_layout, end_layout, spec.axis, n, "ProductStage",
                      "the product-level layouts carry the product axis at `n`, one level per component")
    factor_start = resize_axis(start_layout, spec.axis, 1)
    factor_end   = resize_axis(end_layout,   spec.axis, 1)

    components = ntuple(i -> _bundle_between(spec.components[i], factor_start, factor_end,
                                             interiors[i], T), n)
    V_fused = zeros(T, layout_size(start_layout))
    Λ_fused = zeros(T, layout_size(end_layout))
    return ProductStageBuffer(components, V_fused, Λ_fused, start_layout, end_layout,
                              map(_component_interior, components))
end

# Env slice — union over components #
#-----------------------------------#

static_env_deps(::Type{<:ProductStageSpec}) = NamedTuple()

effective_env_slice(spec::ProductStageSpec) = _union_env_slices(spec.components)
tangent_grade(spec::ProductStageSpec)        = _worst_grade(map(tangent_grade, spec.components))

# Backward / forward — per-component on slices of the fused tensor #
#------------------------------------------------------------------#

function backward!(buffer::ProductStageBuffer, spec::ProductStageSpec, V_end, env;
                   env_changed::Bool = true)
    n    = length(spec.components)
    pdim = axis_position(buffer.start_layout, spec.axis)   # product axis (same position at both ends)
    for i in 1:n
        V_slice      = selectdim(V_end, pdim, i:i)          # i:i keeps the axis at size 1
        V_start_comp = backward!(buffer.components[i], V_slice, env; env_changed)
        selectdim(buffer.V_fused, pdim, i:i) .= V_start_comp
    end
    return buffer.V_fused
end

function forward!(buffer::ProductStageBuffer, spec::ProductStageSpec, Λ_start)
    n    = length(spec.components)
    pdim = axis_position(buffer.start_layout, spec.axis)
    for i in 1:n
        Λ_slice    = selectdim(Λ_start, pdim, i:i)
        Λ_end_comp = forward!(buffer.components[i], copy(Λ_slice))
        selectdim(buffer.Λ_fused, pdim, i:i) .= Λ_end_comp
    end
    return buffer.Λ_fused
end

# Endpoint accessors — fused tensors #
#------------------------------------#

V_start_buffer(stage::ProductStage) = stage.buffer.V_fused
Λ_end_buffer(stage::ProductStage)   = stage.buffer.Λ_fused

# `⊕` operator #
#--------------#

"Direct sum of two stages along the `:group` axis; the Stage form allocates a fresh fused buffer."
⊕(a::AbstractStageSpec, b::AbstractStageSpec) =
    ProductStageSpec((a, b); axis=:group)

⊕(a::AbstractStage, b::AbstractStage) = product(a, b; axis=:group)

"`N` independent copies of `stage` joined along `axis`, each with its own kernel, scratch and cache."
replicate_age(stage::AbstractStage, N::Int; axis::Symbol=:age) =
    product(ntuple(_ -> stage, N)...; axis=axis)
replicate_age(spec::AbstractStageSpec, N::Int; axis::Symbol=:age) =
    ProductStageSpec(ntuple(_ -> spec, N); axis=axis)
