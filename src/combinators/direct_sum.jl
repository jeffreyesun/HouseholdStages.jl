"""
Cartesian product of `n` *uniform* stages along an existing **singleton** axis. The K-operator is
the block-diagonal direct sum `⊕_i K_i`. The product axis must already exist in every factor's
layout at size 1 (the fixed-layout invariant — the product never introduces an axis); it is resized
`1 → n`, one factor per slice, and may sit at any position (not only the last).

v1 restricts to components with identical concrete Spec type and identical input layout.
"""
struct ProductStageSpec{Specs<:Tuple} <: AbstractStageSpec
    components :: Specs
    axis       :: Symbol
end

function ProductStageSpec(components::Tuple; axis::Symbol=:group)
    @assert !isempty(components)
    first_type = typeof(components[1])
    @assert all(s -> typeof(s) === first_type, components)
    return ProductStageSpec{typeof(components)}(components, axis)
end

"""
Buffer for a product. Holds the fused `V` (input-layout shaped) and `Λ`
(output-layout shaped) tensors plus the bundled per-component sub-STAGES, driven
through the per-stage sugar and copied into/out of the fused slices.
"""
# The fused `V`/`Λ` are carried on free `<:AbstractArray` type parameters (not a concrete
# `Array{T,N}`) so the GPU lift `to_device(::ProductStage, …)` can rebuild this buffer with
# device-resident fused tensors — exactly the convention the modern-stage lift documents
# (lifts/gpu.jl). On the host these parameters infer to `Array`, leaving CPU behavior unchanged.
struct ProductStageBuffer{Stages<:Tuple, Av<:AbstractArray, Al<:AbstractArray, LIn, LOut} <: AbstractStageBuffer
    components    :: Stages
    V_fused       :: Av
    Λ_fused       :: Al
    input_layout  :: LIn
    output_layout :: LOut
end

"Product stage: bundled stages of identical type joined along a new axis."
struct ProductStage{Spec<:ProductStageSpec, Buffer<:ProductStageBuffer} <: AbstractLegacyStage
    spec   :: Spec
    buffer :: Buffer
end

# Construct a ProductStage from bundled sub-stages.
function product(stages::AbstractStage...; axis::Symbol=:group)
    @assert !isempty(stages)
    specs  = map(s -> s.spec, stages)
    spec   = ProductStageSpec(specs; axis)
    # All components share an input layout that must carry the product axis as a size-1 singleton.
    comp_layout = input_layout(stages[1])
    @assert axis in axisnames(comp_layout) && _axis_size(comp_layout, axis) == 1 "product: the " *
        "`$axis` axis must be a size-1 singleton in every factor's layout (it is resized to " *
        "$(length(stages)); the product never introduces an axis — declare `$axis` in the block layout)."
    return ProductStage(spec, comp_layout)
end

ProductStage(spec::ProductStageSpec, comp_layout::GriddedLayout, ::Type{T}=Float64) where {T} =
    ProductStage(spec, allocate(spec, _product_layout(spec, comp_layout), T))

"The product layout: the (singleton-in-each-factor) product axis grown to `n` discrete levels `1:n`,
in place (same position). The inverse — back to the per-factor layout — is `resize_axis(·, axis, 1)`."
_product_layout(spec::ProductStageSpec, comp_layout::GriddedLayout) =
    grow_axis(comp_layout, spec.axis, length(spec.components))

bundle(spec::ProductStageSpec, layout::GriddedLayout) = ProductStage(spec, layout)
bundle(spec::ProductStageSpec, layout::GriddedLayout, ::Type{T}) where {T} =
    ProductStage(spec, layout, T)

# Allocate — bundle components, fuse along the product axis #
#----------------------------------------------------------#

function allocate(spec::ProductStageSpec, layout::GriddedLayout,
                  ::Type{T}=Float64) where {T}
    n               = length(spec.components)
    comp_in_layout  = resize_axis(layout, spec.axis, 1)             # factor layout: product axis at size 1
    # Uniform components ⇒ identical output layout; build it from the first.
    comp_out_layout = output_layout(spec.components[1], comp_in_layout)
    out_layout      = _product_layout(spec, comp_out_layout)       # grow the product axis 1 → n

    V_fused = zeros(T, layout_size(layout))
    Λ_fused = zeros(T, layout_size(out_layout))

    components = ntuple(i -> bundle(spec.components[i], comp_in_layout, T), n)
    return ProductStageBuffer(components, V_fused, Λ_fused,
                              layout, out_layout)
end

# Env slice — union over components #
#-----------------------------------#

static_env_deps(::Type{<:ProductStageSpec}) = NamedTuple()

effective_env_slice(spec::ProductStageSpec) = _union_env_slices(spec.components)

# Backward / forward — per-component on slices of the fused tensor #
#------------------------------------------------------------------#
# Each component is a bundled sub-stage driven through the per-stage sugar (uniform
# across legacy/modern); its result is copied into the matching product-axis slice.

function backward!(buffer::ProductStageBuffer, spec::ProductStageSpec, V_end, env;
                   env_changed::Bool = true)
    n    = length(spec.components)
    pdim = axis_position(buffer.input_layout, spec.axis)   # product axis (same position in/out)
    for i in 1:n
        V_slice      = selectdim(V_end, pdim, i:i)          # i:i keeps the axis at size 1
        V_start_comp = backward!(buffer.components[i], V_slice, env; env_changed)
        selectdim(buffer.V_fused, pdim, i:i) .= V_start_comp
    end
    return buffer.V_fused
end

function forward!(buffer::ProductStageBuffer, spec::ProductStageSpec, Λ_start)
    n    = length(spec.components)
    pdim = axis_position(buffer.input_layout, spec.axis)
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

# `⊕` operator — Spec-level primary, Stage-level sugar #
#------------------------------------------------------#

"""
Direct sum of two stages along the default `:group` axis (the block-diagonal
combinator `⊕`). The Spec form returns a `ProductStageSpec`; the Stage form
bundles a fresh buffer.
"""
⊕(a::AbstractStageSpec, b::AbstractStageSpec) =
    ProductStageSpec((a, b); axis=:group)

⊕(a::AbstractStage, b::AbstractStage) = product(a, b; axis=:group)

"`N` uniform copies of `stage` joined along `axis`. Accepts a bundled stage or a Spec."
replicate_age(stage::AbstractStage, N::Int; axis::Symbol=:age) =
    product(ntuple(_ -> stage, N)...; axis=axis)
replicate_age(spec::AbstractStageSpec, N::Int; axis::Symbol=:age) =
    ProductStageSpec(ntuple(_ -> spec, N); axis=axis)
