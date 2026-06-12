"""
Cartesian product of `n` *uniform* stages along a new product axis.
The K-operator is the block-diagonal direct sum `⊕_i K_i`. The
product axis is appended to the component layout as a
`discrete_finite(1:n)` axis with the user-chosen name.

v1 restricts to components with identical concrete Spec type and
identical input layout (when allocated against the product layout).
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
struct ProductStageBuffer{Stages<:Tuple, Tv, Nin, Tl, Nout, LIn, LOut} <: AbstractStageBuffer
    components    :: Stages
    V_fused       :: Array{Tv, Nin}
    Λ_fused       :: Array{Tl, Nout}
    input_layout  :: LIn
    output_layout :: LOut
    cache         :: CacheState
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
    # All components share an input layout: pick the first.
    comp_layout = input_layout(stages[1])
    return ProductStage(spec, comp_layout)
end

# Spec-level product (no allocation).
product_spec(specs::AbstractStageSpec...; axis::Symbol=:group) =
    ProductStageSpec(specs; axis)

ProductStage(spec::ProductStageSpec, comp_layout::GriddedLayout, ::Type{T}=Float64) where {T} =
    ProductStage(spec, allocate(spec, _product_layout(spec, comp_layout), T))

"Append the product axis (`n` discrete levels) to a component layout."
_append_product_axis(comp_layout::GriddedLayout, axis::Symbol, n::Int) =
    GriddedLayout(comp_layout.axes..., StateAxis(axis, discrete_finite(collect(1:n))))

"Wrap a component input layout into the product input layout by appending the product axis."
_product_layout(spec::ProductStageSpec, comp_layout::GriddedLayout) =
    _append_product_axis(comp_layout, spec.axis, length(spec.components))

bundle(spec::ProductStageSpec, layout::GriddedLayout) = ProductStage(spec, layout)
bundle(spec::ProductStageSpec, layout::GriddedLayout, ::Type{T}) where {T} =
    ProductStage(spec, layout, T)

# Allocate — bundle components, fuse along the product axis #
#----------------------------------------------------------#

function allocate(spec::ProductStageSpec, layout::GriddedLayout,
                  ::Type{T}=Float64) where {T}
    n              = length(spec.components)
    comp_in_layout = _strip_product_axis(layout, spec.axis)
    # Uniform components ⇒ identical output layout; build it from the first.
    comp_out_layout = output_layout(spec.components[1], comp_in_layout)
    out_layout      = _append_product_axis(comp_out_layout, spec.axis, n)

    V_fused = zeros(T, layout_size(layout))
    Λ_fused = zeros(T, layout_size(out_layout))

    components = ntuple(i -> bundle(spec.components[i], comp_in_layout, T), n)
    return ProductStageBuffer(components, V_fused, Λ_fused,
                              layout, out_layout, CacheState())
end

"""
Strip the product axis from a fused layout, returning the per-component
layout. Asserts the product axis is the trailing one.
"""
function _strip_product_axis(layout::GriddedLayout, axis::Symbol)
    @assert axisname(last(layout.axes)) === axis
    return GriddedLayout(layout.axes[1:end-1]...)
end

# Env slice — union over components #
#-----------------------------------#

static_env_deps(::Type{<:ProductStageSpec}) = NamedTuple()

function effective_env_slice(spec::ProductStageSpec)
    names = Symbol[]
    for s in spec.components
        for k in effective_env_slice(s)
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end

# Backward / forward — per-component on slices of the fused tensor #
#------------------------------------------------------------------#
# Each component is a bundled sub-stage driven through the per-stage sugar (uniform
# across legacy/modern); its result is copied into the matching product-axis slice.

function backward!(buffer::ProductStageBuffer, spec::ProductStageSpec, V_end, env)
    n    = length(spec.components)
    Nin  = ndims(V_end)
    for i in 1:n
        V_slice      = selectdim(V_end, Nin, i)
        V_start_comp = backward!(buffer.components[i], V_slice, env)
        selectdim(buffer.V_fused, Nin, i) .= V_start_comp
    end
    _seat_cache!(buffer, V_end, env)
    return buffer.V_fused
end

function forward!(buffer::ProductStageBuffer, spec::ProductStageSpec, Λ_start)
    n    = length(spec.components)
    Nin  = ndims(Λ_start)
    Nout = ndims(buffer.Λ_fused)
    for i in 1:n
        Λ_slice    = selectdim(Λ_start, Nin, i)
        Λ_end_comp = forward!(buffer.components[i], Λ_slice)
        selectdim(buffer.Λ_fused, Nout, i) .= Λ_end_comp
    end
    return buffer.Λ_fused
end

# Endpoint accessors — fused tensors #
#------------------------------------#

V_start_buffer(stage::ProductStage) = stage.buffer.V_fused
Λ_end_buffer(stage::ProductStage)   = stage.buffer.Λ_fused

# Cache invalidation walks components #
#-------------------------------------#

function invalidate!(buffer::ProductStageBuffer)
    buffer.cache.kernel_valid = false
    for s in buffer.components
        invalidate!(s)
    end
    return buffer
end

# `×` operator — Spec-level primary, Stage-level sugar #
#------------------------------------------------------#

"""
Product of two stages along the default `:group` axis. The Spec form returns a
`ProductStageSpec`; the Stage form bundles a fresh buffer.
"""
×(a::AbstractStageSpec, b::AbstractStageSpec) =
    ProductStageSpec((a, b); axis=:group)

×(a::AbstractStage, b::AbstractStage) = product(a, b; axis=:group)

"`N` uniform copies of `stage` joined along `axis`. Accepts a bundled stage or a Spec."
replicate_age(stage::AbstractStage, N::Int; axis::Symbol=:age) =
    product(ntuple(_ -> stage, N)...; axis=axis)
replicate_age(spec::AbstractStageSpec, N::Int; axis::Symbol=:age) =
    ProductStageSpec(ntuple(_ -> spec, N); axis=axis)
