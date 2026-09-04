"""
Deterministic transition along one named continuous grid `axis`: each cell moves to the position
`destination(; ax…[, env])` on that grid — backward interpolates `V_end` there, forward splits the
cell's mass over the two bracketing points (Young's method). A destination past either end clamps.
"""
struct DeterministicContinuousStageSpec{F} <: AbstractStageSpec
    destination :: F
    axis        :: Symbol
end

DeterministicContinuousStageSpec(; destination, axis::Symbol) =
    DeterministicContinuousStageSpec{typeof(destination)}(destination, axis)

@definestage DeterministicContinuousStage DeterministicContinuousStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(spec::DeterministicContinuousStageSpec) = spec.axis
tangent_grade(::DeterministicContinuousStageSpec)     = :exact_ae

# Kernel: an `InterpKernel` holding one float destination per origin cell.
allocate_kernel(spec::DeterministicContinuousStageSpec, ::Type{T}, start_layout::GriddedLayout,
                ::GriddedLayout) where {T} =
    InterpKernel(zeros(T, layout_size(start_layout)), Val(axis_position(start_layout, spec.axis)))

"Cache: the destination as a materialised [`ScalarField`](@ref)."
allocate_cache(spec::DeterministicContinuousStageSpec, ::Type{T}, start_layout::GriddedLayout,
               ::GriddedLayout) where {T} =
    (destination = ScalarField(spec.destination, start_layout, T),)

function backward!(V_start, spec::DeterministicContinuousStageSpec, start_layout::GriddedLayout,
                   ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    destinations(kernel) .= materialize_scalar!(cache.destination, spec.destination, start_layout, env; env_changed)
    seat_interp!(kernel, scratch.kernel_scratch.dest_grid)
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)
    return (V_start, kernel)
end

# forward! (the Young-split mass scatter) is the generic default.
