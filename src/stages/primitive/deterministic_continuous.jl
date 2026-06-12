"""
Deterministic transition along one named continuous grid axis. Each cell moves to a
per-cell `destination(cell; env)` on the `axis` grid: backward interpolates `V_end` there,
forward redistributes mass by Young-split shares. Off-grid targets clamp to the grid
endpoints (the unique mass-conserving, adjoint-paired policy — see `kernel.jl`). The
axis-neutral primitive behind [`WealthChangeStage`](@ref) and [`AssetPriceChangeStage`](@ref).
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

# The kernel is the shared `SingleDestinationKernel` (kernel.jl) over FLOAT destinations: each
# cell's off-grid `destinations[cell]` Young-splits its mass (`:share`, forward) and its value
# clip-interpolates back (backward) through the seams in `helper/interpolations.jl`. The kernel
# carries its `axis`; its `kernel_scratch` plan carries the layout grid.

allocate_kernel(spec::DeterministicContinuousStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    SingleDestinationKernel(zeros(T, layout_size(layout)), Val(axis_position(layout, spec.axis)))

"Cache: the memoised `cell_array(layout)` so the destination closure broadcast doesn't reallocate."
allocate_cache(::DeterministicContinuousStageSpec, ::Type, layout::GriddedLayout) =
    (cells = cell_array(layout),)

# Backward / forward — refill `destinations` from the closure each backward (over the memoised
# `cache.cells`), then apply the kernel verb (axis on the kernel, grid in its plan).

function backward!(V_start, spec::DeterministicContinuousStageSpec, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache)
    kernel.destinations .= spec.destination.(cache.cells; env)
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)
    return (V_start, kernel)
end

# forward! (Young-split mass scatter, K·Λ_start) is the generic modern default (abstract.jl).


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. Placeholder marking where the deriv-carrying
# representation's methods will go.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder marking where the dynamic-grid
# representation's methods will go.
