"""
Deterministic transition along one named continuous grid axis. Each cell moves to the dep closure
`destination(; ax…[, env])` evaluated on the `axis` grid: backward interpolates `V_end` there,
forward redistributes mass by Young-split shares. The axis-neutral primitive behind
[`WealthChangeStage`](@ref) and [`AssetPriceChangeStage`](@ref).
"""
# Off-grid pairing: the backward interpolate-gather and the forward Young-split scatter are exact
# transposes for ALL destinations, interior and off-grid. The forward mass pass
# (`convert_distribution!`) clamps overflow/underflow onto the endpoint bins; the backward value
# pass (`reinterpolate!`, `:clip`) now clamps at BOTH ends to match (the right-side clip was the
# missing half — linear extrapolation there would need negative interp weights and break the
# K/Kᵀ pair). So this is a genuine transition off-grid too: V/Λ duality and the reverse-mode
# adjoints hold, with no per-stage off-grid flag (end-goal §8/§13).
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

# The kernel is an `InterpKernel` (kernel.jl) over FLOAT destinations (continuous axis): each
# cell's off-grid `destinations[cell]` Young-splits its mass (`:share`, forward) and its value
# clip-interpolates back (backward) through the seams in `helper/interpolations.jl`. The kernel
# carries its `axis`; its `kernel_scratch` plan carries the layout grid.

allocate_kernel(spec::DeterministicContinuousStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    InterpKernel(zeros(T, layout_size(layout)), Val(axis_position(layout, spec.axis)))

"Cache: the destination as a [`ScalarField`](@ref) (the materialized buffer; the Source lives in the Spec). Env-independent ⇒ filled at construction; env-dependent ⇒ NaN-filled, seated each `backward!`."
allocate_cache(spec::DeterministicContinuousStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    (destination = ScalarField(spec.destination, layout, T),)

# Backward / forward — materialise `destinations` from the source each backward (the `ScalarField`
# broadcasts a dep closure's compact buffer into the full `destinations` array), then apply the
# kernel verb (axis on the kernel, grid in its plan).

function backward!(V_start, spec::DeterministicContinuousStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    destinations(kernel) .= materialize_scalar!(cache.destination, spec.destination, layout, env; env_changed)
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
