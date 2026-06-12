# ForgetfulSum is the rectangular marginalize transition: a ones-row dense kernel
# `K = ones(1, n_forget)` along the forget axis. Forward sums the axis out (`K·Λ`),
# backward broadcasts back (`Kᵀ·V`) — both fall out of the dense contraction; this stage
# allocates the prebuilt ones-row kernel (a self-describing array over a ones parent).
#
# Decision 7 (STRATIFIED_KERNEL_PLAN): the output layout RESIZES the forget axis to a
# single level rather than dropping it — the chain's ordered `(name, kind)` tuple is
# invariant; only the axis's size collapses to 1. The surviving coordinate is the axis's
# first level/grid-point (conventional; a marginalised axis is degenerate).

"""
Layout-changing stage that marginalises one axis. Backed by a ones-row dense kernel
`K = ones(1, n_forget)`: forward sums `Λ_start` along the forget axis,
backward broadcasts `V_end` back across it. The output layout keeps the axis at a single
level (resized, not dropped). No V/θ/env dependence — the fiber is built once.
"""
struct ForgetfulSumStageSpec <: AbstractStageSpec
    forget_axis :: Symbol
end

ForgetfulSumStageSpec(; forget_axis::Symbol) = ForgetfulSumStageSpec(forget_axis)

output_layout(spec::ForgetfulSumStageSpec, layout::GriddedLayout) =
    resize_axis_to_one(layout, spec.forget_axis)

@definestage ForgetfulSumStage ForgetfulSumStageSpec


##########################
# Gridded implementation #
##########################
# Ones-row dense kernel `K = ones(1, n_forget)`: `n_out = 1` (marginalised), `n_in = n_forget`,
# no dep. The gather plan (`kernel_scratch`, merged by `@definestage`) sizes its buffers for the
# larger of input/output — here the full forget axis. `scratch.V_start` is input-shaped (full
# axis), `scratch.Λ_end` output-shaped (axis at size 1) via `io_scratch` + `output_layout`. No
# `allocate_scratch` of its own; both verbs and the lift adjoints ride the shared dense path.

function allocate_kernel(spec::ForgetfulSumStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    # Ones-row fiber K = ones(1, n_forget): n_out = 1 (marginalised), n_in = n_forget, no dep.
    return _dense_kernel(ones(T, 1, _axis_size(layout, spec.forget_axis)), layout, spec.forget_axis, ())
end

function backward!(V_start, ::ForgetfulSumStageSpec, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache)
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)   # Kᵀ·V_end : broadcast across the forget axis
    return (V_start, kernel)
end

# forward! (K·Λ_start : sum out the forget axis) is the generic modern default (abstract.jl).


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
