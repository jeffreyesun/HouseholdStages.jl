"""
Hard discrete-choice stage: actions are the levels of `choice_axis` and K is a sparse
single-destination scatter (the chosen per-cell destination lives on the kernel). The two
callbacks carry the model:

  - `flow_payoff(cell, action; env)` — period payoff; `-Inf` means "unavailable".
  - `next_state_idx(cell, action) -> Int` — index along `choice_axis` reached by the action.
"""
struct ArgmaxStageSpec{F, BF} <: AbstractStageSpec
    choice_axis    :: Symbol
    flow_payoff    :: F
    next_state_idx :: BF
end

ArgmaxStageSpec(; choice_axis, flow_payoff, next_state_idx) =
    ArgmaxStageSpec{typeof(flow_payoff), typeof(next_state_idx)}(
        choice_axis, flow_payoff, next_state_idx,
    )

@definestage ArgmaxStage ArgmaxStageSpec


##########################
# Gridded implementation #
##########################

# Kernel — the shared `SingleDestinationKernel` (kernel.jl) #
#----------------------------------------------------------#
# The argmax `σ(s) = argmax_a R[s,a] + V_out[ν(s,a)]` is selected during `backward!`, which
# writes each cell's resolved destination index `ν(s,σ(s))` into the kernel's `destinations`
# (an O(state) integer scatter/gather — never the dense `(n,n,N_other)` tensor the
# destination's dependence on every off-choice axis would require). The precomputed
# action→destination map `next_ci` is build scratch for the Q-findmax, so it lives in cache.

allocate_kernel(spec::ArgmaxStageSpec, ::Type, layout::GriddedLayout) =
    SingleDestinationKernel(zeros(Int, layout_size(layout)), Val(axis_position(layout, spec.choice_axis)))

"Scratch: the io buffers plus the transient per-(cell, action) payoff buffer `R`."
function allocate_scratch(spec::ArgmaxStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    R = zeros(T, layout_size(layout)..., axissize(layout.axes[axis_dim(layout, spec.choice_axis)]))
    return merge(io_scratch(spec, layout, T), (R = R,))
end

"Cache: the memoised host cells, the reshaped action vector, and the `(cell, action) → destination` map `next_ci` (build scratch for the Q-findmax)."
function allocate_cache(spec::ArgmaxStageSpec, ::Type, layout::GriddedLayout)
    cdim       = axis_dim(layout, spec.choice_axis)
    action_vec = axisvalues(layout.axes[cdim])
    adim       = length(layout.axes) + 1
    actions    = reshape(action_vec, ntuple(_ -> 1, adim - 1)..., :)
    next_ci    = reshape(
        [set_coord(CartesianIndex(Tuple(idx)), layout, spec.choice_axis => spec.next_state_idx(cell, action))
         for (idx, cell) in cells(layout), action in action_vec],
        layout_size(layout)..., length(action_vec),
    )
    return (cells = cell_array(layout), actions = actions, next_ci = next_ci)
end

# Degenerate (K, r): the findmax selects the policy AND yields V_start = max_a Q directly
# (no separate `r + KᵀV` — the flow is already in the max), so there is no constant reward
# to carry. The POLICY — the choice variable, the chosen index ν(s,σ(s)) on `choice_axis` —
# is the kernel's `destinations` (choosing where to go on the axis IS the destination).
# backward writes it; forward scatters mass by it (`:nearest`, the shared seam).

function backward!(V_start, spec::ArgmaxStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache)
    next_ci = cache.next_ci
    adim    = ndims(next_ci)                        # action axis is next_ci's trailing dim
    copyto!(scratch.R, spec.flow_payoff.(cache.cells, cache.actions; env = env))
    Q = scratch.R .+ V_end[next_ci]                # Q[s,a] = r(s,a) + V_end[ν(s,a)]
    best_v, best_ci = findmax(Q; dims = adim)       # ties → first (strict-`>` rule)
    @assert all(isfinite, best_v) "every cell must have at least one finite-payoff action"
    V_start .= reshape(best_v, size(V_start))
    choice_dim = axis_position(layout, spec.choice_axis)
    kernel.destinations .= getindex.(next_ci[reshape(best_ci, size(kernel.destinations))], choice_dim)
    return (V_start, kernel)
end

# forward! (the `:nearest` scatter, K·Λ_start) is the generic modern default (abstract.jl).

"""
The solved policy of an [`ArgmaxStage`](@ref): the chosen `choice_axis` index per cell (the
choice variable). It IS the kernel's destination — for an argmax, choosing where to go on
the axis is the policy.
"""
policy(stage::ArgmaxStage) = stage.kernel.destinations


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. The phase-1 stage methods above do not dispatch on
# layout type, so this is a placeholder marking where the deriv-carrying
# representation's methods will go.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder marking where the dynamic-grid
# representation's methods will go.
