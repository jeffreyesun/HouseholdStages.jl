"""
State-only flow-utility stage — backward-shift `V_start = u .+ V_end`, identity on Λ. `utility` is any
[`ScalarField`](@ref) source held in the Spec: a `(; dep…[, env])` closure varying along an axis subset,
a precomputed layout-shaped array (a `0`/`-Inf` feasibility table — see [`BorrowingConstraintStage`](@ref)),
a scalar, or a `FromEnv`.
"""
struct UtilityStageSpec{F} <: AbstractStageSpec
    utility :: F
end

UtilityStageSpec(; utility) = UtilityStageSpec{typeof(utility)}(utility)

@definestage UtilityStage UtilityStageSpec


##########################
# Gridded implementation #
##########################
# K = I (identity on Λ); the flow utility is a `ScalarField` broadcast-added to V_end.

allocate_kernel(::UtilityStageSpec, ::Type, ::GriddedLayout) = I

"Cache: the flow utility as a `ScalarField` (the materialized buffer; the Source lives in the Spec). Env-independent ⇒ filled at construction; env-dependent ⇒ NaN-filled, seated each `backward!`."
allocate_cache(spec::UtilityStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    (payoff = ScalarField(spec.utility, layout, T),)

function backward!(V_start, spec::UtilityStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    V_start .= materialize_scalar!(cache.payoff, spec.utility, layout, env; env_changed) .+ V_end
    return (V_start, kernel)
end

# forward! (I → copyto!; the utility reward is a backward-only V shifter) is the generic
# modern default (abstract.jl).


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
