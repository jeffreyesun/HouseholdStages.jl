"""
State-only flow-utility stage: `V_start[s] = u[s] + V_end[s]`, identity
on Λ. `utility` is either a `(cell; env)` closure evaluated per cell, or
a precomputed layout-shaped `AbstractArray` of flow utilities (e.g. a
`0`/`-Inf` feasibility table — see [`BorrowingConstraintStage`](@ref)).
"""
struct UtilityStageSpec{F} <: AbstractStageSpec
    utility :: F
end

UtilityStageSpec(; utility) = UtilityStageSpec{typeof(utility)}(utility)

@definestage UtilityStage UtilityStageSpec


##########################
# Gridded implementation #
##########################
# K = I (identity on Λ); the flow utility is a layout-shaped array added to V_end.
# `cache` holds the reward array + (for the closure form) the memoised `cell_array`
# the `(cell; env)` closure broadcasts over, refilled each backward.

allocate_kernel(::UtilityStageSpec, ::Type, ::GriddedLayout) = I

"Cache: the reward buffer and the memoised cells (`nothing` for the precomputed-array form)."
function allocate_cache(spec::UtilityStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    if spec.utility isa AbstractArray
        return (reward = copy(spec.utility), cells = nothing)
    else
        return (reward = zeros(T, layout_size(layout)), cells = cell_array(layout))
    end
end

function backward!(V_start, spec::UtilityStageSpec, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache)
    u = _fill_utility!(cache.reward, spec.utility, cache.cells, env)
    V_start .= u .+ V_end
    return (V_start, kernel)
end

# forward! (I → copyto!; the utility reward is a backward-only V shifter) is the generic
# modern default (abstract.jl).

# Array form: the table is static — already in `reward`, nothing to refill.
_fill_utility!(data, ::AbstractArray, ::Nothing, env) = data

"""
Refill the reward buffer with `u(cell; env)` (closure form). `.(...; env)` not `@.`,
which can't carry the `env` kwarg.
"""
_fill_utility!(data, utility, cells, env) =
    (data .= utility.(cells; env); data)


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
