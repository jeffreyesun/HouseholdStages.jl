"""
State-only flow utility: `V_start = u .+ V_end`, identity on Λ. `utility` is any `ScalarField`
source — a scalar, a `FromEnv`, a `(; dep…[, env])` closure varying along a subset of the axes, or a
layout-shaped array.
"""
struct UtilityStageSpec{F} <: AbstractStageSpec
    utility :: F
end

UtilityStageSpec(; utility) = UtilityStageSpec{typeof(utility)}(utility)

@definestage UtilityStage UtilityStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(::UtilityStageSpec) = nothing

allocate_kernel(::UtilityStageSpec, ::Type, ::GriddedLayout, ::GriddedLayout) = I

"Cache: the flow utility as a materialised `ScalarField`."
allocate_cache(spec::UtilityStageSpec, ::Type{T}, start_layout::GriddedLayout, ::GriddedLayout) where {T} =
    (payoff = ScalarField(spec.utility, start_layout, T),)

function backward!(V_start, spec::UtilityStageSpec, start_layout::GriddedLayout, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    V_start .= materialize_scalar!(cache.payoff, spec.utility, start_layout, env; env_changed) .+ V_end
    return (V_start, kernel)
end

# forward! is the generic default — the utility only shifts V.
