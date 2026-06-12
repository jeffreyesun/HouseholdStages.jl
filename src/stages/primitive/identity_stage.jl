"No-op stage whose K is the identity on `M(S)`. Useful as a `product` branch when one component performs no within-period action."
struct IdentityStageSpec <: AbstractStageSpec end

@definestage IdentityStage IdentityStageSpec


##########################
# Gridded implementation #
##########################
# K = I (UniformScaling): both verbs are copies through the identity transition.

allocate_kernel(::IdentityStageSpec, ::Type, ::GriddedLayout) = I

function backward!(V_start, ::IdentityStageSpec, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache)
    backward!(V_start, kernel, V_end)   # I → copyto!
    return (V_start, kernel)
end

# forward! (I → copyto!) is the generic modern default (abstract.jl).


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
