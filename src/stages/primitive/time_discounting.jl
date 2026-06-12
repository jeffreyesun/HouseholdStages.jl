# Time discounting — scale the continuation value by β. The transition is the
# discount outlier `BackwardScale(β)`: `backward! = β·`, `forward! = copy` (the
# *undiscounted* population is pushed forward; the two directions are NOT an
# adjoint pair — the discount asymmetry, MATH_CONTEXT §1).

"""
Scalar discounting stage: `V_start = β · V_end`, identity on Λ.
`β` is a literal `Real` or a [`FromEnv`](@ref) marker (`FromEnv(:β)`)
resolved at `backward!` time. The kernel is `BackwardScale(β)`; the resolved
β is rewritten into a `Ref` each `backward!` so a `FromEnv`/AD-shocked β tracks env.
"""
@kwdef struct TimeDiscountingStageSpec{B} <: AbstractStageSpec
    β :: B = 1.0
end

@definestage TimeDiscountingStage TimeDiscountingStageSpec


##########################
# Gridded implementation #
##########################
# β lives in a `Ref{Any}` inside `BackwardScale` so an env-resolved or AD-shocked β
# (incl. `Dual`s) can be written in place each backward.

allocate_kernel(::TimeDiscountingStageSpec, ::Type, ::GriddedLayout) =
    BackwardScale(Base.RefValue{Any}(1.0))

function backward!(V_start, spec::TimeDiscountingStageSpec, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache)
    kernel.β[] = resolve(spec.β, env)
    backward!(V_start, kernel, V_end)    # β · V_end
    return (V_start, kernel)
end

# forward! (copy — the undiscounted mass push) is the generic modern default (abstract.jl).


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
