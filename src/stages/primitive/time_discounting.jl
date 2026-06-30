# Time discounting — scale the continuation value by β. A `PointwiseScale(backward = β,
# forward = 1)`: `backward! = β·` (values are discounted), `forward! = copy` (the *undiscounted*
# population is pushed forward; the two directions are NOT an adjoint pair — the discount
# asymmetry, MATH_CONTEXT §1). The public API (`β`, `FromEnv` β) is preserved as a thin
# constructor over [`PointwiseScaleStage`](@ref).

"""
Scalar discounting stage: `V_start = β · V_end`, identity on Λ. `β` is a literal `Real` or a
[`FromEnv`](@ref) marker resolved at `backward!` time. A thin
[`PointwiseScaleStage`](@ref)`(backward = β, forward = 1)`.
"""
TimeDiscountingStage(layout::GriddedLayout; β = 1.0) =
    PointwiseScaleStage(layout; backward = β, forward = 1)
