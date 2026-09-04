"""
Scalar discounting: `V_start = β · V_end`, identity on Λ. `β` is a literal `Real` or a `FromEnv`
marker resolved at `backward!` time.
"""
TimeDiscountingStage(layout::GriddedLayout; β = 1.0) =
    PointwiseScaleStage(layout; backward = β, forward = 1)
