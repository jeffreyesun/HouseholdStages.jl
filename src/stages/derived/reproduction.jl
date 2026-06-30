# Reproduction — domain-named sugar over the measure half of `PointwiseScale`. Leaves V untouched
# (`backward = 1`) and scales Λ on the forward push.

"""
Reproduction / attrition stage — `Λ_end = s · Λ_start`, identity on V. A
[`PointwiseScaleStage`](@ref)`(backward = 1, forward = s)`. `s` is a `Real` or a [`FromEnv`](@ref)
survival/growth factor: `s < 1` is attrition, `s > 1` is growth. Mass is NOT conserved by design
(Λ need not sum to 1).
"""
ReproductionStage(layout::GriddedLayout; s = 1.0) =
    PointwiseScaleStage(layout; backward = 1, forward = s)
