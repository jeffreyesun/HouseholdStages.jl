"""Reproduction or attrition — scale the population, `Λ_end = s·Λ_start`, leaving values untouched; mass is not conserved."""
ReproductionStage(layout::GriddedLayout; s = 1.0) =
    PointwiseScaleStage(layout; backward = 1, forward = s)
