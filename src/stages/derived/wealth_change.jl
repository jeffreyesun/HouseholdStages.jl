"""
Deterministic wealth-change stage: each cell's wealth moves to the dep closure
`wealth_post(; ax…[, env])` on the wealth `axis` grid (off-grid targets clamp to the grid
endpoints). A domain wrapper over the axis-neutral [`DeterministicContinuousStage`](@ref).
"""
WealthChangeStage(layout::GriddedLayout; wealth_post, axis::Symbol=:wealth) =
    DeterministicContinuousStage(layout; destination=wealth_post, axis=axis)
