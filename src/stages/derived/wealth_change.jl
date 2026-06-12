"""
Deterministic wealth-change stage: each cell's wealth moves to `wealth_post(cell; env)`
on the `wealth_axis` grid (off-grid targets clamp to the grid endpoints). A domain wrapper
over the axis-neutral [`DeterministicContinuousStage`](@ref).
"""
WealthChangeStage(layout::GriddedLayout; wealth_post, wealth_axis::Symbol=:wealth) =
    DeterministicContinuousStage(layout; destination=wealth_post, axis=wealth_axis)
