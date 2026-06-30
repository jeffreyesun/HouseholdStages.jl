# ForgetfulSum — marginalize one axis by sum. NOT a primitive: it is exactly a `MarkovStage` with an
# `n×1` all-ones (row-stochastic) transition `T = ones(n, 1)` — every level maps to the single
# surviving level, so the forward sums the axis out (`K·Λ`), the backward broadcasts back (`Kᵀ·V`),
# and Markov's rectangular `to = 1` output layout resizes the axis to one. A derived stage —
# domain-named sugar over the de-squared `MarkovStage`, with no kernel/backward!/forward! of its own.

"""
Marginalise one axis by sum — sugar for a rectangular `MarkovStage` with
`transition_matrix = ones(n, 1)`: the forward sums `Λ_start` along `axis`, the backward
broadcasts `V_end` back, and the output layout keeps the axis at one level (resized, not dropped).
"""
ForgetfulSumStage(layout::GriddedLayout; axis::Symbol) =
    MarkovStage(layout; axis = axis,
                transition_matrix = ones(Float64, _axis_size(layout, axis), 1))
