"""
Logit choice over `axis` when the payoff depends on the destination cell — a
[`LogitChoiceStage`](@ref) after a [`UtilityStage`](@ref), valuing `i → j` in the rest of the state
`s` at `−cost_matrix[i, j] + utility(j, s) + V_end[j, s]`, smoothed at scale `ε`. `utility` is a
closure of the layout axes it names as keyword arguments.
"""
LogitUtilityStage(start_layout::GriddedLayout, end_layout::GriddedLayout = start_layout;
                  axis::Symbol, cost_matrix::AbstractMatrix, utility, ε=1.0) =
    LogitChoiceStage(start_layout, end_layout; axis, cost_matrix, ε) ∘
    UtilityStage(end_layout; utility)
