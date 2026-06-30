# Directed search (Moen; Menzio–Shi) — domain-named sugar over LogitChoiceStage (a soft choice
# over submarkets). A thin wrapper, like MigrationStage: the fill-probability / wage tradeoff that
# distinguishes submarkets enters through the destination value `V_end` (set by the stages composed
# after), and the application/search friction through `search_cost`.

"""
Directed job search over a submarket `axis` — a [`LogitChoiceStage`](@ref) with `search_cost[i, j]`
the cost of aiming from submarket `i` at submarket `j` (any of the cost forms: a matrix, a
`FromEnv`, or a dep-declaring closure). The fill-probability vs. wage tradeoff lives in each
submarket's continuation value, not here. With `ε → 0` this is the hard directed choice (argmax);
`ε > 0` smooths it (the Menzio–Shi / Choo–Siow logit form).
"""
DirectedSearchStage(layout::GriddedLayout; axis::Symbol=:submarket, search_cost, ε=1.0) =
    LogitChoiceStage(layout; axis = axis, cost_matrix = search_cost, ε)
