"""Directed job search (Moen; Menzio–Shi) over a submarket `axis` — a [`LogitChoiceStage`](@ref) whose `search_cost[i, j]` prices aiming `i → j`."""
DirectedSearchStage(start_layout::GriddedLayout, end_layout::GriddedLayout = start_layout;
                    axis::Symbol=:submarket, search_cost, ε=1.0) =
    LogitChoiceStage(start_layout, end_layout; axis = axis, cost_matrix = search_cost, ε)
