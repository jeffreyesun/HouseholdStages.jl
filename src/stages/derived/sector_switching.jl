"""Sector or occupation switching — a [`LogitChoiceStage`](@ref) charging `switching_cost[i, j]` to move `i → j`."""
SectorSwitchingStage(start_layout::GriddedLayout, end_layout::GriddedLayout = start_layout;
                     axis::Symbol=:sector, switching_cost, ε=1.0) =
    LogitChoiceStage(start_layout, end_layout; axis=axis, cost_matrix=switching_cost, ε=ε)
