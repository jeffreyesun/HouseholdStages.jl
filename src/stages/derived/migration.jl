"""Migration over a location `axis` — a [`LogitChoiceStage`](@ref) charging `migration_cost[i, j]` to move `i → j`."""
MigrationStage(start_layout::GriddedLayout, end_layout::GriddedLayout = start_layout;
               axis::Symbol=:location, migration_cost, ε=1.0) =
    LogitChoiceStage(start_layout, end_layout; axis=axis, cost_matrix=migration_cost, ε=ε)
