# Migration — domain-named sugar over LogitChoiceStage (transition-cost logit
# on a location axis). A thin wrapper: all the work is in logit_choice.jl.

"""
Migration over a location axis — a `LogitChoiceStage` with `migration_cost[i, j]`
the cost of moving `i → j`. The cost takes any of `LogitChoiceStage`'s forms (a
matrix, a `FromEnv`, or a dep-declaring closure for moves restricted to a subset
of agents). Location-specific values enter through the destination `V_end` via
stages composed after the move, not here.
"""
MigrationStage(layout::GriddedLayout; axis::Symbol=:location, migration_cost, ε=1.0) =
    LogitChoiceStage(layout; axis=axis, cost_matrix=migration_cost, ε=ε)
