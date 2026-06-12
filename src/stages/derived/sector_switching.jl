# Sector switching — domain-named sugar over LogitChoiceStage (transition-cost
# logit on a sector/occupation axis). Same mechanics as MigrationStage, a
# different name; a thin wrapper to show how cheaply the primitive specialises.

"""
Sector-switching stage — a [`LogitChoiceStage`](@ref) on a sector (or
occupation) axis: agents pay `switching_cost[i, j]` to move from sector `i`
to `j` (zero diagonal = free to stay). Same mechanics as
[`MigrationStage`](@ref); sector-specific values (and any destination amenity)
enter through `V_end` via composed `UtilityStage`s.
"""
SectorSwitchingStage(layout::GriddedLayout; sector_axis::Symbol=:sector, switching_cost, ε=1.0) =
    LogitChoiceStage(layout; choice_axis=sector_axis, cost_matrix=switching_cost, ε=ε)
