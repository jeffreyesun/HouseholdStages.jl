# Borrowing constraint — state feasibility as a UtilityStage paying `-Inf` on
# infeasible cells, `0` elsewhere. All the work is in utility.jl.

"""
State-feasibility stage — a [`UtilityStage`](@ref) adding `-Inf` flow utility
on infeasible cells and `0` elsewhere, so `backward!` sets `V_start = -Inf`
there and is the identity on `V_end` otherwise. Forward is the identity on Λ.
`infeasible` is either an `AbstractArray{Bool}` of layout shape (baked into a
constant `0`/`-Inf` utility table) or a `(cell; env) -> Bool` closure
(re-evaluated each `backward!`, so feasibility may depend on `env`).
"""
function BorrowingConstraintStage(layout::GriddedLayout; infeasible)
    if infeasible isa AbstractArray{Bool}
        @assert size(infeasible) == layout_size(layout) "infeasible mask must match layout shape"
        return UtilityStage(layout; utility = ifelse.(infeasible, -Inf, 0.0))
    else
        return UtilityStage(layout; utility = (cell; env) -> infeasible(cell; env) ? -Inf : 0.0)
    end
end
