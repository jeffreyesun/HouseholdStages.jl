"""turn the infeasibility predicate `(; ax…[, env]) -> Bool` into flow utility, `-Inf` where it holds."""
function _feasibility_utility(infeasible)
    kws  = _closure_kwargs_raw(infeasible)
    env  = :env in kws
    axes = Tuple(filter(!=(:env), kws))
    return DepClosure(nt -> (infeasible(; nt...) ? -Inf : 0.0), axes, env)
end

"""
Rule cells out of the state space — a [`UtilityStage`](@ref) with flow utility `-Inf` on infeasible
cells and `0` elsewhere; the forward pass is the identity on Λ. `infeasible` is an
`AbstractArray{Bool}` of layout shape or a predicate `(; ax…[, env]) -> Bool` re-evaluated at every
`backward!`.
"""
function BorrowingConstraintStage(layout::GriddedLayout; infeasible)
    if infeasible isa AbstractArray{Bool}
        @assert size(infeasible) == layout_size(layout) "infeasible mask must match layout shape"
        return UtilityStage(layout; utility = ifelse.(infeasible, -Inf, 0.0))
    else
        return UtilityStage(layout; utility = _feasibility_utility(infeasible))
    end
end
