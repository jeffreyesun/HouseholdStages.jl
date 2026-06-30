# IncomeStage — the standard cash-on-hand receipt, a domain wrapper over WealthChangeStage.

"""
Cash-on-hand receipt: each cell's wealth moves to `(1+r)·wealth + w·income`, the standard
budget map — a domain wrapper over [`WealthChangeStage`](@ref). `axis` names the wealth axis
written, `income_axis` the shock axis read (default `:income`, so a non-`:income` shock axis
just sets `income_axis`); pass `wealth_post` for a fully custom budget.
"""
function IncomeStage(layout::GriddedLayout; axis::Symbol = :wealth, income_axis::Symbol = :income,
                     wealth_post = nothing)
    wp = wealth_post === nothing ?
        DepClosure((axis, income_axis), true) do nt
            (1 + nt.env.r) * nt[axis] + nt.env.w * nt[income_axis]
        end : wealth_post
    return WealthChangeStage(layout; wealth_post = wp, axis)
end
