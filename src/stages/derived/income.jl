"""
Receipt of income — a [`WealthChangeStage`](@ref) moving each cell's wealth to the cash-on-hand
`(1 + env.r)·wealth + env.w·income`. Pass `wealth_post` to use a different budget map.
"""
function IncomeStage(start_layout::GriddedLayout, end_layout::GriddedLayout = start_layout;
                     axis::Symbol = :wealth, income_axis::Symbol = :income,
                     wealth_post = nothing)
    wp = wealth_post === nothing ?
        DepClosure((axis, income_axis), true) do nt
            (1 + nt.env.r) * nt[axis] + nt.env.w * nt[income_axis]
        end : wealth_post
    return WealthChangeStage(start_layout, end_layout; wealth_post = wp, axis)
end
