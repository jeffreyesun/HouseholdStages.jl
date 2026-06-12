# Sell home — domain-named sugar over ArgmaxStage (a gated discrete choice on a
# housing axis). A thin wrapper: all the work is in argmax.jl. The realtor fee /
# sale proceeds are not applied here; they compose as a following
# WealthChangeStage.

"""
Sell-home stage — an [`ArgmaxStage`](@ref) on a housing axis encoding the
keep-vs-sell choice. A homeowner (`h ≥ 2`) chooses between keeping its own
`h` and selling to the renter level (`renter_index`, default `1`); a renter
has nothing to sell and passes through.

The sale proceeds / realtor fee are **not** applied here — like
[`MigrationStage`](@ref) carrying only the move cost, follow this stage with
a [`WealthChangeStage`](@ref) that credits sellers (the cells just landed at
the renter level). `flow_payoff` defaults to zero on the {keep, sell}
actions; pass one to add a per-action shifter.

#TODO Sellers and pre-existing renters both sit on the renter slice after the
choice — identifying sellers for the wealth credit needs the pre-sell size
carried on another axis, or the fee folded into the buy stage's price.
"""
function SellHomeStage(layout::GriddedLayout; housing_axis::Symbol=:h,
                       renter_index::Int=1, flow_payoff=nothing)
    values = axisvalues(layout.axes[axis_position(layout, housing_axis)])
    renter = values[renter_index]
    own    = getproperty                       # cell.<housing_axis>
    # Available actions per cell: renters → {stay renter}; owners → {keep, sell}.
    available = (cell, action) -> begin
        h = own(cell, housing_axis)
        h == renter ? (action == renter) : (action == h || action == renter)
    end
    fp = flow_payoff === nothing ?
        ((cell, action; env) -> available(cell, action) ? 0.0 : -Inf) :
        ((cell, action; env) -> available(cell, action) ? flow_payoff(cell, action; env) : -Inf)
    return ArgmaxStage(layout;
        choice_axis    = housing_axis,
        flow_payoff    = fp,
        next_state_idx = (cell, action) -> _housing_index(values, action),
    )
end
