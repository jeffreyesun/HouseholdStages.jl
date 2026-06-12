# Buy home — domain-named sugar over ArgmaxStage (a gated discrete choice on a
# housing axis). A thin wrapper: all the work is in argmax.jl. The purchase
# *price* is not deducted here; it composes as a following WealthChangeStage.

"""
Index of housing value `action` within the axis `values` — the chosen
housing index for `next_state_idx`.
"""
_housing_index(values, action) = findfirst(==(action), values)::Int

"""
Buy-home stage — an [`ArgmaxStage`](@ref) on a housing axis encoding the
homebuying choice. A cell at the renter level (`renter_index`, default `1`)
may choose any housing size; a homeowner (`h ≥ 2`) may only keep its own `h`.

The purchase price is **not** deducted here — like [`MigrationStage`](@ref)
carrying only the move cost, the wealth consequence composes as a following
[`WealthChangeStage`](@ref) that reads `cell.<housing_axis>`. `flow_payoff`
defaults to zero on available actions; pass one to add a per-size shifter.
"""
function BuyHomeStage(layout::GriddedLayout; housing_axis::Symbol=:h,
                      renter_index::Int=1, flow_payoff=nothing)
    values = axisvalues(layout.axes[axis_position(layout, housing_axis)])
    renter = values[renter_index]
    own    = getproperty                       # cell.<housing_axis>
    fp = flow_payoff === nothing ?
        # Renters: every action available (payoff 0). Owners: only their own h.
        ((cell, action; env) ->
            own(cell, housing_axis) == renter ? 0.0 :
            (action == own(cell, housing_axis) ? 0.0 : -Inf)) :
        # Custom payoff, but still gate owners to their own h (pass-through).
        ((cell, action; env) ->
            own(cell, housing_axis) == renter ? flow_payoff(cell, action; env) :
            (action == own(cell, housing_axis) ? flow_payoff(cell, action; env) : -Inf))
    return ArgmaxStage(layout;
        choice_axis    = housing_axis,
        flow_payoff    = fp,
        next_state_idx = (cell, action) -> _housing_index(values, action),
    )
end
