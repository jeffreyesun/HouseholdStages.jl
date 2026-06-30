# Buy home — domain-named sugar over ArgmaxStage (a gated discrete choice on a
# housing axis). A thin wrapper: all the work is in argmax.jl. The purchase
# *price* is not deducted here; it composes as a following WealthChangeStage.

"""
Lower a value-gated housing/tenure choice to a reward matrix source via `to_matrix_source`
(end-goal §4.1): the start-and-end `payoff` takes the origin/destination axis GRID VALUES
POSITIONALLY and gates inline — `avail(origin, destination)` infeasible moves score `-Inf`, available
ones the per-size `flow_payoff` shifter (`0` if none). No bespoke `(after, before)` matrix.
"""
function _gate_reward(layout::GriddedLayout, axis::Symbol, avail, flow_payoff)
    payoff = (origin, destination; env) ->
        avail(origin, destination) ? (flow_payoff === nothing ? 0.0 : flow_payoff(destination; env)) : -Inf
    return to_matrix_source(payoff, layout, axis)
end

"""
Buy-home stage — an [`ArgmaxStage`](@ref) on a housing axis encoding the
homebuying choice. A cell at the renter level (`renter_index`, default `1`)
may choose any housing size; a homeowner (`h ≥ 2`) may only keep its own `h`.

The reward is the `(after, before)` gate matrix on the housing axis (available moves score `0`,
unavailable `-Inf`). The purchase price is **not** deducted here — like [`MigrationStage`](@ref)
carrying only the move cost, the wealth consequence composes as a following
[`WealthChangeStage`](@ref) that reads `cell.<axis>`. `flow_payoff`, if given, is a
per-size shifter `(housing_value; env) -> Float64` on the available entries (a destination payoff
is V-additive, so it may equally be a composed [`UtilityStage`](@ref)).
"""
function BuyHomeStage(layout::GriddedLayout; axis::Symbol=:h,
                      renter_index::Int=1, flow_payoff=nothing)
    # The renter level is identified by its grid VALUE (the axis is injective; cf. the `h == 0.0`
    # idiom in the housing examples), so the index-positional gate reads over grid values.
    renter = axis_grid(layout, axis)[renter_index]
    avail  = (origin, destination) -> origin == renter || destination == origin
    reward = _gate_reward(layout, axis, avail, flow_payoff)
    return ArgmaxStage(layout; axis = axis, reward)
end
