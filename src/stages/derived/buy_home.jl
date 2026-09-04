"""gated tenure reward on `axis` — `-Inf` where `avail` bars the move, else `flow_payoff(destination)` or `0`."""
function _gate_reward(layout::GriddedLayout, axis::Symbol, avail, flow_payoff)
    payoff = if _reads_env_declared(flow_payoff)
        (origin, destination; env) -> avail(origin, destination) ? flow_payoff(destination; env) : -Inf
    else
        (origin, destination) -> avail(origin, destination) ? (flow_payoff === nothing ? 0.0 : flow_payoff(destination)) : -Inf
    end
    return to_matrix_source(payoff, layout, layout, axis)
end

"""
The homebuying choice — an [`ArgmaxStage`](@ref) in which a cell at the renter level
(`renter_index`, default `1`) may choose any level of the housing `axis`, while an owner may only
keep its own. No money moves here. `flow_payoff`, if given, shifts the reward of each attainable
size and is called `flow_payoff(housing_value)`, or with `; env` if it declares it.
"""
function BuyHomeStage(layout::GriddedLayout; axis::Symbol=:h,
                      renter_index::Int=1, flow_payoff=nothing)
    renter = axis_grid(layout, axis)[renter_index]
    avail  = (origin, destination) -> origin == renter || destination == origin
    reward = _gate_reward(layout, axis, avail, flow_payoff)
    return ArgmaxStage(layout; axis = axis, reward)
end
