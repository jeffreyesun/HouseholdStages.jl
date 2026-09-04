#TODO Sellers and pre-existing renters both sit on the renter slice after the choice — identifying
# sellers for the wealth credit needs the pre-sell size carried on another axis, or the fee folded
# into the buy stage's price.
"""
Keep-or-sell choice — an [`ArgmaxStage`](@ref) in which an owner picks between keeping its current
level of the housing `axis` and moving to `renter_index`, while a renter passes through. No money
moves here. `flow_payoff`, if given, shifts the reward by `(housing_value; env) -> Float64`.
"""
function SellHomeStage(layout::GriddedLayout; axis::Symbol=:h,
                       renter_index::Int=1, flow_payoff=nothing)
    renter = axis_grid(layout, axis)[renter_index]
    avail  = (origin, destination) -> origin == renter ? (destination == renter) :
                                      (destination == origin || destination == renter)
    reward = _gate_reward(layout, axis, avail, flow_payoff)
    return ArgmaxStage(layout; axis = axis, reward)
end
