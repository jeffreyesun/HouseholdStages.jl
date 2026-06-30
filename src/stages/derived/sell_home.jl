# Sell home — domain-named sugar over ArgmaxStage (a gated discrete choice on a
# housing axis). A thin wrapper: all the work is in argmax.jl. The realtor fee /
# sale proceeds are not applied here; they compose as a following
# WealthChangeStage.

"""
Sell-home stage — an [`ArgmaxStage`](@ref) on a housing axis encoding the keep-vs-sell choice. A
homeowner (`h ≥ 2`) chooses between keeping its own `h` and selling to the renter level
(`renter_index`, default `1`); a renter has nothing to sell and passes through. The sale proceeds /
realtor fee are **not** applied here — like [`MigrationStage`](@ref) carrying only the move cost,
follow this stage with a [`WealthChangeStage`](@ref) that credits the sellers (now at the renter
level). `flow_payoff`, if given, is a per-action shifter `(housing_value; env) -> Float64`.
"""
#TODO Sellers and pre-existing renters both sit on the renter slice after the choice — identifying
# sellers for the wealth credit needs the pre-sell size carried on another axis, or the fee folded
# into the buy stage's price.
function SellHomeStage(layout::GriddedLayout; axis::Symbol=:h,
                       renter_index::Int=1, flow_payoff=nothing)
    # Renters may only stay renters; owners may keep their size or sell to the renter level. The
    # renter level is identified by its grid VALUE (the axis is injective; cf. the `h == 0.0` idiom
    # in the housing examples), so the index-positional gate reads over grid values.
    renter = axis_grid(layout, axis)[renter_index]
    avail  = (origin, destination) -> origin == renter ? (destination == renter) :
                                      (destination == origin || destination == renter)
    reward = _gate_reward(layout, axis, avail, flow_payoff)
    return ArgmaxStage(layout; axis = axis, reward)
end
