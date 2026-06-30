# ============================================================================
# Logit choice with a state-dependent payoff = LogitChoiceStage ∘ UtilityStage
# ============================================================================
#
# `LogitChoiceStage` deliberately has no general per-action payoff closure: its
# only parameter is the origin→destination cost matrix `C[i,j]`. That is *not*
# a limitation, because any payoff that depends on the *destination* (and the
# rest of the state) is V-additive — it just shifts the continuation value the
# logit sees. So you recover a fully state-dependent discrete choice by
# composing a `UtilityStage` (which adds `u(cell)` to V) before the logit:
#
#     choice ∘ value      # backward: value adds u to V_end, then choice does
#                         # the logit over (V_end + u)
#
# The one term this composition *cannot* absorb is an **origin-dependent**
# payoff — utility adds to V at the destination, so it can't see where you came
# from. That origin→destination friction is exactly what `C[i,j]` is for. Hence
# the split: cost matrix on the logit, everything else in a `UtilityStage`.
#
# Run: julia --project=. examples/logit_utility_composition.jl

using HouseholdStages

# State: income z (2 levels) × location ℓ (3 locations). The choice is location.
layout = GriddedLayout(
    :income   => Discrete([0.6, 1.4]),
    :location => Discrete([:A, :B, :C]),
)

ε = 0.8
C = [0.0 0.5 0.7;     # cost of moving origin → destination (origin-dependent!)
     0.5 0.0 0.5;
     0.7 0.5 0.0]

# A destination value that depends on the *whole* cell: location B is worth more
# to high-income households, C more to low-income — a state-dependent payoff
# that a bare cost matrix can't express, but a UtilityStage can.
dest_value(; location, income) = location === :B ? 0.8 * income :
                                 location === :C ? 0.6 / income : 0.0

choice = LogitChoiceStage(layout; axis = :location, cost_matrix = C, ε = ε)
value  = UtilityStage(layout; utility = dest_value)
chain  = choice ∘ value          # logit over (V_end + dest_value)

# A terminal continuation value to push back through the chain.
V_end = Float64[0.3 * zi + 0.1 * li for zi in 1:2, li in 1:3]
V_pre = backward!(chain, V_end, NamedTuple())

# Closed form for the *general* state-dependent logit we just built by
# composition: V_pre[z, i] = ε·log Σ_j exp((−C[i,j] + u(j,z) + V_end[z,j]) / ε).
z_grid = [0.6, 1.4]; locs = [:A, :B, :C]
u(j, z) = locs[j] === :B ? 0.8 * z : locs[j] === :C ? 0.6 / z : 0.0
expected = [ε * log(sum(exp((-C[i, j] + u(j, z_grid[zi]) + V_end[zi, j]) / ε) for j in 1:3))
            for zi in 1:2, i in 1:3]

@assert isapprox(V_pre, expected; atol = 1e-12)

# This composition is packaged as the derived `LogitUtilityStage`, so the same
# state-dependent logit is one constructor call — `choice ∘ value` under the hood.
packaged = LogitUtilityStage(layout; axis = :location, cost_matrix = C,
                             utility = dest_value, ε = ε)
V_pre_packaged = backward!(packaged, V_end, NamedTuple())
@assert V_pre_packaged == V_pre

println("LogitChoiceStage ∘ UtilityStage reproduces the state-dependent logit:")
println("  max |V_pre − closed form| = ", maximum(abs, V_pre .- expected))
println("  LogitUtilityStage matches the manual composition exactly: ",
        V_pre_packaged == V_pre)
println("\nThe destination payoff u(location, income) entered through V via the")
println("UtilityStage; only the origin-dependent cost C[i,j] lives on the logit.")
