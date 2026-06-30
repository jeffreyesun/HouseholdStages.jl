# Continuous investment in a stock (human capital, durables, health) — domain-named sugar over
# ArgmaxStage on the stock axis, reward = stock payoff − convex cost of moving the stock.
# All the work is in continuous_argmax.jl; these wrappers only assemble the `(after, before)` reward
# closure. The policy (next stock) is assumed monotone in the current stock (the usual Ben-Porath /
# (S,s)-smooth supermodular case); pass `monotone_search`/`assume_monotone` through to override.

"""
Investment in a productive stock — a [`ArgmaxStage`](@ref) on a capital axis `axis` (human capital,
health, knowledge, physical capital). From stock `h` the agent picks next stock `h'`, paying
`effort_cost` on gross investment `i = h' − (1−δ)h` (`δ = depreciation`) and earning a production
flow `production(h; env)`:

    reward[h', h] = production(h; env) − effort_cost(max(h' − (1−δ)h, 0); env).

`production` and `effort_cost` are `(value; env)` closures. The canonical instance is Ben-Porath
(1967) human-capital investment; Grossman health investment is the same stage with `production` a
survival/health flow (see the note in this file). Contrast `DurableAdjustmentStage` — a consumption
durable yielding a service flow, adjusted at a cost.
"""
function CapitalInvestmentStage(layout::GriddedLayout; axis::Symbol=:h, β=1.0,
                        production, effort_cost, depreciation::Real=0.0,
                        monotone_search::Symbol=:divide_conquer, assume_monotone::Bool=false)
    # Start-and-end reward closure on the operative `axis` (origin stock `h`, destination `h′` taken
    # POSITIONALLY), lowered to a matrix source by `to_matrix_source` (end-goal §4.1) — no bespoke
    # `(after, before)` matrix.
    payoff = (h, h′; env) -> production(h; env) -
                             effort_cost(max(h′ - (1 - depreciation) * h, zero(h)); env)
    reward = to_matrix_source(payoff, layout, axis)
    # Discount composed as its own stage (end-goal §1): argmax solves `max(reward + V_end)`, the
    # `TimeDiscountingStage` supplies `β·V_end` first in the backward sweep (`∘` is time-ordered).
    return ArgmaxStage(layout; reward, axis = axis, search = monotone_search, assume_monotone) ∘
           TimeDiscountingStage(layout; β)
end

# Grossman health investment (drift-only) is `CapitalInvestmentStage` with `production` the health/survival
# flow and `effort_cost` the medical-expenditure cost — no separate constructor needed.

"""
Smooth durable adjustment — a [`ArgmaxStage`](@ref) on a durable-stock axis. From stock
`d` the agent picks next stock `d'`, earning service flow `service(d'; env)` net of a convex
adjustment cost `adjustment_cost(d' − d; env)`:

    reward[d', d] = service(d'; env) − adjustment_cost(d' − d; env).
"""
function DurableAdjustmentStage(layout::GriddedLayout; axis::Symbol=:durable, β=1.0,
                                service, adjustment_cost,
                                monotone_search::Symbol=:divide_conquer, assume_monotone::Bool=false)
    # Start-and-end reward closure (origin stock `d`, destination `d′` taken POSITIONALLY), lowered by
    # `to_matrix_source` (end-goal §4.1) — no bespoke `(after, before)` matrix.
    payoff = (d, d′; env) -> service(d′; env) - adjustment_cost(d′ - d; env)
    reward = to_matrix_source(payoff, layout, axis)
    # Discount composed as its own stage (end-goal §1), as in `CapitalInvestmentStage` above.
    return ArgmaxStage(layout; reward, axis = axis, search = monotone_search, assume_monotone) ∘
           TimeDiscountingStage(layout; β)
end
