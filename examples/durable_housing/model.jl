###############################################################
# Durable / housing (S,s) adjustment — steady state           #
###############################################################

# A household that holds a lumpy durable (housing) alongside liquid
# wealth, in the (S,s) tradition of Berger–Vavra (2015) and Díaz–
# Luengo-Prado (2010). The point of this example: the entire within-
# period problem is existing library stages, in time order, with
# **no bespoke household stage rolled here**. A leading `Move` Markov
# (the owner→renter moving shock) precedes the income shock —
#
#     Move ∘ IncomeShock ∘ BuyHome ∘ Receipt ∘ UserCost ∘ ConsumptionSavings
#
# `Move`        — `MarkovStage` on the housing axis: owners are ejected to
#                 the renter level w.p. `π_move`, keeping the buy choice live.
# `IncomeShock` — `MarkovStage` on the income axis.
# `BuyHome`     — `BuyHomeStage` on the housing axis `:h`. A renter
#                 (h-index 1) may move to any owned size or stay renting;
#                 an owner is GATED to keep its own size. That gate IS the
#                 (S,s) inaction region — only renters re-optimise the
#                 stock; owners sit until they (here, via a Markov exit
#                 shock) return to the renter level. The buy decision is on
#                 beginning-of-period assets, before income is received.
# `Receipt`     — `WealthChangeStage` `b ↦ (1+r)·b + w·y` (cash-on-hand from
#                 financial wealth plus labour income). Placed after the buy so
#                 every cell entering savings has income credited and the
#                 wealth-grid floor is strictly feasible (`c > 0`) — which the
#                 gated-owner branch of `BuyHomeStage` requires.
# `UserCost`    — `WealthChangeStage` `b ↦ b − u·h` charging the
#                 per-period housing user cost (Jorgenson rental price
#                 `u = (r + δ)·q` of the durable: foregone return plus
#                 depreciation/maintenance). It reads the post-buy
#                 `cell.h`, so every household on the owned slice pays it
#                 each period — exactly the right charge for a *user-cost*
#                 durable formulation (Díaz–Luengo-Prado's per-period
#                 maintenance + depreciation), and the reason the cost is a
#                 *following* `WealthChangeStage` rather than baked into the
#                 choice (the choice stage carries only the gate, like
#                 `MigrationStage`'s move cost). A one-time stock-price
#                 `q·h` would instead need to distinguish a fresh buyer
#                 from a continuing owner — both land on the identical owned
#                 cell — which is not expressible from existing stages
#                 without a pre-buy-size axis no stage populates; the
#                 user-cost flow sidesteps that and is the standard
#                 Díaz–Luengo-Prado timing.
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period
#                 financial wealth `b'`; `c = b − b'`. Utility is over
#                 `c` and the housing SERVICE FLOW `s(h)`, folded in via
#                 `utility_axes = (:h,)` so the chosen size shifts the
#                 flow payoff (a renter consumes a rental service `s(0)`).
#
# Returns/wage are exogenous (partial equilibrium): no market to clear, so
# the "outer loop" is a single `solve_steady_state_given_env!`. The
# housing axis carries a renter level at index 1 (`h = 0`, owned sizes at
# `h ≥ 2`). An exogenous owner→renter "moving shock" on the housing axis
# (a `MarkovStage`) keeps the renter level populated so the buy choice
# stays live in the stationary distribution; without it, mass would
# absorb permanently into the owned states.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct DurableHousingParams
    β :: Float64       = 0.93
    σ :: Float64       = 2.0                      # CRRA over the consumption–housing composite
    ξ :: Float64       = 0.15                     # Cobb–Douglas housing-service weight
    r :: Float64       = 0.03                     # exogenous return on financial wealth
    w :: Float64       = 1.0                      # wage (income scale)
    s_rent :: Float64  = 0.30                     # rental service flow at the renter level (h = 0)
    q :: Float64       = 3.0                      # house price per unit of size (stock value)
    δ :: Float64       = 0.03                     # housing depreciation / maintenance rate
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    h_sizes :: Vector{Float64} = [0.0, 1.0, 2.0]  # index 1 = renter (h = 0), 2,3 = owned sizes
    π_move  :: Float64 = 0.06                     # per-period owner→renter "moving" shock
    N_w   :: Int       = 160
    w_min :: Float64   = 0.0
    w_max :: Float64   = 40.0
end

Base.Broadcast.broadcastable(p::DurableHousingParams) = Ref(p)

const durable_housing_params = DurableHousingParams()


# Utility #
#---------#

"""
Per-period flow utility of the consumption–housing composite. The composite is
Cobb–Douglas `c^(1−ξ)·s^ξ` in nondurable consumption `c` and housing service flow
`s`, put through CRRA. A renter (`h = 0`) gets the rental service `s_rent`; an owner
of size `h` gets a service flow equal to its size.
"""
function u_housing(c, h, p)
    c <= 0 && return -Inf
    s = h == 0.0 ? p.s_rent : h
    composite = c^(1 - p.ξ) * s^p.ξ
    return u_crra(composite, Val(p.σ))
end


# Household chain assembly #
#--------------------------#

"""
Build the durable-housing block
`Move ∘ IncomeShock ∘ BuyHome ∘ Receipt ∘ UserCost ∘ ConsumptionSavings`, with
`mean_wealth = ∫ wealth dΛ`, `mean_house = ∫ h dΛ`, and the homeownership rate
`own_rate = ∫ 1{h>0} dΛ` attached.
"""
function durable_housing_household(p = durable_housing_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
        :h      => Discrete(p.h_sizes),
    )

    # Owner→renter moving shock: owners flip to the renter level (index 1) w.p.
    # π_move each period; renters stay renters.
    n_h = length(p.h_sizes)
    P_h = zeros(n_h, n_h)
    P_h[1, 1] = 1.0
    for h in 2:n_h
        P_h[h, 1]  = p.π_move
        P_h[h, h] += 1 - p.π_move
    end
    move = MarkovStage(layout; axis = :h, transition_matrix = P_h)

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout) # defaults: (; axis = :wealth)
    buy      = BuyHomeStage(layout) # defaults: (; axis = :h, renter_index = 1, flow_payoff = nothing)
    usercost = WealthChangeStage(layout;
        wealth_post = (; wealth, h, env) -> wealth - (env.r + env.δ) * env.q * h) # defaults: (; axis = :wealth) — user cost u·h
    savings  = ConsumptionSavingsStage(layout;
        β            = p.β,
        utility      = (cell, c) -> u_housing(c, cell.h, p),
        utility_axes = (:h,),                  # flow utility reads the chosen housing size
    ) # defaults: (; axis = :wealth, skip_monotonicity_check = false)

    # Buy precedes receipt so every cell entering the savings problem has had
    # income credited — the wealth-grid floor is then strictly feasible (`c > 0`),
    # which the gated owner branch of `BuyHomeStage` requires (a gated owner has
    # only its own continuation).
    hh = move ∘ shock ∘ buy ∘ receipt ∘ usercost ∘ savings
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        mean_house  = at_end(integrand = :h,      reduce = sum),
        own_rate    = at_end(integrand = (; h) -> h > 0.0 ? 1.0 : 0.0,
                             reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"Exogenous env: financial return, wage, house price, and depreciation (no market to clear)."
durable_housing_env(p = durable_housing_params) = (; r = p.r, w = p.w, q = p.q, δ = p.δ)
