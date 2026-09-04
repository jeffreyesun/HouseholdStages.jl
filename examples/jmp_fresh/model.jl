######################################################################
# JMP — "Continuation Value is All You Need" household block          #
######################################################################

# The household side of the neural-VFI testbed of "Continuation Value
# is All You Need" (the JMP model: a spatial heterogeneous-agent GE
# model with aggregate shocks — households choose location, housing,
# and consumption/savings under idiosyncratic income + aggregate
# shocks). The full model runs N_K=129, N_Z=5, N_H=7, N_LOC=2447
# (~11M points) on GPU; this example is a CPU steady state on a
# handful of points per axis, whose ONLY purpose is to show that the
# JMP within-period **household block** is expressible as a `∘`
# composition of stages the library already ships — no `@definestage`,
# no bespoke kernel.
#
# The within-period timing is read straight off the JMP backward sweep
# `iterate_V_backward!` (price → sell → move → buy → income → consume →
# shock), i.e. in TIME order (the leftmost stage acts first):
#
#     hh = price ∘ sell ∘ move ∘ buy ∘ income ∘ savings ∘ shock
#
# stage-by-stage map to the JMP operation it implements:
#
# `price`   `AssetPriceChangeStage(:h)` — house revaluation
#           `wealth ↦ wealth + (q − q_last)·h`. Existing owners gain
#           the capital gain on their housing stock. In steady state
#           `q_last = q`, so it is the identity here; it is present for
#           structural fidelity (it carries the aggregate-shock channel
#           in the full model). JMP `get_V_price` / `get_wealth_postprice`.
# `sell`    `SellHomeStage(:h)` — keep-vs-sell choice on the housing
#           axis. An owner (`h ≥ 2`) may keep its size or sell to the
#           renter level (h-index 1); a renter has nothing to sell.
#           JMP `get_V_choosesell`. (The realtor fee `ϕ·q·h` on the
#           sale is NOT modelled — see README; it is the one JMP
#           household operation that does not compose cleanly.)
# `move`    `MigrationStage(:location)` — Gumbel/logit location choice
#           with a pairwise migration-cost matrix and taste-shock scale
#           `ε`. JMP `get_V_move` / `get_λ_postmove`.
# `buy`     `BuyHomeStage(:h)` — homebuying choice on the housing axis.
#           A renter (h-index 1) may buy any size or stay renting; an
#           owner is gated to keep its own size (the inaction region).
#           JMP `get_V_choosebuy` / `get_P_buy`.
# `income`  `WealthChangeStage(:wealth)` with the JMP budget
#           `get_income`: interest on net bondholdings `(wealth − q·h)`
#           at the bond rate `r` if positive else the mortgage rate
#           `r_m`, plus labour earnings `A·z` and net rental income
#           `ρ·h − (ρ·χ + δ)·h`. JMP `get_V_income` / `get_income`.
# `savings` `ConsumptionSavingsStage(:wealth)` — pick next-period total
#           wealth; `c = cash_on_hand − wealth'`. Flow utility is the
#           JMP log-CES indirect utility `log(c / P(ρ))`, the dual price
#           index `P(ρ)` location-specific through the rent `ρ`. JMP
#           `get_V_consume` / `get_indirect_u`.
# `shock`   `MarkovStage(:income)` — idiosyncratic income (z) Markov
#           transition, applied at period end. JMP `get_V_preshock`.
#
# Wealth is TOTAL wealth (it includes the house value `q·h`); the
# mortgage is implicit, as negative net bondholdings `wealth − q·h < 0`
# charged at `r_m`. This matches the JMP accounting in `get_interest`.
# Two locations differ in productivity `A` and rent `ρ`. Returns,
# rents, and the house price are exogenous here (partial equilibrium):
# the GE tatonnement / aggregate-shock loop that closes the model lives
# outside the household block and is not built in this example.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct JMPParams
    β :: Float64       = 0.95                       # decadal-ish discount factor
    γ :: Float64       = 0.20                       # CES weight on housing services
    σ :: Float64       = 1.50                       # CES elasticity of substitution (goods vs housing)
    r :: Float64       = 0.03                       # bond / liquid return
    r_m :: Float64     = 0.06                       # mortgage rate on negative net bondholdings
    δ :: Float64       = 0.02                       # housing depreciation
    χ :: Float64       = 0.10                       # maintenance, as a fraction of rent
    q :: Float64       = 2.0                        # house price per unit of size (stock value)
    # Location-specific productivity and rent (two locations).
    A_home   :: Float64 = 1.20
    A_abroad :: Float64 = 1.00
    ρ_home   :: Float64 = 0.16
    ρ_abroad :: Float64 = 0.12
    # Idiosyncratic income (z) grid + Markov transition.
    z_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_z    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    # Housing sizes: index 1 = renter (h = 0), 2,3 = owned sizes.
    h_sizes :: Vector{Float64} = [0.0, 1.0, 2.0]
    # Migration.
    migration_cost :: Float64 = 0.30
    ε_logit        :: Float64 = 4.0
    # Wealth grid (small, log-spaced — dense near zero).
    N_w   :: Int     = 12
    w_min :: Float64 = 0.0
    w_max :: Float64 = 20.0
end

Base.Broadcast.broadcastable(p::JMPParams) = Ref(p)

const jmp_params = JMPParams()


# Preferences — JMP log-CES indirect utility #
#--------------------------------------------#

"""
CES dual price index for the goods/housing bundle at rent `ρ` (goods are the
numéraire): `P(ρ) = ((1−γ)^σ + γ^σ·ρ^(1−σ))^(1/(1−σ))`. JMP `get_price_index`.
"""
get_price_index(ρ, γ, σ) = ((1 - γ)^σ + γ^σ * ρ^(1 - σ))^(1 / (1 - σ))

"""
Log indirect utility of total consumption expenditure `c` at rent `ρ`: the
household splits `c` optimally over goods and rented housing services, so flow
felicity is `log(c / P(ρ))`. JMP `get_indirect_u`. `-Inf` when `c ≤ 0`.
"""
function jmp_indirect_u(c, ρ, γ, σ)
    c <= 0 && return -Inf
    return log(c / get_price_index(ρ, γ, σ))
end

"""
Location-specific rent: `:home` vs `:abroad` read off the env.
"""
rent_at(location, env) = location == :home ? env.ρ_home : env.ρ_abroad

"""
Location-specific labour productivity: `:home` vs `:abroad` read off the env.
"""
productivity_at(location, env) = location == :home ? env.A_home : env.A_abroad


# Household chain assembly #
#--------------------------#

"""
Build the JMP within-period household block as a composition of existing stages
(time order, leftmost first):

    price ∘ sell ∘ move ∘ buy ∘ income ∘ savings ∘ shock

with mass, mean wealth, homeownership rate, mean house size, and the home-location
population share attached as moments. No `@definestage`, no bespoke kernel — the
whole JMP household side in the shipped stage vocabulary.
"""
function jmp_household(p = jmp_params)
    layout = GriddedLayout(
        :wealth   => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income   => Discrete(p.z_grid),
        :h        => Discrete(p.h_sizes),
        :location => Discrete([:home, :abroad]),
    )

    # (2) House revaluation — owners gain (q − q_last)·h. Identity in steady state.
    price = AssetPriceChangeStage(layout; holdings_axis = :h)

    # (3) Keep-vs-sell on the housing axis (owner → renter level, or keep).
    sell = SellHomeStage(layout; axis = :h)

    # (4) Location choice — Gumbel logit with a pairwise move cost.
    move = MigrationStage(layout; axis = :location,
        migration_cost = [0.0              p.migration_cost;
                          p.migration_cost 0.0],
        ε              = p.ε_logit)

    # (5) Homebuying — renter may buy any size; owner gated to keep its size.
    buy = BuyHomeStage(layout; axis = :h)

    # (6) Income receipt — the JMP budget `get_income`, on total wealth.
    income = WealthChangeStage(layout; axis = :wealth,
        wealth_post = function (; wealth, h, income, location, env)
            A           = productivity_at(location, env)
            ρ           = rent_at(location, env)
            net_bond    = wealth - env.q * h
            interest    = net_bond * (net_bond >= 0 ? env.r : env.r_m)
            noninterest = A * income + ρ * h - (ρ * env.χ + env.δ) * h
            return wealth + interest + noninterest
        end)

    # (7) Consumption-savings — pick next total wealth; log-CES indirect utility.
    savings = ConsumptionSavingsStage(layout; axis = :wealth,
        β            = p.β,
        utility      = (cell, c; env) -> jmp_indirect_u(c, rent_at(cell.location, env), p.γ, p.σ),
        utility_axes = (:location,))

    # (8) Idiosyncratic income (z) Markov transition at period end.
    shock = MarkovStage(layout; axis = :income, transition_matrix = p.P_z)

    hh = price ∘ sell ∘ move ∘ buy ∘ income ∘ savings ∘ shock
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        mean_house  = at_end(integrand = :h,      reduce = sum),
        own_rate    = at_end(integrand = (; h) -> h > 0.0 ? 1.0 : 0.0, reduce = sum),
        pop_home    = at_end(integrand = (; location) -> location == :home ? 1.0 : 0.0,
                             reduce = sum))
end


# Exogenous env (partial equilibrium — no market cleared here) #
#--------------------------------------------------------------#

"""
Exogenous prices for the steady-state solve: bond/mortgage rates, house price
(with `q_last = q`, so the revaluation stage is the identity), depreciation,
maintenance, and the two locations' productivity and rent.
"""
jmp_env(p = jmp_params) = (; r = p.r, r_m = p.r_m, q = p.q, q_last = p.q,
                             δ = p.δ, χ = p.χ,
                             A_home = p.A_home, A_abroad = p.A_abroad,
                             ρ_home = p.ρ_home, ρ_abroad = p.ρ_abroad)
