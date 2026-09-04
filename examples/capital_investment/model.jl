######################################################################
# Capital investment with convex adjustment cost — Cooper–Haltiwanger #
######################################################################

# A firm holding capital `k` and an idiosyncratic profitability state `z`,
# investing subject to a CONVEX adjustment cost (the convex component of
# Cooper–Haltiwanger 2006). The point of this §1 example: the within-period
# firm block is EXISTING library stages, in time order, with NO bespoke stage —
#
#     ProfitShock ∘ Profit ∘ Invest
#   = MarkovStage(:z) ∘ UtilityStage(z·k^α) ∘ CapitalInvestmentStage(:k)
#
# `ProfitShock` — `MarkovStage` on the profitability axis `:z`: the AR(1)-in-logs
#                 idiosyncratic productivity transitions (Rouwenhorst-discretized).
# `Profit`      — `UtilityStage` adding the flow operating profit `z·k^α` to the
#                 value. This is the LOAD-BEARING stage of the example: profit
#                 depends on BOTH the capital stock `k` AND the profitability `z`,
#                 and a `UtilityStage` closure `(; k, z, env) -> z·k^α` can read
#                 both axes. `CapitalInvestmentStage`'s own `effort_cost`/`production`
#                 closures are `(value; env)` — they see ONLY the operative axis
#                 (`k`) and `env`, so the `z` dependence of profit CANNOT live in
#                 the investment stage's reward. Splitting the z-dependent flow
#                 into a separate `UtilityStage` is what gives `k` a NON-degenerate
#                 cross-section: the capital choice responds to `z` purely through
#                 the continuation value, and the persistent `z` chain spreads the
#                 stationary mass over `(k, z)`.
# `Invest`      — `CapitalInvestmentStage` on `:k`: from `k` the firm picks next capital
#                 `k'`, paying a convex cost `φ·i²` on GROSS investment
#                 `i = k' − (1−δ)k` and discounting by `β = 1/(1+r)`. Production
#                 in this stage is set to 0 (the operating profit lives in the
#                 `UtilityStage`); only the adjustment cost + depreciation are here.
#
# Why depreciation is REQUIRED (and why `CapitalInvestmentStage`, not `DurableAdjustmentStage`).
# With concave operating profit `z·k^α` (α<1), no depreciation, and a one-time
# convex cost, capital is costless to hold forever once installed — its marginal
# profit `α z k^{α−1}` is positive for all `k`, so the firm would accumulate
# without bound (no finite optimum, V unbounded). Depreciation `δ` makes holding
# capital costly every period (maintaining `k` needs gross investment `δk` at cost
# `φ(δk)²`), so the per-period payoff `z·k^α − φ(δk)²` is eventually decreasing in
# `k` and the optimum is finite. `CapitalInvestmentStage` carries depreciation in its
# gross-investment definition `i = k' − (1−δ)k`; `DurableAdjustmentStage` (cost on
# the net change `k'−k`) does not, hence the choice here.
#
# The reward `−φ·(k' − (1−δ)k)²` is supermodular in `(k', k)` (cross-partial
# `2φ(1−δ) ≥ 0`), so `CapitalInvestmentStage`'s divide-and-conquer monotone solve is valid.
# Prices `r` are exogenous (partial equilibrium): no market to clear, so the
# "outer loop" is a single `solve_steady_state_given_env!` over `(k, z)`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CapitalInvestmentParams
    α     :: Float64 = 0.70                 # curvature of operating profit z·k^α (<1: decreasing returns)
    φ     :: Float64 = 1.0                  # convex adjustment-cost coefficient: cost = φ·i²
    δ     :: Float64 = 0.15                 # capital depreciation
    r     :: Float64 = 0.04                 # discount rate ⇒ β = 1/(1+r)
    # Idiosyncratic profitability z: AR(1) in logs, log z' = ρ log z + σ ε, mean 1.
    ρ_z   :: Float64 = 0.80                 # persistence of log profitability
    σ_z   :: Float64 = 0.30                 # innovation std of log profitability
    N_z   :: Int     = 7                    # number of profitability states
    N_k   :: Int     = 120                  # capital grid points
    k_min :: Float64 = 0.3
    k_max :: Float64 = 24.0
end

Base.Broadcast.broadcastable(p::CapitalInvestmentParams) = Ref(p)

const capital_investment_params = CapitalInvestmentParams()


# Profitability process — Rouwenhorst discretization of the AR(1) in logs #
#------------------------------------------------------------------------#

"""
Rouwenhorst discretization of an AR(1) `x' = ρ x + σ ε` into `n` states.
Returns `(grid, P)` with `grid` the evenly spaced state values (symmetric about 0,
half-width `σ·√(n−1)/√(1−ρ²)`) and `P` the `n×n` ROW-stochastic transition matrix.
Rouwenhorst is accurate for highly persistent processes (unlike Tauchen at large ρ).
"""
function rouwenhorst(ρ::Real, σ::Real, n::Integer)
    n == 1 && return ([0.0], reshape([1.0], 1, 1))
    p = (1 + ρ) / 2
    P = [p (1 - p); (1 - p) p]
    for m in 3:n
        Pprev = P
        P = zeros(m, m)
        P[1:m-1, 1:m-1] .+= p .* Pprev
        P[1:m-1, 2:m]   .+= (1 - p) .* Pprev
        P[2:m, 1:m-1]   .+= (1 - p) .* Pprev
        P[2:m, 2:m]     .+= p .* Pprev
        P[2:m-1, :]     ./= 2                 # interior rows double-counted
    end
    ψ    = σ * sqrt((n - 1) / (1 - ρ^2))      # half-width of the state space
    grid = collect(range(-ψ, ψ; length = n))
    return (grid, P)
end


# Household / firm chain assembly — THREE library stages, NO bespoke stage #
#-------------------------------------------------------------------------#

"""
Build the convex-adjustment capital-investment firm block
`MarkovStage(:z) ∘ UtilityStage(z·k^α) ∘ CapitalInvestmentStage(:k)`, with `mean_k`,
`mean_profit`, and `mean_z` moments attached.

The profitability axis `:z` carries `exp(grid)` (so mean log-z ≈ 0, mean z ≈ 1).
Operating profit `z·k^α` lives in a `UtilityStage` reading BOTH axes — the
investment stage's `(value; env)` closures cannot see `z`, so this split is what
gives the capital stock a non-degenerate cross-section over `(k, z)`.
"""
function capital_investment_household(p = capital_investment_params)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid     = exp.(log_z)

    layout = GriddedLayout(
        :k => GriddedContinuous(p.k_min, p.k_max, p.N_k),
        :z => Discrete(z_grid),
    )

    shock  = MarkovStage(layout; axis = :z, transition_matrix = P_z)
    profit = UtilityStage(layout; utility = (; k, z) -> z * k^p.α)   # reads BOTH k and z
    invest = CapitalInvestmentStage(layout;
        axis         = :k,
        β            = 1 / (1 + p.r),
        depreciation = p.δ,
        production   = (k) -> 0.0,                              # profit lives in the UtilityStage
        effort_cost  = (i) -> p.φ * i^2)                        # convex cost on gross investment i ≥ 0

    hh = shock ∘ profit ∘ invest
    return define_moments!(hh;
        mean_k      = at_end(integrand = :k, reduce = sum),
        mean_profit = at_end(integrand = (; k, z) -> z * k^p.α, reduce = sum),
        mean_z      = at_end(integrand = :z, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"Exogenous env: the discount rate only (no market to clear; β is baked into the stage)."
capital_investment_env(p = capital_investment_params) = (; r = p.r)
