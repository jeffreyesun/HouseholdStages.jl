###############################################################
# Portfolio choice — incomplete-markets steady state          #
###############################################################

# An incomplete-markets household that BOTH saves and chooses a risky
# portfolio share — the Aiyagari savings problem with a Merton/Cocco–
# Gomes–Maenhout portfolio decision bolted on. The point of this example:
# the entire within-period problem is FOUR existing library stages, in
# time order, with **no bespoke household stage rolled here** —
#
#     IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Portfolio
#
# `IncomeShock`  — `MarkovStage` on the income axis.
# `Receipt`      — `WealthChangeStage` `a ↦ a + w·y` (cash-on-hand).
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period
#                  financial wealth `b'` on the wealth grid; `c = x − b'`.
# `Portfolio`    — `MeanVarianceStage` picks the risky share `θ`, so next
#                  wealth is `b'·(R_f + θ·(R_k − R_f))` for risky gross
#                  returns `R_k`. Higher `θ` raises the mean and variance
#                  of next wealth; risk-averse CRRA agents pick interior θ.
#
# Returns are exogenous (partial equilibrium): there is no market to clear,
# so the "outer loop" is a single `solve_steady_state_given_env!`. The
# borrowing constraint (`b' ≥ 0`, the grid floor) plus impatience
# (`β·E[R] < 1`) deliver a stationary wealth distribution.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct PortfolioParams
    β :: Float64       = 0.95
    σ :: Float64       = 3.0                      # CRRA (risk aversion drives the share)
    w :: Float64       = 1.0                      # wage (income scale)
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.7 0.25 0.05;
                                 0.2 0.60 0.20;
                                 0.05 0.25 0.70]
    R_f     :: Float64 = 1.02                     # gross risk-free return
    R_risky :: Vector{Float64} = [0.85, 1.25]     # gross risky returns (mean 1.05 ⇒ 3% premium)
    p_risky :: Vector{Float64} = [0.5, 0.5]
    shares  :: Vector{Float64} = collect(0.0:0.1:1.0)
    N_w   :: Int       = 200
    w_min :: Float64   = 0.0
    w_max :: Float64   = 60.0
end

Base.Broadcast.broadcastable(p::PortfolioParams) = Ref(p)

const portfolio_params = PortfolioParams()


# Household chain assembly #
#--------------------------#

"""
Build the portfolio household block `IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Portfolio`, with
`mean_wealth = ∫ wealth dΛ` and `mean_risky_share = ∫ θ*(x) dΛ` attached. Four existing stages, no
bespoke household stage.
"""
function portfolio_household(p = portfolio_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;                                       # cash-on-hand x = a + w·y
        wealth_post = (; wealth, income, env) -> wealth + env.w * income)        # defaults: (; axis = :wealth)
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)))
        # defaults: (; axis = :wealth, monotone_search = :divide_conquer, assume_monotone = false, utility_axes = nothing)
    portfolio = MeanVarianceStage(layout;                                     # defaults: (; axis = :wealth, cost = (θ; env) -> 0.0)
        shares = p.shares, risk_free = p.R_f, risky_returns = p.R_risky, probs = p.p_risky)

    hh = shock ∘ receipt ∘ savings ∘ portfolio
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
