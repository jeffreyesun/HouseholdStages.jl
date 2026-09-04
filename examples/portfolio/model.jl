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
# `Portfolio`    — `GaussianLoadingStage`, here read as the portfolio stage
#                  (anchor = R_f, increment = the Gaussian excess return): picks
#                  the CONTINUOUS risky share
#                  `θ ∈ [0, 1]`, so next wealth is `b'·(R_f + θ·(μ_x + σ_x·Z))`
#                  for a truncated-Gaussian excess return with moments
#                  `(μ_x, σ_x)` matched to the two-point lottery
#                  `(R_risky, p_risky)`. Higher `θ` raises the mean and variance
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
    p_risky :: Vector{Float64} = [0.5, 0.5]       # feed the Gaussian excess moments (μ_x, σ_x) below
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
`mean_wealth = ∫ wealth dΛ` attached (θ* is read via `policy` in the driver). Four existing stages, no
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
        utility         = (cell, c) -> u_crra(c, Val(p.σ)))
        # defaults: (; axis = :wealth, skip_monotonicity_check = false, utility_axes = nothing)
    # Gaussian excess-return moments matched to the two-point lottery's excess returns.
    μx = sum(p.p_risky .* (p.R_risky .- p.R_f))                               # E[R_k] − R_f = 0.03
    σx = sqrt(sum(p.p_risky .* (p.R_risky .- p.R_f) .^ 2) - μx^2)             # sd of R_k − R_f = 0.20
    portfolio = GaussianLoadingStage(layout;                                     # defaults: (; axis = :wealth, loading_bounds = (0.0, 1.0), cost = (θ; env) -> 0.0)
        anchor = p.R_f, increment_mean = μx, increment_sd = σx)

    hh = shock ∘ receipt ∘ savings ∘ portfolio
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
