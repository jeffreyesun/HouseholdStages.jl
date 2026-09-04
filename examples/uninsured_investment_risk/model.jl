#####################################################################
# Uninsured idiosyncratic investment risk — Angeletos (2007), Covas (2006) #
#####################################################################

# UNDIVERSIFIABLE IDIOSYNCRATIC CAPITAL-RETURN RISK adds a precautionary
# wedge to investment (Angeletos, "Uninsured idiosyncratic investment risk
# and aggregate saving", RED 2007; Covas, "Uninsured idiosyncratic
# production risk with borrowing constraints", JEDC 2006). An entrepreneur
# invests in their OWN capital, whose return is risky and cannot be
# diversified away. Because the risk is uninsurable, a risk-averse agent
# tilts away from risky capital toward a safe store — the precautionary
# investment wedge. Higher idiosyncratic variance depresses the risky share.
#
# This is the portfolio block REINTERPRETED: the "risky asset" is the
# agent's own capital with undiversifiable idiosyncratic return risk, and
# `θ` is the fraction of wealth invested in risky capital vs a safe store.
# The whole within-period problem is FOUR existing library stages, in time
# order, with **no bespoke household stage rolled here** —
#
#     IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Investment
#
# `IncomeShock`   — `MarkovStage` on the `:income` axis: labor income risk.
# `Receipt`       — `WealthChangeStage` `a ↦ a + w·y` (cash-on-hand).
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period saved
#                   wealth `b'` on the wealth grid; `c = x − b'` (CRRA).
# `Investment`    — `GaussianLoadingStage` on the wealth axis (the portfolio stage
#                   read as a capital-exposure choice: anchor = R_f, increment =
#                   the capital excess return) picks the CONTINUOUS
#                   share `θ ∈ [0, 1]` of saved wealth placed in RISKY OWN
#                   CAPITAL: next wealth is `b'·(R_f + θ·(μ_x + σ_x·Z))` where
#                   `R_f` is the safe store and the truncated-Gaussian excess
#                   `(μ_x, σ_x)` is moment-matched to the agent's idiosyncratic
#                   capital return (the mean-`μ_k`, half-spread-`Δ` two-point
#                   draw of the calibration). The idiosyncratic capital
#                   risk is exactly `GaussianLoadingStage`'s risky leg — the
#                   "portfolio share" is here the capital share, the variance
#                   is undiversifiable production risk.
#
# Why this is the faithful statement. Angeletos/Covas put undiversifiable
# idiosyncratic risk on the return to OWN capital, and the agent chooses how
# much to expose. That is precisely a mean–variance share choice: raising θ
# raises both the mean and variance of next wealth, and a CRRA agent trades
# them off. The variance is genuinely idiosyncratic (each agent draws their
# own capital return), so `GaussianLoadingStage`'s per-cell return distribution
# is the right primitive — no aggregate risk, no diversification.
#
# Returns, the wage, and the capital-return distribution are exogenous
# (partial equilibrium): there is no market to clear, so the "outer loop" is
# a single `solve_steady_state_given_env!`. Impatience (`β·R_f < 1`) plus the
# borrowing constraint (`b' ≥ 0`, the grid floor) give a stationary
# distribution.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct InvestmentRiskParams
    β :: Float64       = 0.95
    σ :: Float64       = 3.0                       # CRRA risk aversion
    w :: Float64       = 1.0                       # wage (income scale)
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.70 0.25 0.05;
                                 0.20 0.60 0.20;
                                 0.05 0.25 0.70]
    R_f    :: Float64 = 1.02                       # safe store gross return
    μ_k    :: Float64 = 1.08                       # MEAN gross return on risky own capital (6% premium)
    Δ      :: Float64 = 0.20                       # half-spread of the two-point idiosyncratic capital return
    p_up   :: Float64 = 0.5                        # two-point draw: μ_k − Δ w.p. p_up, μ_k + Δ w.p. 1 − p_up
    N_w   :: Int       = 150
    w_min :: Float64   = 0.0
    w_max :: Float64   = 80.0
end

Base.Broadcast.broadcastable(p::InvestmentRiskParams) = Ref(p)

const investment_risk_params = InvestmentRiskParams()


# Household chain assembly #
#--------------------------#

"""
Build the uninsured-investment-risk household block
`IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Investment`, with
`mean_wealth = ∫ wealth dΛ` attached. Four existing stages, no bespoke
household stage. The risky leg of the final `GaussianLoadingStage` is the
agent's OWN-CAPITAL return — a truncated-Gaussian excess moment-matched to
the mean-`μ_k`, half-spread-`Δ` two-point idiosyncratic draw — so `θ` is the
share of saved wealth exposed to undiversifiable capital risk. Raising `Δ`
(the excess sd, at `p_up = ½`) sharpens the precautionary tilt away from
risky capital.
"""
function investment_risk_household(p = investment_risk_params; Δ = p.Δ)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;                                        # cash-on-hand x = a + w·y
        wealth_post = (; wealth, income, env) -> wealth + env.w * income)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))
    # Own-capital investment: share θ of saved wealth in risky capital; the rest
    # earns R_f safe. The Gaussian excess moments match the two-point draw
    # {μ_k − Δ w.p. p_up, μ_k + Δ w.p. 1 − p_up} — general (asymmetric-
    # probability) formulas; they collapse to (μ_k − R_f, Δ) at p_up = ½.
    μx = (p.μ_k - p.R_f) + (1 - 2p.p_up) * Δ
    σx = 2Δ * sqrt(p.p_up * (1 - p.p_up))
    investment = GaussianLoadingStage(layout;                                     # defaults: (; axis = :wealth, loading_bounds = (0.0, 1.0), cost)
        anchor = p.R_f, increment_mean = μx, increment_sd = σx)

    hh = shock ∘ receipt ∘ savings ∘ investment
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end

"The `GaussianLoadingStage` leaf (the last seated stage) — its seated `policy` is the
risky-capital share θ*(x). Same `stages[end]` access the portfolio example uses."
investment_risk_stage(hh) = hh.buffer.stages[end]


# Exogenous environment (plain function, no AbstractBlock) #
#----------------------------------------------------------#

"The exogenous env: wage `w`. (The capital-return distribution is baked into
the `GaussianLoadingStage` at build time via `Δ`, so the variance sweep rebuilds
the block; everything else is held fixed.)"
investment_risk_env(p = investment_risk_params; w = p.w) = (; w)
