#####################################################################
# Fagereng–Guiso–Malacrino–Pistaferri (2020) — return heterogeneity   #
#####################################################################

# PERSISTENT HETEROGENEITY IN RETURNS TO WEALTH. Households differ
# persistently in the gross return they earn on their wealth (Fagereng,
# Guiso, Malacrino & Pistaferri, "Heterogeneity and Persistence in Returns
# to Wealth", ECMA 2020). Such return heterogeneity is a fat-tail driver of
# wealth inequality: a high-return type compounds faster, so the right tail
# of the wealth distribution thickens far beyond what income risk alone
# produces. The paper's empirical point — returns differ across people, and
# the differences persist — is exactly a persistent type that scales the
# return on wealth.
#
# The whole within-period problem is FOUR existing library stages, in time
# order, with **no bespoke household stage rolled here** —
#
#     ReturnType ∘ IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ ReturnReceipt
#
# `ReturnType`    — `MarkovStage` on the return-type axis `:rtype`: a
#                   persistent type governing the return earned on wealth.
# `IncomeShock`   — `MarkovStage` on the `:income` axis: idiosyncratic labor
#                   income risk (the usual Aiyagari shock).
# `Receipt`       — `WealthChangeStage` `a ↦ a + w·y` (cash-on-hand).
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period saved
#                   wealth `b'` on the wealth grid; `c = x − b'` (CRRA).
# `ReturnReceipt` — `WealthChangeStage` `b' ↦ R(rtype)·b'`: the type-specific
#                   gross return realized on saved wealth between periods.
#                   The persistent return heterogeneity is carried HERE.
#
# KEY DESIGN DECISION — why the return rides a `WealthChangeStage`, not
# `GaussianLoadingStage`. The natural portfolio primitive `GaussianLoadingStage`
# takes PLAIN return moments (`anchor`/`increment_mean`/`increment_sd`) — it
# cannot vary the mean return by a persistent `:rtype` axis, because those
# moments are scalars (or `FromEnv`) fixed across cells. Fagereng's contribution is precisely
# heterogeneity in the MEAN return across people, so the return must read the
# `:rtype` axis. A `WealthChangeStage` does exactly that: its `wealth_post`
# closure reads any layout axis, so `b' ↦ R(rtype)·b'` carries the per-type
# return cleanly and faithfully. (Catalog §2: "the return process is the
# model's, not the package's".) `R(rtype)` is supplied via `env`, so the
# heterogeneous and homogeneous-baseline calibrations are a pure `env` swap —
# the household block is identical for both.
#
# We deliberately omit a `GaussianLoadingStage` portfolio leg: a homogeneous
# risky-share choice composes cleanly on top (see `examples/portfolio`), but
# it would add an idiosyncratic-risk margin that is NOT the Fagereng
# mechanism. The mean-return heterogeneity is the model's contribution, and
# it rides the `ReturnReceipt` `WealthChangeStage`.
#
# Returns, the return-type process, income, and the wage are exogenous
# (partial equilibrium): there is no market to clear, so the "outer loop" is
# a single `solve_steady_state_given_env!`. Impatience relative to every
# type's return (`β·R(rtype) < 1` for all types) plus the borrowing
# constraint (`b' ≥ 0`, the grid floor) deliver a stationary distribution.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct FagerengParams
    β :: Float64       = 0.95
    σ :: Float64       = 3.0                       # CRRA risk aversion
    w :: Float64       = 1.0                       # wage (income scale)
    # Persistent idiosyncratic labor income (mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.70 0.25 0.05;
                                 0.20 0.60 0.20;
                                 0.05 0.25 0.70]
    # Persistent return TYPE. The axis values are type labels (1, 2); the
    # gross return each type earns is supplied via env (so hetero vs homo is
    # an env swap, not a relayout). Very persistent ⇒ a near-permanent type.
    rtype_labels :: Vector{Float64} = [1.0, 2.0]
    P_rtype      :: Matrix{Float64} = [0.97 0.03;
                                       0.03 0.97]
    # Heterogeneous returns: a low-return and a high-return type. Stationary
    # type mass is [0.5, 0.5], so the population-average gross return is 1.02
    # — the homogeneous baseline (below) holds that average fixed.
    R_hetero :: Vector{Float64} = [1.00, 1.04]
    R_homog  :: Vector{Float64} = [1.02, 1.02]     # same average return, no spread
    N_w   :: Int       = 150
    w_min :: Float64   = 0.0
    w_max :: Float64   = 300.0
end

Base.Broadcast.broadcastable(p::FagerengParams) = Ref(p)

const fagereng_params = FagerengParams()


# Household chain assembly #
#--------------------------#

"""
Build the Fagereng household block
`ReturnType ∘ IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ ReturnReceipt`,
with `mean_wealth = ∫ wealth dΛ` attached. Five existing stages, no bespoke
household stage. The persistent return heterogeneity rides the final
`WealthChangeStage` (`ReturnReceipt`), whose closure reads the `:rtype` axis
and maps it to the type's gross return `R(rtype)` supplied via `env`.
"""
function fagereng_household(p = fagereng_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
        :rtype  => Discrete(p.rtype_labels),
    )

    rtype_shock  = MarkovStage(layout; axis = :rtype,  transition_matrix = p.P_rtype)
    income_shock = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;                                        # cash-on-hand x = a + w·y
        wealth_post = (; wealth, income, env) -> wealth + env.w * income)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))
    # Type-specific gross return on saved wealth: b' ↦ R(rtype)·b'. The closure
    # reads the persistent :rtype axis (the grid value 1.0/2.0 is the type
    # label) and looks up that type's return in env.R_by_type.
    return_receipt = WealthChangeStage(layout;                                 # defaults: (; axis = :wealth)
        wealth_post = (; rtype, wealth, env) -> env.R_by_type[Int(rtype)] * wealth)

    hh = rtype_shock ∘ income_shock ∘ receipt ∘ savings ∘ return_receipt
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end


# Exogenous environment (plain function, no AbstractBlock) #
#----------------------------------------------------------#

"The exogenous Fagereng env: the wage `w` and the per-type gross-return vector
`R_by_type` (`R_by_type[t]` is type `t`'s gross return on wealth). Swap
`R_by_type` between `R_hetero` and `R_homog` to isolate return heterogeneity."
fagereng_env(p = fagereng_params; w = p.w, R_by_type = p.R_hetero) = (; w, R_by_type)
