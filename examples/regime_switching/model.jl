################################################################
# Regime-switching income (Hamilton 1989-style)                 #
################################################################
#
# Aggregate conditions follow a Markov regime — boom / recession (Hamilton
# 1989). The IDIOSYNCRATIC income transition depends on which regime the
# household is in: in a recession the conditional income distribution is
# shifted toward (and stickier in) the low state. Unlike the
# `countercyclical_risk` example — where the regime is an EXOGENOUS env
# input fixed within a steady state — here the regime is an ENDOGENOUS axis
# of the state space with its own Markov law, so the household carries a
# joint `(regime, income, wealth)` distribution and the two aggregate
# states coexist in one stationary Λ.
#
# The chain is two Markov stages plus the spine:
#
#   MarkovStage(:regime)
#       ∘ MarkovStage(:income; transition_matrix = (; regime) -> T_income(regime))
#       ∘ IncomeStage ∘ ConsumptionSavingsStage
#
# The income transition is a DEP-CLOSURE on the `:regime` AXIS: `MarkovStage`
# reads the closure's `regime` kwarg via `Base.kwarg_decl`, recognizes it as
# a layout axis, and stores ONE income transition per regime value
# (`(n_income, n_income, n_regime)` compact field). At apply time each
# regime cell picks its own income fiber. The closure receives the regime
# GRID VALUE (not an index), so it dispatches on `regime == boom`.

using HouseholdStages


# Parameters #
#------------#

# Regime grid values (boom / recession) — the closure dispatches on these.
const REGIME_BOOM      = 1.0
const REGIME_RECESSION = 2.0

@kwdef struct RegimeSwitchingParams
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0
    r   :: Float64 = 0.03                  # fixed, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0

    # Aggregate regime: a persistent two-state chain (expansions longer than
    # recessions — the Hamilton calibration shape).
    regime_grid :: Vector{Float64} = [REGIME_BOOM, REGIME_RECESSION]
    P_regime    :: Matrix{Float64} = [0.95 0.05;     # boom → mostly boom
                                      0.20 0.80]      # recession → shorter-lived

    # Idiosyncratic income levels (mean ≈ 1), shared across regimes.
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]

    # Income transition IN BOOM: favorable, upward-drifting.
    T_income_boom :: Matrix{Float64} = [0.50 0.40 0.10;
                                        0.10 0.60 0.30;
                                        0.05 0.25 0.70]
    # Income transition IN RECESSION: adverse, sticky-low.
    T_income_recession :: Matrix{Float64} = [0.80 0.18 0.02;
                                             0.40 0.50 0.10;
                                             0.15 0.45 0.40]

    N_w   :: Int     = 250
    w_min :: Float64 = 0.0
    w_max :: Float64 = 80.0
end

Base.Broadcast.broadcastable(p::RegimeSwitchingParams) = Ref(p)

const regime_switching_params = RegimeSwitchingParams()


# Household chain assembly #
#--------------------------#

"""
Build the regime-switching household block. The aggregate regime evolves by
its own `MarkovStage(:regime)`; the idiosyncratic income `MarkovStage` is
handed a DEP-CLOSURE on the `:regime` axis,

    transition_matrix = (; regime) -> regime == REGIME_BOOM ? T_boom : T_recession,

so the income transition is regime-specific (Hamilton 1989). The chain is

    MarkovStage(:regime) ∘ MarkovStage(:income; (; regime) -> T(regime))
        ∘ IncomeStage ∘ ConsumptionSavingsStage.

Attaches the buffer-stock moment `A_mean` plus the recession population
share `recession_share` (a cross-check that both regimes carry mass).
"""
function regime_switching_household(p = regime_switching_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :regime => Discrete(p.regime_grid),
        :income => Discrete(p.y_grid),
    )

    regime_shock = MarkovStage(layout; axis = :regime, transition_matrix = p.P_regime)
    income_shock = MarkovStage(layout; axis = :income,
        transition_matrix = (; regime) ->
            regime == REGIME_BOOM ? p.T_income_boom : p.T_income_recession)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )

    hh = regime_shock ∘ income_shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean          = at_end(integrand = :wealth, reduce = sum),
        recession_share = at_end(integrand = (; regime) -> regime == REGIME_RECESSION ? 1.0 : 0.0,
                                  reduce = sum),
    )
end

"Env for the fixed-r partial-equilibrium experiment: return `r`, wage `w`."
regime_switching_env(p = regime_switching_params) = (; r = p.r, w = p.w)
