#####################################################################
# Risk-shifting / gambling-for-resurrection — Vereshchagina–Hopenhayn #
#####################################################################

# A borrowing-constrained, limited-liability ENTREPRENEUR who gambles for
# resurrection (Vereshchagina & Hopenhayn, "Risk Taking by Entrepreneurs",
# AER 2009). The classic result: where the entrepreneur's value function is
# CONVEX — near the limited-liability floor a poorly-capitalized entrepreneur
# faces — the agent picks a RISKIER project, even at NO higher mean return.
# Risk-taking is DECREASING in net worth.
#
# The whole within-period problem is FIVE existing library stages, in time
# order, with **no bespoke household stage rolled here** —
#
#     OccShock ∘ Receipt ∘ Savings ∘ Gamble ∘ LimitedLiability
#
# `OccShock`         — `MarkovStage` on the entrepreneurial-productivity axis
#                      `z`: a persistent shock to project quality.
# `Receipt`          — `WealthChangeStage` `a ↦ a + w`: a small labor/cash
#                      injection each period, so wealth accumulates and the
#                      stationary distribution spreads across the grid rather
#                      than collapsing to a single point.
# `Savings`          — `ConsumptionSavingsStage` picks the stake `a'` carried
#                      into the project on the wealth grid; `c = x − a'` (CRRA).
#                      The grid floor `a' ≥ a_min` is the borrowing constraint.
# `Gamble`           — `GaussianLoadingStage` on the wealth axis (the portfolio
#                      stage read as a project-risk dial: anchor = R_f,
#                      increment = the project's excess payoff): the entrepreneur
#                      picks project risk `θ ∈ [0, 1]` CONTINUOUSLY, so the
#                      stake becomes `a'·(R_f + θ·(μ_x + σ_x·Z))` for a
#                      truncated-Gaussian excess return moment-matched to the
#                      two-point succeed/fail project — the Gaussian carries
#                      that project through its first two moments
#                      (μ_x = 0.02, σ_x = 0.59). The V–H mechanism does not
#                      live in the bet's SHAPE: `θ` is a VARIANCE dial pulled at
#                      a convex region of V, the risky leg is essentially
#                      MEAN-NEUTRAL, and the right-tail emphasis emerges
#                      ENDOGENOUSLY from the limited-liability floor below — so
#                      no skewness primitive is needed.
# `LimitedLiability` — `WealthChangeStage` `a ↦ max(z·a, a_floor)`: limited
#                      liability / the occupational outside option. A failed
#                      gamble cannot push the entrepreneur below `a_floor` (the
#                      creditor absorbs the shortfall; the agent walks to the
#                      worker option worth ≈ holding `a_floor`). This wealth
#                      FLOOR convexifies V just above `a_floor` — the engine of
#                      risk-shifting. (Same primitive the Aiyagari/portfolio
#                      examples use for the budget map; here the map is a max,
#                      the V–H floor.) `z` also scales realized wealth, so a
#                      better entrepreneur's project compounds faster.
#
# Why no occupation axis / no separate `max(V_continue, W)` stage. The V–H
# floor is a floor on REALIZED WEALTH (limited liability), and a wealth-floor
# `WealthChangeStage` states it directly and library-only — the faithful
# statement of the same convexity (V–H §II: the outside option is a lower bound
# on the continuation; here, a lower bound on the carried collateral), with no
# occupation axis to carry.
#
# Returns and `z` are exogenous (partial equilibrium): there is no market to
# clear, so the "outer loop" is a single `solve_steady_state_given_env!`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct RiskShiftingParams
    β :: Float64       = 0.96                       # discount factor
    σ :: Float64       = 2.0                        # CRRA risk aversion
    w :: Float64       = 0.40                       # per-period labor/cash injection
    # Entrepreneurial productivity z scales the project's compounding.
    z_grid :: Vector{Float64} = [0.98, 1.02]
    P_z    :: Matrix{Float64} = [0.85 0.15;
                                 0.30 0.70]
    # Succeed/fail project calibration on the risky leg, nearly MEAN-NEUTRAL vs
    # the safe return (E[R_k] = 0.4·1.8 + 0.6·0.6 = 1.08 ≈ R_f): gambling is
    # driven by CONVEXITY, not a return premium — the pure V–H bet. Feeds the
    # moment-matched Gaussian excess (μ_x = 0.02, σ_x = 0.59) below.
    R_f    :: Float64 = 1.06                        # gross return of the safe (θ = 0) project
    R_up   :: Float64 = 1.80                        # success multiple
    R_dn   :: Float64 = 0.60                        # failure multiple
    p_up   :: Float64 = 0.40                        # success probability
    a_floor :: Float64 = 0.50                       # limited-liability wealth floor (the V–H outside option)
    N_a   :: Int       = 200
    a_min :: Float64   = 0.0
    a_max :: Float64   = 200.0
end

Base.Broadcast.broadcastable(p::RiskShiftingParams) = Ref(p)

const risk_shifting_params = RiskShiftingParams()


# Household chain assembly #
#--------------------------#

"""
Build the risk-shifting household block
`OccShock ∘ Receipt ∘ Savings ∘ Gamble ∘ LimitedLiability`, with
`mean_wealth = ∫ wealth dΛ` attached (the carried collateral after the
limited-liability floor — robustly `≥ a_floor`). Five existing stages, no
bespoke household stage.
"""
function risk_shifting_household(p = risk_shifting_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :z => Discrete(p.z_grid),
    )

    # Persistent entrepreneurial-productivity shock.
    occshock = MarkovStage(layout; axis = :z, transition_matrix = p.P_z)

    # Per-period cash injection so the stationary distribution spreads.
    receipt = WealthChangeStage(layout;                                  # defaults: (; axis = :wealth)
        wealth_post = (; wealth, env) -> wealth + env.w)

    # Choose the stake a' carried into the project (consumption-savings).
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c) -> u_crra(c, Val(p.σ)))
        # defaults: (; axis = :wealth, skip_monotonicity_check = false, utility_axes = nothing)

    # The project gamble: stake a' → a'·(R_f + θ·(μ_x + σ_x·Z)), the Gaussian
    # excess moment-matched to the (succeed, fail) pair {R_up − R_f w.p. p_up,
    # R_dn − R_f}. Mean-neutral ⇒ convexity drives θ.
    μx = p.p_up * (p.R_up - p.R_f) + (1 - p.p_up) * (p.R_dn - p.R_f)     # 0.02
    σx = sqrt(p.p_up * (1 - p.p_up)) * abs(p.R_up - p.R_dn)              # 0.59
    gamble = GaussianLoadingStage(layout;                                   # defaults: (; axis = :wealth, loading_bounds = (0.0, 1.0), cost = (θ; env) -> 0.0)
        anchor = p.R_f, increment_mean = μx, increment_sd = σx)

    # Limited liability / outside option: a failed gamble cannot drop wealth
    # below a_floor. z also scales realized wealth, so productivity compounds.
    liability = WealthChangeStage(layout;                               # defaults: (; axis = :wealth)
        wealth_post = (; z, wealth, env) -> max(z * wealth, env.a_floor))

    hh = occshock ∘ receipt ∘ savings ∘ gamble ∘ liability
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end

"The `GaussianLoadingStage` leaf — its seated `policy` is the project-risk choice `θ*(x)`.
(Index 5, not 4: the upstream `ConsumptionSavingsStage` expands to `argmax ∘ TimeDiscounting`,
inserting the discount leaf just before `gamble`.)"
risk_shifting_gamble_stage(hh) = hh.buffer.stages[5]


# Exogenous environment (plain function, no AbstractBlock) #
#----------------------------------------------------------#

"The exogenous risk-shifting env: the cash injection `w` and the limited-liability floor `a_floor` (the V–H outside option)."
risk_shifting_env(p = risk_shifting_params; w = p.w, a_floor = p.a_floor) = (; w, a_floor)
