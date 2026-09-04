######################################################################
# Hall–Jones (2007) — health spending & the value of life            #
######################################################################

# Health spending as a share of income RISES with income because the value of
# life is a luxury good: with curved (CRRA, σ>1) consumption utility the marginal
# utility of an extra unit of consumption falls as people get richer, so trading
# consumption for additional life-years — i.e. health/survival — becomes
# relatively more attractive. This is the Hall–Jones (2007) mechanism. The catalog
# marks it ◐ because the headline is a CALIBRATION result that lives in the flow
# closures (the value-of-life flow + CRRA curvature), not in any structural
# primitive: the *model* is just `examples/health` extended with a wealth/
# consumption margin and a value-of-life flow, all from existing stages.
#
# Household block (time order; `∘` runs the LEFT stage first), existing stages
# only, NO bespoke stage:
#
#   IncomeReceipt ∘ [ ChooseHealth' ∘ DebitHealth ∘ CommitHealth ∘ Forget ]
#                 ∘ Consume ∘ Survive
#
# `IncomeReceipt`  — `IncomeStage(:wealth)`: cash-on-hand `(1+r)·wealth + w·y`.
# `ChooseHealth'`  — `ArgmaxStage` picks next health `h'` onto the auxiliary
#                    `:hc` axis (reward 0; the survival benefit is in the
#                    continuation value, the cost is debited downstream — exactly
#                    the two_asset_hank auxiliary-choice-axis pattern).
# `DebitHealth`    — `WealthChangeStage(:wealth)` debits the medical cost of the
#                    chosen `h'` from wealth (reads `:hc`, `:health`, `:wealth`).
#                    THIS is what couples health spending to the budget, so health
#                    competes with consumption — the precondition for an
#                    income-share result.
# `CommitHealth`   — `WealthChangeStage(:health)` writes `:health ← h'`.
# `Forget`         — `ForgetfulSumStage(:hc)` collapses the auxiliary axis.
# `Consume`        — `ConsumptionSavingsStage(:wealth)`, utility
#                    `u_crra(c) + value-of-life flow` (the flow value of being
#                    alive, the VSL primitive). The flow is read off `:alive` so
#                    dead cells carry zero flow and zero value.
# `Survive`        — `MarkovStage(:alive | health)`: health-dependent survival on
#                    the CHOSEN health `h'`, dead absorbing (as in examples/health).
#
# Finite-horizon cohort driver (mortality ⇒ no stationary alive-mass; the cohort's
# alive-mass decays along the survival curve, as in examples/health). The Hall–Jones
# headline is demonstrated by a comparative-statics sweep over the income level
# (`env.w`) in `steady_state.jl`: richer cohorts spend a higher share of income on
# health.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HallJonesParams
    β        :: Float64 = 0.96                  # patience
    σ        :: Float64 = 3.0                   # CRRA curvature (>1 ⇒ value of life a luxury)
    r        :: Float64 = 0.03                  # interest rate
    vsl      :: Float64 = 4.0                   # flow value of being alive (value-of-life primitive)
    # Health (Grossman / Ben-Porath) technology
    γ        :: Float64 = 0.6                   # health-production curvature ∈ (0,1)
    δ        :: Float64 = 0.07                  # health-capital depreciation
    a_health :: Float64 = 2.2                   # efficiency of producing health (price scale = 1/a)
    # Health-dependent survival — `s_max − gap·exp(−κ·h)`: rises with health,
    # diminishing returns, NEVER saturating within range (so the value of buying
    # more survival persists as people get richer — the Hall–Jones precondition).
    s_max    :: Float64 = 0.995                 # asymptotic per-age survival (h → ∞)
    s_gap    :: Float64 = 0.18                  # survival deficit at h = 0
    s_kappa  :: Float64 = 0.07                  # how fast survival approaches s_max in health
    # Horizon, endowments, grids
    N_age    :: Int     = 30                    # life-cycle length
    h0       :: Float64 = 4.0                   # health capital at birth
    w0       :: Float64 = 2.0                   # wealth at birth
    y        :: Float64 = 1.0                   # labour productivity (income = w·y)
    floor_w  :: Float64 = 0.05                  # subsistence floor on post-medical wealth
    N_w      :: Int     = 44                    # wealth grid points
    w_min    :: Float64 = 0.0
    w_max    :: Float64 = 110.0
    N_h      :: Int     = 42                    # health grid points
    h_min    :: Float64 = 1.0
    h_max    :: Float64 = 75.0
end

Base.Broadcast.broadcastable(p::HallJonesParams) = Ref(p)

const hall_jones_params = HallJonesParams()

"""
Per-age survival probability at health `h` — `s_max − gap·exp(−κ·h)`: rises with
health toward the asymptote `s_max` with diminishing returns, never saturating
within range (so buying more survival stays valuable as people get richer — the
Hall–Jones precondition). Assembled by the driver into the survival `MarkovStage`'s
transition matrix; healthier agents survive with higher probability.
"""
function hj_survival_prob(h::Real, p = hall_jones_params)
    return p.s_max - p.s_gap * exp(-p.s_kappa * h)
end

"""
Medical cost of moving health from `h` to `h'` — the inverted Ben-Porath /
Grossman production function: delivering gross investment `i = h' − (1−δ)h` at
efficiency `a` costs `(i / a)^{1/γ}` goods (convex for `γ ∈ (0,1)`). Zero for
non-positive investment. This is the goods cost DEBITED from wealth.
"""
function hj_medical_cost(h_next::Real, h::Real, p = hall_jones_params)
    i = h_next - (1 - p.δ) * h
    return i <= 0 ? 0.0 : (i / p.a_health)^(1 / p.γ)
end


# Household chain assembly — existing stages, auxiliary-choice-axis pattern #
#--------------------------------------------------------------------------#

"""
Build the Hall–Jones health-spending household block via the auxiliary-choice-axis
pattern (existing stages only, NO bespoke stage). Health spending is financed from
wealth (a `WealthChangeStage` debit), so it competes with consumption; a
value-of-life flow in the consumption utility makes survival a luxury good. Returns
the chain with `frac_alive`, `mean_health`, `mean_wealth` moments attached.
"""
function hall_jones_household(p = hall_jones_params)
    hgrid = collect(range(p.h_min, p.h_max; length = p.N_h))
    axes_base = (:wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
                 :health => GriddedContinuous(hgrid),
                 :alive  => Discrete([1, 0]))
    block = GriddedLayout(axes_base..., :hc => Discrete([1]))            # singleton aux
    full  = GriddedLayout(axes_base..., :hc => Discrete(collect(1:p.N_h)))

    # Income receipt: cash-on-hand. Single labour productivity p.y; the income
    # LEVEL is set by env.w (swept in steady_state.jl for the Hall–Jones result).
    receipt = WealthChangeStage(block; axis = :wealth,
        wealth_post = (; wealth, env) -> (1 + p.r) * wealth + env.w * p.y)

    # Choose next health h' onto the auxiliary axis (reward 0; benefit via
    # continuation, cost via downstream debit).
    choose = ArgmaxStage(block, full; axis = :hc, reward = zeros(p.N_h, 1))

    # Debit the medical cost from wealth — this couples health to the budget.
    debit = WealthChangeStage(full; axis = :wealth,
        wealth_post = (; hc, health, wealth) ->
            max(wealth - hj_medical_cost(hgrid[Int(hc)], health, p), p.floor_w))

    # Commit the chosen health, then collapse the auxiliary axis.
    commit = WealthChangeStage(full; axis = :health,
        wealth_post = (; hc) -> hgrid[Int(hc)])
    forget = ForgetfulSumStage(full; axis = :hc)

    # Consume from post-medical wealth. Period utility = CRRA(c) + value-of-life
    # flow while alive (zero when dead, so dead cells carry zero value).
    consume = ConsumptionSavingsStage(block; β = p.β, axis = :wealth,
        utility_axes = (:alive,),
        utility = (cell, c) -> cell.alive == 1 ? u_crra(c, Val(p.σ)) + p.vsl : 0.0)

    # Survive on the CHOSEN health; dead absorbing.
    survival_T(h) = [hj_survival_prob(h, p)  1 - hj_survival_prob(h, p);
                     0.0                     1.0]
    survive = MarkovStage(block; axis = :alive,
        transition_matrix = (; health) -> survival_T(health))

    hh = receipt ∘ choose ∘ debit ∘ commit ∘ forget ∘ consume ∘ survive
    return define_moments!(hh;
        frac_alive  = at_end(integrand = (; alive) -> alive == 1 ? 1.0 : 0.0,            reduce = sum),
        mean_health = at_end(integrand = (; alive, health) -> alive == 1 ? health : 0.0, reduce = sum),
        mean_wealth = at_end(integrand = (; alive, wealth) -> alive == 1 ? wealth : 0.0, reduce = sum))
end
