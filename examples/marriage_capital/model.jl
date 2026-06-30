###################################################################
# Becker (1973) marriage-specific capital — within-match life cycle #
###################################################################

# Marriage-specific capital accumulation as a within-match life-cycle household
# block built from EXISTING library stages only — NO bespoke household stage
# rolled here. This example is STRUCTURALLY IDENTICAL to `examples/health`
# (Grossman): a Ben-Porath investment in a stock, composed with a stock-dependent
# survival/separation Markov. Only the labels change. The whole point:
#
#     Household block = MatchInvestStage ∘ SeparationStage
#                     = CapitalInvestmentStage(:match_capital) ∘ MarkovStage(:married | match_capital)
#
# State = `(match_capital, married)`. Two existing stages, time-ordered (`∘` runs
# the LEFT stage first — invest in the match, THEN draw the separation shock on the
# chosen match capital):
#
#   MatchInvestStage — `CapitalInvestmentStage` on the `:match_capital` axis `m`. From match
#       capital `m` the couple picks next stock `m'`, paying a convex investment
#       cost on gross investment `i = m' − (1−δ)m` (`δ` = match-capital
#       depreciation: the match decays absent investment) and earning the match
#       flow `production(m; env)` (Becker's match-specific surplus — the value of
#       staying together, rising in shared/match-specific capital). This is exactly
#       the Ben-Porath (1967) technology with a MARRIAGE-MATCH stock instead of
#       human capital — the same stage as `examples/human_capital` and
#       `examples/health`, a different flow.
#
#   SeparationStage — `MarkovStage` on a 2-state `:married` axis (`[1, 0]` =
#       married, separated) whose ROW-stochastic transition is MATCH-CAPITAL-
#       dependent via a dep closure `(; match_capital) -> [stay(m) 1−stay(m); 0 1]`.
#       Couples with more match-specific capital survive (stay together) with higher
#       probability — the per-period separation hazard `1 − stay(m)` FALLS with match
#       capital. The separated state is ABSORBING (no remarriage modelled here). On
#       the married sub-mass the rows sum to `stay(m) < 1` — the sub-stochastic
#       survival the spec calls for — while total mass on the full
#       `(match_capital, married)` grid is conserved (it accumulates in the
#       absorbing separated state). The retention probability `stay(m)` is a plain
#       economic primitive (a logistic in match capital) assembled into a matrix and
#       handed to the EXISTING `MarkovStage`: DATA fed to a stage, not a new stage.
#
# Why finite-horizon (within-match cohort), not stationary. A stationary
# distribution with separation but no match FORMATION leaks all mass into the
# absorbing separated state — there is no nondegenerate stationary married-mass
# without a forward mass-injection SOURCE (new couples entering), which is the
# two-sided matching/formation problem and is OUT OF SCOPE here (see README — the
# catalog's "within-match ✅", with a matching gap on formation/bargaining). The
# finite-horizon within-match cohort sidesteps that honestly: a couple is "born"
# (married) at match capital `m0`, invests each period, and its married-mass decays
# along the separation curve over the duration of the relationship. No
# stationarity-of-mass requirement, so no formation source needed.
#
# What is example-side (and allowed): the FINITE-HORIZON DRIVER. The age/duration-
# specific investment-efficiency profile (the ease of building match capital, which
# can decline with relationship duration) and the separation matrix are threaded
# through `env`/the stage by the backward/forward sweep in `steady_state.jl` —
# driver logic, never inside a household stage.
#
# The effort cost is the inverted match-production function: new match capital
# `Q = a·i^γ`-style, so to deliver gross investment `i` at investment efficiency `a`
# costs `effort_cost(i; env) = env.R · (i / env.a)^{1/γ}` (convex for `γ ∈ (0,1)`,
# exponent `1/γ > 1`). The reward `production(m) − effort_cost(m' − (1−δ)m)` is then
# supermodular in `(m', m)`, so `CapitalInvestmentStage`'s `:divide_conquer` monotone solve
# is valid. Higher efficiency `a` lowers the cost — the engine of capital build-up.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MarriageParams
    β        :: Float64 = 0.97                # patience
    γ        :: Float64 = 0.7                 # match-production curvature ∈ (0,1)
    δ        :: Float64 = 0.08                # match-capital depreciation (decay of the match)
    R        :: Float64 = 1.0                 # value of a unit of match flow (price of match surplus)
    a_young  :: Float64 = 0.42                # efficiency of building match capital, duration 1
    a_old    :: Float64 = 0.12                # efficiency at the last duration (declines over time)
    N_age    :: Int     = 35                  # max relationship duration (periods)
    m0       :: Float64 = 4.0                 # match capital at marriage (cohort starts here)
    # Retention (= 1 − separation hazard) — logistic in match capital:
    #   stay(m) = s_max / (1 + exp(−k(m − m_mid))), INCREASING in m.
    s_max    :: Float64 = 0.99                # retention cap (best attainable per-period survival)
    s_k      :: Float64 = 0.45                # steepness of retention in match capital
    s_m_mid  :: Float64 = 3.0                 # match capital at the retention inflection
    N_m      :: Int     = 150                 # match-capital grid points
    m_min    :: Float64 = 0.5
    m_max    :: Float64 = 25.0
end

Base.Broadcast.broadcastable(p::MarriageParams) = Ref(p)

const marriage_params = MarriageParams()

"Duration-`t` investment efficiency (`t = 1…N_age`): a log-linear taper from
`a_young` to `a_old` — match capital is easier to build early in a relationship."
function efficiency_at_age(t::Integer, p = marriage_params)
    s = (t - 1) / max(p.N_age - 1, 1)
    return exp((1 - s) * log(p.a_young) + s * log(p.a_old))
end

"Per-period retention probability `stay(m)` at match capital `m` — a logistic
capped at `s_max`, INCREASING in `m`. The per-period separation hazard `1 − stay(m)`
therefore FALLS with match capital (Becker). This is a plain economic primitive; the
driver assembles it into the separation `MarkovStage`'s transition matrix."
function retention_prob(m::Real, p = marriage_params)
    return p.s_max / (1 + exp(-p.s_k * (m - p.s_m_mid)))
end


# Household chain assembly — TWO library stages, NO bespoke stage #
#----------------------------------------------------------------#

"""
Build the Becker marriage-capital household block — `MarkovStage(:married |
match_capital) ∘ CapitalInvestmentStage(:match_capital)`, two existing library stages, with
`mean_match_capital`, `married_rate`, and `mean_surplus` moments attached.

The `:married` axis is `[1, 0]` (married, separated). The separation `MarkovStage`'s
transition is a MATCH-CAPITAL-DEPENDENT dep closure `(; match_capital) -> T` with the
separated state absorbing — `stay(m)` on the married→married entry, `1−stay(m)` on
married→separated. No bespoke household stage; the duration-efficiency profile is
threaded through `env` by the finite-horizon driver, and the retention probability
is economic data fed to `MarkovStage`.
"""
function marriage_household(p = marriage_params)
    layout = GriddedLayout(
        :match_capital => GriddedContinuous(p.m_min, p.m_max, p.N_m),
        :married       => Discrete([1, 0]),          # 1 = married, 0 = separated
    )

    invest = CapitalInvestmentStage(layout;
        axis            = :match_capital,
        β               = p.β,
        depreciation    = p.δ,
        production      = (m; env) -> env.R * m,
        effort_cost     = (i; env) -> i <= 0 ? 0.0 : env.R * (i / env.a)^(1 / p.γ))
        # defaults: (; monotone_search = :divide_conquer, assume_monotone = false)

    # Match-capital-dependent retention as a 2×2 ROW-stochastic matrix per match
    # value; separated is absorbing. `(; match_capital)` receives the :match_capital
    # axis VALUE (see MarkovStage's dep-closure contract).
    separation_T(m) = [retention_prob(m, p)  1 - retention_prob(m, p);
                       0.0                    1.0]
    separation = MarkovStage(layout; axis = :married,
        transition_matrix = (; match_capital) -> separation_T(match_capital))

    hh = invest ∘ separation
    return define_moments!(hh;
        mean_match_capital = at_end(integrand = (; married, match_capital) -> married == 1 ? match_capital : 0.0, reduce = sum),
        married_rate       = at_end(integrand = (; married) -> married == 1 ? 1.0 : 0.0,                          reduce = sum),
        mean_surplus       = at_end(integrand = (; married, match_capital, env) -> married == 1 ? env.R * match_capital : 0.0, reduce = sum))
end
