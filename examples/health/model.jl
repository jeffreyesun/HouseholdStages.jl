##################################################################
# Grossman (1972) health capital & mortality — life-cycle cohort  #
##################################################################

# The Grossman health-capital model as a life-cycle household block built from
# EXISTING library stages only — NO bespoke household stage rolled here. The
# whole point of this Part-3 example:
#
#     Household block = HealthInvestStage ∘ MortalityStage
#                     = CapitalInvestmentStage(:health)  ∘  MarkovStage(:alive | health)
#
# State = `(health, alive)`. Two existing stages, time-ordered (`∘` runs the LEFT
# stage first — invest in health, THEN survive on the chosen health):
#
#   HealthInvestStage — `CapitalInvestmentStage` on the `:health` axis. From health stock
#       `h` the agent picks next stock `h'`, paying a convex medical-expenditure
#       cost on gross investment `i = h' − (1−δ)h` (`δ` = health depreciation) and
#       earning the health flow `production(h; env)` (Grossman's consumption +
#       production value of being healthy). This is exactly the Ben-Porath (1967)
#       technology with a HEALTH stock instead of human capital — the catalog's
#       "Health investment (Grossman) = CapitalInvestmentStage with a survival/health flow"
#       row. Identical stage to `examples/human_capital`, different flow.
#
#   MortalityStage — `MarkovStage` on a 2-state `:alive` axis (`[alive, dead]`)
#       whose ROW-stochastic transition is HEALTH-DEPENDENT via a dep closure
#       `(; health) -> [survival(health) 1−survival(health); 0 1]`. Healthier
#       agents survive with higher probability; the dead state is ABSORBING. On
#       the alive sub-mass the rows sum to `survival(health) < 1` — the
#       sub-stochastic survival the model calls for — while total mass on the full
#       `(health, alive)` grid is conserved (it accumulates in the absorbing dead
#       state). The survival probability is a plain economic primitive (a logistic
#       in health) assembled into a matrix and handed to the EXISTING `MarkovStage`:
#       DATA fed to a stage, not a new stage.
#
# Why finite-horizon (life-cycle), not stationary. A stationary distribution with
# mortality but no birth leaks all mass into the dead state — there is no
# nondegenerate stationary alive-mass without a forward mass-injection SOURCE
# (catalog gap G2: `Λ' = K·Λ + M·g`), which is NOT a household stage and is out of
# scope by construction. The finite-horizon cohort sidesteps G2 honestly: a cohort
# is born alive at health `h0`, invests in health each age, and its alive-mass
# decays along the survival curve over the life cycle. No stationarity-of-mass
# requirement, so no birth source needed. This is the Grossman model in the
# Galama–van Kippersluis life-cycle form. See `PART3_LITERATURE_MODELS.md` (the
# Grossman health-investment and Grossman-mortality rows) and `examples/human_capital`
# / `examples/life_cycle` (the finite-horizon driver precedent).
#
# What is example-side (and allowed): the FINITE-HORIZON DRIVER. The age-specific
# medical-care efficiency profile (Grossman's efficiency of health production,
# declining with age) and the survival matrix are threaded through `env`/the stage
# by the backward/forward sweep in `steady_state.jl` — driver logic, never inside a
# household stage.
#
# The effort cost is the inverted health-production function: new health
# `Q = a·i^γ`-style, so to deliver gross investment `i` at medical efficiency `a`
# costs `effort_cost(i; env) = env.R · (i / env.a)^{1/γ}` (convex for `γ ∈ (0,1)`,
# exponent `1/γ > 1`). The reward `production(h) − effort_cost(h' − (1−δ)h)` is then
# supermodular in `(h', h)`, so `CapitalInvestmentStage`'s divide-and-conquer monotone solve
# is valid. Higher efficiency `a` lowers the cost — the engine of the health profile.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HealthParams
    β        :: Float64 = 0.97                # patience
    γ        :: Float64 = 0.7                 # health-production curvature ∈ (0,1)
    δ        :: Float64 = 0.08                # health-capital depreciation
    R        :: Float64 = 1.0                 # value of a unit of health flow (price of being healthy)
    a_young  :: Float64 = 0.42                # efficiency of producing health, age 1
    a_old    :: Float64 = 0.10                # efficiency at the last age (declines with age)
    N_age    :: Int     = 50                  # life-cycle length (ages)
    h0       :: Float64 = 6.0                 # health capital at birth (cohort starts here)
    # Survival hazard — logistic in health: surv(h) = s_max / (1 + exp(−k(h − h_mid))).
    s_max    :: Float64 = 0.995               # survival cap (best attainable per-age survival)
    s_k      :: Float64 = 0.50                # steepness of survival in health
    s_h_mid  :: Float64 = 2.0                 # health at the survival inflection
    N_h      :: Int     = 200                 # health grid points
    h_min    :: Float64 = 1.0
    h_max    :: Float64 = 30.0
end

Base.Broadcast.broadcastable(p::HealthParams) = Ref(p)

const health_params = HealthParams()

"Age-`t` medical-care efficiency (`t = 1…N_age`): a log-linear taper from `a_young`
to `a_old` — the Grossman declining efficiency-of-health-production profile."
function efficiency_at_age(t::Integer, p = health_params)
    s = (t - 1) / max(p.N_age - 1, 1)
    return exp((1 - s) * log(p.a_young) + s * log(p.a_old))
end

"Per-age survival probability at health `h` — a logistic capped at `s_max`. This is
a plain economic primitive (Grossman's health-dependent mortality hazard); the
driver assembles it into the survival `MarkovStage`'s transition matrix."
function survival_prob(h::Real, p = health_params)
    return p.s_max / (1 + exp(-p.s_k * (h - p.s_h_mid)))
end


# Household chain assembly — TWO library stages, NO bespoke stage #
#----------------------------------------------------------------#

"""
Build the Grossman health-capital household block — `MarkovStage(:alive | health)
∘ CapitalInvestmentStage(:health)`, two existing library stages, with `mean_health`,
`survival_rate`, and `mean_medical` moments attached.

The `:alive` axis is `[1, 0]` (alive, dead). The survival `MarkovStage`'s
transition is a HEALTH-DEPENDENT dep closure `(; health) -> T` with the dead state
absorbing — `survival(h)` on the alive→alive entry, `1−survival(h)` on alive→dead.
No bespoke household stage; the age-efficiency profile is threaded through `env` by
the finite-horizon driver, and the survival probability is economic data fed to
`MarkovStage`.
"""
function health_household(p = health_params)
    layout = GriddedLayout(
        :health => GriddedContinuous(p.h_min, p.h_max, p.N_h),
        :alive  => Discrete([1, 0]),          # 1 = alive, 0 = dead
    )

    invest = CapitalInvestmentStage(layout;
        axis            = :health,
        β               = p.β,
        depreciation    = p.δ,
        production      = (h; env) -> env.R * h,
        effort_cost     = (i; env) -> i <= 0 ? 0.0 : env.R * (i / env.a)^(1 / p.γ))

    # Health-dependent survival as a 2×2 ROW-stochastic matrix per health value;
    # dead is absorbing. `(; health)` receives the :health axis VALUE (see
    # MarkovStage's dep-closure contract).
    survival_T(h) = [survival_prob(h, p)  1 - survival_prob(h, p);
                     0.0                   1.0]
    mortality = MarkovStage(layout; axis = :alive,
        transition_matrix = (; health) -> survival_T(health))

    hh = invest ∘ mortality
    return define_moments!(hh;
        mean_health   = at_end(integrand = (; alive, health) -> alive == 1 ? health : 0.0, reduce = sum),
        survival_rate = at_end(integrand = (; alive) -> alive == 1 ? 1.0 : 0.0,            reduce = sum),
        mean_medical  = at_end(integrand = (; alive, health, env) -> alive == 1 ? env.R * health : 0.0, reduce = sum))
end
