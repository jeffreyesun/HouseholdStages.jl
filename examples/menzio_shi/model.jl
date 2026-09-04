###########################################################################
# Menzio–Shi (2011) — directed search ON and OFF the job                    #
###########################################################################

# "Efficient Search on the Job and the Business Cycle" (JPE 2011): BOTH the
# unemployed AND the employed direct their search across posted-wage
# submarkets, trading a higher target wage against a lower fill probability;
# an employed searcher who fails to match keeps their current job. The
# distinguishing feature versus `../directed_search` (where the employed are
# LOCKED) is on-the-job search with a fall-back to the current wage. The
# within-period problem is still a pure `∘`-composition of existing stages,
# in time order:
#
#     Aim ∘ Match ∘ Receipt ∘ ConsumptionSavings
#
# The state carries TWO discrete axes that make on-the-job search expressible
# without any bespoke stage:
#   :submarket — the current job: `unemployed` (a sentinel wage −1) or one of
#                the employed wage tiers.
#   :target    — the wage tier the worker AIMS at this period.
#
# Library stages (NO bespoke household stage in this file):
#   Aim     — `LogitChoiceStage` on the :target axis: choose which posted-wage
#             submarket to apply to. The fill/wage tradeoff is NOT a kwarg — it
#             enters through the target's continuation value (the Match below).
#             The cost is the flat search friction (constant across targets, so
#             the logit is driven purely by the match value).
#   Match   — `MarkovStage` on the :submarket axis whose transition VARIES
#             along :target. Given a target wage `w_t` with fill probability
#             `f(w_t)` (decreasing): the unemployed match → employed at `w_t`
#             w.p. `f`; the employed separate w.p. `s`, else move to `w_t` w.p.
#             `f` ONLY IF `w_t` beats the current wage (on-the-job upgrade),
#             else keep the current job. This choice-conditional probabilistic
#             move IS the Menzio–Shi on-the-job search, expressed as a
#             transition reading the :target dep.
#   Receipt — `WealthChangeStage`: `(1+r)·a + income`, income the current wage
#             (the benefit `b` if unemployed).
#   ConsumptionSavings — `ConsumptionSavingsStage` on the wealth grid.
#
# Block recursivity (Menzio–Shi's elegance) is exactly the condition that
# makes the value functions distribution-independent — so the household block
# is a clean stage chain with no distribution-as-state dependence. Prices
# `(r, b)` and the fill schedule `f(·)` (the free-entry tightness summary) are
# exogenous, so the "outer loop" is a single `solve_steady_state_given_env!`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MenzioShiParams
    β :: Float64 = 0.95
    σ :: Float64 = 2.0
    r :: Float64 = 0.03
    benefit :: Float64 = 0.4           # unemployment income b
    sep     :: Float64 = 0.04          # separation rate s (emp → unemp)

    wages :: Vector{Float64} = collect(0.8:0.1:1.3)   # 6 posted-wage submarkets (employed tiers)
    f_hi  :: Float64 = 0.95            # fill prob at the lowest posted wage
    f_lo  :: Float64 = 0.30            # fill prob at the highest posted wage

    ε :: Float64 = 0.05                # logit dispersion over targets (ε→0 ⇒ hard)
    search_cost :: Float64 = 0.0       # flat application cost (cancels in the logit; here for clarity)

    N_w   :: Int     = 120
    w_min :: Float64 = 0.0
    w_max :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::MenzioShiParams) = Ref(p)

const params = MenzioShiParams()

const UNEMPLOYED = -1.0    # sentinel :submarket level for the unemployed state


# Directed-search primitives (plain economic functions) #
#-------------------------------------------------------#

"""
Job-fill probability `f(w)` at posted wage `w` — a clamped DECREASING linear
schedule from `f_hi` (lowest posted wage) to `f_lo` (highest). Aiming higher
is less likely to land the job (the Moen/Menzio–Shi tradeoff).
"""
function fill_prob(w::Real, p = params)
    w_lo, w_hi = first(p.wages), last(p.wages)
    t = (w - w_lo) / (w_hi - w_lo)
    return clamp(p.f_hi + t * (p.f_lo - p.f_hi), 0.0, 1.0)
end

"The :submarket axis levels: the unemployed sentinel, then the employed wage tiers."
submarket_levels(p = params) = vcat(UNEMPLOYED, p.wages)

"""
The `n_sub × n_sub` row-stochastic Match transition `T[from, to]` over the
:submarket axis, for a worker aiming at target wage `w_t`. Unemployed → employed
at `w_t` w.p. `f(w_t)`; employed separate w.p. `s`, else upgrade to `w_t` w.p.
`f(w_t)` IF `w_t` beats the current wage (on-the-job search), else stay put.
Plain data handed to `MarkovStage` via a `(; target)` closure.
"""
function match_transition(target::Real, p = params)
    sub = submarket_levels(p)
    n   = length(sub)
    f   = fill_prob(target, p)
    t_idx = findfirst(==(target), sub)          # destination submarket index of the target tier
    T = zeros(n, n)
    # Row 1 — currently unemployed: match into the target w.p. f, else stay unemployed.
    T[1, 1]     = 1 - f
    T[1, t_idx] = f
    # Rows 2…n — currently employed at wage sub[k].
    for k in 2:n
        w_k  = sub[k]
        T[k, 1] += p.sep                        # separation → unemployed
        rest = 1 - p.sep
        if target > w_k                         # on-the-job upgrade only to a better submarket
            T[k, t_idx] += rest * f
            T[k, k]     += rest * (1 - f)
        else
            T[k, k]     += rest                 # no upgrade available / not better → keep job
        end
    end
    return T
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Menzio–Shi on-the-job directed-search household
block `Aim ∘ Match ∘ Receipt ∘ ConsumptionSavings` over (wealth, submarket,
target). Moments: mean wealth, unemployment rate, and the employed wage bill.
Four existing stages; the on-the-job match rides the :target dep of `Match`.
"""
function menzio_shi_household(p = params)
    sub = submarket_levels(p)
    layout = GriddedLayout(
        :wealth    => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :submarket => Discrete(sub),
        :target    => Discrete(p.wages),
    )

    # (1) Aim: a logit over target submarkets. The cost is flat (cancels in the
    # softmax); the fill/wage tradeoff lives in V_end (the Match continuation).
    n_t = length(p.wages)
    aim = LogitChoiceStage(layout;
        axis        = :target,
        cost_matrix = fill(p.search_cost, n_t, n_t),   # flat cost (cancels in the softmax)
        ε           = p.ε)

    # (2) Match: a Markov draw on :submarket whose transition varies along :target.
    match = MarkovStage(layout; axis = :submarket,
        transition_matrix = (; target) -> match_transition(target, p))

    # (3) Receipt: cash-on-hand. Employed earn their submarket wage; unemployed get b.
    receipt = WealthChangeStage(layout;       # defaults: (; axis = :wealth)
        wealth_post = function (; submarket, wealth, env)
            income = submarket < 0 ? env.benefit : submarket
            return (1 + env.r) * wealth + income
        end)

    # (4) Savings on the wealth grid.
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = aim ∘ match ∘ receipt ∘ savings
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        unemp_rate  = at_end(integrand = (; submarket) -> submarket < 0 ? 1.0 : 0.0, reduce = sum),
        wage_bill   = at_end(integrand = (; submarket) -> submarket < 0 ? 0.0 : submarket, reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"The env consumed by the chain: prices fixed (partial equilibrium)."
menzio_shi_env(p = params) = (; r = p.r, benefit = p.benefit)
