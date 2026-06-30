###############################################################
# Directed job search (Moen 1997; Menzio–Shi 2011)           #
#  — income-fluctuation savings + a submarket choice          #
###############################################################

# A worker who BOTH saves and directs their search across posted-wage
# submarkets — the Aiyagari savings problem with a Moen/Menzio–Shi directed-
# search block bolted on. The point of this example: the entire within-period
# problem is FOUR existing library stages, in time order, with **no bespoke
# household stage rolled here** —
#
#     DirectedSearch ∘ Matching ∘ Receipt ∘ ConsumptionSavings
#
# `DirectedSearch` — `DirectedSearchStage` (a `LogitChoiceStage` over the
#                  `:submarket` axis): the unemployed pick which posted-wage
#                  submarket to apply to. The wage/fill-probability tradeoff is
#                  NOT a kwarg here — it enters through `V_end[submarket]`, the
#                  continuation value of being in submarket `j` going into the
#                  matching draw, which the following stages produce. The search
#                  cost varies along `:employment`: the EMPLOYED pay `+Inf` to
#                  switch submarket, so they keep their current job's wage
#                  (immobile, like "owners" in the migration test); the
#                  unemployed pay a finite cost and aim freely.
# `Matching`     — `MarkovStage` on `:employment` whose transition varies along
#                  `:submarket`: an unemployed worker who aimed at submarket `j`
#                  matches (→ employed at wage `w_j`) with the fill probability
#                  `f(w_j)`, a DECREASING schedule (high posted wage ⇒ tight
#                  market ⇒ low fill prob — the Moen/Menzio–Shi tradeoff). The
#                  employed separate at rate `s`.
# `Receipt`      — `WealthChangeStage` `a ↦ (1+r)·a + income`, income `= w_j`
#                  when employed in submarket `j`, the benefit `b` when unemployed.
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period wealth.
#
# Because `DirectedSearch` runs FIRST and `Matching` SECOND, the logit's
# `V_end[submarket]` already integrates "match w.p. f(w_j) into a job paying w_j
# vs. stay unemployed" — exactly the Menzio–Shi submarket-value channel. The
# tradeoff lives entirely in the continuation value; the directed-search stage
# is a plain logit over the submarket axis.
#
# Equilibrium is partial: prices `(r, b)` and the fill schedule `f(·)` are
# exogenous (the Moen block's free-entry tightness schedule is summarised by
# `f(w)`), so the "outer loop" is a single `solve_steady_state_given_env!`.
# Impatience (`β(1+r) < 1`) plus the wealth floor deliver a stationary
# distribution.

using HouseholdStages


# Parameters #
#------------#

# CRRA felicity `u_crra` is provided by HouseholdStages.

@kwdef struct DirectedSearchParams
    β :: Float64       = 0.95
    σ :: Float64       = 2.0                       # CRRA
    r :: Float64       = 0.03                      # gross return on wealth − 1 (exogenous)
    benefit :: Float64 = 0.4                       # unemployment income b
    sep     :: Float64 = 0.05                      # separation rate s (emp → unemp)

    # Posted-wage submarkets. The submarket axis stores the posted wage w_j.
    wages   :: Vector{Float64} = collect(0.8:0.1:1.4)   # 7 submarkets
    # Fill-probability schedule f(w): DECREASING in the posted wage (the
    # Moen/Menzio–Shi tradeoff). Linear between (w_lo ⇒ f_hi) and (w_hi ⇒ f_lo).
    f_hi    :: Float64 = 0.95                      # fill prob at the lowest posted wage
    f_lo    :: Float64 = 0.25                      # fill prob at the highest posted wage

    ε       :: Float64 = 0.05                      # logit dispersion over submarkets (ε→0 ⇒ hard)
    search_cost :: Float64 = 0.0                   # flat application cost (unemployed)

    N_w   :: Int       = 150
    w_min :: Float64   = 0.0
    w_max :: Float64   = 40.0
end

Base.Broadcast.broadcastable(p::DirectedSearchParams) = Ref(p)

const directed_search_params = DirectedSearchParams()


# Directed-search primitives (plain economic functions) #
#-------------------------------------------------------#

"""
Job-fill probability `f(w)` at posted wage `w` — a clamped DECREASING linear
schedule from `f_hi` (at the lowest posted wage) to `f_lo` (at the highest).
This is the Moen/Menzio–Shi wage/fill-probability tradeoff: aiming at a
higher-paying submarket is less likely to land a job.
"""
function fill_prob(w::Real, p = directed_search_params)
    w_lo, w_hi = first(p.wages), last(p.wages)
    t = (w - w_lo) / (w_hi - w_lo)
    return clamp(p.f_hi + t * (p.f_lo - p.f_hi), 0.0, 1.0)
end


# Household chain assembly #
#--------------------------#

"""
Build the directed-search household block `DirectedSearch ∘ Matching ∘ Receipt ∘
ConsumptionSavings`, with `mean_wealth`, `unemp_rate`, and `mean_wage` (over the
employed) attached. Four existing stages, no bespoke household stage.

Employment axis is `[:unemp, :emp]`; the submarket axis stores the posted wage.
The employed are locked to their submarket (`+Inf` switch cost); the unemployed
direct their search across submarkets via the logit.
"""
function directed_search_household(p = directed_search_params)
    layout = GriddedLayout(
        :wealth     => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :employment => Discrete([:unemp, :emp]),
        :submarket  => Discrete(p.wages),
    )

    # (1) Directed search: a logit over submarkets. The unemployed aim freely at a
    # flat cost; the employed pay +Inf to switch (keep their current job's wage).
    # The cost matrix is C[origin_submarket, dest_submarket]; it varies along the
    # :employment dep (the choice axis is the two positional dims, so :employment
    # is the only kwarg). The fill-prob/wage tradeoff is NOT here — it is in V_end.
    n_sub = length(p.wages)
    free   = [i == j ? 0.0 : p.search_cost for i in 1:n_sub, j in 1:n_sub]   # unemployed: aim anywhere
    locked = [i == j ? 0.0 : Inf          for i in 1:n_sub, j in 1:n_sub]    # employed: stay put
    search_cost(; employment) = employment == :emp ? locked : free
    search = DirectedSearchStage(layout;
        search_cost = search_cost, ε = p.ε)   # defaults: (; axis = :submarket)

    # (2) Matching: a Markov draw on :employment whose transition varies along
    # :submarket. Unemployed → employed w.p. f(w_submarket); employed → unemployed
    # w.p. p.sep. T[from, to] is row-stochastic with from,to ∈ {unemp=1, emp=2}.
    function match_transition(; submarket)
        f = fill_prob(submarket, p)
        return [1 - f          f;          # unemp → (stay unemp, match)
                p.sep    1 - p.sep]         # emp   → (separate, keep job)
    end
    matching = MarkovStage(layout; axis = :employment, transition_matrix = match_transition)

    # (3) Receipt: cash-on-hand. Employed earn their submarket wage; unemployed get b.
    receipt = WealthChangeStage(layout;       # defaults: (; axis = :wealth)
        wealth_post = function (; employment, submarket, wealth, env)
            income = employment == :emp ? submarket : env.benefit
            return (1 + env.r) * wealth + income
        end)

    # (4) Savings on the wealth grid.
    savings = ConsumptionSavingsStage(layout;   # defaults: (; axis = :wealth, monotone_search = :divide_conquer)
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)))

    hh = search ∘ matching ∘ receipt ∘ savings
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        unemp_rate  = at_end(
            integrand = (; employment) -> employment == :unemp ? 1.0 : 0.0, reduce = sum),
        # ∫ w · 1{employed} dΛ — divide by employment rate downstream for the mean wage.
        wage_bill   = at_end(
            integrand = (; employment, submarket) -> employment == :emp ? submarket : 0.0, reduce = sum),
    )
end
