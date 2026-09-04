###############################################################
# De Nardi–French–Jones (2010) — medical expenses & mortality  #
###############################################################

# Retired elderly save against two intertwined old-age risks: uncertain
# out-of-pocket MEDICAL EXPENSES (high and rising near death) and uncertain
# MORTALITY (health- and age-dependent). The central finding: the medical-expense
# risk that spikes near the end of life rationalises why the elderly DISSAVE so
# slowly — they hold a precautionary buffer against a costly, uncertain death.
#
# The household block is **one existing library object** — a `replicate_age` of a
# `∘`-chain of EXPORTED stages:
#
#   replicate_age(HealthShock ∘ Medical ∘ ConsumptionSavings ∘ Exit, N; axis=:age)
#
# Each age-slice is:
#   HealthShock — `MarkovStage` on the `:health` axis (good/bad health dynamics).
#   Medical     — `WealthChangeStage` mapping wealth to cash-on-hand net of the
#                 stochastic medical expense `m(health, age)`:
#                   x = max( (1+r)·w + pension − m(health, age),  c_floor ),
#                 the `c_floor` a means-tested consumption floor (Medicaid). The
#                 expense is stochastic because `health` follows a Markov chain
#                 and the expense reads BOTH the (random) health state and age.
#   ConsumptionSavings — `ConsumptionSavingsStage` picks next assets `a'`, `c = x − a'`.
#   Exit        — `ExogenousExit(survival = s(health, age), bequest)`. The survival
#                 hazard is exactly a dep closure reading the `:health` axis AND the
#                 age via `env` — health-and-age-dependent mortality. The dying
#                 fraction `(1−s)` LEAVES the cohort each age, so the cohort shrinks
#                 with age (correct: a retired cohort dies out).
#
# `replicate_age` requires identical age-slices, so the medical expense and the
# survival hazard both read the age through the per-age `env` — every age-slice is
# the SAME chain. The finite-horizon backward + forward-cohort sweep lives in
# `steady_state.jl` (a `ProductStage` does not thread continuations across ages).
# Partial equilibrium: `r`, pension exogenous.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct DFJParams
    β :: Float64       = 0.985
    σ :: Float64       = 2.0
    r :: Float64       = 0.02
    N :: Int           = 20                        # retirement ages (e.g. 70 … 89)
    pension :: Float64 = 0.55                      # flat public pension
    c_floor :: Float64 = 0.05                      # means-tested consumption floor (Medicaid)
    # Health: 2-state Markov (1 = good, 2 = bad). Bad health is persistent.
    P_h :: Matrix{Float64} = [0.90 0.10;
                              0.30 0.70]
    # Medical expense m(health, age) = med_base[health] · (1 + med_growth·(age−1)).
    # Rises with age (the near-death spike) and is far larger in bad health.
    med_base   :: Vector{Float64} = [0.05, 0.40]
    med_growth :: Float64 = 0.18
    # Survival s(health, age): good health survives better; both decline with age.
    # s(h, age) = clamp(s0[h] − s_decline·(age−1), s_min, 1). Strictly positive.
    s0        :: Vector{Float64} = [0.985, 0.93]
    s_decline :: Float64 = 0.018
    s_min     :: Float64 = 0.55
    # Warm-glow bequest b(a') = φ·u_crra(a' + κ) over assets left behind.
    φ :: Float64 = 14.0
    κ :: Float64 = 3.0
    N_w   :: Int       = 100
    w_min :: Float64   = 0.0
    w_max :: Float64   = 30.0
    # Newborn (newly-retired) initial assets — a point mass at the nearest grid node.
    w_init :: Float64  = 8.0
end

Base.Broadcast.broadcastable(p::DFJParams) = Ref(p)

const dfj_params = DFJParams()


# Age/health profiles #
#---------------------#

"""
Out-of-pocket medical expense `m(health, age)`: a health-specific base scaled up
with age (the near-death medical-expense spike), far larger in bad health.
"""
dfj_medical(health::Integer, age::Integer, p = dfj_params) =
    p.med_base[health] * (1 + p.med_growth * (age - 1))

"""
Survival probability `s(health, age)`: a health-specific level declining linearly
with age, clamped to `[p.s_min, 1]`. Good health survives better than bad; both
fall with age. Strictly positive at every age.
"""
dfj_survival(health::Integer, age::Integer, p = dfj_params) =
    clamp(p.s0[health] - p.s_decline * (age - 1), p.s_min, 1.0)

"""
Stationary distribution of the health Markov chain `p.P_h` — the newly-retired
health draw. Power-iterates the row-stochastic transpose.
"""
function dfj_health_stationary(p = dfj_params)
    n = size(p.P_h, 1)
    π = fill(1 / n, n)
    for _ in 1:10_000
        π_next = p.P_h' * π
        maximum(abs, π_next - π) < 1e-14 && (π = π_next; break)
        π = π_next
    end
    return π ./ sum(π)
end


# Household chain assembly #
#--------------------------#

"""
The DFJ household block: `replicate_age(HealthShock ∘ Medical ∘ ConsumptionSavings
∘ Exit, N; axis = :age)` with a `mean_wealth` moment attached. The within-period
chain and the `(wealth, health, exiting, age)` layout are inlined; `:exiting` is the
size-1 axis the exit composite grows, `:age` the size-1 axis `replicate_age` grows to
`N`. Medical expense and survival both read the age through the per-age `env`, so
every age-slice is the SAME chain. The finite-horizon sweep lives in
`steady_state.jl`.
"""
function dfj_household(p = dfj_params)
    layout = GriddedLayout(
        :wealth  => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :health  => Discrete([1, 2]),
        :exiting => Discrete([0]),
        :age     => Discrete([1]),
    )

    shock = MarkovStage(layout; axis = :health, transition_matrix = p.P_h)
    # Cash-on-hand net of the stochastic medical expense, floored at the Medicaid level.
    medical = WealthChangeStage(layout;
        wealth_post = (; wealth, health, env) ->
            max((1 + env.r) * wealth + env.pension - dfj_medical(health, env.age, p), p.c_floor))
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )
    # Health-and-age-dependent mortality is exactly ExogenousExit's `survival` dep closure.
    exit = ExogenousExit(layout;
        survival = (; health, env) -> dfj_survival(health, env.age, p),
        bequest  = (; wealth) -> p.φ * u_crra(wealth + p.κ, Val(p.σ)),
    )

    age_chain = shock ∘ medical ∘ savings ∘ exit
    hh = replicate_age(age_chain, p.N; axis = :age)
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
