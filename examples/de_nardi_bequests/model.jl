###############################################################
# De Nardi (2004) — bequests, inheritance & wealth concentration #
###############################################################

# Finite-lived agents save over the life cycle and leave bequests (voluntary
# warm-glow + accidental) when they die. Inherited wealth links one generation
# to the next, forming a DYNASTY whose stationary wealth distribution has a
# fatter top tail than a model without bequests.
#
# The household block is **one existing library object** — a `replicate_age`
# of a `∘`-chain of EXPORTED stages, exactly the catalog decomposition:
#
#   replicate_age(IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Exit, N; axis=:age)
#
# Each age-slice is:
#   IncomeShock — `MarkovStage` on the persistent earnings state.
#   Receipt     — `WealthChangeStage` `b ↦ (1+r)·b + y(age)·ε` (cash-on-hand).
#   ConsumptionSavings — `ConsumptionSavingsStage` picks next-period assets `a'`;
#                 `c = x − a'` on the wealth grid.
#   Exit        — `ExogenousExit(survival = s(age), bequest = warm-glow over a')`.
#                 Backward: continuation `V` mixed with the value of dying,
#                 `s·V_end + (1−s)·b(a')`; forward: the `(1−s)` fraction leaves
#                 the cohort (mass leaks — that is mortality). The warm-glow
#                 bequest `b(a') = φ·u(a'+κ)` is the value of leaving assets `a'`.
#
# `replicate_age` requires identical age-slices, so exit applies at EVERY age via
# an env-dependent survival closure `s(env.age)` (accidental bequests at every
# age, not only the terminal one). All age slices are structurally one chain; the
# age only enters through the per-age `env`.
#
# What is example-side (and allowed): the FINITE-HORIZON DRIVER plus the DYNASTIC
# CLOSURE, both in `steady_state.jl`. A `ProductStage`'s `backward!` runs each age
# independently, so the life-cycle solve is a backward sweep `a = N…1` threading
# each age's continuation into the previous, then a forward cohort sweep `a = 1…N`.
# The dynasty is closed by an OUTER loop: the cross-age distribution of assets
# bequeathed by the dying must equal the assets newborns inherit — a fixed point
# in the newborn inherited-wealth distribution `g`, seeded each pass through an
# `EntryStage(entry = g)`. Returns are exogenous (partial equilibrium).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct DeNardiParams
    β :: Float64       = 0.95
    σ :: Float64       = 2.0                       # CRRA over consumption AND bequests
    r :: Float64       = 0.04
    N :: Int           = 20                        # life-cycle periods (ages)
    # Persistent earnings state (3-state Markov on log earnings).
    ε_grid :: Vector{Float64} = [0.6, 1.0, 1.6]
    P_ε    :: Matrix{Float64} = [0.80 0.15 0.05;
                                 0.10 0.80 0.10;
                                 0.05 0.15 0.80]
    # Hump-shaped deterministic age-earnings, retirement replacement after retire_age.
    peak_age   :: Int     = 12
    retire_age :: Int     = 14
    y_peak     :: Float64 = 1.0
    y_curv     :: Float64 = 0.5
    repl       :: Float64 = 0.4
    # Survival s(age): high early, declining with age (accidental bequests rise late).
    # `survival_floor` at the last working ages; everyone in age N dies for sure in the
    # dynastic accounting (see steady_state.jl). All entries are strictly > 0.
    s_young :: Float64 = 0.995
    s_old   :: Float64 = 0.90
    # Warm-glow (joy-of-giving) bequest: b(a') = φ·u_crra(a' + κ). κ large ⇒ bequests a
    # luxury good (only the rich leave them) — De Nardi's mechanism for the fat tail.
    φ :: Float64 = 8.0
    κ :: Float64 = 4.0
    N_w   :: Int       = 100
    w_min :: Float64   = 0.0
    w_max :: Float64   = 80.0
end

Base.Broadcast.broadcastable(p::DeNardiParams) = Ref(p)

const de_nardi_params = DeNardiParams()


# Age profiles #
#-------------#

"""
Deterministic age-earnings `y(age)`: a downward quadratic peaking at `p.peak_age`,
dropping to `p.y_peak·(1−p.y_curv)` at the endpoints, flat retirement replacement
`p.repl·p.y_peak` after `p.retire_age`.
"""
function dn_age_earnings(age::Integer, p = de_nardi_params)
    age > p.retire_age && return p.repl * p.y_peak
    span = max(p.peak_age - 1, p.N - p.peak_age)
    drop = p.y_curv * ((age - p.peak_age) / span)^2
    return p.y_peak * (1 - drop)
end

"""
Age-specific survival probability `s(age)`: linearly declines from `p.s_young` at
age 1 to `p.s_old` at age `N`. Strictly positive at every age (the terminal forced
death is handled in the dynastic accounting, not by a zero survival here).
"""
function dn_survival(age::Integer, p = de_nardi_params)
    p.N == 1 && return p.s_young
    t = (age - 1) / (p.N - 1)
    return p.s_young + t * (p.s_old - p.s_young)
end

"""
Stationary distribution of the earnings Markov chain `p.P_ε` (the newborn earnings
draw). Power-iterates the row-stochastic transpose.
"""
function dn_income_stationary(p = de_nardi_params)
    n = length(p.ε_grid)
    π = fill(1 / n, n)
    for _ in 1:10_000
        π_next = p.P_ε' * π
        maximum(abs, π_next - π) < 1e-14 && (π = π_next; break)
        π = π_next
    end
    return π ./ sum(π)
end


# Household chain assembly #
#--------------------------#

"""
The De Nardi household block: `replicate_age(IncomeShock ∘ Receipt ∘
ConsumptionSavings ∘ Exit, N; axis = :age)` with a `mean_wealth` moment attached.
The within-period chain and the `(wealth, income, exiting, age)` layout are inlined;
`:exiting` is the size-1 axis the exit composite grows `1 → 2 → 1`, `:age` the size-1
axis `replicate_age` grows to `N`. Survival rides the per-age `env` (`s(env.age)`),
so every age-slice is the SAME chain. The bequest is the warm-glow `φ·u_crra(a'+κ)`
over the assets `a'` left behind. The finite-horizon sweep + dynastic closure live in
`steady_state.jl`.
"""
function de_nardi_household(p = de_nardi_params)
    layout = GriddedLayout(
        :wealth  => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income  => Discrete(p.ε_grid),
        :exiting => Discrete([0]),
        :age     => Discrete([1]),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_ε)
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.y * income)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )
    # Warm-glow bequest over the assets `a'` carried into death; env-independent
    # (the exit composite materialises it once, against the wealth grid).
    exit = ExogenousExit(layout;
        survival = (; env) -> dn_survival(env.age, p),
        bequest  = (; wealth) -> p.φ * u_crra(wealth + p.κ, Val(p.σ)),
    )

    age_chain = shock ∘ receipt ∘ savings ∘ exit
    hh = replicate_age(age_chain, p.N; axis = :age)
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
