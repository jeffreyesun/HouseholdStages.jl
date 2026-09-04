###############################################################
# Life-cycle / OLG consumption-savings — finite horizon        #
###############################################################

# A finite-horizon life-cycle household (Gourinchas–Parker 2002;
# Cocco–Gomes–Maenhout 2005). The household block is **one existing
# library object**:
#
#     replicate_age(IncomeShock ∘ Receipt ∘ ConsumptionSavings, N; axis = :age)
#
# i.e. a `ProductStage` that stacks `N` age-specific copies of the
# Aiyagari within-period chain along an `:age` axis. Each age-slice is:
#
#   IncomeShock — `MarkovStage` on the income axis (persistent earnings risk).
#   Receipt     — `WealthChangeStage` `b ↦ (1+r)·b + y(age)·ε` (cash-on-hand,
#                 deterministic hump-shaped age earnings `y(age)` times the
#                 transitory/persistent income state `ε`).
#   ConsumptionSavings — `ConsumptionSavingsStage` picks next-period wealth
#                 `b'` on the wealth grid; `c = x − b'`.
#
# **No bespoke household stage is rolled here** — the block is `replicate_age`
# of a `∘`-chain of existing stages, exactly the catalog decomposition.
#
# What is example-side (and allowed): the FINITE-HORIZON DRIVER. A
# `ProductStage`'s own `backward!` runs each age-slice independently
# (block-diagonal direct sum) — it does NOT thread age-(a+1)'s continuation
# value into age a. So the life-cycle solve is rolled in `steady_state.jl`
# as (i) a single backward sweep `a = N…1` feeding each age's continuation
# value into the previous age's slice, then (ii) a forward cohort simulation
# `a = 1…N` from newborns. Both drive the ProductStage's per-age components
# (`hh.buffer.components[a]`) — plain example outer-loop logic, no new stage.
#
# Returns and the age-earnings profile are exogenous (partial equilibrium,
# the standard GP/CGM setup): there is no market to clear.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct LifeCycleParams
    β :: Float64       = 0.96
    σ :: Float64       = 2.0                       # CRRA
    r :: Float64       = 0.03                      # exogenous gross-of-depreciation real rate
    N :: Int           = 30                        # number of life-cycle periods (ages)
    # Persistent income state (CGM-style 3-state Markov on log earnings).
    ε_grid :: Vector{Float64} = [0.7, 1.0, 1.3]
    P_ε    :: Matrix{Float64} = [0.8 0.15 0.05;
                                 0.1 0.80 0.10;
                                 0.05 0.15 0.80]
    # Hump-shaped deterministic age-earnings (Gourinchas–Parker): a quadratic
    # in age peaking at `peak_age`, scaled to `y_peak`, with a retirement
    # replacement `repl` after `retire_age`.
    peak_age   :: Int     = 18
    retire_age :: Int     = 24
    y_peak     :: Float64 = 1.0
    y_curv     :: Float64 = 0.45                    # depth of the hump at the endpoints
    repl       :: Float64 = 0.5                     # retirement replacement of peak earnings
    N_w   :: Int       = 120
    w_min :: Float64   = 0.0
    w_max :: Float64   = 60.0
end

Base.Broadcast.broadcastable(p::LifeCycleParams) = Ref(p)

const life_cycle_params = LifeCycleParams()


# Earnings profile and the ergodic newborn income distribution #
#-------------------------------------------------------------#

"""
Deterministic age-earnings `y(age)` (Gourinchas–Parker hump): a downward
quadratic in age peaking at `p.peak_age`, normalised to `p.y_peak` at the
peak and dropping to `p.y_peak·(1−p.y_curv)` at the life endpoints; after
`p.retire_age` earnings are the flat retirement replacement `p.repl·p.y_peak`.
"""
function age_earnings(age::Integer, p = life_cycle_params)
    age > p.retire_age && return p.repl * p.y_peak
    span = max(p.peak_age - 1, p.N - p.peak_age)        # half-width to the farther endpoint
    drop = p.y_curv * ((age - p.peak_age) / span)^2
    return p.y_peak * (1 - drop)
end

"""
Stationary distribution of the income Markov chain `p.P_ε` — the newborn draw
over the persistent income state. Power-iterates the row-stochastic transpose.
"""
function income_stationary(p = life_cycle_params)
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
The life-cycle household block: `replicate_age(IncomeShock ∘ Receipt ∘
ConsumptionSavings, N; axis = :age)` with a life-cycle `mean_wealth` moment
attached. The `:age` axis enters the layout as a size-1 singleton;
`replicate_age` grows it to `N`, one slice per age.
"""
function life_cycle_household(p = life_cycle_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.ε_grid),
        :age => Discrete([1]),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_ε)
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.y * income) # defaults: (; axis = :wealth)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    ) # defaults: (; axis = :wealth, utility_axes = nothing, skip_monotonicity_check = false)

    age_chain = shock ∘ receipt ∘ savings
    hh = replicate_age(age_chain, p.N; axis = :age)
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
