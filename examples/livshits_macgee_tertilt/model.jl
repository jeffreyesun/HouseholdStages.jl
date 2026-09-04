###############################################################
# Livshits–MacGee–Tertilt (2007) — life-cycle bankruptcy       #
###############################################################

# Livshits, MacGee & Tertilt (2007), "Consumer Bankruptcy: A Fresh Start",
# in its partial-equilibrium form. Finite-lived households face persistent
# earnings risk and a hump-shaped age-earnings profile, can borrow
# (`a_min < 0`), and each period in good standing choose REPAY vs FILE for
# Chapter-7 bankruptcy. Filing discharges all debt (assets reset to 0) at
# the cost of an exclusion/garnishment spell: a filer's debt is wiped, but
# while excluded it cannot borrow, it suffers an income garnishment
# `(1−λ)·earnings`, and it must wait (geometrically) to be readmitted to
# credit markets.
#
# This is the `examples/default` within-period chain — five existing
# library stages — wrapped in `replicate_age` and solved with the life_cycle
# finite-horizon driver. **No bespoke household stage** is rolled:
#
#     replicate_age( IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘
#                    ConsumptionSavings ∘ Readmission , N; axis = :age )
#
#   IncomeShock  — `MarkovStage` on the `:income` axis (persistent earnings
#                  risk; the deterministic age-hump `y(age)` rides `env`).
#   DefaultChoice— `DefaultStage`: a gated argmax on the 2-level `:status`
#                  axis (1 = good standing, 2 = excluded). A good-standing
#                  household chooses repay (stay 1, score 0) or file (→2,
#                  score `−χ`); an excluded household is gated to stay
#                  excluded (it cannot re-enter at will).
#   DebtReset    — `WealthChangeStage` reading `cell.status`: an excluded /
#                  filing household carries zero assets (debt discharged); a
#                  good-standing repayer keeps `a`.
#   Receipt      — `WealthChangeStage` reading `cell.status`: cash-on-hand
#                  `x = (1+r)·a + y(age)·ε` in good standing; an excluded
#                  household has no debt and suffers the garnishment
#                  `(1−λ)·y(age)·ε`.
#   ConsumptionSavings — `ConsumptionSavingsStage` picks next assets
#                  `a' ∈ grid` (floor `a_min < 0`); `c = x − a'`.
#   Readmission  — `MarkovStage` on `:status`: an excluded household regains
#                  good standing next period w.p. `ψ` (mean spell `1/ψ`); a
#                  good-standing household stays good.
#
# The `:status` and `:income` axes are PER-AGE state — `replicate_age`
# stacks the full six-stage chain per age. The finite-horizon DRIVER in
# `steady_state.jl` (copied from life_cycle) threads the continuation value
# across ages and runs a forward cohort sim; the terminal condition forbids
# dying in debt (a no-bequest "no negative estate" rule on the terminal
# continuation array — example code, not a stage). Prices are exogenous
# (risk-free unit bond, partial equilibrium).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct LMTParams
    β :: Float64       = 0.96                       # mild impatience drives the borrowing motive
    σ :: Float64       = 2.0                         # CRRA
    r :: Float64       = 0.02                         # exogenous risk-free rate
    N :: Int           = 24                           # number of life-cycle periods (ages)
    λ :: Float64       = 0.45                          # garnishment / income cost while excluded
    χ :: Float64       = 0.05                           # flat per-period utility cost of exclusion (filing stigma)
    ψ :: Float64       = 0.12                            # readmission prob (mean exclusion spell 1/ψ ≈ 8 periods)
    # Persistent earnings risk (3-state Markov on idiosyncratic earnings).
    ε_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_ε    :: Matrix{Float64} = [0.7 0.25 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.25 0.70]
    # Hump-shaped deterministic age-earnings (peaks at `peak_age`); modest
    # endpoints so that young low-income households want to borrow.
    peak_age :: Int     = 14
    y_peak   :: Float64 = 1.0
    y_curv   :: Float64 = 0.40
    N_a   :: Int       = 120
    a_min :: Float64   = -0.30                        # borrowing limit (debt is a < 0)
    a_max :: Float64   = 6.0
end

Base.Broadcast.broadcastable(p::LMTParams) = Ref(p)

const lmt_params = LMTParams()


# Earnings profile and the ergodic newborn income distribution #
#-------------------------------------------------------------#

"""
Deterministic age-earnings `y(age)` (hump): a downward quadratic in age,
normalised to `p.y_peak` at `p.peak_age` and dropping to `p.y_peak·(1−p.y_curv)`
at the life endpoints. Low young earnings + persistent downside risk are what
make borrowing and bankruptcy live in the life-cycle bankruptcy model.
"""
function age_earnings(age::Integer, p = lmt_params)
    span = max(p.peak_age - 1, p.N - p.peak_age)
    drop = p.y_curv * ((age - p.peak_age) / span)^2
    return p.y_peak * (1 - drop)
end

"""
Stationary distribution of the income Markov chain `p.P_ε` — the newborn draw
over the persistent earnings state. Power-iterates the row-stochastic transpose.
"""
function income_stationary(p = lmt_params)
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
The Livshits–MacGee–Tertilt life-cycle bankruptcy household block:
`replicate_age(IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘
ConsumptionSavings ∘ Readmission, N; axis = :age)` with `mean_assets` and
`excluded_rate` moments attached. The within-period chain is the
`examples/default` chain verbatim (the file/credit-market mechanics are
identical); the only life-cycle additions are the hump earnings `y(age)` and
the finite horizon, both supplied by the driver. The `(wealth, income, status,
age)` layout is inlined with `:age` a size-1 singleton (the product grows it to
`N`). Returns the moment-attached `ProductStage`.
"""
function lmt_household(p = lmt_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :linear),
        :income => Discrete(p.ε_grid),
        :status => Discrete([1, 2]),   # 1 = good standing, 2 = excluded / just filed
        :age    => Discrete([1]),
    )

    shock = MarkovStage(layout; axis = :income, transition_matrix = p.P_ε)

    # DefaultChoice: a good-standing household chooses repay (stay 1, score 0)
    # or file (→ 2, score −χ); an excluded household is gated to stay excluded.
    excl_gate(before, after) = before == 1 ? true : after == 2
    default = DefaultStage(layout; default_penalty = p.χ, avail = excl_gate) # defaults: (; axis = :status, default_index = 2)

    # DebtReset: an excluded / filing household carries zero assets (debt
    # discharged); a good-standing repayer keeps `a`.
    reset = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; status, wealth) -> status == 2 ? 0.0 : wealth)

    # Receipt: cash-on-hand. An excluded household earns the garnished income
    # `(1−λ)·y(age)·ε` and has no debt; a good-standing repayer earns
    # `y(age)·ε` and services its debt (`a < 0`) out of cash-on-hand. The hump
    # `env.y = y(age)` is seated per age by the driver.
    receipt = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; status, income, wealth, env) ->
            status == 2 ? (1 - env.λ) * env.y * income
                        : (1 + env.r) * wealth + env.y * income)

    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth)

    # Readmission: an excluded household regains good standing next period w.p.
    # ψ; a good-standing household stays good. `T[from, to]` is row-stochastic.
    readmit = MarkovStage(layout; axis = :status,
        transition_matrix = [1.0 0.0; p.ψ (1 - p.ψ)])

    age_chain = shock ∘ default ∘ reset ∘ receipt ∘ savings ∘ readmit
    hh = replicate_age(age_chain, p.N; axis = :age)
    return define_moments!(hh;
        mean_assets   = at_end(integrand = :wealth, reduce = sum),
        excluded_rate = at_end(integrand = (; status) -> status == 2 ? 1.0 : 0.0, reduce = sum),
    )
end
