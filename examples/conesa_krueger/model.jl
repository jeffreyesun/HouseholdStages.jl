###############################################################
# Conesa–Krueger (1999) — OLG with PAYG social security        #
###############################################################

# Conesa & Krueger (1999), "Social Security Reform with Heterogeneous
# Agents", in its steady-state, partial-equilibrium form. The household is
# the life_cycle household PLUS an unfunded pay-as-you-go (PAYG) social
# security system: working-age agents pay a flat payroll tax `τ` on labor
# earnings; retirees (age > `retire_age`) draw a flat benefit `b`. With
# each cohort carrying unit mass, the unfunded PAYG balance is
#
#     b · (#retired cohorts) = τ · (aggregate labor earnings).
#
# The household block is the SAME `replicate_age(…)` of existing stages as
# `examples/life_cycle`, with **no bespoke household stage** — the social
# security tax/benefit rides the receipt `WealthChangeStage` closure, which
# reads the SS rule (net wage, benefit) out of the per-age `env`:
#
#     replicate_age( IncomeShock ∘ Receipt+SS ∘ ConsumptionSavings , N; axis = :age )
#
#   IncomeShock  — `MarkovStage` on the income axis (persistent earnings risk).
#   Receipt+SS   — `WealthChangeStage` `b ↦ (1+r)·b + (1−τ)·y(age)·ε + benefit`.
#                  A worker pays the payroll tax `τ` on labor earnings
#                  `y(age)·ε` and draws no benefit; a retiree has no labor
#                  earnings (`y=0`) and draws the flat benefit `b`. Both the
#                  net wage and the benefit arrive through the per-age `env`
#                  (the `:age` axis is a size-1 singleton inside each
#                  `replicate_age` component, so age-dependence is threaded
#                  by the driver's `env_age(a)`, exactly as life_cycle does
#                  with the hump `y(age)`).
#   ConsumptionSavings — `ConsumptionSavingsStage` picks next-period wealth
#                  `b'` on the wealth grid; `c = x − b'`.
#
# What is example-side (and allowed): (i) the FINITE-HORIZON DRIVER — a
# `ProductStage`'s own `backward!` runs each age-slice independently and
# does NOT thread age-(a+1)'s continuation into age a, so the life-cycle
# solve is a custom backward-sweep + forward-cohort driver in
# `steady_state.jl` (copied from life_cycle); and (ii) the PAYG
# BUDGET-BALANCE fixed point on the benefit `b` (a plain outer loop, the
# same status as a tatonnement on K̄). Prices `r`, `w` and the earnings
# profile are exogenous (partial equilibrium); the SS system clears.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct ConesaKruegerParams
    β :: Float64       = 0.96
    σ :: Float64       = 2.0                        # CRRA
    r :: Float64       = 0.04                       # exogenous real rate
    N :: Int           = 24                         # number of life-cycle periods (ages)
    retire_age :: Int  = 16                         # work for ages 1…retire_age; retired after
    τ :: Float64       = 0.10                       # PAYG payroll tax on labor earnings
    # Persistent income state (3-state Markov on idiosyncratic earnings).
    ε_grid :: Vector{Float64} = [0.7, 1.0, 1.3]
    P_ε    :: Matrix{Float64} = [0.8 0.15 0.05;
                                 0.1 0.80 0.10;
                                 0.05 0.15 0.80]
    # Hump-shaped deterministic age-earnings (peaks at `peak_age`), zero after
    # retirement (retirees live purely off the SS benefit + their savings).
    peak_age :: Int     = 12
    y_peak   :: Float64 = 1.0
    y_curv   :: Float64 = 0.45                       # depth of the hump at the working-life endpoints
    N_w   :: Int       = 100
    w_min :: Float64   = 0.0
    w_max :: Float64   = 40.0
end

Base.Broadcast.broadcastable(p::ConesaKruegerParams) = Ref(p)

const conesa_krueger_params = ConesaKruegerParams()


# Earnings profile and the ergodic newborn income distribution #
#-------------------------------------------------------------#

"""
Deterministic age-earnings `y(age)` for a worker (Conesa–Krueger hump): a
downward quadratic in age over the working life, normalised to `p.y_peak` at
`p.peak_age` and dropping to `p.y_peak·(1−p.y_curv)` at the working-life
endpoints. Retirees (`age > p.retire_age`) have NO labor earnings — they live
off the social-security benefit and their savings — so this returns `0`.
"""
function age_earnings(age::Integer, p = conesa_krueger_params)
    age > p.retire_age && return 0.0
    span = max(p.peak_age - 1, p.retire_age - p.peak_age)   # half-width to the farther work endpoint
    drop = p.y_curv * ((age - p.peak_age) / span)^2
    return p.y_peak * (1 - drop)
end

"""
Stationary distribution of the income Markov chain `p.P_ε` — the newborn draw
over the persistent income state. Power-iterates the row-stochastic transpose.
"""
function income_stationary(p = conesa_krueger_params)
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
The Conesa–Krueger OLG household block: `replicate_age(IncomeShock ∘ Receipt+SS ∘
ConsumptionSavings, N; axis = :age)` with a `mean_wealth` moment attached. The
`:age` axis enters the layout as a size-1 singleton; `replicate_age` grows it to
`N`, one slice per age.
"""
function conesa_krueger_household(p = conesa_krueger_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.ε_grid),
        :age    => Discrete([1]),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_ε)
    # Receipt with the PAYG social-security wedge. `env.netwage = (1−τ)·y(age)`
    # is the post-tax wage per efficiency unit for a worker (0 for a retiree),
    # and `env.benefit` is the flat SS benefit for a retiree (0 for a worker);
    # both are seated per age by the driver's `env_age(a)`.
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) ->
            (1 + env.r) * wealth + env.netwage * income + env.benefit) # defaults: (; axis = :wealth)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    ) # defaults: (; axis = :wealth, utility_axes = nothing, skip_monotonicity_check = false)

    age_chain = shock ∘ receipt ∘ savings
    hh = replicate_age(age_chain, p.N; axis = :age)
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
