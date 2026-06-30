###############################################################
# Castañeda–Díaz-Giménez–Ríos-Rull (2003) — earnings + OLG     #
###############################################################
#
# A heterogeneous-agent OLG model in the spirit of Castañeda, Díaz-Giménez
# and Ríos-Rull (2003, JPE): a persistent idiosyncratic EARNINGS Markov, an
# OLG age structure with RETIREMENT and STOCHASTIC DEATH, and NEWBORNS who
# inherit the ACCIDENTAL BEQUESTS of the deceased.
#
# The within-period BLOCK is a straight `∘`-composition of FOUR EXISTING
# exported stages — no bespoke stage:
#
#     MarkovStage(:earnings) ∘ MarkovStage(:age, sub-stochastic)
#                            ∘ IncomeStage ∘ ConsumptionSavingsStage
#
#   * `MarkovStage(:earnings)`  — the persistent idiosyncratic earnings draw.
#   * `MarkovStage(:age)` with a SUB-STOCHASTIC age matrix — deterministic
#     aging conditional on survival, with a per-period death hazard in
#     retirement and certain death at the maximum age. The rows that sum to
#     `< 1` bleed mass off the age axis: that lost mass IS the deceased.
#     (`AdvanceAgeStage` is exactly a `MarkovStage(:age)` with a shift matrix;
#     here we hand `MarkovStage(:age)` a survival-hazard matrix directly, the
#     same primitive, so the death process is genuinely stochastic.)
#   * `IncomeStage` — cash-on-hand `(1+r)·a + y`, where `y` is `w·earnings`
#     for workers and a flat `pension` for retirees (retirement read off the
#     `age` coordinate inside the receipt closure).
#   * `ConsumptionSavingsStage` — choose next-period wealth; CRRA felicity.
#
# What is CUSTOM example-side code (expected, required, NOT a violation): the
# DEMOGRAPHICS OUTER LOOP in `steady_state.jl` — the newborn inflow, the
# accidental-bequest transfer to newborns, and the death renormalization that
# keeps `ΣΛ = 1`. The block's `MarkovStage(:age)` deletes the deceased; the
# driver re-injects an equal newborn mass at age 1, carrying the deceased's
# wealth as bequests. That cohort-iteration loop is the part no single stage
# expresses, and it is rolled by hand exactly as the life_cycle example rolls
# its finite-horizon sweep.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CastanedaParams
    β :: Float64 = 0.96
    σ :: Float64 = 2.0
    r :: Float64 = 0.03                   # FIXED real return, < 1/β − 1 ≈ 0.0417
    w :: Float64 = 1.0                    # wage scaling worker earnings

    # Persistent idiosyncratic earnings Markov (workers).
    e_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_e    :: Matrix{Float64} = [0.80 0.15 0.05;
                                 0.10 0.80 0.10;
                                 0.05 0.15 0.80]

    # OLG age structure.
    N_age      :: Int     = 8             # number of age stages (each ~ a decade)
    retire_age :: Int     = 5             # ages ≥ retire_age are retired (pension income)
    surv_ret   :: Float64 = 0.80          # per-period survival prob in retirement (death hazard 0.20)
    pension    :: Float64 = 0.6           # flat retirement replacement income

    # Wealth grid.
    N_w   :: Int     = 300
    w_min :: Float64 = 0.0
    w_max :: Float64 = 150.0
end

Base.Broadcast.broadcastable(p::CastanedaParams) = Ref(p)

const castaneda_params = CastanedaParams()


# Demographics primitives (used by the block's age matrix and the driver) #
#------------------------------------------------------------------------#

"""
Sub-stochastic age-transition matrix for `MarkovStage(:age)`. Workers
(`age < retire_age`) advance deterministically and surely survive
(`T[a,a+1] = 1`). Retirees (`retire_age ≤ age < N_age`) advance with survival
probability `surv_ret`; the residual `1 − surv_ret` bleeds off the axis as
death. The maximum age has an all-zero row — certain death. The off-axis
leakage is exactly the deceased mass the demographics driver re-injects as
newborns.
"""
function age_transition(p = castaneda_params)
    n = p.N_age
    T = zeros(Float64, n, n)
    for a in 1:(n - 1)
        survive = a < p.retire_age ? 1.0 : p.surv_ret
        T[a, a + 1] = survive          # age one step, conditional on survival
    end
    # Row n is left all-zero: the oldest cohort dies for sure.
    return T
end

"""
Stationary distribution of the earnings Markov `p.P_e` — the newborn draw over
the persistent earnings state. Power-iterates the row-stochastic transpose.
"""
function earnings_stationary(p = castaneda_params)
    n = length(p.e_grid)
    π = fill(1 / n, n)
    for _ in 1:10_000
        π_next = p.P_e' * π
        maximum(abs, π_next - π) < 1e-14 && (π = π_next; break)
        π = π_next
    end
    return π ./ sum(π)
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Castañeda household BLOCK
`MarkovStage(:earnings) ∘ MarkovStage(:age, sub-stochastic) ∘ IncomeStage
∘ ConsumptionSavingsStage` over the `(wealth, earnings, age)` layout. The age
Markov is the survival-hazard matrix from `age_transition`; the receipt
closure pays `pension` to retirees (`age ≥ retire_age`) and `w·earnings` to
workers. Attaches the aggregate-wealth moment `K_supplied = ∫ wealth dΛ`.
The death/newborn/bequest demographics live in the driver, not the block.
"""
function castaneda_household(p = castaneda_params)
    layout = GriddedLayout(
        :wealth   => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :earnings => Discrete(p.e_grid),
        :age      => Discrete(collect(1.0:p.N_age)),
    )

    earn_shock = MarkovStage(layout; axis = :earnings, transition_matrix = p.P_e)
    aging      = MarkovStage(layout; axis = :age, transition_matrix = age_transition(p))
    receipt    = IncomeStage(layout;
        # Cash-on-hand: gross interest plus labor income. Retirees (age ≥
        # retire_age) get the flat pension; workers get w·earnings.
        wealth_post = (; wealth, earnings, age, env) ->
            (1 + env.r) * wealth + (age ≥ env.retire_age ? env.pension : env.w * earnings),
    )
    savings    = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )

    hh = earn_shock ∘ aging ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the Castañeda block: fixed real return `r`, wage `w`, retirement
threshold `retire_age`, and flat `pension`. The receipt closure reads
`retire_age`/`pension`/`w`; the savings stage reads nothing extra.
"""
castaneda_env(p = castaneda_params) =
    (; r = p.r, w = p.w, retire_age = p.retire_age, pension = p.pension)
