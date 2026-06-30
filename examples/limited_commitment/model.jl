###############################################################
# Alvarez–Jermann (2000) — Limited Commitment, Decentralized   #
###############################################################

# The DECENTRALIZED household side of an Alvarez–Jermann limited-commitment
# economy: households trade a single asset subject to a STATE-CONTINGENT
# "not-too-tight" solvency constraint — a per-income-state lower bound on
# wealth, tighter in low-income states (where the outside option of default is
# more tempting). Given those bounds, the household's problem is the canonical
# income-fluctuation spine with the solvency floor enforced on the savings
# choice:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage ∘ SolvencyFloor
#
# `IncomeShock` (MarkovStage, axis = :income) — the income Markov draw.
# `IncomeReceipt` (IncomeStage)               — `a ↦ (1+r) a + w·y`.
# `ConsumptionSavingsStage`                   — choose next-period assets;
#       implicit budget `c = a_in − a_end`, CRRA utility.
# `SolvencyFloor` (UtilityStage)              — a per-income-state borrowing
#       floor `a_end ≥ B(y)` enforced as a flow penalty on the chosen wealth:
#       `(; wealth, income, env) -> wealth < env.solvency_bound(income) ? −PEN : 0`.
#       Placed AFTER the savings choice, it masks the continuation value
#       `ConsumptionSavingsStage` optimises over, so the policy never violates
#       the bound. The per-state bound vector `B(y)` rides in `env`.
#
# Because the penalty sits on the CHOSEN end-of-period wealth and the income
# Markov mixes states between periods, the *effective* binding limit a household
# respects is shaped by the bounds of the states it is LIKELY to transition to:
# with persistent income, a high-income household uses more of its (looser) slack
# than a low-income one, so the per-state bounds genuinely shape the stationary
# distribution (high-income households carry more debt). With a hard −∞ bound the
# limit would collapse to the worst-case `min_y B(y)`; the finite penalty retains
# the state-dependence (and is the only form that converges — see below).
#
# OUT OF SCOPE: the planner's promised-utility / Pareto-weight fixed point that
# DERIVES the not-too-tight bounds `B(y)` from primitives. That outer fixed point
# is the caller's; here the bounds are taken as given (calibrated in `env`) and
# only the decentralized household block is built. Fixed-`r` single solve.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY A `UtilityStage` PENALTY, NOT `BorrowingConstraintStage`. The natural API
# is `BorrowingConstraintStage(infeasible = …)`, which masks with `−Inf`. That is
# unusable in the VFI loop: its convergence metric `maximum(abs, V_new .- V)`
# evaluates to `NaN` at steady `−Inf` cells (`−Inf − −Inf = NaN`), so iteration
# stops after 2 passes with an unconverged V (see
# `examples/borrowing_constraint/model.jl` for the full diagnosis). A FINITE
# penalty `−PEN` is the within-constraints stand-in: large enough to deter
# violations, finite so the VFI norm stays well-defined. The grid floor sits well
# inside the natural limit so no cell is an infeasible CRRA trap.
# ─────────────────────────────────────────────────────────────────────────────

using HouseholdStages


# Parameters #
#------------#

@kwdef struct LimitedCommitmentParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    r :: Float64       = 0.03      # FIXED return, strictly < 1/β − 1 ≈ 0.0417
    w :: Float64       = 1.0
    # Three-state income process (persistent — persistence is what lets the
    # per-state bounds shape the distribution).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.80 0.15 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.15 0.80]
    # Not-too-tight solvency bounds B(y), aligned to y_grid: TIGHTER (less
    # negative) in low-income states. Calibrated, not derived (planner's fixed
    # point is out of scope).
    solvency_bounds :: Vector{Float64} = [-0.5, -1.5, -2.5]
    penalty :: Float64 = 50.0      # finite stand-in for the −∞ feasibility mask
    N_a   :: Int       = 250
    a_min :: Float64   = -3.0      # grid floor = loosest bound; well inside the
                                   # natural limit (−y_min/r ≈ −16.7) ⇒ no traps
    a_max :: Float64   = 40.0
end

Base.Broadcast.broadcastable(p::LimitedCommitmentParams) = Ref(p)

const limited_commitment_params = LimitedCommitmentParams()


# Asset grid #
#------------#

"""
Asset grid spanning the borrowing region `[a_min, 0)` (linear) glued to a
log-spaced positive region `[0, a_max]`. `a_min` is the loosest solvency bound
and sits well inside the natural limit, so no cell is an infeasible CRRA trap.
"""
function limited_commitment_asset_grid(p = limited_commitment_params)
    neg = collect(range(p.a_min, 0.0; length = 100 + 1))[1:end-1]
    pos = GriddedContinuous(0.0, p.a_max, p.N_a; spacing = :log).grid
    return GriddedContinuous(vcat(neg, pos))
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached decentralized limited-commitment block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage ∘ SolvencyFloor`. The
`SolvencyFloor` `UtilityStage` penalises chosen wealth below the per-income-state
bound `env.solvency_bound(income)`, placed after the savings choice so it
constrains the policy. Attaches `A_mean` and per-income mean assets (to read the
state-dependent debt capacity off the stationary distribution).
"""
function limited_commitment_household(p = limited_commitment_params)
    grid   = limited_commitment_asset_grid(p)
    layout = GriddedLayout(:wealth => grid, :income => Discrete(p.y_grid))

    shock     = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt   = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis        = :wealth,
    )
    savings   = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    )
    solvency  = UtilityStage(layout;
        utility = (; wealth, income, env) -> wealth < env.solvency_bound(income) ? -env.penalty : 0.0,
    )

    hh = shock ∘ receipt ∘ savings ∘ solvency
    return define_moments!(hh;
        A_mean   = at_end(integrand = :wealth, reduce = sum),
        A_lowy   = at_end(integrand = (; wealth, income) -> income == minimum(p.y_grid) ? wealth : 0.0, reduce = sum),
        A_highy  = at_end(integrand = (; wealth, income) -> income == maximum(p.y_grid) ? wealth : 0.0, reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the fixed-`r` decentralized solvency-constrained household: return `r`,
wage `w`, the finite `penalty`, and `solvency_bound`, a closure mapping an income
grid value to its calibrated per-state lower wealth bound `B(y)` (the vector
`p.solvency_bounds`, aligned to `p.y_grid`).
"""
function limited_commitment_env(p = limited_commitment_params)
    solvency_bound = let yg = p.y_grid, sv = p.solvency_bounds
        y -> sv[findfirst(==(y), yg)]
    end
    return (; r = p.r, w = p.w, penalty = p.penalty, solvency_bound)
end
