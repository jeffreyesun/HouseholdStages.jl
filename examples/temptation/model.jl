###############################################################
# Temptation & Self-Control (Gul–Pesendorfer 2001)            #
###############################################################
#
# A heterogeneous-agent consumption-savings model with Gul–Pesendorfer (GP)
# temptation preferences. The within-period ranking of a feasible plan is
#
#     u(c) + w(c) − max_{c̃ feasible} w(c̃)
#
# `u` is the commitment (long-run) utility and `w` is the temptation utility;
# the bracketed term `w(c) − max_{c̃} w(c̃) ≤ 0` is the self-control cost of
# resisting the most tempting feasible option.
#
# KEY INSIGHT (what makes this a clean compositional build). If `w` is
# monotone increasing in consumption, the most-tempting feasible option is to
# consume EVERYTHING this period — i.e. drive next-period wealth to the
# borrowing floor `b_min`. With post-income cash-on-hand `m = cell.wealth`,
# that corner consumption is `c̃* = m − b_min`, and the temptation peak is the
# CLOSED-FORM constant
#
#     max_{c̃} w(c̃) = w(m − b_min).
#
# It depends only on the cell's cash-on-hand `m`, NOT on the savings choice
# `c`, so as a function of the choice it is a CONSTANT additive shift. It
# therefore shifts the within-period value `V` (the GP self-control cost is
# real) WITHOUT distorting the savings policy that maximises `u(c) + w(c)` —
# exactly the GP solution, and time-consistent.
#
# The catalog flags general temptation as ◐ because the temptation max ranges
# over the SAME feasible set as the choice (a shared-feasible-set coupling).
# That caveat DISSOLVES under monotone `w`: the max collapses to the single
# corner `b' = b_min`, which is a closed form in the origin cash-on-hand, so
# there is nothing to couple — it folds into one felicity closure.
#
# The within-period BLOCK is thus the canonical Aiyagari spine — three
# EXISTING exported stages, no bespoke stage:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#     = MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
#
# Receipt is placed BEFORE savings (as in Aiyagari), so inside the savings
# closure `cell.wealth` is post-receipt cash-on-hand `m = (1+r)b + w·y`. The
# felicity closure is
#
#     u_crra(c, σ) + λ·w_crra(c) − λ·w_crra(m − b_min)
#
# the GP ranking with the corner temptation peak folded in as a constant.
#
# Partial equilibrium: `r`, `w` fixed and exogenous; one inner solve.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct TemptationParams
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0                 # CRRA on commitment utility u
    σ_t :: Float64 = 2.0                 # CRRA curvature of temptation utility w
    λ   :: Float64 = 0.15                # strength of the temptation term
    r   :: Float64 = 0.02                # FIXED real return, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0                 # FIXED wage (scales the endowment)
    # Three-state idiosyncratic income process (persistent, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_w   :: Int     = 150               # wealth grid points
    w_min :: Float64 = 0.0               # borrowing floor b_min (grid floor)
    w_max :: Float64 = 100.0
end

Base.Broadcast.broadcastable(p::TemptationParams) = Ref(p)

const temptation_params = TemptationParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Gul–Pesendorfer temptation household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. With a monotone
temptation utility the temptation peak is the closed-form corner
`w(m − b_min)` (consume to the borrowing floor), constant in the savings
choice, so it folds into ONE felicity closure as a value shift that does not
distort policy. Attaches the aggregate-wealth moment `K_supplied = ∫ wealth dΛ`.
"""
function temptation_household(p = temptation_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    b_min = p.w_min   # borrowing floor = grid floor; consume-everything corner

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)            # (1+r)·b + w·y  → cell.wealth = cash-on-hand m
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        # GP felicity: commitment u(c) + temptation λ·w(c) − temptation peak
        # λ·w(m − b_min). The peak is constant in c (it depends only on the
        # cell's cash-on-hand cell.wealth), so it shifts V without moving policy.
        utility = (cell, c) ->
            u_crra(c, Val(p.σ)) +
            p.λ * u_crra(c, Val(p.σ_t)) -
            p.λ * u_crra(cell.wealth - b_min, Val(p.σ_t)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the partial-equilibrium temptation experiment: fixed real return `r`
and wage `w`. One inner solve delivers the stationary wealth distribution.
"""
temptation_env(p = temptation_params) = (; r = p.r, w = p.w)
