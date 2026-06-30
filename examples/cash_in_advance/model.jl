###############################################################
# Cash-in-Advance (Clower 1967 / Lucas–Stokey 1987)           #
###############################################################
#
# A heterogeneous-agent cash-in-advance model. Money is the SINGLE asset
# carried between periods, and consumption is capped by the real value of
# the money the household carried IN:
#
#     c ≤ m / P.
#
# The within-period BLOCK is the Aiyagari spine — three EXISTING exported
# stages, no bespoke stage:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# (`MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage`.) The CIA
# constraint is imposed INSIDE the `ConsumptionSavingsStage` utility closure
# by masking infeasible consumption with `-Inf`:
#
#     utility = (cell, c; env) -> c > cell.wealth/env.P ? -Inf : u_crra(c, σ)
#
# `cell.wealth` is the money carried in (the savings-axis start coordinate),
# so `cell.wealth / env.P` is real money on hand and the mask enforces the
# Clower constraint exactly. Returning `-Inf` from the closure is the same
# masking mechanism the stage uses internally for `c ≤ 0`, and it lives
# inside the argmax (unlike a standalone constraint on STATES, which would
# break the VFI metric) — so the constraint composes cleanly with no new
# stage and no second axis.
#
# Partial equilibrium: real return `r`, wage `w`, price level `P` all fixed
# and exogenous; one `solve_steady_state_given_env!`. Stationarity: `r < 1/β − 1`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CIAParams
    β :: Float64 = 0.96
    σ :: Float64 = 2.0                   # CRRA on consumption
    r :: Float64 = 0.02                  # FIXED real return on money, < 1/β − 1 ≈ 0.0417
    w :: Float64 = 1.0                   # FIXED wage (scales the endowment)
    P :: Float64 = 1.0                   # FIXED price level (real money = m/P)
    # Three-state idiosyncratic income process (persistent, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_m   :: Int     = 400               # money grid points
    m_min :: Float64 = 0.0
    m_max :: Float64 = 120.0
end

Base.Broadcast.broadcastable(p::CIAParams) = Ref(p)

const cia_params = CIAParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached cash-in-advance household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. Money is the
`:wealth` axis; the CIA constraint `c ≤ cell.wealth/P` is imposed by the
savings utility closure masking infeasible `c` with `-Inf`. Attaches the
aggregate money moment `M_supplied = ∫ m dΛ`.
"""
function cia_household(p = cia_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.m_min, p.m_max, p.N_m; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)            # (1+r)·m + w·y  — money is the asset
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        # Clower constraint: consumption cannot exceed real money carried in.
        utility = (cell, c; env) ->
            c > cell.wealth / env.P ? -Inf : u_crra(c, Val(p.σ)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        M_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the partial-equilibrium CIA experiment: fixed real return `r`, wage
`w`, and price level `P`. No market clears — one inner solve at this env
delivers the stationary money distribution.
"""
cia_env(p = cia_params) = (; r = p.r, w = p.w, P = p.P)
