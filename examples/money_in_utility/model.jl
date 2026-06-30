###############################################################
# Money-in-the-Utility (Sidrauski 1967 / Brock 1974)          #
###############################################################
#
# A heterogeneous-agent money-in-the-utility model. Money is the SINGLE
# asset the household carries between periods, and the real value of the
# money it ends the period holding enters felicity directly.
#
# The within-period BLOCK is the canonical Aiyagari spine — three EXISTING
# exported stages, no bespoke stage:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# (`MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage`.) The
# ONLY thing that makes this Sidrauski rather than Aiyagari is the utility
# closure handed to `ConsumptionSavingsStage`:
#
#     u(c) + χ · u_m( m'/P )
#
# where `m'` is the next-period money holding the household chooses. On the
# `:wealth` (= money) axis, the saving choice is `m' = cell.wealth − c`
# (start-of-period money minus consumption), so real balances are
# `(cell.wealth − c) / env.P`. That single additive term — read straight off
# the savings closure's `cell` and `c` — is the whole MIU modification. No
# extra `utility_axes` are needed: the money holding lives on the savings
# axis itself.
#
# Partial equilibrium: the real return `r`, the wage `w`, and the price
# level `P` are all fixed and exogenous (`steady_state.jl` does a single
# `solve_steady_state_given_env!`). For stationarity we need `r < 1/β − 1`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MIUParams
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0                 # CRRA on consumption
    σ_m :: Float64 = 2.0                 # curvature of money felicity
    χ   :: Float64 = 0.15                # weight on real balances in utility
    r   :: Float64 = 0.02                # FIXED real return on money, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0                 # FIXED wage (scales the endowment)
    P   :: Float64 = 1.0                 # FIXED price level (real balances = m'/P)
    # Three-state idiosyncratic income process (persistent, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_m   :: Int     = 400               # money grid points
    m_min :: Float64 = 0.0
    m_max :: Float64 = 120.0
end

Base.Broadcast.broadcastable(p::MIUParams) = Ref(p)

const miu_params = MIUParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached money-in-utility household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. Money is the
`:wealth` axis; the savings closure's `cell.wealth − c` is the chosen
next-period money `m'`, so the additive felicity term `χ·u_crra(m'/P, σ_m)`
turns the Aiyagari spine into Sidrauski. Attaches the aggregate real-balance
moment `M_supplied = ∫ m dΛ`.
"""
function miu_household(p = miu_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.m_min, p.m_max, p.N_m; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)            # (1+r)·m + w·y  — money is the asset
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        # MIU felicity: consumption utility + utility of real balances m'/P,
        # where m' = cell.wealth − c is the chosen next-period money holding.
        utility = (cell, c; env) ->
            u_crra(c, Val(p.σ)) + p.χ * u_crra((cell.wealth - c) / env.P, Val(p.σ_m)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        M_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the partial-equilibrium MIU experiment: the fixed real return `r`,
wage `w`, and price level `P`. No market clears here — a single inner solve
at this env delivers the stationary money distribution.
"""
miu_env(p = miu_params) = (; r = p.r, w = p.w, P = p.P)
