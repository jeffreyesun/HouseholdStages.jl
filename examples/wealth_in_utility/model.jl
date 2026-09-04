###############################################################
# Wealth-in-Utility / "Capitalist Spirit"                      #
# (Carroll 2000; Francis 2009)                                 #
###############################################################
#
# A heterogeneous-agent consumption-savings model in which wealth is held
# not only for the future consumption it buys but also for its own sake —
# the "capitalist spirit" / "wealth-in-utility" (WIU) motive. Felicity is
#
#     u(c) + v(b')
#
# where `b'` is the next-period wealth the household chooses to carry. The
# direct taste for wealth `v(·)` raises saving relative to a pure Aiyagari
# agent and is the textbook route to a thick-tailed wealth distribution.
#
# The within-period BLOCK is the canonical Aiyagari spine — three EXISTING
# exported stages, no bespoke stage:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#     = MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
#
# The ONLY thing that turns Aiyagari into wealth-in-utility is the felicity
# closure handed to `ConsumptionSavingsStage`:
#
#     u_crra(c, σ) + χ · u_crra(b', σ_w)
#
# On the `:wealth` axis the saving choice is `b' = cell.wealth − c`
# (post-receipt cash-on-hand minus consumption), so the bequest/wealth term
# is `χ · u_crra(cell.wealth − c, σ_w)`, read straight off `cell` and `c`.
# `u_crra` masks any argument `≤ 0` to `-Inf`, so the `b' ≤ 0` corner is
# guarded automatically (consuming all cash-on-hand is feasible only when it
# also satisfies the savings stage's own `c > 0` mask). No extra
# `utility_axes` are needed — wealth lives on the savings axis itself.
#
# Stationarity. The WIU motive pushes the agent to save even at a low return,
# so the patience condition `r < 1/β − 1` is necessary but not by itself
# sufficient to bound wealth: the CRRA curvature `σ_w > 0` on the wealth
# term supplies the diminishing marginal taste for wealth that pins a finite
# stationary target. Partial equilibrium — `r`, `w` fixed and exogenous; a
# single `solve_steady_state_given_env!` delivers the stationary distribution.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct WIUParams
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0                 # CRRA on consumption
    σ_w :: Float64 = 2.0                 # curvature of the wealth-in-utility term
    χ   :: Float64 = 0.10                # weight on next-period wealth in felicity
    r   :: Float64 = 0.02                # FIXED real return, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0                 # FIXED wage (scales the endowment)
    # Three-state idiosyncratic income process (persistent, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_w   :: Int     = 150               # wealth grid points
    w_min :: Float64 = 0.0
    w_max :: Float64 = 300.0             # wide: the WIU motive thickens the upper tail
end

Base.Broadcast.broadcastable(p::WIUParams) = Ref(p)

const wiu_params = WIUParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached wealth-in-utility household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. The savings
closure's `cell.wealth − c` is the chosen next-period wealth `b'`, so the
additive felicity term `χ·u_crra(b', σ_w)` turns the Aiyagari spine into the
"capitalist spirit" model. Attaches the aggregate-wealth moment
`K_supplied = ∫ wealth dΛ`.
"""
function wiu_household(p = wiu_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)            # (1+r)·b + w·y
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        # WIU felicity: consumption utility + direct taste for next-period
        # wealth b' = cell.wealth − c. u_crra masks b' ≤ 0 to -Inf.
        utility = (cell, c) ->
            u_crra(c, Val(p.σ)) + p.χ * u_crra(cell.wealth - c, Val(p.σ_w)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the partial-equilibrium wealth-in-utility experiment: fixed real
return `r` and wage `w`. No market clears here — one inner solve at this env
delivers the stationary wealth distribution.
"""
wiu_env(p = wiu_params) = (; r = p.r, w = p.w)
