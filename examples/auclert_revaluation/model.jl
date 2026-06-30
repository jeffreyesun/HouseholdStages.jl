###################################################################
# Auclert (2019) revaluation channel — the Fisher/balance-sheet leg #
###################################################################

# Auclert (2019, "Monetary Policy and the Redistribution Channel")
# decomposes the transmission of an aggregate shock into (i) substitution,
# (ii) income, and (iii) a REVALUATION / Fisher balance-sheet channel: when
# the price `q` of an existing asset (or the real value of nominal claims)
# moves, holders' net worth is revalued by `(q − q_last) · holdings` BEFORE
# they make any new decision. That within-period balance-sheet jump is a
# transfer across the wealth distribution and is the redistributive heart of
# the channel.
#
# The within-period block inserts exactly this revaluation step into the
# canonical chain, right after the income shock resolves and before the cash-
# on-hand receipt:
#
#     IncomeShock ∘ AssetPriceChange ∘ IncomeReceipt ∘ ConsumptionSavings
#
#   IncomeShock     — `MarkovStage` on the `:income` axis.
#   AssetPriceChange— a `WealthChangeStage` (exported) with the revaluation
#                     closure `wealth ↦ wealth + (env.q − env.q_last)·wealth`.
#                     This IS the Fisher/balance-sheet revaluation: existing
#                     holdings are marked to the new price. (See the API-FRICTION
#                     note below on why the `AssetPriceChangeStage` convenience
#                     wrapper cannot be used when holdings live ON the wealth
#                     axis — the single-asset case here.)
#   IncomeReceipt   — `IncomeStage`: `b ↦ (1+r)·b + w·income`.
#   ConsumptionSavings — choose next-period wealth on the grid.
#
# In STEADY STATE `q = q_last`, so the revaluation term is identically zero
# and the stage is inert — the SS is the plain Aiyagari/Bewley fixed point.
# The channel is activated OUT of steady state, when `q ≠ q_last`; the full
# MIT-shock transition path is outer-loop machinery (see steady_state.jl for
# a one-pass demonstration of the revaluation transfer). No bespoke stage:
# the revaluation is an existing exported `WealthChangeStage` parameterized by
# a closure — exactly what `AssetPriceChangeStage` itself reduces to.
#
# API-FRICTION NOTE. The shipped `AssetPriceChangeStage(layout;
# holdings_axis = :wealth)` is the natural call here, but with its default
# `wealth_axis = :wealth` it builds a dep closure declaring the axis tuple
# `(:wealth, :wealth)`; downstream `dep_combos` then constructs
# `NamedTuple{(:wealth, :wealth)}`, which throws "duplicate field name in
# NamedTuple". The wrapper is therefore usable only when the revalued holdings
# sit on an axis DISTINCT from wealth (a genuine two-asset state). For the
# single-asset balance sheet — where wealth IS the holdings — we drop to the
# `WealthChangeStage` the wrapper is built on, with the same revaluation law.
# This is still a pure composition of exported stages.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct AuclertParams
    β :: Float64       = 0.96
    σ :: Float64       = 2.0
    r :: Float64       = 0.03               # fixed exogenous return, < 1/β − 1 ≈ 0.0417
    w :: Float64       = 1.0
    # Idiosyncratic income shock (mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_w   :: Int       = 250
    w_min :: Float64   = 0.0
    w_max :: Float64   = 120.0
end

Base.Broadcast.broadcastable(p::AuclertParams) = Ref(p)

const auclert_params = AuclertParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Auclert household block
`IncomeShock ∘ Revalue ∘ IncomeReceipt ∘ ConsumptionSavings`.
The revaluation `WealthChangeStage` inserts the Fisher transfer
`wealth ↦ wealth + (q − q_last)·wealth`, inert at the steady
state where `q = q_last`. The aggregate `A_mean = ∫ wealth dΛ` moment is
attached — it doubles as the revaluation BASE, since the aggregate
balance-sheet transfer at a price move is `(q − q_last)·A_mean`.
"""
function auclert_household(p = auclert_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock     = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    # Fisher/balance-sheet revaluation: wealth ↦ wealth + (q − q_last)·wealth.
    # (`AssetPriceChangeStage` would be the convenience call, but its
    # holdings == wealth case hits a duplicate-axis bug — see header note.)
    revalue   = WealthChangeStage(layout;
        wealth_post = (; wealth, env) -> wealth + (env.q - env.q_last) * wealth)
    receipt   = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income)
    savings   = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )

    hh = shock ∘ revalue ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (fixed prices; q = q_last at the steady state) #
#----------------------------------------------------#

"Env for the Auclert block. `q`/`q_last` are the current/last asset price;
`q = q_last` at the steady state makes the revaluation stage inert."
auclert_env(p = auclert_params; q = 1.0, q_last = 1.0) = (; p.r, p.w, q, q_last)
