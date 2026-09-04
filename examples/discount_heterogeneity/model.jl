###################################################################
# Discount-factor heterogeneity — Krusell–Smith / CSTW (2017)      #
###################################################################

# A standard incomplete-markets economy in which households differ in their
# discount factor β (Krusell–Smith 1998's stochastic-β extension; Carroll,
# Slacalek, Tokuoka & White 2017, "The distribution of wealth and the
# marginal propensity to consume"). A spread of fixed β-types is the
# canonical device for matching the empirical wealth distribution: patient
# types accumulate the top tail, impatient types stay near the constraint.
#
# Each β-type solves the SAME canonical L03/L04 within-period chain,
# differing ONLY in the discount factor fed to its `ConsumptionSavingsStage`:
#
#     block_i = IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage(β = β_i)
#
# The household block is the DIRECT SUM (`⊕` / `product`) of these per-type
# blocks along a `:beta` axis:
#
#     household = product(block_1, …, block_n; axis = :beta)
#
# Because β is a plain `Float64` field of the ConsumptionSavingsStage spec,
# all per-type blocks are ONE chain at different parameter values, sharing the
# start and end layouts `product` asks of its factors. The `:beta` axis is
# declared a size-1 SINGLETON in the
# block layout; `product` grows it `1 → n`, one slice per β-type.
#
# The direct sum is BLOCK-DIAGONAL: there is no transition across β-types
# (a household keeps its β forever), so each slice is its own independent
# stationary infinite-horizon problem. This is exactly what block-diagonal
# `⊕` delivers — and, unlike the finite-horizon life-cycle product (which
# needs cross-age threading the block-diagonal form does not supply), here
# the STANDARD solver works directly: `define_moments!` wraps the product in
# a singleton ChainStage, and `solve_steady_state_given_env!` runs VFI to a
# fixed point and Λ to stationarity on the fused tensor, converging each
# β-slice independently.
#
# Partial equilibrium: fixed exogenous `r`, `w` (no market cleared here).
# `r` is set below the MOST PATIENT type's impatience knife-edge
# `1/β_max − 1` so every type's wealth distribution is stationary.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct DiscountHetParams
    σ :: Float64       = 2.0
    r :: Float64       = 0.015              # fixed; < 1/β_max − 1 = 1/0.98 − 1 ≈ 0.0204
    w :: Float64       = 1.0
    # The β-type spread (CSTW-style: a small band of fixed discount factors).
    β_grid :: Vector{Float64} = [0.94, 0.96, 0.98]
    # Persistent idiosyncratic income shock (mean ≈ 1).
    ε_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_ε    :: Matrix{Float64} = [0.80 0.15 0.05;
                                 0.10 0.80 0.10;
                                 0.05 0.15 0.80]
    N_w   :: Int       = 220
    w_min :: Float64   = 0.0
    w_max :: Float64   = 250.0
end

Base.Broadcast.broadcastable(p::DiscountHetParams) = Ref(p)

const discount_het_params = DiscountHetParams()


# Household chain assembly #
#--------------------------#

"""
Build one β-type's within-period block `IncomeShock ∘ IncomeReceipt ∘
ConsumptionSavingsStage(β)` against the SHARED layout (the `:beta` axis a
size-1 singleton that `product` grows to the type count, so every β-block sits
between the same two layouts). Only the discount factor `β` differs across calls.
"""
function discount_het_block(β::Float64, p = discount_het_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.ε_grid),
        :beta   => Discrete([1.0]),          # size-1 singleton; product grows it 1 → n
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_ε)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income)
    savings = ConsumptionSavingsStage(layout;
        β       = β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )

    return shock ∘ receipt ∘ savings
end

"""
The discount-heterogeneity household block: the direct sum
`product(block(β_1), …, block(β_n); axis = :beta)` of the per-type blocks,
with the aggregate `A_mean = ∫ wealth dΛ` moment attached. Per-β wealth is
computed example-side from the `:beta`-indexed slices of `Λ`.
"""
function discount_het_household(p = discount_het_params)
    blocks = [discount_het_block(β, p) for β in p.β_grid]
    hh = product(blocks...; axis = :beta)
    return define_moments!(hh;
        A_mean = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (no production, fixed prices) #
#-----------------------------------#

"Env for the discount-heterogeneity experiment: fixed return `r` and wage `w`."
discount_het_env(p = discount_het_params) = (; p.r, p.w)
