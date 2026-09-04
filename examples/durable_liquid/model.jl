####################################################################
# Durable + liquid asset, lumpy (S,s) adjustment (Berger–Vavra)    #
####################################################################

# A household that holds a LUMPY durable `d` (a few discrete sizes) alongside a
# LIQUID asset `b`. Each period it picks a durable target `d'`; CHANGING the
# durable (`d' ≠ d`) triggers a transaction that (a) moves the durable stock and
# (b) debits the liquid balance by the down-payment — the purchase price of the
# net stock change `p·(d' − d)` plus a FIXED adjustment cost `F`. A single choice
# therefore sets TWO axes at once (durable stock AND liquid), the same cross-axis
# coupling that halts naive attempts; it IS expressible from existing stages via
# the **auxiliary-choice-axis pattern** (Route A; cf. examples/two_asset_hank and
# examples/habit). The fixed cost makes adjustment lumpy — an (S,s) inaction band
# — in the Berger–Vavra (2015) / Díaz–Luengo-Prado (2010) tradition.
#
# Household block (time order), existing stages only, NO bespoke stage:
#
#   Depreciate ∘ IncomeShock ∘ Receipt ∘ [ ChooseD' ∘ Debit ∘ SetDurable ∘ Forget ] ∘ ConsumptionSavings
#
# `Depreciate`  — `MarkovStage` on the `:durable` axis: w.p. `π_dep` the stock drops one level
#                 (breakdown / depreciation), else holds; level 1 (no durable) is absorbing under the
#                 shock. This is the churn source — like durable_housing's moving shock keeps the buy
#                 choice live, it keeps the (S,s) adjustment margin live, so households periodically
#                 let the stock run down and then lumpily replace. Without it the durable is absorbing
#                 (everyone climbs to the top and never moves: a zero adjustment rate).
# `ChooseD'`    — `ArgmaxStage` picks `d'` onto the auxiliary `:durable_choice` axis (reward 0;
#                 brute, since the lumpy policy is non-monotone).
# `Debit`       — `WealthChangeStage(:liquid, b ↦ max(b − outlay(d',d), ε))`, the down-payment +
#                 fixed cost. Reads BOTH `:durable_choice` (d') and `:durable` (old d). The subsistence
#                 floor `ε` on post-transaction liquid keeps consumption feasible for EVERY action — the
#                 brute `ArgmaxStage` requires every cell to have a feasible action (a corner like
#                 cash-on-hand `= 0`, no durable to sell, would otherwise have none). This is exactly
#                 the two-asset-HANK floor; the consumption hit a `ε`-floored over-purchase takes is what
#                 deters unaffordable adjustments, so no explicit affordability gate is needed.
# `SetDurable`  — `WealthChangeStage(:durable, d ↦ d')` commits the chosen stock.
# `Forget`      — `ForgetfulSumStage(:durable_choice)` collapses the auxiliary axis.
# `ConsumptionSavings` — `ConsumptionSavingsStage(:liquid)`; `c = b_post − b'`, flow utility over `c`
#                 and the durable service `θ·log(d+1)` folded in via `utility_axes = (:durable,)`.
#
# The fixed adjustment cost + down-payment ride the FOLLOWING `Debit` WealthChange — never the choice
# reward (mirroring BuyHomeStage ∘ WealthChangeStage and the two-asset-HANK deposit). Returns/wage are
# exogenous (partial equilibrium): the outer loop is a single `solve_steady_state_given_env!`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct DurableLiquidParams
    β    :: Float64 = 0.93
    σ    :: Float64 = 2.0                         # CRRA over nondurable consumption
    θ    :: Float64 = 0.55                        # weight on the durable service flow θ·log(d+1)
    r_b  :: Float64 = 0.02                        # liquid return
    p    :: Float64 = 1.0                         # durable price (per unit)
    F    :: Float64 = 0.12                        # FIXED adjustment cost (the (S,s) friction)
    ε    :: Float64 = 0.05                        # subsistence floor on post-transaction liquid
    w    :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.7, 1.3]
    P_y    :: Matrix{Float64} = [0.85 0.15; 0.15 0.85]
    d_sizes :: Vector{Float64} = [0.0, 1.0, 2.0, 3.0]   # lumpy durable levels (index 1 = no durable)
    π_dep :: Float64 = 0.15                       # per-period prob. the durable drops one level
    N_b  :: Int     = 100
    b_max :: Float64 = 12.0
end

Base.Broadcast.broadcastable(p::DurableLiquidParams) = Ref(p)

const durable_liquid_params = DurableLiquidParams()


# Durable service flow #
#----------------------#

"""
Per-period utility, additively separable in nondurable consumption `c` (CRRA) and the durable
service flow `θ·log(d+1)` (finite at `d=0`, so the no-durable state is always feasible).
"""
u_durable(c, d, p) = c <= 0 ? -Inf : u_crra(c, Val(p.σ)) + p.θ * log(d + 1)


# Household chain assembly #
#--------------------------#

"""
Build the durable+liquid (S,s) block via the auxiliary-choice-axis pattern (existing stages only, no
bespoke stage). `mean_liquid`, `mean_durable` attached as moments; the adjustment rate is computed
driver-side from the choice policy (it is a transition statistic, invisible to an `at_end` integrand).
"""
function durable_liquid_household(p = durable_liquid_params)
    bgrid = collect(range(0.0, p.b_max; length = p.N_b))
    N_d   = length(p.d_sizes)
    axes_base = (:liquid  => GriddedContinuous(bgrid),
                 :durable => GriddedContinuous(p.d_sizes),
                 :income  => Discrete(p.y_grid))
    block = GriddedLayout(axes_base..., :durable_choice => Discrete([1]))            # pre/post-choice: singleton
    full  = GriddedLayout(axes_base..., :durable_choice => Discrete(collect(1:N_d))) # choice block: full

    # Cash outlay of moving the stock d → d': buy/sell the net change at price p, plus a fixed cost F
    # whenever the stock changes (the (S,s) friction). Zero on keep.
    outlay(dc, d) = (dprime = p.d_sizes[Int(dc)];
                     dprime == d ? 0.0 : p.p * (dprime - d) + p.F)

    # Depreciation shock on the durable axis: level k ≥ 2 drops to k−1 w.p. π_dep; level 1 absorbing.
    P_d = zeros(N_d, N_d)
    P_d[1, 1] = 1.0
    for k in 2:N_d
        P_d[k, k - 1] = p.π_dep
        P_d[k, k]     = 1 - p.π_dep
    end
    depreciate = MarkovStage(block; axis = :durable, transition_matrix = P_d)

    shock   = MarkovStage(block; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(block; axis = :liquid,                               # cash on hand
        wealth_post = (; liquid, income) -> (1 + p.r_b) * liquid + p.w * income)
    choose  = ArgmaxStage(block, full; axis = :durable_choice, reward = zeros(N_d, 1)) # grows 1 → N_d
    debit   = WealthChangeStage(full; axis = :liquid,                                # b ↦ max(b − outlay(d',d), ε)
        wealth_post = (; durable_choice, durable, liquid) -> max(liquid - outlay(durable_choice, durable), p.ε))
    setdur  = WealthChangeStage(full; axis = :durable,                               # d ↦ d'
        wealth_post = (; durable_choice) -> p.d_sizes[Int(durable_choice)])
    forget  = ForgetfulSumStage(full; axis = :durable_choice)
    savings = ConsumptionSavingsStage(block; β = p.β, axis = :liquid,                # c = b_post − b'
        utility = (cell, c) -> u_durable(c, cell.durable, p),
        utility_axes = (:durable,))

    hh = depreciate ∘ shock ∘ receipt ∘ choose ∘ debit ∘ setdur ∘ forget ∘ savings
    return define_moments!(hh;
        mean_liquid  = at_end(integrand = :liquid,  reduce = sum),
        mean_durable = at_end(integrand = :durable, reduce = sum))
end


# Exogenous prices (partial equilibrium) #
#----------------------------------------#

"Exogenous env: liquid return and wage (no market to clear)."
durable_liquid_env(p = durable_liquid_params) = (; r_b = p.r_b, w = p.w)
