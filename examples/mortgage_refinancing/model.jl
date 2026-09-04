####################################################################
# Mortgage refinancing / cash-out, LTV-gated (fixed cost, (S,s))   #
####################################################################

# A homeowner with a fixed-value house `H` (subject to a house-PRICE shock) and a
# mortgage balance `m` (a few discrete levels) alongside liquid wealth `a`. Each
# period the household may refinance — adjust its mortgage balance up (cash-out,
# the borrowing margin) or down (prepay) — paying a FIXED refinancing cost `κ`,
# GATED by the loan-to-value constraint `m'/H ≤ θ_ltv`. The refi choice moves TWO
# axes at once: the mortgage balance AND (via the principal change + fixed cost)
# the liquid balance. That cross-axis coupling — and the fact that the cash flow
# `(m' − m)` reads BOTH the new and the OLD balance — make it the
# auxiliary-choice-axis pattern (Route A; cf. examples/two_asset_hank, habit,
# durable_liquid), NOT the simpler default/buy-home gated-choice pattern (a
# following WealthChange there sees only the post-choice state — see README).
#
# Household block (time order), existing stages only, NO bespoke stage:
#
#   IncomeShock ∘ HousePriceShock ∘ Receipt ∘ [ ChooseM' ∘ LTVGate ∘ Pay ∘ SetMortgage ∘ Forget ] ∘ ConsumptionSavings
#
# `IncomeShock`     — `MarkovStage` on the income axis.
# `HousePriceShock` — `MarkovStage` on the `:hp` axis (boom/bust house value). It shifts the LTV gate:
#                     in a bust, high balances are underwater and locked out of refinancing.
# `Receipt`         — `WealthChangeStage(:wealth, a ↦ (1+r_a)·a + w·y)`, cash on hand.
# `ChooseM'`        — `ArgmaxStage` picks the new balance `m'` onto the auxiliary `:refi_choice` axis
#                     (reward 0; brute, lumpy/non-monotone). Grows the singleton axis 1 → K.
# `LTVGate`         — `BorrowingConstraintStage`: a CASH-OUT (`m' > m`) is infeasible (`-Inf`) when it
#                     violates the LTV cap `m'/hp > θ_ltv`. Keep and prepay are never gated, so the
#                     brute argmax always has a feasible action.
# `Pay`             — `WealthChangeStage(:wealth, a ↦ max(a + (m'−m) − κ·1{m'≠m} − r_m·m', ε))`: the
#                     principal change (cash-out is +, prepay is −), the fixed refi cost, and the
#                     interest on the new balance. Reads BOTH `:refi_choice` (m') and `:mortgage` (old m).
# `SetMortgage`     — `WealthChangeStage(:mortgage, m ↦ m')` commits the new balance.
# `Forget`          — `ForgetfulSumStage(:refi_choice)`.
# `ConsumptionSavings` — `ConsumptionSavingsStage(:wealth)`; `c = a_post − a'`, CRRA over `c`.
#
# The fixed cost rides the FOLLOWING `Pay` WealthChange — never the choice reward. The ε floor on
# post-pay liquid keeps consumption feasible for every action (the brute-argmax requirement; cf.
# two-asset-HANK / durable_liquid). Prices are exogenous (partial equilibrium): a single
# `solve_steady_state_given_env!`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MortgageParams
    β    :: Float64 = 0.94
    σ    :: Float64 = 2.0
    r_a  :: Float64 = 0.02                        # liquid return
    r_m  :: Float64 = 0.05                        # mortgage interest rate (> r_a ⇒ debt is costly)
    κ    :: Float64 = 0.05                        # FIXED refinancing cost (the (S,s) friction)
    θ_ltv :: Float64 = 0.50                       # loan-to-value cap on cash-out refis
    ε    :: Float64 = 0.02                        # subsistence floor on post-pay liquid
    w    :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.7, 1.3]
    P_y    :: Matrix{Float64} = [0.85 0.15; 0.15 0.85]
    hp_grid :: Vector{Float64} = [3.0, 5.0]       # house value: bust / boom
    P_hp    :: Matrix{Float64} = [0.90 0.10; 0.10 0.90]
    m_grid :: Vector{Float64} = [0.0, 1.0, 2.0]   # mortgage balance levels (index 1 = no mortgage)
    N_a  :: Int     = 100
    a_max :: Float64 = 10.0
end

Base.Broadcast.broadcastable(p::MortgageParams) = Ref(p)

const mortgage_params = MortgageParams()


# Household chain assembly #
#--------------------------#

"""
Build the LTV-gated mortgage-refinancing block via the auxiliary-choice-axis pattern (existing stages
only, no bespoke stage). `mean_wealth`, `mean_mortgage` attached as moments; the refinancing rate is
computed driver-side from the choice policy (a transition statistic, invisible to an `at_end` integrand).
"""
function mortgage_household(p = mortgage_params)
    agrid = collect(range(0.0, p.a_max; length = p.N_a))
    K     = length(p.m_grid)
    axes_base = (:wealth   => GriddedContinuous(agrid),
                 :mortgage => GriddedContinuous(p.m_grid),
                 :income   => Discrete(p.y_grid),
                 :hp       => Discrete(p.hp_grid))
    block = GriddedLayout(axes_base..., :refi_choice => Discrete([1]))            # pre/post-choice: singleton
    full  = GriddedLayout(axes_base..., :refi_choice => Discrete(collect(1:K)))   # choice block: full

    # Net cash drawn from liquid when moving the balance m → m': principal change (cash-out +, prepay −),
    # less the fixed cost on any change, less interest on the new balance.
    cash(mc, m) = (mprime = p.m_grid[Int(mc)];
                   (mprime - m) - (mprime == m ? 0.0 : p.κ) - p.r_m * mprime)

    shock_y = MarkovStage(block; axis = :income, transition_matrix = p.P_y)
    shock_h = MarkovStage(block; axis = :hp,     transition_matrix = p.P_hp)
    receipt = WealthChangeStage(block; axis = :wealth,                            # cash on hand
        wealth_post = (; wealth, income) -> (1 + p.r_a) * wealth + p.w * income)
    choose  = ArgmaxStage(block, full; axis = :refi_choice, reward = zeros(K, 1))   # grows 1 → K
    ltvgate = BorrowingConstraintStage(full;                                      # cash-out beyond the LTV cap
        infeasible = (; refi_choice, mortgage, hp) -> (mprime = p.m_grid[Int(refi_choice)];
                                                       mprime > mortgage && mprime / hp > p.θ_ltv))
    pay     = WealthChangeStage(full; axis = :wealth,                             # a ↦ max(a + cash(m',m), ε)
        wealth_post = (; refi_choice, mortgage, wealth) -> max(wealth + cash(refi_choice, mortgage), p.ε))
    setm    = WealthChangeStage(full; axis = :mortgage,                           # m ↦ m'
        wealth_post = (; refi_choice) -> p.m_grid[Int(refi_choice)])
    forget  = ForgetfulSumStage(full; axis = :refi_choice)
    savings = ConsumptionSavingsStage(block; β = p.β, axis = :wealth,             # c = a_post − a'
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = shock_y ∘ shock_h ∘ receipt ∘ choose ∘ ltvgate ∘ pay ∘ setm ∘ forget ∘ savings
    return define_moments!(hh;
        mean_wealth   = at_end(integrand = :wealth,   reduce = sum),
        mean_mortgage = at_end(integrand = :mortgage, reduce = sum))
end


# Exogenous prices (partial equilibrium) #
#----------------------------------------#

"Exogenous env: liquid return, mortgage rate, wage (no market to clear)."
mortgage_env(p = mortgage_params) = (; r_a = p.r_a, r_m = p.r_m, w = p.w)
