#####################################################
# Taste Shocks — Preference Risk in a Bewley Economy #
#####################################################

# A standard incomplete-markets self-insurance economy with an extra
# EXOGENOUS PREFERENCE (taste) shock layered onto the income process. Beyond
# income risk, the household's flow utility is shifted up or down each period by
# a persistent taste state — periods of high marginal value of consumption
# (a "want to spend" shock) versus low. The taste shock is a pure
# `UtilityStage` additive shifter on the value function, driven by its own
# Markov axis; it changes the consumption/saving incentive without touching the
# budget.
#
# The within-period problem extends the canonical spine with a second Markov
# draw and a utility shifter:
#
#     TasteShock ∘ IncomeShock ∘ IncomeReceipt ∘ UtilityStage(taste) ∘ ConsumptionSavingsStage
#
# `TasteShock`  (MarkovStage, axis = :taste)    — the preference Markov draw.
# `IncomeShock` (MarkovStage, axis = :income)   — the idiosyncratic income draw.
# `IncomeReceipt` (IncomeStage)                 — `a ↦ (1+r) a + w·y`.
# `UtilityStage(taste)`                         — adds the taste state's flow
#       value `taste` (an additive shift to V; reads the `:taste` axis). A
#       positive taste raises the value of being in that state this period; the
#       household leans on its buffer stock differently across taste states.
# `ConsumptionSavingsStage`                     — choose next-period wealth;
#       implicit budget `c = a_in − a_end`, CRRA utility.
#
# Fixed-`r` partial equilibrium (the Bewley framing): a single inner V/Λ solve
# at the exogenous return `r < 1/β − 1` delivers the stationary joint
# distribution over (wealth, income, taste). No market clearing.
#
# NOTE on the utility shifter. `UtilityStage` adds a STATE-dependent flow value;
# it is the value of being in the taste state, not a multiplicative twist on the
# consumption felicity (that would require the taste to enter the
# `ConsumptionSavingsStage` utility closure, e.g. `utility_axes = (:taste,)`).
# The additive form is the clean composable demonstration of an exogenous taste
# process moving the value function.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct TasteShockParams
    β :: Float64       = 0.95
    σ :: Float64       = 2.0
    r :: Float64       = 0.03      # FIXED exogenous return, < 1/β − 1 ≈ 0.0526
    w :: Float64       = 1.0
    # Idiosyncratic income process (persistent, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    # Taste states as additive flow-utility shifts (low / neutral / high
    # marginal value of being in this state), with their own persistent Markov.
    taste_grid :: Vector{Float64} = [-0.10, 0.0, 0.10]
    P_taste    :: Matrix{Float64} = [0.80 0.15 0.05;
                                     0.10 0.80 0.10;
                                     0.05 0.15 0.80]
    N_a   :: Int       = 250
    a_min :: Float64   = 0.0
    a_max :: Float64   = 100.0
end

Base.Broadcast.broadcastable(p::TasteShockParams) = Ref(p)

const taste_shock_params = TasteShockParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached taste-shock household block
`TasteShock ∘ IncomeShock ∘ IncomeReceipt ∘ UtilityStage(taste) ∘ ConsumptionSavingsStage`.
The `UtilityStage` reads the `:taste` axis and adds its grid value as an additive
flow shift to V; the two Markov stages resolve the taste and income draws.
Attaches `A_mean = ∫ a dΛ` and the taste-conditional mean assets.
"""
function taste_shock_household(p = taste_shock_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :income => Discrete(p.y_grid),
        :taste  => Discrete(p.taste_grid),
    )

    taste_shock  = MarkovStage(layout; axis = :taste,  transition_matrix = p.P_taste)
    income_shock = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt      = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis        = :wealth,
    )
    taste_value  = UtilityStage(layout; utility = (; taste) -> taste)
    savings      = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    )

    hh = taste_shock ∘ income_shock ∘ receipt ∘ taste_value ∘ savings
    return define_moments!(hh;
        A_mean      = at_end(integrand = :wealth, reduce = sum),
        A_hightaste = at_end(integrand = (; wealth, taste) -> taste > 0 ? wealth : 0.0,
                            reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the fixed-`r` taste-shock self-insurance experiment: the exogenous
return `r` and wage `w` read by the receipt closure. (The taste shift is read
off the `:taste` axis, not `env`.)
"""
taste_shock_env(p = taste_shock_params) = (; r = p.r, w = p.w)
