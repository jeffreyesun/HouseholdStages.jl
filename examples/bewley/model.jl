#####################################################
# Bewley — Standard Incomplete-Markets Self-Insurance #
#####################################################

# The canonical Bewley / standard-incomplete-markets self-insurance model
# (Bewley 1977/1986; the imperfect-insurance tradition of Aiyagari 1994,
# Huggett 1993). A household faces idiosyncratic income risk and a SINGLE
# risk-free asset; it cannot insure the income shock directly, so it
# self-insures by building a precautionary buffer stock of wealth.
#
# The within-period problem is the canonical L03 / L04 three-stage
# decomposition — the SAME chain as Aiyagari and Huggett:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# `IncomeShock`  (MarkovStage)        — the idiosyncratic income Markov draw.
# `IncomeReceipt`(WealthChangeStage)  — the receipt `a ↦ (1+r) a + y`: gross
#                                       interest on wealth plus this period's
#                                       endowment.
# `ConsumptionSavingsStage`           — choose next-period wealth `a_end` on
#                                       the wealth grid; implicit budget
#                                       `c = a_in − a_end`; CRRA utility.
#
# What makes this *Bewley framing* (rather than Aiyagari/Huggett) is NOT
# the household block — it is byte-near the sibling examples, as expected —
# but the OUTER layer (in steady_state.jl):
#
#   * The interest rate `r` is FIXED and EXOGENOUS, strictly below the
#     impatience knife-edge `1/β − 1`. There is NO market to clear (no
#     production, no bond-clearing): a single `solve_steady_state_given_env!`
#     at the fixed `r` delivers the stationary distribution. This is the
#     pure partial-equilibrium self-insurance experiment — "given a return
#     `r`, how much precautionary wealth does idiosyncratic risk generate?"
#   * `r < 1/β − 1` is what makes the wealth distribution stationary: with
#     `β(1+r) < 1` the household is impatient enough to run wealth down in
#     good states, so the precautionary motive (not a drift to the grid top)
#     pins the ergodic distribution. At `r = 1/β − 1` exactly, the Chamberlain–
#     Wilson result says wealth diverges; below it, self-insurance is
#     imperfect and a non-degenerate distribution emerges.
#
# The wealth grid is log-spaced (dense near the borrowing constraint, where
# the precautionary buffer-stock policy is most nonlinear; coarse at the
# top). `WealthChangeStage.backward` interpolates `V_end` linearly along the
# wealth axis, so the grid top must be far enough that `(1+r) a + y` stays
# inside the grid for active cells — the standard Aiyagari log-grid argument.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct BewleyParams
    β :: Float64       = 0.95
    σ :: Float64       = 2.0                # CRRA — risk aversion drives precautionary saving
    r :: Float64       = 0.03              # FIXED exogenous return, strictly < 1/β − 1 ≈ 0.0526
    # Three-state idiosyncratic income process (persistent, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_a   :: Int       = 400
    a_min :: Float64   = 0.0               # zero-borrowing (ad-hoc) constraint
    a_max :: Float64   = 120.0
end

Base.Broadcast.broadcastable(p::BewleyParams) = Ref(p)

const bewley_params = BewleyParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Bewley household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`, with the
precautionary-wealth moments attached:

  * `A_mean`         — `∫ a dΛ`, the aggregate self-insurance buffer stock.
  * `frac_constrained` — `∫ 𝟙{a ≈ a_min} dΛ`, the mass pinned at the
    borrowing constraint (the hand-to-mouth share).
"""
function bewley_household(p = bewley_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + income,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    ) # defaults: (; utility_axes = nothing, skip_monotonicity_check = false)

    a_floor = p.a_min
    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean           = at_end(integrand = :wealth, reduce = sum),
        frac_constrained = at_end(integrand = (; wealth) -> wealth <= a_floor + 1e-9 ? 1.0 : 0.0,
                                  reduce = sum),
    )
end


# Prices (plain function, no AbstractBlock) #
#-------------------------------------------#

"""
Env for the Bewley self-insurance experiment at the fixed exogenous return
`r`. There is no production and no market to clear — the only price is the
risk-free return `r`, and income is the pure endowment `cell.income`.
"""
bewley_env(r::Real = bewley_params.r) = (; r)
