#####################################################
# İmrohoroğlu (1989) — Liquidity-Constrained Self-Insurance #
#####################################################

# İmrohoroğlu's (1989) cost-of-business-cycles economy in its
# steady-state, partial-equilibrium form. A household faces a two-state
# EMPLOYMENT Markov process (employed / unemployed) and self-insures with
# a single storage asset against the unemployment risk. There is no
# explicit insurance market: the unemployment "benefit" is simply a low
# endowment in the unemployed state, carried directly in the employment-
# axis grid (`y_grid = [0.25, 1.0]` — replacement income ≈ 0.25 when
# unemployed, full income 1.0 when employed).
#
# The within-period problem is the canonical three-stage decomposition,
# the SAME spine as Aiyagari/Bewley/Huggett, with the income shock
# relabelled as an employment shock:
#
#     EmploymentShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# `EmploymentShock` (MarkovStage, axis = :employment) resolves the
# employed/unemployed draw. `IncomeReceipt` (IncomeStage) is the receipt
# `a ↦ (1+r) a + w·e`, where `e` is the employment-axis endowment value
# (1.0 employed, 0.25 unemployed) and `w` a scale (1.0). `Consumption-
# SavingsStage` chooses next-period assets on the wealth grid; implicit
# budget `c = a_in − a_end`, CRRA utility.
#
# What makes this İmrohoroğlu (not Aiyagari) is the OUTER framing: the
# return `r` is FIXED and exogenous, strictly below `1/β − 1`, so a single
# steady-state solve delivers the precautionary wealth distribution that
# the employment risk generates. No market is cleared.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct ImrohorogluParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    r :: Float64       = 0.03   # FIXED exogenous return, strictly < 1/β − 1 ≈ 0.0417
    w :: Float64       = 1.0    # income scale
    # Employment-axis endowment values: unemployed (benefit) / employed.
    y_grid :: Vector{Float64} = [0.25, 1.0]
    # Two-state employment Markov (rows = current state, order matches
    # y_grid: [unemployed, employed]). Persistent employment, moderate
    # unemployment duration.
    P_e    :: Matrix{Float64} = [0.50 0.50;
                                 0.04 0.96]
    N_a   :: Int       = 250
    a_min :: Float64   = 0.0    # zero-borrowing (liquidity) constraint
    a_max :: Float64   = 80.0
end

Base.Broadcast.broadcastable(p::ImrohorogluParams) = Ref(p)

const imrohoroglu_params = ImrohorogluParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached İmrohoroğlu household block
`EmploymentShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. The employment
axis carries the endowment values directly (1.0 employed, 0.25 unemployed),
so the receipt closure reads the `employment` axis as the income flow:
`a ↦ (1+r) a + w·e`. Two precautionary moments are attached: `A_mean`
(aggregate buffer stock) and `frac_constrained` (mass at the liquidity
constraint). Identical spine to Aiyagari/Bewley; the content is the fixed-`r`
self-insurance framing and the employment relabelling.
"""
function imrohoroglu_household(p = imrohoroglu_params)
    layout = GriddedLayout(
        :wealth     => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :employment => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :employment, transition_matrix = p.P_e)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, employment, env) -> (1 + env.r) * wealth + env.w * employment,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    )

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
Env for the İmrohoroğlu self-insurance experiment at the fixed exogenous
return `r`. No production and no market to clear: the only prices are the
risk-free return `r` and the income scale `w`; the unemployment benefit is
the low employment-axis endowment value.
"""
imrohoroglu_env(p = imrohoroglu_params) = (; r = p.r, w = p.w)
