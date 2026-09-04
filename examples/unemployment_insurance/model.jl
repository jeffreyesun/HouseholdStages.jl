#####################################################
# Hansen–İmrohoroğlu (1992) — Unemployment Insurance #
#####################################################

# Hansen & İmrohoroğlu's (1992) economy with an explicit unemployment-
# insurance scheme. As in İmrohoroğlu (1989), a household faces a two-state
# EMPLOYMENT Markov process and self-insures with a single storage asset.
# What is new here is that the income structure is a genuine UI POLICY, not
# just a low endowment: the employed pay a flat payroll tax `τ` and the
# unemployed receive a replacement rate `ρ` of the wage. Net labour income
# is therefore
#
#     employed:    w·(1 − τ)
#     unemployed:  ρ·w
#
# carried in the receipt budget closure (NOT in the endowment grid — the
# employment axis is a 0/1 indicator, and the closure maps it to net income
# via `τ` and `ρ` from `env`). This is the distinction from the
# `examples/imrohoroglu` sibling, whose benefit lives directly in `y_grid`.
#
# The within-period problem is the canonical three-stage spine:
#
#     EmploymentShock ∘ IncomeReceipt(UI) ∘ ConsumptionSavingsStage
#
# `EmploymentShock` (MarkovStage, axis = :employment) — employed/unemployed
# draw. `IncomeReceipt(UI)` (IncomeStage) — receipt
# `a ↦ (1+r) a + [e·w(1−τ) + (1−e)·ρw]` with `e ∈ {0,1}` the employment
# indicator. `ConsumptionSavingsStage` — choose next-period assets;
# implicit budget `c = a_in − a_end`, CRRA utility.
#
# Single fixed-`r` solve: partial equilibrium, no market clearing.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct UIParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    r :: Float64       = 0.03    # FIXED exogenous return, strictly < 1/β − 1 ≈ 0.0417
    w :: Float64       = 1.0     # wage
    ρ :: Float64       = 0.25    # UI replacement rate (unemployed get ρ·w)
    τ :: Float64       = 0.03    # flat payroll tax on the employed
    # Employment axis as a 0/1 indicator: unemployed (0) / employed (1).
    e_grid :: Vector{Float64} = [0.0, 1.0]
    # Two-state employment Markov (rows = current state, order matches
    # e_grid: [unemployed, employed]).
    P_e    :: Matrix{Float64} = [0.50 0.50;
                                 0.04 0.96]
    N_a   :: Int       = 250
    a_min :: Float64   = 0.0     # zero-borrowing (liquidity) constraint
    a_max :: Float64   = 80.0
end

Base.Broadcast.broadcastable(p::UIParams) = Ref(p)

const ui_params = UIParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Hansen–İmrohoroğlu UI household block
`EmploymentShock ∘ IncomeReceipt(UI) ∘ ConsumptionSavingsStage`. The receipt
closure reads the 0/1 employment indicator and maps it to NET labour income
through the UI policy in `env`: employed receive `w(1−τ)`, unemployed `ρ·w`.
Attaches `A_mean` (buffer stock) and `frac_constrained` (mass at the
constraint). The UI policy lives in the budget closure — that is what
distinguishes this from `examples/imrohoroglu`.
"""
function ui_household(p = ui_params)
    layout = GriddedLayout(
        :wealth     => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :employment => Discrete(p.e_grid),
    )

    shock   = MarkovStage(layout; axis = :employment, transition_matrix = p.P_e)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, employment, env) -> (1 + env.r) * wealth +
            employment * env.w * (1 - env.τ) + (1 - employment) * env.ρ * env.w,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
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
Env for the Hansen–İmrohoroğlu UI experiment at the fixed exogenous return
`r`. Carries the wage `w`, the UI replacement rate `ρ`, and the payroll tax
`τ` consumed by the receipt budget closure.
"""
ui_env(p = ui_params) = (; r = p.r, w = p.w, ρ = p.ρ, τ = p.τ)
