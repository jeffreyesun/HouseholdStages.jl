###################################################################
# Heterogeneous Income Profiles (HIP) — Guvenen (2007, 2009)        #
###################################################################

# A standard incomplete-markets household whose earnings carry TWO
# distinct sources of heterogeneity:
#
#   1. A PERMANENT lifetime-earnings TYPE `θ` (the "profile heterogeneity"
#      of the HIP literature). Different households are born onto
#      systematically different earnings levels and never switch type —
#      there is NO Markov stage on this axis, so mass is conserved within
#      each type.
#   2. An AR(1)-style persistent income shock `ε` on top, the usual
#      idiosyncratic risk resolved by a `MarkovStage`.
#
# Earnings are the product `θ · ε`, so the permanent type SCALES the whole
# stochastic earnings profile (the HIP "heterogeneous slopes/levels"
# channel), rather than entering additively.
#
# The within-period block is the canonical L03/L04 three-stage chain,
# UNCHANGED except that the receipt closure reads the extra permanent axis:
#
#     IncomeShock ∘ IncomeReceipt(θ·ε) ∘ ConsumptionSavingsStage
#
#   IncomeShock — `MarkovStage` on the `:income` (ε) axis only.
#   IncomeReceipt — `IncomeStage` with a budget closure that reads BOTH the
#                   `:income` shock and the permanent `:income_type` axis:
#                   `b ↦ (1+r)·b + w·θ·ε`. The closure declares `income_type`
#                   as a kwarg, so the field machinery resolves it against the
#                   `:income_type` layout axis (no new stage needed).
#   ConsumptionSavings — choose next-period wealth on the wealth grid.
#
# No bespoke stage: the permanent type is a plain extra Discrete axis read
# inside the existing `IncomeStage` budget closure. The experiment is
# partial-equilibrium (fixed exogenous `r`, `w`): "how does a permanent
# earnings-type gap map into a wealth-distribution gap?"

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HIPParams
    β :: Float64       = 0.96
    σ :: Float64       = 2.0
    r :: Float64       = 0.03               # fixed exogenous return, < 1/β − 1 ≈ 0.0417
    w :: Float64       = 1.0
    # Persistent idiosyncratic income shock ε (mean ≈ 1).
    ε_grid :: Vector{Float64} = [0.7, 1.0, 1.3]
    P_ε    :: Matrix{Float64} = [0.80 0.15 0.05;
                                 0.10 0.80 0.10;
                                 0.05 0.15 0.80]
    # Permanent lifetime-earnings TYPE θ (low / high profile). NO transitions.
    θ_grid :: Vector{Float64} = [0.75, 1.25]
    N_w   :: Int       = 250
    w_min :: Float64   = 0.0
    w_max :: Float64   = 150.0
end

Base.Broadcast.broadcastable(p::HIPParams) = Ref(p)

const hip_params = HIPParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached HIP household block
`IncomeShock ∘ IncomeReceipt(θ·ε) ∘ ConsumptionSavingsStage`. The
permanent earnings type enters as a plain extra `:income_type` Discrete
axis with NO Markov stage; the `IncomeStage` budget closure reads it
(`(; wealth, income, income_type, env)`), scaling the stochastic shock:
`b ↦ (1+r)·b + w·income_type·income`. The aggregate `A_mean = ∫ wealth dΛ`
moment is attached; per-type wealth is computed example-side from `Λ`.
"""
function hip_household(p = hip_params)
    layout = GriddedLayout(
        :wealth      => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income      => Discrete(p.ε_grid),
        :income_type => Discrete(p.θ_grid),     # permanent: no MarkovStage on this axis
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_ε)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, income_type, env) ->
            (1 + env.r) * wealth + env.w * income_type * income)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (no production, fixed prices) #
#-----------------------------------#

"Env for the HIP partial-equilibrium experiment: fixed return `r` and wage `w`."
hip_env(p = hip_params) = (; p.r, p.w)
