#####################################################################
# Kaplan–Violante (2014) — wealthy hand-to-mouth, two assets         #
#####################################################################
#
# A household with a LIQUID asset b (low return r_b, freely adjusted) and an
# ILLIQUID asset a (high return r_a > r_b) that can only be rebalanced by paying
# a FIXED (lumpy) transaction cost κ_f. The defining KV mechanism: many
# households optimally DO NOT pay the cost — they hold illiquid wealth but act
# hand-to-mouth in liquid, generating large MPCs (the "wealthy hand-to-mouth").
#
# This is the SAME two-asset auxiliary-choice-axis pattern as
# examples/two_asset_hank, with two changes:
#   * the adjustment cost is a FIXED κ_f (a non-convexity) rather than convex
#     κ·d²; brute `ArgmaxStage` maximisation needs no convexity, so the lumpy
#     cost is just a different closure;
#   * the choice axis carries an explicit NO-ADJUST option (index N_a+1): keep
#     illiquid at its accrued level a' = (1+r_a)a, pay nothing, touch nothing in
#     liquid. This is the option whose taking makes a household wealthy-HtM.
#
# Household block (time order), existing stages only, NO bespoke stage:
#
#   IncomeShock ∘ Receipt(:liquid) ∘ ChooseA' ∘ DebitLiquid ∘ CreditIlliquid
#     ∘ Forget ∘ ConsumptionSavings(:liquid)
#
# `ChooseA'`        — `ArgmaxStage` picks the illiquid action (adjust-to-grid-i,
#                     or no-adjust) onto the auxiliary `:illiquid_choice` axis.
# `DebitLiquid`     — `WealthChangeStage(:liquid)`: if adjusting, debit the net
#                     deposit d and the fixed cost κ_f; if not, leave liquid
#                     untouched. Reads `:illiquid_choice` and old `:illiquid`.
# `CreditIlliquid`  — `WealthChangeStage(:illiquid)`: a ↦ a' (grid point if
#                     adjusting, (1+r_a)a if not).
# `Forget`          — `ForgetfulSumStage(:illiquid_choice)` collapses the axis.
# `ConsumptionSavings` — `ConsumptionSavingsStage(:liquid)` over post-deposit liquid.
#
# A subsistence floor ε on post-deposit liquid keeps consumption feasible for
# every cell (the brute argmax requires a feasible action everywhere).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct KVParams
    β    :: Float64 = 0.94
    σ    :: Float64 = 2.0
    r_b  :: Float64 = 0.005                       # liquid return (low)
    r_a  :: Float64 = 0.040                       # illiquid return (high, > r_b)
    κ_f  :: Float64 = 0.05                        # FIXED adjustment cost (lumpy)
    ε    :: Float64 = 0.02                        # subsistence floor on post-deposit liquid
    w    :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.7, 1.3]
    P_y    :: Matrix{Float64} = [0.8 0.2; 0.2 0.8]
    N_b  :: Int = 24
    b_max :: Float64 = 16.0
    N_a  :: Int = 12
    a_max :: Float64 = 16.0
end

Base.Broadcast.broadcastable(p::KVParams) = Ref(p)


# Household chain assembly #
#--------------------------#

"""
Build the Kaplan–Violante wealthy-hand-to-mouth two-asset block via the
auxiliary-choice-axis pattern (existing stages only). The illiquid choice axis
has `N_a + 1` actions: indices `1..N_a` adjust to `agrid[i]` paying the FIXED
cost `κ_f`; index `N_a+1` is the no-adjust option (illiquid accrues to
`(1+r_a)a`, liquid untouched). `mean_liquid`, `mean_illiquid`, and `frac_htm`
(mass with near-zero liquid but positive illiquid — the wealthy-HtM share) are
attached.
"""
function kv_household(p = KVParams())
    bgrid = collect(range(0.0, p.b_max; length = p.N_b))
    agrid = collect(range(0.0, p.a_max; length = p.N_a))
    n_choice = p.N_a + 1                                   # +1 = the no-adjust action
    noadjust = n_choice                                    # index meaning "don't adjust"

    axes_base = (:liquid   => GriddedContinuous(bgrid),
                 :illiquid => GriddedContinuous(agrid),
                 :income   => Discrete(p.y_grid))
    block = GriddedLayout(axes_base..., :illiquid_choice => Discrete([1]))           # singleton
    full  = GriddedLayout(axes_base..., :illiquid_choice => Discrete(collect(1:n_choice)))

    # New illiquid level a' implied by a choice index, given the old level a.
    a_next(ac, a) = Int(ac) == noadjust ? (1 + p.r_a) * a : agrid[Int(ac)]
    # Net deposit d into illiquid (debited from liquid), and the fixed cost paid.
    deposit(ac, a) = Int(ac) == noadjust ? 0.0 : agrid[Int(ac)] - (1 + p.r_a) * a
    fixedcost(ac)  = Int(ac) == noadjust ? 0.0 : p.κ_f

    shock   = MarkovStage(block; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(block; axis = :liquid,                               # liquid cash = (1+r_b)b + w·y
        wealth_post = (; liquid, income) -> (1 + p.r_b) * liquid + p.w * income)
    choose  = ArgmaxStage(block, full; axis = :illiquid_choice, reward = zeros(n_choice, 1))
    debit   = WealthChangeStage(full; axis = :liquid,                                # b ↦ max(b − d − κ_f, ε)
        wealth_post = (; illiquid_choice, illiquid, liquid) ->
            max(liquid - deposit(illiquid_choice, illiquid) - fixedcost(illiquid_choice), p.ε))
    credit  = WealthChangeStage(full; axis = :illiquid,                              # a ↦ a'
        wealth_post = (; illiquid_choice, illiquid) -> a_next(illiquid_choice, illiquid))
    forget  = ForgetfulSumStage(full; axis = :illiquid_choice)
    savings = ConsumptionSavingsStage(block; β = p.β, axis = :liquid,                # consume from post-deposit liquid
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = shock ∘ receipt ∘ choose ∘ debit ∘ credit ∘ forget ∘ savings
    return define_moments!(hh;
        mean_liquid   = at_end(integrand = :liquid,   reduce = sum),
        mean_illiquid = at_end(integrand = :illiquid, reduce = sum),
        frac_htm      = at_end(
            integrand = (; liquid, illiquid) ->
                (liquid <= bgrid[2] + 1e-9 && illiquid > agrid[2]) ? 1.0 : 0.0,
            reduce = sum),
    )
end
