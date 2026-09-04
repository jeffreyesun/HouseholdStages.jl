####################################################################
# Two-asset HANK (Kaplan–Moll–Violante 2018) — steady state        #
####################################################################

# A household with a LIQUID asset b (return r_b, freely adjusted) and an ILLIQUID asset a (return
# r_a > r_b, COSTLY to adjust). Each period it picks the illiquid target a' (paying a convex deposit
# cost on the net flow d = a' − (1+r_a)a, which is debited from liquid) and then consumes/saves
# liquid. The illiquid choice therefore sets TWO axes at once — illiquid AND (via the deposit)
# liquid — which existing stages express through the **auxiliary-choice-axis pattern**: route the
# choice onto its own axis, act on both wealth axes downstream, then sum it away (see
# `examples/habit` for the other user of the pattern).
#
# Household block (time order), existing stages only, NO bespoke stage:
#
#   IncomeShock ∘ Receipt ∘ [ ChooseA' ∘ DebitLiquid ∘ CreditIlliquid ∘ Forget ] ∘ ConsumptionSavings
#
# `ChooseA'`        — `ArgmaxStage` picks a' onto the auxiliary `:illiquid_choice` axis.
# `DebitLiquid`     — `WealthChangeStage(:liquid, b ↦ b − d − χ(d))` reads BOTH `:illiquid_choice` (a') and `:illiquid` (old a).
# `CreditIlliquid`  — `WealthChangeStage(:illiquid, a ↦ a')` commits the choice.
# `Forget`          — `ForgetfulSumStage(:illiquid_choice)`.
# `ConsumptionSavings` — `ConsumptionSavingsStage(:liquid)` consumes from post-deposit liquid (β here).
#
# The illiquid block's deposit (the cross-axis flow) is pinned to its brute reference to machine
# precision in test/test_example_two_asset_hank.jl. A subsistence floor `ε` on post-deposit liquid
# keeps consumption feasible (the brute `ArgmaxStage` requires every cell to have a feasible action).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct TwoAssetParams
    β    :: Float64 = 0.94
    σ    :: Float64 = 2.0
    r_b  :: Float64 = 0.01                       # liquid return
    r_a  :: Float64 = 0.05                       # illiquid return (> r_b)
    κ    :: Float64 = 0.08                       # convex deposit cost χ(d) = κ·d²
    ε    :: Float64 = 0.05                       # subsistence floor on post-deposit liquid
    w    :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.7, 1.3]
    P_y    :: Matrix{Float64} = [0.8 0.2; 0.2 0.8]
    N_b  :: Int = 24
    b_max :: Float64 = 16.0
    N_a  :: Int = 12
    a_max :: Float64 = 16.0
end

Base.Broadcast.broadcastable(p::TwoAssetParams) = Ref(p)


# Household chain assembly #
#--------------------------#

"""
Build the two-asset HANK block via the auxiliary-choice-axis pattern, with `mean_liquid` and
`mean_illiquid` attached as moments.
"""
function two_asset_household(p = TwoAssetParams())
    bgrid = collect(range(0.0, p.b_max; length = p.N_b))
    agrid = collect(range(0.0, p.a_max; length = p.N_a))
    axes_base = (:liquid   => GriddedContinuous(bgrid),
                 :illiquid => GriddedContinuous(agrid),
                 :income   => Discrete(p.y_grid))
    block = GriddedLayout(axes_base..., :illiquid_choice => Discrete([1]))      # singleton
    full  = GriddedLayout(axes_base..., :illiquid_choice => Discrete(collect(1:p.N_a)))

    deposit(ac, a) = agrid[Int(ac)] - (1 + p.r_a) * a        # net illiquid flow d

    shock   = MarkovStage(block; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(block; axis = :liquid,                           # liquid cash = (1+r_b)b + w·y
        wealth_post = (; liquid, income) -> (1 + p.r_b) * liquid + p.w * income)
    choose  = ArgmaxStage(block, full; axis = :illiquid_choice, reward = zeros(p.N_a, 1))
    debit   = WealthChangeStage(full; axis = :liquid,                            # b ↦ max(b − d − κd², ε)
        wealth_post = (; illiquid_choice, illiquid, liquid) -> (d = deposit(illiquid_choice, illiquid);
                                      max(liquid - d - p.κ * d^2, p.ε)))
    credit  = WealthChangeStage(full; axis = :illiquid,                          # a ↦ a'
        wealth_post = (; illiquid_choice) -> agrid[Int(illiquid_choice)])
    forget  = ForgetfulSumStage(full; axis = :illiquid_choice)
    savings = ConsumptionSavingsStage(block; β = p.β, axis = :liquid,            # consume from post-deposit liquid
        utility = (cell, c) -> u_crra(c, Val(p.σ)))   # defaults: (; skip_monotonicity_check = false, utility_axes = nothing)

    hh = shock ∘ receipt ∘ choose ∘ debit ∘ credit ∘ forget ∘ savings
    return define_moments!(hh;
        mean_liquid   = at_end(integrand = :liquid,   reduce = sum),
        mean_illiquid = at_end(integrand = :illiquid, reduce = sum))
end
