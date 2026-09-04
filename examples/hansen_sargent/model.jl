###############################################################
# Hansen–Sargent multiplier preferences / robust savings       #
###############################################################
#
# A standard incomplete-markets (Aiyagari/Huggett) saver, except the
# household is ROBUST: it does not trust the income transition `P_y`. Instead
# of taking the ordinary expectation `Σ_j P[i,j]·V_end[j]` over next-period
# income, it minimises over entropy-penalised distortions of `P_y`, paying a
# penalty `θ·KL(distorted‖P)` for each unit of relative entropy. The minimised
# object is the exponential/entropic certainty equivalent
#
#     V_start[i] = −θ·log Σ_j P[i,j]·exp(−V_end[j]/θ),      θ = robustness multiplier > 0,
#
# the risk-sensitive recursion of Hansen–Sargent (2008) multiplier preferences
# and Hansen–Sargent–Tallarini (1999). Small θ ⇒ very robust / pessimistic
# (heavy weight on bad income draws); θ→∞ ⇒ ordinary expectation (risk-neutral
# in the continuation).
#
# The headline point of this example: this risk-sensitive operator IS NOT a
# bespoke stage. It is `LogitChoiceStage` on the `:income` axis at NEGATIVE ε.
# The logit backward computes the log-sum-exp
#
#     V_start[i] = ε·log Σ_j exp((−C[i,j] + V_end[j])/ε),
#
# and the choice of cost matrix `C[i,j] = −ε·log P[i,j]` makes `exp(−C/ε) = P`,
# so the recursion collapses to
#
#     V_start[i] = ε·log Σ_j P[i,j]·exp(V_end[j]/ε)
#                = −θ·log Σ_j P[i,j]·exp(−V_end[j]/θ)      (ε = −θ),
#
# EXACTLY the entropic certainty equivalent with multiplier θ = |ε|. Robustness
# "comes for free" as the soft-MIN member of the very same log-sum-exp that
# gives logit discrete choice at ε>0 (see MODEL_CATALOG.md §7). The household
# block is three existing library stages, no bespoke stage rolled here —
#
#     RobustExpectation ∘ Receipt ∘ ConsumptionSavings
#
# `RobustExpectation` — `LogitChoiceStage(:income, ε<0)`, `C = −ε·log P_y`.
# `Receipt`           — `IncomeStage`: `a ↦ (1+r)·a + w·y` (cash-on-hand).
# `ConsumptionSavings`— `ConsumptionSavingsStage` picks next-period wealth.
#
# Returns are exogenous (partial equilibrium): a single inner V/Λ solve, no
# market to clear. β·(1+r)<1 plus the borrowing-constraint grid floor give a
# stationary wealth distribution. We solve across a grid of θ = |ε| and show
# the precautionary comparative static: a more robust (smaller θ) household
# holds MORE wealth. As a sanity check, a near-risk-neutral robust solve
# (θ = 1e8) reproduces the ordinary-expectation `MarkovStage` chain to several
# digits — the soft-MIN limits to the linear expectation.
#
# NOTE on the forward pass. `LogitChoiceStage.forward!` pushes the wealth
# distribution through the SEATED Gibbs kernel π(j|i) ∝ P[i,j]·exp(−V_end[j]/θ),
# i.e. the WORST-CASE (distorted) income measure. The stationary Λ reported
# here is therefore the ergodic distribution under the household's own
# pessimistic belief — the natural object for "how much does the robust agent
# accumulate," and the standard reading of the distorted-measure stationary
# distribution in robust-control models.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HansenSargentParams
    β :: Float64 = 0.96
    σ :: Float64 = 2.0                                 # CRRA flow curvature
    r :: Float64 = 0.03                                # fixed gross return − 1 (< 1/β − 1 ≈ 0.0417)
    w :: Float64 = 1.0                                 # wage / income scale

    # Idiosyncratic income process. STRICTLY POSITIVE transition (no log 0):
    # a persistent five-state chain with non-trivial spread, so precautionary
    # motives bite and the robust tilt has bad states to fear.
    y_grid :: Vector{Float64} = [0.5, 0.75, 1.0, 1.25, 1.5]
    P_y    :: Matrix{Float64} = [0.60 0.25 0.10 0.04 0.01;
                                 0.20 0.45 0.25 0.08 0.02;
                                 0.07 0.20 0.46 0.20 0.07;
                                 0.02 0.08 0.25 0.45 0.20;
                                 0.01 0.04 0.10 0.25 0.60]

    N_w   :: Int     = 120
    w_min :: Float64 = 0.0
    w_max :: Float64 = 80.0
end

Base.Broadcast.broadcastable(p::HansenSargentParams) = Ref(p)

const hansen_sargent_params = HansenSargentParams()


# Household chain assembly #
#--------------------------#

"""
Build the robust-savings household block `RobustExpectation ∘ Receipt ∘
ConsumptionSavings` at robustness multiplier `θ` (so logit scale `ε = −θ < 0`).
The risk-sensitive expectation over next-period income is a `LogitChoiceStage`
on the `:income` axis with `cost_matrix = −ε·log P_y` and `ε = −θ`: this makes
`exp(−C/ε) = P_y`, so the logit log-sum-exp backward reduces to the entropic
certainty equivalent `−θ·log Σ_j P_y[i,j]·exp(−V_end[j]/θ)`. Three existing
stages, no bespoke household stage. `θ` large ⇒ ordinary expectation.
"""
function hansen_sargent_household(p = hansen_sargent_params; θ::Float64 = 1.0)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    ε = -θ                                                       # ε < 0 ⇒ soft-MIN / robustness
    robust_expectation = LogitChoiceStage(layout;
        axis        = :income,
        cost_matrix = -ε .* log.(p.P_y),                         # C = −ε·log P_y  ⇒  exp(−C/ε) = P_y
        ε           = ε)
    receipt = IncomeStage(layout)                                # default (1+r)·wealth + w·income
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = robust_expectation ∘ receipt ∘ savings
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end

"""
Reference NON-robust household: the same block with the robust expectation
replaced by an ordinary `MarkovStage(:income, P_y)`. Used only to confirm that
the robust chain at large `θ` reproduces the ordinary-expectation steady state
— both are library-stage compositions.
"""
function reference_household(p = hansen_sargent_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end

"Env for the fixed-`r` partial-equilibrium experiment: gross return `r`, wage `w`."
hansen_sargent_env(p = hansen_sargent_params) = (; r = p.r, w = p.w)


# Policy extraction (reporting only — outside the block) #
#-------------------------------------------------------#

"""
Recover the chosen next-period wealth VALUES `b'(wealth, income)` from a solved
household. The savings leaf is the `ContinuousArgmaxStage` inside
`ConsumptionSavingsStage` on the `:wealth` axis (the unique `(N_w,
n_income)`-shaped policy-bearing leaf — the `LogitChoiceStage` income leaf also
carries a policy, sized `(n_income, …)`, so we select by shape); its policy
holds the chosen next-wealth values directly. This is a distribution-free read
of the savings policy, used to show the precautionary tilt cleanly (the
own-measure stationary mean wealth couples the policy with the worst-case
forward measure; see steady_state.jl).
"""
function hansen_sargent_savings_policy(hh, p = hansen_sargent_params)
    n_y   = length(p.y_grid)
    leaves = filter(s -> !(s isa HouseholdStages.ChainStage) &&
                         hasmethod(HouseholdStages.policy, Tuple{typeof(s)}),
                    collect(hh.buffer.stages))
    savings = only(filter(s -> size(HouseholdStages.policy(s)) == (p.N_w, n_y), leaves))
    return HouseholdStages.policy(savings)                   # (wealth, income) → next-wealth value
end
