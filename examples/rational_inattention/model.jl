###################################################################
# Gaussian / variance rational inattention — savings + attention   #
###################################################################

# An incomplete-markets savings household that ALSO chooses the dispersion
# of its next-period wealth state at an information cost — the Sims (2003)
# / Maćkowiak–Wiederholt variance-RI reading. The point of this example:
# the entire within-period problem is FOUR existing library stages, in
# time order, with **no bespoke household stage rolled here** —
#
#     IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Attention
#
# `IncomeShock`        — `MarkovStage` on the income axis.
# `Receipt`            — `WealthChangeStage` `b ↦ (1+r) b + w·y`.
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period
#                        wealth `b'` on the wealth grid; `c = b_in − b'`.
# `Attention`          — `MeanPreservingSpreadStage` picks the CONTINUOUS
#                        dispersion `θ ∈ [0, θ_max]` of a Gaussian
#                        mean-preserving spread of `b'` (sd θ, clamped to the
#                        grid) at the information cost `c(θ) = λ·θ²` — a
#                        stand-in for `λ·KL(θ)`. `θ = 0` is perfect attention
#                        (no noise, no cost); larger `θ` is a noisier signal
#                        about the state, paid for inversely through `λ`.
#
# Why θ* is interior (not the naive θ*=0). A pure mean-preserving spread on
# a globally concave continuation always picks θ=0 (Jensen + a positive
# cost). The bite here is the **borrowing constraint**: for poor, near-
# constrained households the continuation value in wealth is locally
# convex (the low draw is floored at the constraint, the high draw escapes
# it), so a small dispersion carries option value. `MeanPreservingSpreadStage`
# solves the continuous θ per cell (scan + Newton on the smooth Gaussian
# objective) and seats the optimum θ*(x); the quadratic information cost
# λ·θ² disciplines it. The result is a sensible
# attention gradient — θ* high for the constrained poor, → 0 for the
# wealthy — and θ* falls everywhere as λ rises (the RI comparative static).
#
# Returns are exogenous (partial equilibrium): there is no market to clear,
# so the "outer loop" is a single `solve_steady_state_given_env!`. The
# borrowing constraint (`b' ≥ 0`, the grid floor) plus impatience
# (`β·(1+r) < 1`) deliver a stationary wealth distribution.
#
# Literature: Sims (2003, JME) "Implications of Rational Inattention";
# Maćkowiak & Wiederholt (2009, AER) "Optimal Sticky Prices under RI".

using HouseholdStages


# Parameters #
#------------#

@kwdef struct RationalInattentionParams
    β :: Float64       = 0.95
    σ :: Float64       = 3.0                       # CRRA (concavity drives the attention choice)
    r :: Float64       = 0.03                      # exogenous net return on wealth
    w :: Float64       = 1.0                       # wage (income scale)
    λ :: Float64       = 0.01                      # information-cost scale  c(θ) = λ·θ²
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    θ_max :: Float64   = 1.0                       # dispersion cap (θ = 0 is perfect attention)
    N_w   :: Int       = 80
    w_min :: Float64   = 0.0
    w_max :: Float64   = 8.0
end

Base.Broadcast.broadcastable(p::RationalInattentionParams) = Ref(p)

const rational_inattention_params = RationalInattentionParams()

# CRRA felicity `u_crra` is provided by HouseholdStages.


# Household chain assembly #
#--------------------------#

"""
Build the variance-RI household block `IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Attention`,
with `mean_wealth = ∫ wealth dΛ` attached. Four existing stages, no bespoke household stage: the
`Attention` leaf is a `MeanPreservingSpreadStage` choosing the continuous dispersion θ ∈ [0, θ_max]
of next-period wealth at the information cost `c(θ) = λ·θ²` (a stand-in for `λ·KL`).
"""
function rational_inattention_household(p = rational_inattention_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)               # (1+r)·wealth + w·income, the standard receipt
    savings = ConsumptionSavingsStage(layout;   # defaults: (; axis = :wealth)
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))
    attention = MeanPreservingSpreadStage(layout; axis = :wealth, θ_max = p.θ_max,
        cost = (θ; env) -> env.λ * θ^2)               # λ·θ² — stand-in for λ·KL(θ)

    hh = shock ∘ receipt ∘ savings ∘ attention
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
