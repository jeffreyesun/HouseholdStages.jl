####################################################################
# Costly diffusion — deliberate variance-INCREASE (the θ↑ dual)     #
####################################################################

# The "negative" of rational inattention: a household that pays to ADD
# dispersion to its next-period wealth — a deliberate mean-preserving spread —
# rather than to sharpen it. This is the θ↑ ("diffuse") reading of
# `MeanPreservingSpreadStage` flagged in MODEL_CATALOG.md §7 (the one ◐ on the
# opposites table). It realizes that sign-flip as a SHIPPED, SOLVED example
# while staying a pure composition of existing library stages.
#
# The engine is the SAME value-function convexity that drives Vereshchagina–
# Hopenhayn risk-shifting (`examples/risk_shifting`): a limited-liability wealth
# floor `max(wealth, a_floor)` makes the continuation locally CONVEX just above
# the floor (the downside is absorbed by the floor, the upside is kept). A
# mean-preserving spread on a convex region raises `E[V]`, so a household near
# the floor deliberately DIFFUSES — picks θ↑ — even paying a small dispersion
# cost. Far from the floor V is concave and the household picks θ = 0.
#
# Why this is genuinely the dual, not a rebuild of two existing examples:
#   • `risk_shifting` gambles via `GaussianLoadingStage` — a MULTIPLICATIVE risky
#     SHARE, Gaussian rows at mean `a'·(R_f + θμ)` and sd `|a'|·θ·σ` over the
#     `GaussianLoadingKernel`. Here the lever is `MeanPreservingSpreadStage` — an
#     ADDITIVE Gaussian mean-preserving spread of `b'` (mean `b'`, sd θ) over
#     the `MeanPreservingSpreadKernel` row. Same row family, different subspace.
#   • `rational_inattention` / `mackowiak_wiederholt` use the SAME stage
#     (`MeanPreservingSpreadStage`) but in the θ↓ "sharpen" reading: the dispersion
#     is an unwanted byproduct of a noisy signal, θ = 0 is the perfect-attention
#     benchmark, and θ* is interior only because of the borrowing constraint.
#     Here θ↑ is the DELIBERATE choice — the household wants the spread for its
#     own sake (convex-V option value), the opposite economic direction.
#
# The whole within-period problem is FIVE existing library stages, in time
# order, with NO bespoke household stage rolled here —
#
#     IncomeShock ∘ Receipt ∘ Savings ∘ Diffuse ∘ LimitedLiability
#
# `IncomeShock`      — `MarkovStage` on the income axis `y`: a persistent
#                      earnings shock that spreads the stationary distribution.
# `Receipt`          — `WealthChangeStage` `b ↦ b + w·y`: cash-on-hand each
#                      period (impatience keeps wealth bounded; the injection
#                      refills it, so mass churns near the floor where the
#                      diffusion choice is active).
# `Savings`          — `ConsumptionSavingsStage` picks next-period wealth `b'`;
#                      `c = x − b'`, CRRA. Grid floor is the borrowing limit.
# `Diffuse`          — `MeanPreservingSpreadStage` on `:wealth`: the household
#                      picks the continuous dispersion `θ ∈ [0, θ_max]` of a
#                      Gaussian mean-preserving spread of `b'` (sd θ, clamped)
#                      at the dispersion cost `c(θ) = λ·θ²`. This is the
#                      COSTLY-DIFFUSION cost direction: more spread costs more,
#                      θ = 0 is free.
# `LimitedLiability` — `WealthChangeStage` `b ↦ max(b, a_floor)`: limited
#                      liability. A bad diffusion draw cannot push wealth below
#                      `a_floor`. This floor CONVEXIFIES V just above `a_floor`
#                      — the engine that makes deliberate diffusion (θ↑) pay.
#
# Backward order reads right-to-left: limited liability floors the continuation,
# the diffusion stage sees that floored (convex) V and seats θ*(x), savings picks
# b', receipt relabels cash, the Markov stage takes the income expectation.
#
# Returns and income are exogenous (partial equilibrium): there is no market to
# clear, so the "outer loop" is a single `solve_steady_state_given_env!`.
#
# Literature: the convexity mechanism is Vereshchagina & Hopenhayn (AER 2009);
# the deliberate-diffusion / experimentation reading is the θ↑ member of the
# variance-RI family (Sims 2003 read in reverse) — see MODEL_CATALOG.md §7.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CostlyDiffusionParams
    β :: Float64       = 0.94                       # discount factor (impatience ⇒ bounded wealth)
    σ :: Float64       = 3.0                        # CRRA risk aversion
    w :: Float64       = 0.50                       # income scale (cash injection w·y)
    y_grid :: Vector{Float64} = [0.7, 1.3]          # 2-state persistent income
    P_y    :: Matrix{Float64} = [0.80 0.20;
                                 0.20 0.80]
    λ :: Float64       = 0.02                       # dispersion-cost scale  c(θ) = λ·θ²
    θ_max :: Float64   = 2.0                        # dispersion cap (θ = 0 is no diffusion)
    a_floor :: Float64 = 0.50                       # limited-liability floor (convexifies V)
    N_w   :: Int       = 120
    w_min :: Float64   = 0.0
    w_max :: Float64   = 12.0
end

Base.Broadcast.broadcastable(p::CostlyDiffusionParams) = Ref(p)

const costly_diffusion_params = CostlyDiffusionParams()

# CRRA felicity `u_crra` is provided by HouseholdStages.


# Household chain assembly #
#--------------------------#

"""
Build the costly-diffusion household block
`IncomeShock ∘ Receipt ∘ Savings ∘ Diffuse ∘ LimitedLiability`, with
`mean_wealth = ∫ wealth dΛ` attached. Five existing stages, no bespoke
household stage: the `Diffuse` leaf is a `MeanPreservingSpreadStage` choosing
the continuous dispersion θ ∈ [0, θ_max] of a Gaussian mean-preserving spread on
next-period wealth at the cost `c(θ) = λ·θ²`, and the `LimitedLiability` floor
convexifies the continuation so that θ↑ (deliberate diffusion) is optimal near
the floor.
"""
function costly_diffusion_household(p = costly_diffusion_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)               # (1+r)·wealth + w·income; r defaults to 0 in env
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    # Deliberate diffusion: pick the continuous dispersion θ of a Gaussian
    # mean-preserving spread of b' at cost λ·θ². The floor below convexifies
    # V ⇒ θ↑ near it.
    diffuse = MeanPreservingSpreadStage(layout; axis = :wealth, θ_max = p.θ_max,
        cost = (θ; env) -> env.λ * θ^2)

    # Limited liability: a bad spread draw cannot push wealth below a_floor.
    # This floor is the convexity engine that makes deliberate diffusion pay.
    liability = WealthChangeStage(layout;
        wealth_post = (; wealth, env) -> max(wealth, env.a_floor))

    hh = shock ∘ receipt ∘ savings ∘ diffuse ∘ liability
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end

"""
The `MeanPreservingSpreadStage` (`Diffuse`) leaf of a built costly-diffusion
block — its seated `policy` is the chosen dispersion θ*(x). Located by type
rather than a hard index (the upstream `ConsumptionSavingsStage` expands to
`argmax ∘ TimeDiscounting`, so the position is not fixed).
"""
function costly_diffusion_diffuse_stage(hh)
    stages = hh.buffer.stages
    idx = findfirst(s -> s isa MeanPreservingSpreadStage, stages)
    return stages[idx]
end


# Exogenous environment (plain function, no AbstractBlock) #
#----------------------------------------------------------#

"""
The exogenous costly-diffusion env: net return `r`, income scale `w`,
dispersion-cost scale `λ`, and the limited-liability floor `a_floor`.
"""
costly_diffusion_env(p = costly_diffusion_params; r = 0.0, w = p.w, λ = p.λ, a_floor = p.a_floor) =
    (; r, w, λ, a_floor)
