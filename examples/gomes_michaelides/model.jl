####################################################################
# Gomes–Michaelides (2005) — participation cost + preference types  #
####################################################################

# Limited stock-market PARTICIPATION matched the Gomes–Michaelides way: a
# FIXED participation cost plus HETEROGENEOUS preferences (Gomes & Michaelides,
# "Optimal Life-Cycle Asset Allocation: Understanding the Empirical Evidence",
# JF 2005). Two margins coexist: an EXTENSIVE margin (pay the fixed cost and
# hold any equity, or stay out) and, conditional on entry, an INTENSIVE margin
# (the risky share). The fixed cost endogenously sorts households into
# participants and non-participants; the preference spread lets the model match
# both the participation rate and conditional shares at once.
#
# The whole within-period problem is the `⊕`-over-preference-type direct sum of
# the Merton/portfolio chain, with **no bespoke household stage** —
#
#     block_i = IncomeShock ∘ Receipt ∘ ConsumptionSavings(σ_i) ∘ Portfolio(F)
#     household = product(block_1, …, block_n; axis = :ptype)
#
# `IncomeShock`        — `MarkovStage` on the income axis.
# `Receipt`            — `WealthChangeStage` `a ↦ a + w·y` (cash-on-hand).
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next financial wealth
#                        `b'`; CRRA felicity with PER-TYPE curvature `σ_i`
#                        captured in the utility closure.
# `Portfolio`          — `GaussianLoadingStage`, here read as the portfolio stage
#                        (anchor = R_f, increment = the Gaussian excess return):
#                        picks the CONTINUOUS risky share
#                        `θ ∈ [0, 1]`, next wealth `b'·(R_f + θ·(μ_x + σ_x·Z))`
#                        for a truncated-Gaussian excess moment-matched to
#                        (R_risky, p_risky). The FIXED PARTICIPATION COST rides
#                        its `cost` closure:
#                        `cost = (θ; env) -> θ > 0 ? env.F : 0.0` — staying out
#                        (`θ = 0`) is free, ANY positive share pays `F`. This is
#                        the extensive margin: the continuous solver's scan +
#                        exact-endpoint logic compares the free `θ = 0` corner
#                        against the best interior share net of `F`, so
#                        non-participants sit at EXACTLY `θ* = 0` and
#                        participants pick the interior (or cap) share — both
#                        margins from one closure.
#
# THE PARTICIPATION COST IS IN VALUE UNITS. `GaussianLoadingStage`'s backward sets
# `V_start(b') = max_θ[ E_Z V_end(b'·(R_f + θ·(μ_x + σ_x·Z))) − cost(θ) ]`, so `F` is an
# additive penalty in continuation-value (utils), NOT a wealth/consumption
# subtraction. The library `cost` closure sees only `(θ; env)` — never the
# cell's wealth — so a wealth-DENOMINATED fixed cost (the literal G–M monetary
# cost) is NOT expressible through this primitive. The faithful statement here
# is a flat per-period utility cost of entry. One consequence is calibration-
# relevant: with a flat utility cost the participation–wealth gradient flips
# sign at `σ = 1`. For `σ > 1` the utils gain from the premium scales as
# `b'^(1-σ)` and FALLS in wealth (the rich would drop out); for `σ < 1` it RISES
# in wealth — the textbook "rich participate" gradient. We therefore calibrate
# the preference types with `σ < 1`, so participation rises with wealth as in
# the data. See README for the full fidelity note.
#
# Per-type DISTINCT σ rides the `ConsumptionSavingsStage` utility CLOSURE: the
# closure captures `σ::Float64`, so its TYPE is identical across types (only the
# captured value differs) — exactly the uniform-Spec-type invariant `product`
# requires. The `:ptype` axis is a size-1 SINGLETON in the block layout, grown
# `1 → n` by `product`. The direct sum is block-diagonal (a household keeps its
# type forever), so the STANDARD solver runs directly — each type's slice
# converges independently.
#
# Returns are exogenous (partial equilibrium): no market to clear.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct GomesMichaelidesParams
    β :: Float64       = 0.955
    # Preference types: CRRA curvature spread. ALL σ < 1 so the flat utility
    # participation cost yields the textbook "participation rises with wealth"
    # gradient (see header / README). Lower σ ⇒ more risk-tolerant ⇒ larger
    # desired share ⇒ more willing to pay the fixed cost.
    σ_grid :: Vector{Float64} = [0.50, 0.85]
    w :: Float64       = 1.0                       # wage (income scale)
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    R_f     :: Float64 = 1.02                       # gross risk-free return
    # Risky leg: mean 1.05 (3% premium) with a wide spread, so even risk-tolerant
    # σ < 1 types pick an interior share rather than cornering. Feeds the
    # moment-matched Gaussian excess (μ_x = 0.03, σ_x = 0.25) below.
    R_risky :: Vector{Float64} = [0.80, 1.30]
    p_risky :: Vector{Float64} = [0.5, 0.5]
    F       :: Float64 = 0.025                       # fixed participation cost (utils, per period)
    N_w   :: Int       = 160
    w_min :: Float64   = 0.0
    w_max :: Float64   = 120.0
end

Base.Broadcast.broadcastable(p::GomesMichaelidesParams) = Ref(p)

const gm_params = GomesMichaelidesParams()


# Household chain assembly #
#--------------------------#

"""
Build one preference type's Merton block `IncomeShock ∘ Receipt ∘
ConsumptionSavings(σ) ∘ Portfolio(F)` against the SHARED layout (the `:ptype`
axis a size-1 singleton that `product` grows to the type count, so every block
sits between the same two layouts). The CRRA curvature `σ` is captured in the
savings utility closure (only the value differs across types) and
the fixed participation cost `F` rides the portfolio `cost` closure
(`θ > 0 ⇒ F`, `θ = 0` free). Only the captured `σ` differs across calls.
"""
function gm_block(σ::Float64, p = gm_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
        :ptype  => Discrete([1.0]),          # size-1 singleton; product grows it 1 → n
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;                                       # cash-on-hand x = a + w·y
        wealth_post = (; wealth, income, env) -> wealth + env.w * income)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(σ)))                        # σ captured: same closure TYPE per type
    # Gaussian excess-return moments matched to the two-point lottery's excess returns.
    μx = sum(p.p_risky .* (p.R_risky .- p.R_f))
    σx = sqrt(sum(p.p_risky .* (p.R_risky .- p.R_f) .^ 2) - μx^2)
    portfolio = GaussianLoadingStage(layout;
        anchor = p.R_f, increment_mean = μx, increment_sd = σx,
        cost = (θ; env) -> θ > 0 ? env.F : 0.0)                               # fixed participation cost (extensive margin)

    return shock ∘ receipt ∘ savings ∘ portfolio
end

"""
The Gomes–Michaelides household block: the direct sum
`product(gm_block(σ_1), …, gm_block(σ_n); axis = :ptype)` of the per-type Merton
blocks, with the aggregate `mean_wealth = ∫ wealth dΛ` moment attached. The
per-type risky-share policy `θ*(x)` and participation rate are read example-side
from the `:ptype`-indexed slices (see `steady_state.jl`).
"""
function gm_household(p = gm_params)
    blocks = [gm_block(σ, p) for σ in p.σ_grid]
    hh = product(blocks...; axis = :ptype)
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end


# Env (no production, fixed prices) #
#-----------------------------------#

"Env for the Gomes–Michaelides experiment: wage `w` and the fixed participation cost `F` (utils)."
gm_env(p = gm_params) = (; p.w, p.F)
