####################################################################
# Maćkowiak–Wiederholt (2009, 2015) — single-margin attention       #
####################################################################

# A savings household that allocates FINITE ATTENTION to its idiosyncratic
# state while an AGGREGATE condition evolves exogenously — the single-margin
# reading of Maćkowiak & Wiederholt (2009, AER; 2015, REStud), "Optimal Sticky
# Prices under Rational Inattention" / "Business Cycle Dynamics under Rational
# Inattention". The whole within-period problem is FIVE existing library
# stages, in time order, with **no bespoke household stage rolled here** —
#
#     AggShock ∘ IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Attention
#
# `AggShock`           — `MarkovStage` on the AGGREGATE axis `:z`: an economy-
#                        wide productivity state evolving on its own Markov
#                        chain. In this single-margin version the agent does
#                        NOT buy precision about `z`; it is the exogenous
#                        aggregate condition the MW agent could (in the full
#                        model) choose to attend to.
# `IncomeShock`        — `MarkovStage` on the idiosyncratic `:income` axis.
# `Receipt`            — `WealthChangeStage` `b ↦ (1+r)·b + w·z·y`: the
#                        aggregate `z` scales the wage, so good aggregate times
#                        raise everyone's labor income (the channel through
#                        which the aggregate condition matters).
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next wealth `b'`;
#                        `c = b_in − b'`, CRRA.
# `Attention`          — `ScaleVarianceStage` on `:wealth` picks the dispersion
#                        `θ` of a mean-preserving spread `b' ↦ b' + θ·ξ`
#                        (ξ mean-zero) at the KL-style information cost
#                        `c(θ) = λ·θ²`. This is the agent's ONE attention
#                        margin — precision about its own (idiosyncratic)
#                        carried wealth state. `θ = 0` is perfect attention;
#                        larger `θ` is a noisier (cheaper) read of the state.
#
# What is faithful, and what is the gap. The MW mechanism is the ALLOCATION of
# a finite attention capacity across MULTIPLE signals — aggregate vs
# idiosyncratic — coupling their precisions through ONE budget constraint.
# That coupled multi-signal budget is a recorded G3-adjacent gap (MODEL_CATALOG
# §2 / gaps item 5): a single Shannon constraint over several precisions is not
# one univariate streaming stage, because each per-axis `ScaleVarianceStage`
# optimises its own θ independently — there is no shared-budget coupler in the
# per-axis streaming vocabulary. What IS faithful, and what this example builds,
# is the SINGLE-MARGIN household: one attention choice (here, to the
# idiosyncratic wealth state) priced at `λ·θ²`, with the aggregate condition
# entering exogenously. The attention gradient — θ* responding to λ and to the
# state — is the genuine MW comparative static, just on one margin.
#
# Why θ* is interior (the economics, as in `examples/rational_inattention`). A
# mean-preserving spread on a globally concave continuation picks θ=0 (Jensen +
# positive cost). The bite is the borrowing constraint: for poor, near-
# constrained households the continuation in wealth is locally convex (a low
# draw is floored at the constraint, a high draw escapes it), so a small
# dispersion carries option value; the cost `λ·θ²` disciplines it. Result:
# θ* high for the constrained poor, → 0 for the wealthy, and θ* falls
# everywhere as λ rises.
#
# Returns are exogenous (partial equilibrium): no market to clear, so the outer
# loop is a single `solve_steady_state_given_env!`. The borrowing constraint
# (`b' ≥ 0`) plus impatience (`β·(1+r) < 1`) deliver a stationary distribution.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MWParams
    β :: Float64       = 0.95
    σ :: Float64       = 3.0                       # CRRA (concavity drives the attention choice)
    r :: Float64       = 0.03                      # exogenous net return on wealth
    w :: Float64       = 1.0                       # wage scale (multiplied by the aggregate z)
    λ :: Float64       = 0.01                      # information-cost scale  c(θ) = λ·θ²
    # Idiosyncratic income.
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    # Aggregate productivity z (the MW "aggregate condition") — a two-state
    # chain scaling the wage. Persistent, mean ≈ 1.
    z_grid :: Vector{Float64} = [0.95, 1.05]
    P_z    :: Matrix{Float64} = [0.8 0.2;
                                 0.2 0.8]
    # Attention margin (mean-preserving spread of next wealth).
    dispersions :: Vector{Float64} = collect(0.0:0.25:1.0)   # θ grid (0 = perfect attention)
    shocks      :: Vector{Float64} = [-1.0, 1.0]             # mean-zero signal-noise support ξ
    weights     :: Vector{Float64} = [0.5, 0.5]
    N_w   :: Int       = 80
    w_min :: Float64   = 0.0
    w_max :: Float64   = 8.0
end

Base.Broadcast.broadcastable(p::MWParams) = Ref(p)

const mw_params = MWParams()

# CRRA felicity `u_crra` is provided by HouseholdStages.


# Household chain assembly #
#--------------------------#

"""
Build the single-margin Maćkowiak–Wiederholt household block
`AggShock ∘ IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Attention`, with
`mean_wealth = ∫ wealth dΛ` attached. Five existing stages, no bespoke
household stage: an aggregate `:z` `MarkovStage`, an idiosyncratic `:income`
`MarkovStage`, the receipt (aggregate `z` scaling the wage), the savings spine,
and the `Attention` leaf — a `ScaleVarianceStage` choosing the dispersion θ of
next-period wealth at the information cost `c(θ) = λ·θ²` (a stand-in for `λ·KL`).
"""
function mw_household(p = mw_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
        :z      => Discrete(p.z_grid),
    )

    aggshock = MarkovStage(layout; axis = :z, transition_matrix = p.P_z)
    incshock = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    # Aggregate z scales the wage: good aggregate times raise labor income.
    receipt  = WealthChangeStage(layout;          # defaults: (; axis = :wealth)
        wealth_post = (; wealth, income, z, env) -> (1 + env.r) * wealth + env.w * z * income)
    savings  = ConsumptionSavingsStage(layout;    # defaults: (; axis = :wealth, monotone_search = :divide_conquer)
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)))
    attention = ScaleVarianceStage(layout; axis = :wealth,
        dispersions = p.dispersions, shocks = p.shocks, weights = p.weights,
        cost = (θ; env) -> env.λ * θ^2)           # λ·θ² — stand-in for λ·KL(θ)

    hh = aggshock ∘ incshock ∘ receipt ∘ savings ∘ attention
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end

"The `Attention` (`ScaleVarianceStage`) leaf — its seated `policy` is θ*(x), the chosen dispersion."
mw_attention_stage(hh) = hh.buffer.stages[end]
