######################################################################
# Promotion / career-effort ladder — convex-cost promotion hazard     #
######################################################################

# A savings household on a job ladder that can pay a CONVEX EFFORT cost to raise the
# hazard of climbing one rung up. Higher rungs pay more, so workers invest in promotion;
# the convex cost keeps the promotion intensity interior. The whole point of this Part-3
# example: the within-period problem is a pure composition of EXISTING library stages, in
# time order, with NO bespoke household stage —
#
#     Promotion ∘ Demotion ∘ Receipt ∘ ConsumptionSavings
#
# (`∘` runs the LEFT stage first.)
#
# `Promotion` — `MixingStage` on the `:rung` axis (the centerpiece). The worker blends
#               `θ ∈ [0,1]` of a CLIMB kernel `K_A` (rung i → i+1 for sure; top rung
#               absorbing) and a STAY kernel `K_B = I` (identity), at convex cost
#               `c(θ) = θ²/(2κ)`. `θ` is the promotion intensity. The closed form is
#                   `V = K_B·V + c*(K_A·V − K_B·V) = stay_value + c*(climb − stay)`,
#               so `θ*(rung) = clamp(κ·(climb − stay), 0, 1)` rises with the value gap
#               between climbing and staying — workers on low rungs, who have the most to
#               gain, invest most. `κ` is chosen so `θ*` is INTERIOR, not pinned at 1.
#
#               This is `MixingStage` used in the SEARCH direction — pay to transition UP —
#               the exact MIRROR of `examples/insurance`'s `RetentionStage`, which pays NOT
#               to transition (`K_A = I`, `K_B = exit_kernel`). Here `K_A = climb`,
#               `K_B = I`: the two directions of one hazard control (Part 3's contrast).
#
# `Demotion`  — `MarkovStage` on the `:rung` axis: an EXOGENOUS job-loss / separation shock
#               that knocks a worker down one rung w.p. `p_demote` (bottom rung reflecting).
#               This closes the ladder: with the top rung absorbing under promotion alone,
#               all mass would pile at the top and the wage path would be risk-free; the
#               separation shock supplies a birth-death balance (non-degenerate rung
#               distribution) AND the income risk that makes the savings spine bite. It is
#               an existing `MarkovStage` fed a banded kernel — no bespoke stage. (The
#               minimal `Promotion ∘ Receipt ∘ Savings` chain also solves, but absorbs all
#               mass at the top with ≈0 wealth — see README.)
#
# `Receipt`   — `WealthChangeStage` `b ↦ (1+r) b + wage_at_rung(rung)`. The wage rises with
#               rung; the `:rung` axis VALUES are the wage levels directly.
#
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period wealth `b'`; implicit
#               budget `c = b_in − b'`; CRRA utility.
#
# Partial equilibrium: return `r` is exogenous, so the "outer loop" is a single
# `solve_steady_state_given_env!`. The stationary distribution spreads mass up the ladder,
# with an interior, rung-decreasing promotion policy `θ*(rung)`.
#
# Library stages used (NO bespoke household stage):
#   MixingStage, MarkovStage, WealthChangeStage, ConsumptionSavingsStage.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct PromotionParams
    β   :: Float64 = 0.96          # discount factor
    σ   :: Float64 = 2.0           # CRRA curvature
    r   :: Float64 = 0.03          # exogenous (PE) interest rate on savings

    # Job ladder: the rung VALUES are the per-period wages (increasing).
    wage_grid      :: Vector{Float64} = [1.0, 1.5, 2.0, 2.5]
    cost_curvature :: Float64 = 0.6   # κ in c(θ) = θ²/(2κ); chosen so θ* is interior
    p_demote       :: Float64 = 0.12  # P(separation knocks worker down one rung)

    # Wealth grid.
    N_w   :: Int     = 100
    w_min :: Float64 = 0.0
    w_max :: Float64 = 60.0
end

Base.Broadcast.broadcastable(p::PromotionParams) = Ref(p)

const promotion_params = PromotionParams()


# Utility: CRRA felicity `u_crra` is provided by HouseholdStages.


# Promotion kernels (plain economic helpers — row-stochastic on the rung ladder) #
#-------------------------------------------------------------------------------#

"""
Build the row-stochastic CLIMB kernel `K_A[from, to]` on an `n`-rung ladder: rung `i`
moves UP to rung `i+1` with probability 1, and the top rung is absorbing (stays). This
is the `θ = 1` corner of the promotion `MixingStage` — plain data, not a new stage.
"""
function climb_kernel(n::Integer)
    K = zeros(Float64, n, n)
    for i in 1:n
        K[i, min(i + 1, n)] = 1.0
    end
    return K
end

"""
The `n×n` identity (STAY) kernel — the `θ = 0` corner of the promotion `MixingStage`
(no rung change). Plain data handed to `MixingStage` as `K_B`.
"""
identity_kernel(n::Integer) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

"""
Build the row-stochastic DEMOTION kernel `T[from, to]` on an `n`-rung ladder: rung `i`
drops one rung DOWN w.p. `p` (a job-loss / separation shock) and stays w.p. `1−p`; the
bottom rung is reflecting (stays w.p. 1). Plain data handed to `MarkovStage`, the
exogenous turnover that closes the ladder.
"""
function demotion_kernel(n::Integer, p::Real)
    T = zeros(Float64, n, n)
    for i in 1:n
        if i == 1
            T[i, i] = 1.0                           # bottom rung reflects
        else
            T[i, i - 1] = p
            T[i, i]     = 1 - p
        end
    end
    return T
end


# Household chain assembly — THREE library stages, NO bespoke stage #
#------------------------------------------------------------------#

"""
Build the promotion-ladder household block
`Promotion ∘ Demotion ∘ Receipt ∘ ConsumptionSavings`, with the rung distribution
(per-rung indicator sums) and mean wealth attached as moments. The `:rung` axis is the
wage grid itself. `Promotion` is a `MixingStage` blending a CLIMB kernel `K_A` (rung
i → i+1) and the identity `K_B = I` (stay) at convex cost — the SEARCH direction of the
hazard control, mirror of `examples/insurance`'s `RetentionStage`. `Demotion` is a
`MarkovStage` job-loss shock that knocks workers down a rung, closing the ladder. No
bespoke household stage; the climb/stay/demotion kernels are economic data fed to
existing stages.
"""
function promotion_household(p = promotion_params)
    n_rung = length(p.wage_grid)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :rung   => Discrete(p.wage_grid),          # axis VALUES are the wages
    )

    promotion = MixingStage(layout; axis = :rung,
        K_A            = climb_kernel(n_rung),      # θ=1: climb one rung
        K_B            = identity_kernel(n_rung),   # θ=0: stay
        cost_curvature = p.cost_curvature)

    demotion = MarkovStage(layout; axis = :rung,    # exogenous job-loss separation shock
        transition_matrix = demotion_kernel(n_rung, p.p_demote))

    receipt = WealthChangeStage(layout;            # defaults: (; axis = :wealth)
        wealth_post = (; wealth, rung, env) -> (1 + env.r) * wealth + rung,
    )

    savings = ConsumptionSavingsStage(layout;      # defaults: (; axis = :wealth, monotone_search = :divide_conquer)
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )

    # Per-rung mass indicators (one moment per rung) for the rung distribution.
    rung_moments = (; (Symbol("rung_$k") =>
            at_end(integrand = (; rung) -> rung == p.wage_grid[k] ? 1.0 : 0.0, reduce = sum)
        for k in 1:n_rung)...)

    hh = promotion ∘ demotion ∘ receipt ∘ savings
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        mean_rung   = at_end(integrand = :rung,   reduce = sum),
        rung_moments...,
    )
end


# Exogenous env (plain function, partial equilibrium) #
#-----------------------------------------------------#

"The exogenous price env for the promotion-ladder household: net return `r`."
promotion_env(p = promotion_params) = (; r = p.r)
