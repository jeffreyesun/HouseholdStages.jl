#####################################################################
# Two-asset HANK (Kaplan–Moll–Violante 2018) — Route A′ (portfolio   #
# value) — steady state                                             #
#####################################################################

# Same economics as examples/two_asset_hank, a DIFFERENT (cleaner) state. A household holds a LIQUID
# asset b (return r_b, free to adjust) and an ILLIQUID asset a (return r_a > r_b, costly to adjust).
# Each period it picks an illiquid target a' (paying a convex adjustment cost on the net deposit
# d = a' − (1+r_a)a) and consumes/saves the liquid balance.
#
# Route A′ — the PORTFOLIO-VALUE reformulation. Instead of two stock axes (b, a) we track
#
#     W = b + a   (portfolio value)   and   a   (illiquid),    with liquid  b = W − a  DERIVED.
#
# A pure transfer between the two stocks (the deposit) conserves W, so the illiquid choice a' moves
# ONLY the :illiquid axis — an ordinary single-axis argmax. W is untouched by the rebalance, and
# liquid = W − a' automatically reflects the deposit. There is NO auxiliary :illiquid_choice axis and
# NO n_choice× memory blow-up (contrast examples/two_asset_hank, which routes the choice through a
# separate axis).
#
# Household block (time order), existing stages only, NO bespoke stage:
#
#   IncomeShock ∘ ReturnsW ∘ ReturnsA ∘ Rebalance ∘ Consume
#
# `IncomeShock` — `MarkovStage(:income)`.
# `ReturnsW`    — `WealthChangeStage(:wealth)`: W ↦ (1+r_b)·W + (r_a−r_b)·a + w·income.
#                 (Since b = W−a, gross liquid+illiquid receipts are (1+r_b)(W−a) + (1+r_a)a + w·y.)
#                 Reads the OLD illiquid a, so it must run time-before `ReturnsA`.
# `ReturnsA`    — `WealthChangeStage(:illiquid)`: a ↦ (1+r_a)·a.
# `Rebalance`   — `ArgmaxStage(:illiquid)` choosing a', reward M[a', a] = −χ(a' − a) where a is the
#                 post-return illiquid. A single-axis argmax; W is left untouched. The adjustment cost
#                 enters as a UTILITY cost (see "Why a utility cost" below).
# `Consume`     — `ArgmaxStage(:wealth)` choosing next W', consumption c = W − W',
#                 utility u(c), with the LIQUID constraint W' ≥ a (so liquid = W' − a ≥ 0) baked into
#                 the reward by masking W' < a to −Inf. β here.
#
# Why a utility cost. The deposit cost χ(d) depends on both endpoints — the chosen a' and the OLD
# (post-return) a. In Route A′ the rebalance OVERWRITES the illiquid axis with a', so by the time a
# downstream resource debit would run, the old a is gone. Charging χ as a UTILITY cost inside the
# rebalance reward M[a', a] sidesteps this: the reward sees both a' (after) and a (before) in one
# matrix, so no overwrite issue arises. Making χ a RESOURCE cost (debit W by χ) would reintroduce the
# small overwrite that the auxiliary-axis route exists to handle; we accept the utility-cost reading
# (the strictly cleaner KMV route). The original two_asset example uses the resource-cost reading.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct TwoAssetPvParams
    β    :: Float64 = 0.92
    σ    :: Float64 = 2.0
    r_b  :: Float64 = 0.01                        # liquid return
    r_a  :: Float64 = 0.03                        # illiquid return (> r_b); β(1+r_a) < 1 keeps the
                                                  #   illiquid stock grid-INTERIOR (not pinned at a_max)
    κ    :: Float64 = 0.08                        # convex adjustment cost χ(d) = κ·d²  (utility units)
    ε    :: Float64 = 1e-3                         # subsistence consumption floor (keeps every cell feasible:
                                                  #   an all-infeasible cell would carry V = −Inf, incl. the
                                                  #   constrained grid bottom)
    w    :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.7, 1.3]
    P_y    :: Matrix{Float64} = [0.8 0.2; 0.2 0.8]
    N_W  :: Int = 28
    W_min :: Float64 = 0.1                         # grid bottom > 0: post-return W = (1+r_b)(W−a)+(1+r_a)a+w·y
                                                  #   is always > 0, so the empty-budget corner W=0 is never
                                                  #   visited; keeping it off-grid guarantees a feasible
                                                  #   consume (a'=0) at every backward cell (an all-infeasible
                                                  #   cell would carry V = −Inf).
    W_max :: Float64 = 32.0                       # portfolio-value grid top (must cover W = b + a)
    N_a  :: Int = 12
    a_max :: Float64 = 16.0
end

Base.Broadcast.broadcastable(p::TwoAssetPvParams) = Ref(p)


# Household chain assembly #
#--------------------------#

"""
Build the two-asset HANK block in the portfolio-value (Route A′) state `(W, a, income)`, existing
stages only, no auxiliary axis. `mean_liquid`, `mean_illiquid` attached as moments (liquid is the
DERIVED `W − a`).
"""
function two_asset_pv_household(p = TwoAssetPvParams())
    Wgrid = collect(range(p.W_min, p.W_max; length = p.N_W))
    agrid = collect(range(0.0, p.a_max; length = p.N_a))
    layout = GriddedLayout(
        :wealth => GriddedContinuous(Wgrid),
        :illiquid => GriddedContinuous(agrid),
        :income   => Discrete(p.y_grid),
    )

    χ(d) = p.κ * d^2

    shock = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)

    # Returns. W picks up gross receipts on BOTH stocks plus labour income; reads the OLD a, so it
    # runs time-before the illiquid-return update.
    returnsW = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; wealth, illiquid, income) -> (1 + p.r_b) * wealth +
                                     (p.r_a - p.r_b) * illiquid + p.w * income)
    returnsA = WealthChangeStage(layout; axis = :illiquid,
        wealth_post = (; illiquid) -> (1 + p.r_a) * illiquid)

    # Rebalance. Single-axis argmax on :illiquid choosing a'; reward M[after=a', before=a] = −χ(a'−a).
    # `a` here is the post-return illiquid. W is untouched. (Square reward over the illiquid grid.)
    rebalance = ArgmaxStage(layout; axis = :illiquid,
        reward = [-χ(agrid[ap] - agrid[a]) for ap in 1:p.N_a, a in 1:p.N_a])   # M[after=a', before=a]

    # Consume. Argmax on :wealth choosing W'; c = W − W', utility u(c), with the liquid
    # constraint W' ≥ a (liquid = W' − a ≥ 0) masking W' < a as −Inf. The reward depends on illiquid.
    # Discount is its own composed stage (end-goal §1): the argmax solves `max(reward + V_end)`, the
    # `TimeDiscountingStage` supplies `β·V_end` first in the backward sweep (`∘` is time-ordered).
    consume = ArgmaxStage(layout; axis = :wealth, reward = _PvConsumeReward(Wgrid, p)) ∘
              TimeDiscountingStage(layout; β = p.β)

    hh = shock ∘ returnsW ∘ returnsA ∘ rebalance ∘ consume
    return define_moments!(hh;
        mean_liquid   = at_end(integrand = (; wealth, illiquid) -> wealth - illiquid, reduce = sum),
        mean_illiquid = at_end(integrand = :illiquid, reduce = sum))
end

"""
Reward source for the Route A′ consumption choice — a `ArgmaxStage` on `:wealth`. Builds
the `(after, before)` face `U[W', W] = u(max(W − W', ε))` over the portfolio grid, feasible iff the
liquid constraint `W' ≥ a` (liquid `= W' − a ≥ 0`) AND `W' ≤ W` hold (else `−Inf`). The `W'=W`
boundary is the subsistence-`ε` fallback. The liquid mask makes `a` (illiquid) a field dep, so `U` is
full along the illiquid axis. Mirrors `_CSReward` in consumption_savings.jl, plus the liquid mask.
"""
struct _PvConsumeReward{T}
    Wgrid :: Vector{T}
    p     :: TwoAssetPvParams
end
_PvConsumeReward(Wgrid::AbstractVector, p) = _PvConsumeReward{eltype(Wgrid)}(collect(Wgrid), p)

# `a` (illiquid) is a dep: the liquid-constraint mask W' ≥ a depends on it.
HouseholdStages.declared_deps(::_PvConsumeReward, layout::GriddedLayout) =
    (:illiquid in axisnames(layout)) ? (:illiquid,) : ()

function HouseholdStages.evaluate(r::_PvConsumeReward, combo, env)
    g = r.Wgrid; n = length(g); a = combo.illiquid; ε = r.p.ε
    M = fill(-Inf, n, n)
    @inbounds for before in 1:n, after in 1:n
        # Feasible iff liquid = W'−a ≥ 0 AND W' ≤ W. Consumption c = W − W' is floored at the
        # subsistence ε, so the boundary W'=W (zero saving) is a finite fallback — the constrained
        # grid bottom keeps a feasible action (required by the upstream rebalance ArgmaxStage).
        if g[after] >= a && g[after] <= g[before]
            M[after, before] = u_crra(max(g[before] - g[after], ε), Val(r.p.σ))
        end
    end
    return M
end
