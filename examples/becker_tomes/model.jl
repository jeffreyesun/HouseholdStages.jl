######################################################################
# Becker–Tomes (1979, 1986) — child human-capital investment          #
######################################################################

# Altruistic parents invest in their child's human capital AND leave a financial
# bequest, generating intergenerational persistence. A dynasty is a recursive
# problem with the SAME value function across generations, so the "period" is a
# GENERATION and the model is a STATIONARY infinite-horizon problem over two
# endogenous states — financial wealth `b` and human capital `h`:
#
#   A parent enters adulthood with (b, h). It earns `w·h`, then chooses (i) the
#   child's human capital `h'`, (ii) a financial bequest `b'`, and (iii) its own
#   consumption `c`. The child becomes the next generation, entering with (b', h').
#
# Two coupled continuous choices (`h'`, `b'`) financed from one budget → the
# AUXILIARY-CHOICE-AXIS pattern (as in two_asset_hank). Household block (time order,
# `∘` runs the LEFT stage first), existing stages only, NO bespoke stage:
#
#   Earn ∘ [ ChooseChildHC ∘ DebitInvestment ∘ CommitChildHC ∘ Forget ]
#        ∘ Reproduce ∘ ConsumeBequeath
#
# `Earn`            — `IncomeStage(:wealth, income_axis=:h)`: `b ↦ (1+r)b + w·h`.
# `ChooseChildHC`   — `ArgmaxStage` picks the child's HC `h'` onto the auxiliary
#                     `:hc` axis (reward 0; the benefit — the child's dynastic
#                     value — is in the continuation, the cost is debited
#                     downstream).
# `DebitInvestment` — `WealthChangeStage(:wealth)` debits the goods cost of
#                     producing `h'` (reads `:hc`, `:h`, `:wealth`). The cost falls
#                     in PARENT `h` (`κ·h'^θ / h^ψ`): high-h parents produce child
#                     HC more cheaply — one of the two persistence channels.
# `CommitChildHC`   — `WealthChangeStage(:h)` writes `:h ← h'` (the child's HC IS
#                     the next generation's state).
# `Forget`          — `ForgetfulSumStage(:hc)` collapses the auxiliary axis.
# `Reproduce`       — `ReproductionStage(s = fertility)` scales mass for fertility
#                     (`s = 1` ⇒ one child per parent, mass conserved; `s ≠ 1` is
#                     the fertility lever — the cross-generation mass map).
# `ConsumeBequeath` — `ConsumptionSavingsStage(:wealth)` with β = degree of
#                     altruism: picks `c`, leaving the bequest `b' = wealth − c`.
#                     The continuation is the child's dynastic value, discounted by
#                     altruism — the second persistence channel.
#
# The cross-generation closure is the STATIONARY fixed point itself: the chosen
# (b', h') seeds the next generation as (b, h), and `solve_steady_state_given_env!`
# finds the dynastic value `V(b,h)` (VFI) and the stationary distribution of
# dynasties `Λ(b,h)`. No custom cohort iteration is needed — the recursion IS the
# steady state. Intergenerational persistence (children of high-h parents have
# higher h) is read off the seated policy in `steady_state.jl`.
#
# Reuse note: Becker–Tomes is the canonical child-HC dynasty block. Caucutt–Lochner
# / Lee–Seshadri / Daruich layer MORE child-HC phases (a longer ∘-chain of
# choose/debit/commit triples, one per phase) and public transfers (an extra budget
# term) onto this SAME block — see manuelli_seshadri/README.md and below.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct BeckerTomesParams
    σ        :: Float64 = 2.0                   # CRRA curvature of parental consumption
    α        :: Float64 = 0.70                  # degree of altruism (per-generation discount)
    r        :: Float64 = 0.40                  # net return on bequeathed wealth (per generation)
    w        :: Float64 = 1.0                   # wage per unit of human capital (earnings = w·h)
    fertility:: Float64 = 1.0                   # children per parent (ReproductionStage scale)
    # Child human-capital production cost: κ·h'^θ / h^ψ goods to deliver child HC h'
    κ        :: Float64 = 0.18                  # cost scale
    θ        :: Float64 = 2.1                   # convexity in child HC (>1 ⇒ diminishing returns)
    ψ        :: Float64 = 0.45                  # parental-HC efficiency (>0 ⇒ persistence channel)
    h_endow  :: Float64 = 0.4                   # child's baseline HC endowment (floor on h')
    floor_w  :: Float64 = 0.03                  # subsistence floor on post-investment wealth
    # Grids
    N_b      :: Int     = 44                    # wealth grid points
    b_max    :: Float64 = 45.0
    N_h      :: Int     = 30                    # human-capital grid points
    h_min    :: Float64 = 0.4
    h_max    :: Float64 = 9.0
end

Base.Broadcast.broadcastable(p::BeckerTomesParams) = Ref(p)

const becker_tomes_params = BeckerTomesParams()

"""
Goods cost of endowing the child with human capital `h_next` given parental human
capital `h` — the Becker–Tomes child-HC production cost `κ·h_next^θ / h^ψ`. Convex
in `h_next` (`θ>1`, diminishing returns ⇒ `h` mean-reverts, a stationary
distribution exists) and decreasing in parental `h` (`ψ>0`, the cost-side
transmission channel). Zero below the child's baseline endowment.
"""
function bt_invest_cost(h_next::Real, h::Real, p = becker_tomes_params)
    h_next <= p.h_endow && return 0.0
    return p.κ * h_next^p.θ / h^p.ψ
end


# Household chain assembly — existing stages, auxiliary-choice-axis pattern #
#--------------------------------------------------------------------------#

"""
Build the Becker–Tomes dynasty block via the auxiliary-choice-axis pattern
(existing stages only, NO bespoke stage). One generation is one period; the chosen
`(b', h')` seeds the next generation as `(b, h)`, so the block is a self-map solved
as a stationary problem. `mean_wealth`, `mean_h` moments attached.
"""
function becker_tomes_household(p = becker_tomes_params)
    hgrid = collect(range(p.h_min, p.h_max; length = p.N_h))
    axes_base = (:wealth => GriddedContinuous(0.0, p.b_max, p.N_b; spacing = :log),
                 :h      => GriddedContinuous(hgrid))
    block = GriddedLayout(axes_base..., :hc => Discrete([1]))             # singleton aux
    full  = GriddedLayout(axes_base..., :hc => Discrete(collect(1:p.N_h)))

    # Parent earns from its own human capital.
    earn = IncomeStage(block; axis = :wealth, income_axis = :h)           # b ↦ (1+r)b + w·h

    # Choose the child's HC h' onto the auxiliary axis (benefit via continuation,
    # cost via downstream debit).
    choose = ArgmaxStage(full; axis = :hc, reward = zeros(p.N_h, 1))

    # Debit the child-HC production cost from wealth.
    debit = WealthChangeStage(full; axis = :wealth,
        wealth_post = (; hc, h, wealth) ->
            max(wealth - bt_invest_cost(hgrid[Int(hc)], h, p), p.floor_w))

    # Commit the child's HC (= next generation's :h), then collapse the aux axis.
    commit = WealthChangeStage(full; axis = :h,
        wealth_post = (; hc) -> hgrid[Int(hc)])
    forget = ForgetfulSumStage(full; axis = :hc)

    # Fertility: scale the dynasty's mass (one child per parent ⇒ identity).
    reproduce = ReproductionStage(block; s = p.fertility)

    # Consume and bequeath: c chosen, bequest b' = wealth − c. β = altruism; the
    # continuation is the child's dynastic value.
    consume = ConsumptionSavingsStage(block; β = p.α, axis = :wealth,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)))

    hh = earn ∘ choose ∘ debit ∘ commit ∘ forget ∘ reproduce ∘ consume
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        mean_h      = at_end(integrand = :h,      reduce = sum))
end
