####################################################################
# Soft default — a convex-cost probability-of-default hazard         #
####################################################################

# An incomplete-markets borrower that each period can SOFTLY DEFAULT: it scales
# the PROBABILITY `θ ∈ [0,1]` of wiping its balance sheet to a clean slate, at a
# convex cost `c(θ) = θ²/(2κ)`. The whole point of this Part-3 example: the
# within-period problem is FOUR existing library stages, in time order, with
# **no bespoke household stage rolled here** —
#
#     IncomeShock ∘ DefaultChoice ∘ Receipt ∘ ConsumptionSavings
#
# `IncomeShock`  — `MarkovStage` on the income axis (endowment process).
# `DefaultChoice`— `MixingStage` on the `:wealth` axis: a blend `K_θ = θ·K_A +
#                  (1−θ)·K_B` of a RESET kernel `K_A` (every wealth cell → the
#                  zero-wealth grid point: debt discharged, assets wiped — a clean
#                  slate) and the IDENTITY `K_B = I` (keep the balance sheet,
#                  repay), at convex cost `c(θ)=θ²/(2κ)`. The weight on the reset
#                  corner `θ` IS the default probability. The closed form
#                  `V = (keep value) + c*(reset value − keep value)` makes `θ*(w)
#                  = clamp(κ·(V(0) − V(w)), 0, 1)` rise exactly where the clean
#                  slate beats keeping the balance sheet — i.e. for DEEPLY
#                  INDEBTED agents (`w < 0`), so default concentrates in the debt
#                  region, as it should.
# `Receipt`      — `WealthChangeStage` `w ↦ (1+r)·w + income`: return/interest on
#                  the (post-default) balance sheet plus income. Can push agents
#                  into the debt region (`w < 0`).
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period wealth on the
#                  grid (the floor `w_min < 0` is the borrowing limit); `c = x − w'`,
#                  CRRA utility.
#
# Why MixingStage, NOT LogitEndogenousExit. There are two library readings of a
# soft default. (a) `LogitEndogenousExit`: default = a soft EXIT, the defaulting
# mass LEAVES the population (and a bequest is paid). (b) `MixingStage` with a
# RESET kernel: default = a clean-slate RESET, the defaulting mass STAYS in the
# population at zero wealth. Reading (b) is the right one for consumer/sovereign
# default where the agent persists after discharging its debt, and it is MASS-
# CONSERVING by construction (the reset kernel is row-stochastic, so total mass on
# the wealth grid is preserved) — whereas (a) leaks a default-rate share of mass
# out each period, which would need a re-entry/birth source to stay stationary.
# This example takes the mass-conserving reset reading. (The HARD discrete
# repay-vs-default stopping problem — Eaton–Gersovitz / Arellano — is §5's
# `DefaultStage`, a gated argmax; see `examples/default`. This is the SOFT,
# smoothly-scaled-hazard reading of the same margin.)
#
# Prices are exogenous (partial equilibrium): no market to clear, so the "outer
# loop" is a single `solve_steady_state_given_env!`. Impatience `β·(1+r) < 1` plus
# the borrowing floor and the clean-slate reset deliver a stationary distribution.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct SoftDefaultParams
    β :: Float64       = 0.98                      # patience (β·(1+r) ≈ 0.9996 < 1 ⇒ stationary, but high enough for a saving motive)
    σ :: Float64       = 2.0                       # CRRA
    r :: Float64       = 0.02                      # exogenous net return / interest on the balance sheet
    y_grid :: Vector{Float64} = [0.7, 1.0, 1.3]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    cost_curvature :: Float64 = 0.2                # κ in the convex default cost c(θ)=θ²/(2κ); larger ⇒ cheaper default
    N_w   :: Int       = 120
    w_min :: Float64   = -1.0                      # borrowing limit (debt is w < 0)
    w_max :: Float64   = 8.0
end

Base.Broadcast.broadcastable(p::SoftDefaultParams) = Ref(p)

const soft_default_params = SoftDefaultParams()


# Reset kernel (plain economic data — a row-stochastic wealth transition) #
#-------------------------------------------------------------------------#

"""
Build the row-stochastic RESET kernel `K_A[from, to]` on a wealth `grid`: EVERY
cell moves to the fixed `target` wealth (the clean-slate value, here 0), split by
linear-interpolation weights onto the two bracketing grid points so the kernel
stays on-grid and mass-conserving. This is plain economic data — the discrete
analogue of the deterministic map `w ↦ target` — not a household stage; it is
handed to `MixingStage` as its `K_A` (default) corner. Same construction as
`examples/insurance`'s `loss_kernel`, but the destination is a FIXED value rather
than `loss_factor·w`.
"""
function reset_kernel(grid::AbstractVector, target::Real = 0.0)
    n = length(grid)
    K = zeros(Float64, n, n)
    j = searchsortedlast(grid, target)             # grid[j] ≤ target < grid[j+1]
    if j <= 0
        col_lo, col_hi, wgt = 1, 1, 0.0            # target below the floor → pile on the lowest point
    elseif j >= n
        col_lo, col_hi, wgt = n, n, 0.0            # target above the top → pile on the highest point
    else
        col_lo, col_hi = j, j + 1
        wgt = (target - grid[j]) / (grid[j + 1] - grid[j])
    end
    for i in 1:n
        K[i, col_lo] += 1 - wgt
        K[i, col_hi] += wgt
    end
    return K
end

"""
The identity kernel `I[from, to]` on an `n`-state axis — the KEEP / repay corner
`K_B` of the default mix (the balance sheet is carried forward unchanged). Plain
data fed to `MixingStage`.
"""
identity_kernel(n::Integer) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]


# Household chain assembly — FOUR library stages, NO bespoke stage #
#-----------------------------------------------------------------#

"""
Build the soft-default household block `IncomeShock ∘ DefaultChoice ∘ Receipt ∘
ConsumptionSavings` with `mean_wealth`, `debt_share` (mass with `w < 0`), and
`mean_debt` moments attached. Four existing stages, no bespoke household stage:
the default choice is a `MixingStage(K_A = reset, K_B = I)` blending the clean-slate
reset and keep-the-balance-sheet kernels at convex cost, so the weight `θ` on the
reset corner is the default probability. Mass-conserving by construction.
"""
function soft_default_household(p = soft_default_params)
    wealth_axis = GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :linear)
    w_grid      = axisvalues(wealth_axis)
    layout = GriddedLayout(
        :wealth => wealth_axis,
        :income => Discrete(p.y_grid),
    )

    shock = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    default = MixingStage(layout; axis = :wealth,
        K_A            = reset_kernel(w_grid, 0.0),    # default: clean slate (→ wealth 0)
        K_B            = identity_kernel(p.N_w),       # keep: carry the balance sheet (repay)
        cost_curvature = p.cost_curvature)
    receipt = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + income)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)))

    hh = shock ∘ default ∘ receipt ∘ savings
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        debt_share  = at_end(integrand = (; wealth) -> wealth < 0 ? 1.0 : 0.0,    reduce = sum),
        mean_debt   = at_end(integrand = (; wealth) -> wealth < 0 ? wealth : 0.0, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"The exogenous price env for the soft-default household: the net return / interest rate `r`."
soft_default_env(p = soft_default_params) = (; r = p.r)
