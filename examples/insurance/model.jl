###############################################################
# Insurance / annuitization — convex-cost loss insurance       #
###############################################################

# An incomplete-markets savings household that can pay a CONVEX PREMIUM to
# insure against a loss shock on its cash-on-hand. The point of this example
# (Part 3: literature household blocks expressible from the library alone):
# the entire within-period problem is FOUR existing library stages, in time
# order, with **no bespoke household stage rolled here** —
#
#     IncomeShock ∘ Insurance ∘ Receipt ∘ ConsumptionSavings
#
# `IncomeShock`  — `MarkovStage` on the income axis.
# `Insurance`    — `MixingStage` on the wealth axis: the household chooses a
#                  blend `θ ∈ [0,1]` of a NO-LOSS kernel `K_A = I` (asset stock
#                  retained) and a LOSS kernel `K_B` (asset stock shrinks by
#                  `loss_factor`), at convex cost `c(θ) = θ²/(2κ)`. The loss
#                  hits BEGINNING-OF-PERIOD wealth (before income is received),
#                  the canonical casualty/health-loss timing. `θ` is the
#                  insurance coverage; `θ = 1` fully insures (stay on the
#                  no-loss kernel), `θ = 0` is uninsured exposure. The closed
#                  form is `V = K_B·V + c*(K_A·V − K_B·V)` (Fenchel conjugate
#                  of the premium) — see `src/stages/derived/lottery_mixing.jl`.
# `Receipt`      — `WealthChangeStage` `a ↦ (1+r)·a + w·y` (cash-on-hand x):
#                  return on the post-loss asset stock plus labour income.
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period wealth
#                  `b'` on the wealth grid; `c = x − b'`, CRRA utility.
#
# This is the actuarially-fair-with-loading insurance / annuitization block:
# the convex premium `c(θ)` is the loading, so coverage is interior. It maps
# the Yaari (1965) annuity / standard insurance-demand problem (and the
# "pay convex cost to mix toward a safe kernel" reading of Grossman-type
# mortality retention) onto existing stages. `RetentionStage` IS this block's
# `MixingStage` with `K_A = I` named for the "pay not to transition" framing.
#
# Returns/wage are exogenous (partial equilibrium): no market to clear, so the
# "outer loop" is a single `solve_steady_state_given_env!`. The grid floor
# (`b' ≥ 0`) plus impatience (`β·(1+r) < 1`) deliver a stationary distribution.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct InsuranceParams
    β :: Float64       = 0.95
    σ :: Float64       = 2.0                       # CRRA
    r :: Float64       = 0.03                      # exogenous net return on wealth
    w :: Float64       = 1.0                       # wage (income scale)
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.7 0.25 0.05;
                                 0.2 0.60 0.20;
                                 0.05 0.25 0.70]
    loss_factor    :: Float64 = 0.70               # uninsured cash-on-hand multiplier under the loss
    cost_curvature :: Float64 = 4.0                # κ in the premium c(θ)=θ²/(2κ); larger ⇒ cheaper coverage
    N_w   :: Int       = 200
    w_min :: Float64   = 0.0
    w_max :: Float64   = 60.0
end

Base.Broadcast.broadcastable(p::InsuranceParams) = Ref(p)

const insurance_params = InsuranceParams()


# Loss kernel (plain economic helper — a row-stochastic wealth transition) #
#--------------------------------------------------------------------------#

"""
Build the row-stochastic LOSS kernel `K_B[from, to]` on a wealth `grid`: a cell at wealth
`grid[i]` moves to `loss_factor·grid[i]`, split (linear-interpolation weights) onto the two
bracketing grid points so the kernel stays on-grid and mass-conserving. This is plain economic
data — the discrete analogue of the deterministic map `x ↦ loss_factor·x` — not a household
stage; it is handed to `MixingStage` as its `K_B` corner.
"""
function loss_kernel(grid::AbstractVector, loss_factor::Real)
    n = length(grid)
    K = zeros(Float64, n, n)
    for i in 1:n
        target = loss_factor * grid[i]
        j = searchsortedlast(grid, target)         # grid[j] ≤ target < grid[j+1]
        if j <= 0
            K[i, 1] = 1.0                           # below the floor → pile on the lowest point
        elseif j >= n
            K[i, n] = 1.0                           # above the top → pile on the highest point
        else
            wgt = (target - grid[j]) / (grid[j + 1] - grid[j])
            K[i, j]     = 1 - wgt
            K[i, j + 1] = wgt
        end
    end
    return K
end


# Household chain assembly #
#--------------------------#

"""
Build the insurance household block `IncomeShock ∘ Insurance ∘ Receipt ∘ ConsumptionSavings`, with
`mean_wealth = ∫ wealth dΛ` attached. The insurance stage sits on beginning-of-period asset wealth,
before receipt: the loss hits the asset stock, and the post-receipt gather lifts the savings-floor
`-Inf` away from the mixing closed form.
"""
function insurance_household(p = insurance_params)
    wealth_axis = GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log)
    w_grid      = axisvalues(wealth_axis)
    layout = GriddedLayout(
        :wealth => wealth_axis,
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout) # defaults: (; axis = :wealth)
    insurance = RetentionStage(layout; axis = :wealth,
        exit_kernel    = loss_kernel(w_grid, p.loss_factor),            # K_B: the loss; K_A = I (stay)
        cost_curvature = p.cost_curvature)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    ) # defaults: (; axis = :wealth, utility_axes = nothing, skip_monotonicity_check = false)

    hh = shock ∘ insurance ∘ receipt ∘ savings
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"The exogenous price env for the insurance household: net return `r` and wage `w`."
insurance_env(p = insurance_params) = (; r = p.r, w = p.w)
