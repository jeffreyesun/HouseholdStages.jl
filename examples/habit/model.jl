####################################################################
# Habit formation / rational addiction (Becker–Murphy 1988)        #
####################################################################

# A household whose felicity depends on consumption RELATIVE to an addiction stock S, where the SAME
# chosen consumption `c = x − b'` both (a) is consumed now and (b) builds next period's stock
# `S' = (1−δ_S)·S + c`. A single choice that must set TWO axes (liquid wealth AND the habit stock)
# — exactly the coupling that naively looks un-expressible. It IS expressible from existing stages via
# the **auxiliary-choice-axis pattern**: route the savings choice through a separate `:savings_choice`
# axis so a downstream stage can read BOTH the pre-choice wealth `x` and the chosen `b'` and form `c`.
#
# Household block (time order), all existing library stages, NO bespoke stage:
#
#   IncomeShock ∘ Receipt ∘ Choose ∘ Utility ∘ Discount ∘ HabitUpdate ∘ SetLiquid ∘ Forget
#
# `Choose`      — `ArgmaxStage` picks `b'` onto the auxiliary `:savings_choice` axis (reward 0).
# `Utility`     — `UtilityStage` adds felicity `u(c, S)`, `c = x − b'` (reads `:wealth`, `:savings_choice`, `:habit`).
# `Discount`    — `TimeDiscountingStage(β)` (scales the continuation only; felicity stays undiscounted).
# `HabitUpdate` — `WealthChangeStage(:habit, S ↦ (1−δ_S)S + (x − b'))` (reads old `:wealth` and the choice).
# `SetLiquid`   — `WealthChangeStage(:wealth, x ↦ b')` (commit the choice to the wealth axis).
# `Forget`      — `ForgetfulSumStage(:savings_choice)` (collapse the auxiliary axis).
#
# The pattern is structurally the kernel-choice sandwich (`Collapse ∘ transforms ∘ ForgetfulSum`); the
# verified equivalence to the brute Bellman `V(x,S)=max_{b'} u(x−b',S)+βV(b',(1−δ_S)S+x−b')` is pinned
# in test/test_example_habit.jl. Cost: the `:savings_choice` grid makes the intermediate tensor
# `n_w×` larger (gridded). NB the savings choice can be non-monotone, so this uses the brute `ArgmaxStage`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HabitParams
    β    :: Float64 = 0.94
    α    :: Float64 = 0.06                      # adjacent-complementarity weight (∂²u/∂c∂S = α > 0)
    δ_S  :: Float64 = 0.5                       # habit-stock depreciation
    r    :: Float64 = 0.03
    w    :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.7, 1.3]
    P_y    :: Matrix{Float64} = [0.8 0.2; 0.2 0.8]
    N_w   :: Int = 24
    w_max :: Float64 = 14.0
    N_S   :: Int = 24
    S_max :: Float64 = 40.0    # ergodic habit range is c/δ_S; with c up to w_max and δ_S∈[0.4,0.5]
                               # the stock reaches ~30, so the grid top must clear it (was 10.0 — too
                               # short, the savings policy S'=(1−δ_S)S+c landed off-grid-right and the
                               # off-grid clamp then diverged from the brute's extrapolation).
end

Base.Broadcast.broadcastable(p::HabitParams) = Ref(p)


# Felicity — adjacent complementarity: the marginal utility of c rises with the habit stock S #
#---------------------------------------------------------------------------------------------#

"Concave felicity with adjacent complementarity `α·c·S` (Becker–Murphy): higher addiction stock S raises the marginal utility of consumption. Finite at `c=0`, so `b'=0` is always feasible."
u_habit(c, S, p) = c >= 0 ? sqrt(c) + p.α * c * S : -Inf


# Household chain assembly #
#--------------------------#

"""
Build the rational-addiction household block via the auxiliary-choice-axis pattern (8 existing stages,
no bespoke household stage). `mean_wealth` and `mean_habit` are attached as moments.
"""
function habit_household(p = HabitParams())
    wgrid = collect(range(0.0, p.w_max; length = p.N_w))
    axes_base = (:wealth => GriddedContinuous(wgrid),
                 :habit  => GriddedContinuous(range(0.0, p.S_max; length = p.N_S)),
                 :income => Discrete(p.y_grid))
    # Pre-choice stages see `:savings_choice` as a SINGLETON (block boundary); the choice-block stages
    # see it FULL (the candidate-b' grid). `Choose` grows 1→N_w, `Forget` collapses N_w→1.
    block = GriddedLayout(axes_base..., :savings_choice => Discrete([1]))
    full  = GriddedLayout(axes_base..., :savings_choice => Discrete(collect(1:p.N_w)))

    shock   = MarkovStage(block; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(block;                                    # cash-on-hand x = (1+r)a + w·y
        wealth_post = (; wealth, income) -> (1 + p.r) * wealth + p.w * income)   # defaults: (; axis = :wealth)
    choose  = ArgmaxStage(block, full; axis = :savings_choice, reward = zeros(p.N_w, 1))
    felicity = UtilityStage(full; utility = (; wealth, savings_choice, habit) ->     # u(c, S), c = x − b'
        u_habit(wealth - wgrid[Int(savings_choice)], habit, p))
    discount = TimeDiscountingStage(full; β = p.β)
    habitup  = WealthChangeStage(full; axis = :habit,                     # S' = (1−δ_S)S + (x − b')
        wealth_post = (; habit, wealth, savings_choice) -> (1 - p.δ_S) * habit + (wealth - wgrid[Int(savings_choice)]))
    setliq   = WealthChangeStage(full;                                    # commit wealth ← b'  (defaults: (; axis = :wealth))
        wealth_post = (; savings_choice) -> wgrid[Int(savings_choice)])
    forget   = ForgetfulSumStage(full; axis = :savings_choice)

    hh = shock ∘ receipt ∘ choose ∘ felicity ∘ discount ∘ habitup ∘ setliq ∘ forget
    return define_moments!(hh;
        mean_wealth = at_end(integrand = :wealth, reduce = sum),
        mean_habit  = at_end(integrand = :habit,  reduce = sum))
end
