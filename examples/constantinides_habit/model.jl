####################################################################
# Constantinides (1990) internal habit formation                   #
####################################################################

# A household whose felicity depends on consumption RELATIVE to an internal
# habit stock `S` built from its OWN past consumption: utility is CRRA over the
# SURPLUS `c − γ·S` (Constantinides 1990, "Habit Formation: A Resolution of the
# Equity Premium Puzzle"). The same chosen `c = x − b'` is consumed now AND
# builds next period's stock `S' = (1−δ_S)·S + c`. This is the SAME machinery as
# `examples/habit` — the auxiliary-choice-axis pattern routing the single savings
# choice so it can set BOTH liquid wealth and the habit stock — with a ONE-LINE
# felicity change: Becker–Murphy adjacent complementarity (`√c + α·c·S`) is
# replaced by the Constantinides internal-habit surplus `u(c − γS)`.
#
# Household block (time order), all existing library stages, NO bespoke stage:
#
#   IncomeShock ∘ Receipt ∘ Choose ∘ Utility ∘ Discount ∘ HabitUpdate ∘ SetLiquid ∘ Forget
#
# `Choose`      — `ArgmaxStage` picks `b'` onto the auxiliary `:savings_choice` axis (reward 0).
# `Utility`     — `UtilityStage` adds felicity `u(c − γS)`, `c = x − b'` (reads `:wealth`, `:savings_choice`, `:habit`).
# `Discount`    — `TimeDiscountingStage(β)` (scales the continuation only; felicity stays undiscounted).
# `HabitUpdate` — `WealthChangeStage(:habit, S ↦ (1−δ_S)S + (x − b'))` (reads old `:wealth` and the choice).
# `SetLiquid`   — `WealthChangeStage(:wealth, x ↦ b')` (commit the choice to the wealth axis).
# `Forget`      — `ForgetfulSumStage(:savings_choice)` (collapse the auxiliary axis).
#
# The internal habit is a SUBSISTENCE level: the surplus `c − γS` must stay
# positive, so a high habit stock forces higher consumption — the channel that,
# in asset-pricing form, raises the price of risk. Here it is solved as an
# income-fluctuations consumption-savings problem.
#
# A FEASIBILITY NOTE (the one departure from a strict `−∞` subsistence). The
# auxiliary-choice-axis pattern grows the savings axis through a (brute)
# `ArgmaxStage` (`Choose`) — an origin cell with NO finite-reward action carries
# value `−∞`. The minimum-wealth cell structurally has `c = 0` (the
# cash floor and the savings floor are the same grid point), so a strict
# `surplus > 0 ? u : −∞` felicity makes that whole column `−∞`, and the `−∞`
# poisons the value function at the grid bottom. (`examples/habit` is immune
# only because Becker–Murphy `√c + αcS`
# is FINITE at `c = 0`.) The standard subsistence treatment resolves it: floor
# the surplus at a small `ε > 0` (`c_floor`), so felicity stays finite and the
# `−∞` region becomes a steep-but-finite penalty the agent avoids in
# equilibrium. With `c_floor` tiny the floored region is unreachable (the
# stationary mass keeps the surplus well above it — verified in the driver), so
# this is numerically the Constantinides subsistence model. This is exactly the
# "consumption floor" regularization used throughout the quantitative
# subsistence-utility literature.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct ConstantinidesHabitParams
    β    :: Float64 = 0.94
    σ    :: Float64 = 2.0                       # CRRA curvature over the surplus c − γS
    γ    :: Float64 = 0.2                       # habit weight (subsistence share of the stock)
    δ_S  :: Float64 = 0.6                       # habit-stock depreciation (faster decay ⇒ lower stock)
    r    :: Float64 = 0.03
    w    :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.85, 1.15]    # mild income risk: a deep income crash into the
    P_y    :: Matrix{Float64} = [0.85 0.15; 0.15 0.85]   # subsistence region is what the floor guards
    N_w   :: Int = 24
    w_max :: Float64 = 14.0
    N_S   :: Int = 24
    S_max :: Float64 = 40.0    # ergodic habit range is ≈ c/δ_S; with c up to w_max and δ_S = 0.5 the
                               # stock reaches ~28, so the grid top must clear it (as in `examples/habit`).
    c_floor :: Float64 = 1.0e-3   # surplus floor: felicity = u_crra(max(c − γS, c_floor)). Keeps the
                                  # brute `Choose` finite where strict −∞ subsistence cannot (see header).
end

Base.Broadcast.broadcastable(p::ConstantinidesHabitParams) = Ref(p)


# Felicity — Constantinides internal habit: CRRA over the consumption SURPLUS c − γS #
#------------------------------------------------------------------------------------#

"""
Constantinides (1990) internal-habit felicity: CRRA over the consumption surplus `c − γ·S`. The
internal habit `S` acts as a subsistence reference — the surplus must stay positive, so a higher
habit stock forces higher consumption. The surplus is floored at a small `c_floor > 0` (strict `−∞`
below zero would trip the brute `Choose`'s finite-action assertion at the minimum-wealth cell — see
the model header); with `c_floor` tiny the floored region is unreachable in equilibrium, so this is
the Constantinides subsistence model numerically. (Contrast `examples/habit`, whose Becker–Murphy
`√c + α·c·S` makes the stock a complement rather than a subsistence reference, and is finite at `c=0`.)
"""
u_constantinides(c, S, p) = u_crra(max(c - p.γ * S, p.c_floor), Val(p.σ))


# Household chain assembly #
#--------------------------#

"""
Build the Constantinides internal-habit household block via the auxiliary-choice-axis pattern (8
existing stages, no bespoke household stage). Identical wiring to `examples/habit`; only the felicity
(`u_constantinides`) differs. `mean_wealth` and `mean_habit` are attached as moments.
"""
function constantinides_habit_household(p = ConstantinidesHabitParams())
    wgrid = collect(range(0.0, p.w_max; length = p.N_w))
    axes_base = (:wealth => GriddedContinuous(wgrid),
                 :habit  => GriddedContinuous(range(0.0, p.S_max; length = p.N_S)),
                 :income => Discrete(p.y_grid))
    # Pre-choice stages see `:savings_choice` as a SINGLETON; the choice-block stages see it FULL
    # (the candidate-b' grid). `Choose` grows 1→N_w, `Forget` collapses N_w→1.
    block = GriddedLayout(axes_base..., :savings_choice => Discrete([1]))
    full  = GriddedLayout(axes_base..., :savings_choice => Discrete(collect(1:p.N_w)))

    shock   = MarkovStage(block; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(block;                                    # cash-on-hand x = (1+r)a + w·y
        wealth_post = (; wealth, income) -> (1 + p.r) * wealth + p.w * income)   # defaults: (; axis = :wealth)
    choose  = ArgmaxStage(block, full; axis = :savings_choice, reward = zeros(p.N_w, 1))
    felicity = UtilityStage(full; utility = (; wealth, savings_choice, habit) ->     # u(c − γS), c = x − b'
        u_constantinides(wealth - wgrid[Int(savings_choice)], habit, p))
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
