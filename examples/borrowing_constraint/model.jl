#########################################################
# Aiyagari (1994) — Natural vs Ad-hoc Borrowing Limits   #
#########################################################

# A demonstration of the two classical borrowing limits in the
# income-fluctuation problem, on an asset grid that spans negative wealth. The
# household block is the canonical three-stage spine; the borrowing limit lives
# in the asset GRID FLOOR, not in a separate stage:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# Two limits:
#
#   * AD-HOC limit `a ≥ φ`. An exogenous floor (a policy/contract choice),
#     state-INDEPENDENT, implemented as the lower end of the asset grid `a_min`.
#
#   * NATURAL limit `a ≥ −y_min / r`. The deepest debt a household can ever
#     repay: at `a = −y_min/r`, rolling the debt over leaves exactly `c = 0` in
#     the worst income state (`r·a + y_min = 0`). Borrowing past it forces `c<0`
#     in some future history, which CRRA (u(0) = −∞ for σ>1) rules out. So the
#     natural limit is enforced AUTOMATICALLY by the `c ≥ 0` feasibility already
#     inside `ConsumptionSavingsStage` — set `a_min = −y_min/r` and no extra
#     stage is needed.
#
# The example solves the SAME spine at two grid floors and compares the
# stationary wealth distributions: a near-natural floor (households borrow up to
# the natural limit) versus a tighter ad-hoc floor (the constraint binds sooner,
# fewer households in debt, a larger constrained mass).
#
# Fixed-`r` partial equilibrium: a single inner V/Λ solve per economy.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY GRID FLOORS RATHER THAN A `-Inf` MASK. With CRRA the natural limit is the
# `c → 0` boundary, so `c ≥ 0` feasibility already enforces it — and any grid
# cell strictly BELOW the natural limit is infeasible-to-sustain, an absorbing
# trap for mass. Setting the grid floor at (or inside) the limit keeps every
# cell sustainable with no masking machinery at all. A hard `-Inf` gate
# (`BorrowingConstraintStage`) is the tool for genuinely inadmissible cells —
# see `examples/mortgage_refinancing`'s LTV cap.
# ─────────────────────────────────────────────────────────────────────────────

using HouseholdStages


# Parameters #
#------------#

@kwdef struct BorrowingConstraintParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    r :: Float64       = 0.03     # FIXED return, strictly < 1/β − 1 ≈ 0.0417
    w :: Float64       = 1.0      # wage
    # Two-state endowment (Huggett's calibration); y_min = 0.1 ⇒ natural limit
    # −y_min/r ≈ −3.33.
    y_grid :: Vector{Float64} = [0.1, 1.0]
    P_y    :: Matrix{Float64} = [0.5   0.5;
                                 0.075 0.925]
    a_max :: Float64   = 24.0
    N_neg :: Int       = 120      # grid points in the borrowing region [a_min, 0)
    N_pos :: Int       = 200      # grid points in [0, a_max] (log-spaced)
end

Base.Broadcast.broadcastable(p::BorrowingConstraintParams) = Ref(p)

const borrowing_constraint_params = BorrowingConstraintParams()

"Natural borrowing limit `−y_min/r` (the deepest sustainable debt)."
natural_limit(p = borrowing_constraint_params) = -minimum(p.y_grid) / p.r


# Asset grid #
#------------#

"""
Asset grid spanning the borrowing region: a linear segment `[a_min, 0)` glued to
a log-spaced positive segment `[0, a_max]` (Huggett's grid idiom). The linear
lower segment resolves the sharp policy bend at the borrowing constraint; the log
upper segment keeps `(1+r)a + w·y` inside the grid for active cells with modest
`N_pos`. `a_min` must sit strictly inside the natural limit (`a_min > −y_min/r`)
or the lowest cells become infeasible-to-sustain traps.
"""
function borrowing_constraint_asset_grid(a_min::Real, p = borrowing_constraint_params)
    neg = collect(range(a_min, 0.0; length = p.N_neg + 1))[1:end-1]       # [a_min, 0)
    pos = GriddedContinuous(0.0, p.a_max, p.N_pos; spacing = :log).grid    # [0, a_max]
    return GriddedContinuous(vcat(neg, pos))
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached spine `IncomeShock ∘ IncomeReceipt ∘
ConsumptionSavingsStage` on an asset grid with lower floor `a_min`. The borrowing
limit IS the grid floor `a_min`: a near-natural floor exercises the natural
limit, a tighter floor an ad-hoc one. Attaches `A_mean = ∫ a dΛ` and
`frac_constrained` (mass within one grid step of the floor).
"""
function borrowing_constraint_household(a_min::Real, p = borrowing_constraint_params)
    grid   = borrowing_constraint_asset_grid(a_min, p)
    layout = GriddedLayout(:wealth => grid, :income => Discrete(p.y_grid))

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    )

    floor_a = grid.grid[1]
    step    = grid.grid[2] - grid.grid[1]
    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean           = at_end(integrand = :wealth, reduce = sum),
        frac_borrowing   = at_end(integrand = (; wealth) -> wealth < 0.0 ? 1.0 : 0.0, reduce = sum),
        frac_constrained = at_end(integrand = (; wealth) -> wealth <= floor_a + step + 1e-9 ? 1.0 : 0.0,
                                  reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"Env for the fixed-`r` borrowing-limit experiment: bond return `r`, wage `w`."
borrowing_constraint_env(p = borrowing_constraint_params) = (; r = p.r, w = p.w)
