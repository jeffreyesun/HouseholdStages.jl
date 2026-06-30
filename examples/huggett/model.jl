#####################################################
# Huggett (1993) — Pure-Exchange Bond Economy        #
#####################################################

# Huggett's incomplete-markets bond economy. The within-period household
# problem is the *same three-stage decomposition* as Aiyagari (the
# canonical L03 / L04 chain):
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# `IncomeShock` (MarkovStage) resolves the endowment Markov draw.
# `IncomeReceipt` (WealthChangeStage) is the deterministic
# `a ↦ (1+r) a + y` — bonds pay gross return `1+r`, income is the
# pure endowment `y` (no wage/labor: this is an exchange economy).
# `ConsumptionSavingsStage` chooses next-period bond holdings on the
# asset grid; the implicit budget is `c = a_in - a_end`.
#
# What makes this Huggett, not Aiyagari, is the OUTER loop (in
# steady_state.jl), not the household block:
#
#   * The one-period risk-free bond is in ZERO NET SUPPLY. There is no
#     capital, no production. The equilibrium price (the interest rate r)
#     is the one at which aggregate bond demand clears to zero:
#     ∫ a dΛ ≈ 0.
#   * A borrowing limit `a ≥ ā` (here `a_min < 0`) is the lower end of
#     the asset grid; with the natural/ad-hoc limit binding for the
#     low-endowment state, equilibrium r sits strictly below 1/β − 1.
#
# The asset grid spans negative wealth (the borrowing region) and is
# kept linear-ish near the constraint, log-spaced above zero so the
# post-receipt point `(1+r) a + y` stays inside the grid for all active
# cells (`WealthChangeStage.backward` interpolates V linearly and would
# amplify V on extrapolation past the top — same constraint as Aiyagari).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HuggettParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    # Two-state endowment process (employed / unemployed), Huggett's
    # canonical pure-exchange calibration.
    y_grid :: Vector{Float64} = [0.1, 1.0]
    P_y    :: Matrix{Float64} = [0.5 0.5;
                                 0.075 0.925]
    # Asset grid: spans the borrowing region [a_min, 0] and a positive
    # log-spaced region (0, a_max]. a_min is the ad-hoc borrowing limit ā.
    a_min :: Float64   = -2.0
    a_max :: Float64   = 24.0
    N_neg :: Int       = 100   # grid points in the borrowing region [a_min, 0)
    N_pos :: Int       = 200   # grid points in [0, a_max] (log-spaced)
end

Base.Broadcast.broadcastable(p::HuggettParams) = Ref(p)

const huggett_params = HuggettParams()


# Asset grid #
#------------#

"""
Asset grid for Huggett's bond economy: a linear borrowing region
`[a_min, 0)` glued to a log-spaced positive region `[0, a_max]`.

The linear lower segment resolves the borrowing constraint (where
policies are highly nonlinear) without crowding grid points into the
deep-negative tail; the log upper segment keeps the post-receipt point
`(1+r) a + y` inside the grid for active cells with modest `N_pos`
(the Aiyagari log-grid argument). The two segments share the `0.0` node.
"""
function huggett_asset_grid(p = huggett_params)
    neg = collect(range(p.a_min, 0.0; length = p.N_neg + 1))[1:end-1]      # [a_min, 0)
    pos = GriddedContinuous(0.0, p.a_max, p.N_pos; spacing = :log).grid  # [0, a_max]
    return GriddedContinuous(vcat(neg, pos))   # wrap as one interpolated axis
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Huggett household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage` with the
`A_supplied = ∫ asset dΛ` moment attached (aggregate bond demand, which
the outer loop drives to zero). Identical block to Aiyagari; only the
receipt law (`(1+r) a + y`, no wage) and the attached moment differ.
"""
function huggett_household(p = huggett_params)
    layout = GriddedLayout(
        :wealth => huggett_asset_grid(p),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + income,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    ) # defaults: (; utility_axes = nothing, monotone_search = :divide_conquer, assume_monotone = false)

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Prices (plain function, no AbstractBlock) #
#-------------------------------------------#

"Env for the bond economy at interest rate `r`. No production: the only
price is the bond return `r`; income is the pure endowment `cell.income`."
huggett_env(r::Real, p = huggett_params) = (; r)
