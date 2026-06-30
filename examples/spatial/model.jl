#####################################################
# Spatial Aiyagari — Two Locations, MigrationStage   #
#####################################################

# Smallest spatial extension of the Aiyagari household. The state space
# carries a third axis `:location` over `[:home, :abroad]`, and the
# within-period problem decomposes into four stages, in time order:
#
#     IncomeShock ∘ Migration ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# Wealth + income are the canonical L03/L04 decomposition (same as
# `../aiyagari/model.jl`); the migration stage is inserted *after* the
# income shock and *before* income receipt. Households see their
# income draw, then decide whether to move (logit-smoothed with a
# migration cost), and finally receive location-specific income and
# choose savings. `MigrationStage` acts on the `:location` axis with
# action set `{:home, :abroad}` and stores the destination-choice
# probability tensor. Conditional on the new location,
# `WealthChangeStage` reads the location-specific `(r, w)` from env and
# applies the deterministic wealth update.
#
# The wealth grid is log-spaced (dense near zero, coarse at the top).
# The two locations clear their capital markets independently in the
# outer damped tatonnement on `(K_home, K_abroad)`; no inter-location
# trade.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct SpatialParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36
    δ :: Float64       = 0.08
    L :: Float64       = 1.0
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 30.0
    A_home    :: Float64 = 1.0
    A_abroad  :: Float64 = 1.0
    ε_logit         :: Float64 = 5.0
    migration_cost  :: Float64 = 0.5
end

Base.Broadcast.broadcastable(p::SpatialParams) = Ref(p)

const params = SpatialParams()


# Utility: CRRA felicity `u_crra` is provided by HouseholdStages.


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached spatial household block
`IncomeShock ∘ Migration ∘ IncomeReceipt ∘ ConsumptionSavings`,
with per-location capital and population moments attached at the end.
The wealth-axis log grid and the four-stage layout are inlined here.
"""
function spatial_household(p = params)
    layout = GriddedLayout(
        :wealth   => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income   => Discrete(p.y_grid),
        :location => Discrete([:home, :abroad]),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    move    = MigrationStage(layout;   # defaults: (; axis = :location)
        migration_cost = [0.0           p.migration_cost;
                          p.migration_cost 0.0],
        ε              = p.ε_logit,
    )
    receipt = WealthChangeStage(layout;   # defaults: (; axis = :wealth)
        wealth_post = function (; location, wealth, income, env)
            r_loc = location == :home ? env.r_home : env.r_abroad
            w_loc = location == :home ? env.w_home : env.w_abroad
            return (1 + r_loc) * wealth + w_loc * income
        end,
    )
    savings = ConsumptionSavingsStage(layout;   # defaults: (; axis = :wealth, monotone_search = :divide_conquer)
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )

    hh = shock ∘ move ∘ receipt ∘ savings
    return define_moments!(hh;
        K_home     = at_end(
            integrand = (; location, wealth) -> location == :home ? wealth : 0.0,
            reduce = sum),
        K_abroad   = at_end(
            integrand = (; location, wealth) -> location == :abroad ? wealth : 0.0,
            reduce = sum),
        pop_home   = at_end(
            integrand = (; location) -> location == :home ? 1.0 : 0.0,
            reduce = sum),
        pop_abroad = at_end(
            integrand = (; location) -> location == :abroad ? 1.0 : 0.0,
            reduce = sum),
    )
end


# Production prices per location (plain function, no AbstractBlock) #
#-------------------------------------------------------------------#

"""
Cobb-Douglas factor prices at each location. Aggregate labor `p.L` is
split equally between the two locations (`L_each = L / 2`); each
location's capital market clears independently in the outer loop.
"""
function spatial_prices(K_home::Real, K_abroad::Real, p = params)
    (; α, δ, L, A_home, A_abroad) = p
    L_each   = L / 2
    r_home   = α * A_home * (K_home / L_each)^(α - 1) - δ
    w_home   = (1 - α) * A_home * (K_home / L_each)^α
    r_abroad = α * A_abroad * (K_abroad / L_each)^(α - 1) - δ
    w_abroad = (1 - α) * A_abroad * (K_abroad / L_each)^α
    return (; r_home, w_home, r_abroad, w_abroad)
end
