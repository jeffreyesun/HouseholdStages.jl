###########################################################################
# Bayer–Ferreira–McMillan (2007) — neighborhood sorting (LogitChoiceStage)  #
###########################################################################

# "A Unified Framework for Measuring Preferences for Schools and
# Neighborhoods" (JPE 2007): households of different income types choose a
# neighborhood via a (mixed) logit, valuing amenities, neighborhood
# composition, and price — and PRICES CLEAR each neighborhood's fixed
# housing supply. The within-period household block is the catalog's exact
# `∘`-composition, in time order:
#
#     NbhdChoice ∘ Flow ∘ Discount
#
# Library stages (NO bespoke household stage in this file):
#   NbhdChoice — `LogitChoiceStage` on the :neighborhood axis with a small
#                per-move cost (zero diagonal). The amenity/composition/price
#                tradeoff is NOT a kwarg — it is the destination value below.
#   Flow       — `UtilityStage`: `amenity[n] + λ·rich_share[n] − price[n]/y`,
#                with price disutility scaled by `1/income` so richer types
#                are LESS price-sensitive (the source of income sorting). The
#                price vector and rich-share composition are read from `env`.
#   Discount   — `TimeDiscountingStage`: the contraction.
#
# Income type is a FIXED type; the neighborhood choice makes the axis ergodic.
# The clearing price vector (and the endogenous composition) are the caller's
# OUTER loop — `steady_state.jl` runs a tatonnement that raises a
# neighborhood's price until its population equals its housing capacity, and
# updates the rich-share composition each pass. This is the BFM sorting
# equilibrium; the household block never changes, only the driver closes it.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct BFMParams
    β :: Float64 = 0.92
    income_types :: Vector{Symbol}  = [:poor, :rich]
    income       :: Vector{Float64} = [0.7, 1.6]      # marginal utility of money ∝ 1/income
    neighborhoods :: Vector{Symbol}  = [:south, :midtown, :heights]
    amenity       :: Vector{Float64} = [0.0, 0.3, 0.6]  # exogenous neighborhood amenity
    capacity      :: Vector{Float64} = [1/3, 1/3, 1/3]  # housing supply share per neighborhood
    λ_composition :: Float64 = 0.4    # weight on neighborhood rich-share (the peer externality)
    move_cost :: Float64 = 0.15
    ε :: Float64 = 0.20
end

Base.Broadcast.broadcastable(p::BFMParams) = Ref(p)

const params = BFMParams()


# Move-cost matrix (plain primitive, not a stage) #
#-------------------------------------------------#

"""
The `n_nbhd × n_nbhd` neighborhood-move cost matrix: zero diagonal, constant
`move_cost` off it. Plain data handed to `LogitChoiceStage`.
"""
move_cost_matrix(p = params) =
    [i == j ? 0.0 : p.move_cost for i in 1:length(p.neighborhoods), j in 1:length(p.neighborhoods)]


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached BFM neighborhood-sorting household block
`NbhdChoice ∘ Flow ∘ Discount` over (income_type, neighborhood). Prices and
the rich-share composition are read from `env`, so the driver iterates the
sorting equilibrium without rebuilding the chain. Moments: population by
neighborhood, and the rich population by neighborhood (the sorting diagnostic).
"""
function bfm_household(p = params)
    layout = GriddedLayout(
        :income_type  => Discrete(p.income_types),
        :neighborhood => Discrete(p.neighborhoods),
    )

    nbhd_choice = LogitChoiceStage(layout;
        axis        = :neighborhood,
        cost_matrix = move_cost_matrix(p),
        ε           = p.ε)
    flow = UtilityStage(layout;
        utility = (; income_type, neighborhood, env) -> begin
            n = findfirst(==(neighborhood), p.neighborhoods)
            y = p.income[findfirst(==(income_type), p.income_types)]
            return p.amenity[n] + p.λ_composition * env.rich_share[n] - env.price[n] / y
        end)
    discount = TimeDiscountingStage(layout; β = p.β)

    hh = nbhd_choice ∘ flow ∘ discount

    in_nbhd(n)      = (; neighborhood) -> neighborhood == n ? 1.0 : 0.0
    rich_in(n)      = (; neighborhood, income_type) -> (neighborhood == n && income_type == :rich) ? 1.0 : 0.0
    return define_moments!(hh;
        pop_south     = at_end(integrand = in_nbhd(:south),   reduce = sum),
        pop_midtown   = at_end(integrand = in_nbhd(:midtown), reduce = sum),
        pop_heights   = at_end(integrand = in_nbhd(:heights), reduce = sum),
        rich_south    = at_end(integrand = rich_in(:south),   reduce = sum),
        rich_midtown  = at_end(integrand = rich_in(:midtown), reduce = sum),
        rich_heights  = at_end(integrand = rich_in(:heights), reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"""
Assemble the chain's env at a price vector `price` and rich-share composition
`rich_share` (both length n_nbhd) — the two endogenous objects the sorting
outer loop iterates.
"""
bfm_env(price::AbstractVector, rich_share::AbstractVector, p = params) =
    (; price, rich_share)
