#########################################################################
# Diamond (2016) — skill-specific city choice (LogitChoiceStage)          #
#########################################################################

# "The Determinants and Welfare Implications of US Workers' Diverging
# Location Choices by Skill" (AER 2016): high- and low-skill workers choose
# a city trading off the city wage (skill-specific), the city rent, and the
# city amenity — where the amenity is ENDOGENOUS to the city's skill
# composition (high-skill inflows raise amenities, attracting more high
# skill). The within-period household problem is a pure `∘`-composition of
# existing stages, in time order:
#
#     CityChoice ∘ Amenity ∘ Receipt ∘ ConsumptionSavings
#
# Library stages (NO bespoke household stage in this file):
#   CityChoice — `LogitChoiceStage` on the :city axis with a per-move cost
#                `move_cost` (zero diagonal). The wage/rent tradeoff enters
#                the destination value, not a kwarg.
#   Amenity    — `UtilityStage`: the city amenity `A[city]` (read from `env`),
#                a state-only flow shifter. This is where the endogenous
#                amenity enters the household problem; the driver closes the
#                amenity fixed point in the OUTER loop.
#   Receipt    — `WealthChangeStage`: `(1+r)·a + wage(city, skill) −
#                rent(city)`. Skill-specific wages and city rents read from
#                `env`.
#   ConsumptionSavings — `ConsumptionSavingsStage` on the wealth grid.
#
# Skill is a FIXED type; the city choice makes the :city axis ergodic. The
# amenity fixed point (amenity rises with the high-skill share) and the
# wage/rent equilibrium are the caller's OUTER loop — `steady_state.jl`
# closes the amenity fixed point explicitly to demonstrate the sorting
# feedback, holding wages/rents exogenous.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct DiamondParams
    β :: Float64 = 0.96
    σ :: Float64 = 1.5
    r :: Float64 = 0.03
    cities :: Vector{Symbol} = [:rustbelt, :sunbelt, :hub]
    skills :: Vector{Symbol} = [:low, :high]
    # Wage by (city, skill): the hub pays a large high-skill premium.
    wage_low  :: Vector{Float64} = [0.85, 0.90, 1.00]
    wage_high :: Vector{Float64} = [1.05, 1.15, 1.60]
    rent_base :: Vector{Float64} = [0.10, 0.20, 0.35]   # exogenous city rent
    amenity_base :: Vector{Float64} = [0.0, 0.05, 0.05] # exogenous amenity floor
    amenity_spillover :: Float64 = 0.8   # amenity gain per unit high-skill share
    move_cost :: Float64 = 0.30
    ε :: Float64 = 0.40
    N_w   :: Int     = 160
    w_min :: Float64 = 0.0
    w_max :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::DiamondParams) = Ref(p)

const params = DiamondParams()


# Move-cost matrix (plain primitive, not a stage) #
#-------------------------------------------------#

"""
The `n_city × n_city` city-move cost matrix: zero diagonal (stay free),
constant `move_cost` off-diagonal. Plain data handed to `LogitChoiceStage`.
"""
move_cost_matrix(p = params) =
    [i == j ? 0.0 : p.move_cost for i in 1:length(p.cities), j in 1:length(p.cities)]


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Diamond city-choice household block
`CityChoice ∘ Amenity ∘ Receipt ∘ ConsumptionSavings` over
(wealth, skill, city). The amenity is read from `env` (`FromEnv`), so the
driver can iterate the amenity fixed point without rebuilding the chain.
Moments: high-skill population by city (the sorting diagnostic).
"""
function diamond_household(p = params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :skill  => Discrete(p.skills),
        :city   => Discrete(p.cities),
    )

    citychoice = LogitChoiceStage(layout;
        axis        = :city,
        cost_matrix = move_cost_matrix(p),
        ε           = p.ε)
    # The amenity flow is env-supplied per city (declaring `env` makes it
    # re-seated each backward, so the outer loop can vary it).
    amenity = UtilityStage(layout;
        utility = (; city, env) -> env.amenity[findfirst(==(city), p.cities)])
    receipt = WealthChangeStage(layout;                          # defaults: (; axis = :wealth)
        wealth_post = function (; city, skill, wealth, env)
            j = findfirst(==(city), p.cities)
            w = skill == :high ? env.wage_high[j] : env.wage_low[j]
            return (1 + env.r) * wealth + w - env.rent[j]
        end)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = citychoice ∘ amenity ∘ receipt ∘ savings

    hi_in(c) = (; city, skill) -> (city == c && skill == :high) ? 1.0 : 0.0
    pop_in(c) = (; city) -> city == c ? 1.0 : 0.0
    return define_moments!(hh;
        K_supplied   = at_end(integrand = :wealth, reduce = sum),
        pop_rustbelt = at_end(integrand = pop_in(:rustbelt), reduce = sum),
        pop_sunbelt  = at_end(integrand = pop_in(:sunbelt),  reduce = sum),
        pop_hub      = at_end(integrand = pop_in(:hub),      reduce = sum),
        hi_rustbelt  = at_end(integrand = hi_in(:rustbelt), reduce = sum),
        hi_sunbelt   = at_end(integrand = hi_in(:sunbelt),  reduce = sum),
        hi_hub       = at_end(integrand = hi_in(:hub),      reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"""
Assemble the chain's env at a given amenity vector `A` (length n_city). The
amenity is the endogenous object the outer loop iterates; wages and rents
are held exogenous here.
"""
diamond_env(A::AbstractVector, p = params) =
    (; r = p.r, wage_low = p.wage_low, wage_high = p.wage_high,
       rent = p.rent_base, amenity = A)
