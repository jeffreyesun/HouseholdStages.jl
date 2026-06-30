#########################################################################
# Bryan–Morten (2019) — internal migration with selection (MigrationStage) #
#########################################################################

# "The Aggregate Productivity Effects of Internal Migration" (JPE 2019):
# workers choose a location under a fixed + variable moving cost; because
# location wages multiply individual ability, the gain from moving to a
# high-wage location rises with ability — so migration SELECTS on ability.
# This is the catalog's exact block — a discrete location choice whose
# destination value is a `UtilityStage` reading `wage × ability` — assembled
# as a pure `∘`-composition, in time order:
#
#     Migrate ∘ Flow ∘ Discount
#
# Library stages (NO bespoke household stage in this file):
#   Migrate  — `MigrationStage` (sugar over `LogitChoiceStage`) on the
#              :location axis, with `M[i, j] = fixed + variable·|i − j|` for
#              any move (zero diagonal). The ability×wage payoff that drives
#              selection is NOT a kwarg — it is the destination value below.
#   Flow     — `UtilityStage`: the destination flow payoff `wage_j · ability +
#              amenity_j`, read from `env`. The wage×ability complementarity is
#              the selection channel; the decreasing location amenity makes
#              low types prefer the rural amenity (selection shows in stock).
#   Discount — `TimeDiscountingStage`: `V_start = β·V_end`, supplying the
#              contraction for the stationary value recursion.
#
# So `V(i, ability) = logsumexp_j[ −M[i,j] + wage_j·ability + β·V(j, ability) ]`,
# the canonical dynamic-discrete-choice migration value with Gumbel scale ε.
# Ability is a FIXED type; migration makes the :location axis ergodic. Wages
# and the spatial measure clear in the regional GE — the caller's outer loop.
# This file solves the household block at a fixed `env` (partial equilibrium).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct BryanMortenParams
    β :: Float64 = 0.94
    locations :: Vector{Symbol}  = [:rural, :town, :city]
    wage      :: Vector{Float64} = [0.90, 1.00, 1.20]  # location wage per ability unit
    # Location flow amenity (a non-multiplicative "home" premium), DECREASING
    # toward the city: low-ability types — whose wage gain from the city is
    # small — value the rural amenity more and stay; high-ability types
    # overcome it. This is what makes the SELECTION show in the stationary
    # stock rather than everyone collapsing into the highest-wage location.
    amenity   :: Vector{Float64} = [0.30, 0.15, 0.0]
    ability   :: Vector{Float64} = [0.6, 1.0, 1.6]     # permanent ability type (no shock — fixed)
    fixed_cost    :: Float64 = 0.30   # flow-utility cost paid for ANY move
    variable_cost :: Float64 = 0.15   # extra cost per unit distance
    ε :: Float64 = 0.10               # Gumbel scale of the migration logit
end

Base.Broadcast.broadcastable(p::BryanMortenParams) = Ref(p)

const params = BryanMortenParams()


# Migration-cost matrix (plain economic primitive, not a stage) #
#---------------------------------------------------------------#

"""
The `n_loc × n_loc` moving-cost matrix `M[i, j]`: zero on the diagonal (free
to stay), `fixed_cost + variable_cost·|i − j|` off it — a fixed component plus
a distance-linear variable component (Bryan–Morten's cost structure). Plain
data handed to `MigrationStage`; not household-stage logic.
"""
migration_cost_matrix(p = params) =
    [i == j ? 0.0 : p.fixed_cost + p.variable_cost * abs(i - j)
     for i in 1:length(p.locations), j in 1:length(p.locations)]


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Bryan–Morten household block
`Migrate ∘ Flow ∘ Discount` over (ability, location). Ability is a fixed
type; location is the migration choice. Moments: population by location, and
the city-population share within each ability type (the SELECTION diagnostic).
"""
function bryan_morten_household(p = params)
    layout = GriddedLayout(
        :ability  => Discrete(p.ability),
        :location => Discrete(p.locations),
    )

    migrate = MigrationStage(layout;                              # defaults: (; axis = :location)
        migration_cost = migration_cost_matrix(p),
        ε              = p.ε)
    flow = UtilityStage(layout;
        utility = (; location, ability, env) -> begin
            j = findfirst(==(location), p.locations)
            return env.wage[j] * ability + p.amenity[j]
        end)
    discount = TimeDiscountingStage(layout; β = p.β)

    hh = migrate ∘ flow ∘ discount

    at_loc(loc) = (; location) -> location == loc ? 1.0 : 0.0
    # City population restricted to a given ability type (selection diagnostic).
    city_of(a)  = (; location, ability) -> (location == :city && ability == a) ? 1.0 : 0.0
    return define_moments!(hh;
        pop_rural  = at_end(integrand = at_loc(:rural), reduce = sum),
        pop_town   = at_end(integrand = at_loc(:town),  reduce = sum),
        pop_city   = at_end(integrand = at_loc(:city),  reduce = sum),
        city_lo    = at_end(integrand = city_of(p.ability[1]), reduce = sum),
        city_mid   = at_end(integrand = city_of(p.ability[2]), reduce = sum),
        city_hi    = at_end(integrand = city_of(p.ability[3]), reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"The env consumed by the chain: location wages (fixed in partial equilibrium)."
bryan_morten_env(p = params) = (; wage = p.wage)
