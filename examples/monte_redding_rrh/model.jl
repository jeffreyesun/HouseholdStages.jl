##########################################################################
# Monte–Redding–Rossi-Hansberg (2018) — commuting choice (LogitChoiceStage) #
##########################################################################

# "Commuting, Migration, and Local Employment Elasticities" (AER 2018):
# a resident of location `i` chooses a WORKPLACE `j` by a gravity/logit
# rule, trading the workplace wage `w_j` against the bilateral commute
# cost `κ_{ij}`. The within-period problem is a pure `∘`-composition of
# existing stages, in time order:
#
#     Commute ∘ Amenity ∘ Receipt ∘ ConsumptionSavings
#
# Library stages (NO bespoke household stage in this file):
#   Commute  — `LogitChoiceStage` on the :workplace axis. The cost is a
#              dep-closure varying along :residence: aiming from residence
#              `i` at workplace `j` costs the commute `κ_{ij}` (the rows are
#              identical across the immaterial "origin workplace" — workplace
#              is re-chosen fresh each period). The wage tradeoff is NOT a
#              kwarg: it enters through the destination continuation value
#              `V_end[workplace]`, supplied by `Receipt ∘ ConsumptionSavings`.
#   Amenity  — `UtilityStage`: a residence-specific flow amenity `B_i`
#              (state-only V-shifter; constant across workplace, so it does
#              not bias the commute logit — it is there for fidelity to
#              `wage − rent + amenity`).
#   Receipt  — `WealthChangeStage`: cash-on-hand `(1+r)·a + w_workplace −
#              rent_residence`. The workplace wage and residence rent enter
#              the budget here (read from `env`).
#   ConsumptionSavings — `ConsumptionSavingsStage` on the wealth grid.
#
# Residence is a FIXED type (each agent lives where born — `MRRH`'s
# residential margin is a separate, slower migration choice; here it is a
# permanent type, so the distribution over :residence is the initial split,
# while :workplace is re-chosen every period). Wages, rents, and the
# residential measure clear in the spatial GE — the caller's OUTER loop, as
# always. This file solves the household block at a fixed `env` (partial
# equilibrium), which is all that is needed to exercise the commuting block.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MRRHParams
    β :: Float64 = 0.96
    σ :: Float64 = 1.5
    r :: Float64 = 0.03                 # exogenous (partial-equilibrium) return on wealth
    # Locations on a line; commute cost grows with distance |i − j|.
    locations :: Vector{Symbol}  = [:west, :center, :east]
    wage      :: Vector{Float64} = [0.9, 1.2, 1.0]   # workplace wage w_j
    rent      :: Vector{Float64} = [0.10, 0.30, 0.15] # residence rent (housing) by location
    amenity   :: Vector{Float64} = [0.05, 0.0, 0.10]  # residence flow amenity B_i
    κ :: Float64 = 0.20                 # commute cost per unit distance
    ε :: Float64 = 0.30                 # Gumbel scale of the commuting logit
    N_w   :: Int     = 200
    w_min :: Float64 = 0.0
    w_max :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::MRRHParams) = Ref(p)

const params = MRRHParams()


# Commute-cost field (plain economic primitive, not a stage) #
#------------------------------------------------------------#

"""
The `n_wp × n_wp` commute-cost matrix faced by a resident of `residence`:
`C[i, j] = κ · |residence − j|`, the cost of commuting to workplace `j`. The
"origin workplace" `i` is immaterial (workplace is re-chosen each period), so
every row is identical — only the destination `j` and the agent's residence
matter. Plain data handed to `LogitChoiceStage` via a `(; residence)` closure.
"""
function commute_cost(residence::Symbol, p = params)
    res_idx = findfirst(==(residence), p.locations)
    n = length(p.locations)
    return [p.κ * abs(res_idx - j) for _ in 1:n, j in 1:n]
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached MRRH commuting household block
`Commute ∘ Amenity ∘ Receipt ∘ ConsumptionSavings` over (wealth, residence,
workplace). Residence is a fixed type; workplace is the logit choice. Moments:
aggregate wealth, and employment + commute inflow by workplace.
"""
function mrrh_household(p = params)
    layout = GriddedLayout(
        :wealth    => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :residence => Discrete(p.locations),
        :workplace => Discrete(p.locations),
    )

    commute = LogitChoiceStage(layout;
        axis        = :workplace,
        cost_matrix = (; residence) -> commute_cost(residence, p),
        ε           = p.ε,
    )
    amenity = UtilityStage(layout;
        utility = (; residence) -> p.amenity[findfirst(==(residence), p.locations)])
    receipt = WealthChangeStage(layout;                       # defaults: (; axis = :wealth)
        wealth_post = function (; residence, workplace, wealth, env)
            i = findfirst(==(residence), p.locations)
            j = findfirst(==(workplace), p.locations)
            return (1 + env.r) * wealth + env.wage[j] - env.rent[i]
        end)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = commute ∘ amenity ∘ receipt ∘ savings

    work_at(loc) = (; workplace) -> workplace == loc ? 1.0 : 0.0
    return define_moments!(hh;
        K_supplied  = at_end(integrand = :wealth, reduce = sum),
        emp_west    = at_end(integrand = work_at(:west),   reduce = sum),
        emp_center  = at_end(integrand = work_at(:center), reduce = sum),
        emp_east    = at_end(integrand = work_at(:east),   reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"The env consumed by the chain: prices fixed (partial equilibrium)."
mrrh_env(p = params) = (; r = p.r, wage = p.wage, rent = p.rent)
