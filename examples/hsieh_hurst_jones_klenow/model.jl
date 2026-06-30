###########################################################################
# Hsieh–Hurst–Jones–Klenow (2019) — occupational choice with wedges        #
###########################################################################

# "The Allocation of Talent and U.S. Economic Growth" (Econometrica 2019):
# people of different GROUPS sort across occupations under Gumbel preference
# shocks, but face group-specific DISCRIMINATORY WEDGES `τ` that tax entry
# into some occupations. Removing the wedges reallocates talent and raises
# output — the misallocation channel. The household block is the catalog's
# exact `∘`-composition, in time order:
#
#     OccChoice ∘ Flow ∘ Discount
#
# Library stages (NO bespoke household stage in this file):
#   OccChoice — `LogitChoiceStage` on the :occupation axis. The cost is a
#               dep-closure over :group: entering occupation `j` costs the
#               group's wedge `τ[group, j]` (the discriminatory friction).
#               Group-varying via the closure is exactly the `FromEnv`-style
#               heterogeneity the catalog flags. The wage payoff is NOT a
#               kwarg — it is the destination value below.
#   Flow      — `UtilityStage`: the occupation flow payoff `wage[occupation]`
#               (occupation productivity), read from `env`.
#   Discount  — `TimeDiscountingStage`: `V_start = β·V_end`, the contraction.
#
# So `V(occ, group) = logsumexp_j[ −τ[group,j] + wage_j + β·V(j, group) ]`.
# Group is a FIXED type; the occupation choice makes the :occupation axis
# ergodic. Occupation wages clear in the talent-allocation GE — the caller's
# outer loop. This file solves the household block at a fixed `env`
# (partial equilibrium), reporting the wedge-driven occupational segregation.
#
# (Fidelity note: HHJK's comparative advantage is a Fréchet occupation-talent
# draw; here occupations differ by a group-neutral productivity `wage_j`, so
# the segregation is driven purely by the wedges. Adding an explicit talent
# axis with an occupation-specific draw would be one more `MarkovStage`/type
# axis and leave the block a pure composition.)

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HHJKParams
    β :: Float64 = 0.94
    groups      :: Vector{Symbol} = [:advantaged, :disadvantaged]
    occupations :: Vector{Symbol} = [:home, :routine, :skilled]
    wage        :: Vector{Float64} = [0.6, 1.0, 1.6]   # occupation productivity (group-neutral)
    # Discriminatory wedge τ[group, occupation]: the disadvantaged group is
    # taxed (utils) on entry into the high-wage skilled occupation.
    wedge_disadvantaged :: Vector{Float64} = [0.0, 0.0, 0.9]
    ε :: Float64 = 0.30                # Gumbel scale of the occupation logit
end

Base.Broadcast.broadcastable(p::HHJKParams) = Ref(p)

const params = HHJKParams()


# Wedge cost matrix (plain economic primitive, not a stage) #
#-----------------------------------------------------------#

"""
The `n_occ × n_occ` entry-cost matrix faced by `group`: every row equals the
group's wedge vector `τ[group, ·]` (occupation re-chosen each period, so the
origin occupation is immaterial — only the destination's wedge matters). The
advantaged group faces zero wedges; the disadvantaged group is taxed on the
skilled occupation. Plain data handed to `LogitChoiceStage` via a `(; group)`
closure.
"""
function wedge_cost(group::Symbol, p = params)
    τ = group == :disadvantaged ? p.wedge_disadvantaged : zero(p.wedge_disadvantaged)
    n = length(p.occupations)
    return [τ[j] for _ in 1:n, j in 1:n]
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached HHJK occupational-choice household block
`OccChoice ∘ Flow ∘ Discount` over (group, occupation). Group is a fixed type;
occupation is the logit choice with group-specific wedges. Moments: the
skilled-occupation share within each group (the misallocation diagnostic).
"""
function hhjk_household(p = params)
    layout = GriddedLayout(
        :group      => Discrete(p.groups),
        :occupation => Discrete(p.occupations),
    )

    occchoice = LogitChoiceStage(layout;
        axis        = :occupation,
        cost_matrix = (; group) -> wedge_cost(group, p),
        ε           = p.ε)
    flow = UtilityStage(layout;
        utility = (; occupation, env) -> env.wage[findfirst(==(occupation), p.occupations)])
    discount = TimeDiscountingStage(layout; β = p.β)

    hh = occchoice ∘ flow ∘ discount

    in_occ(g, o) = (; group, occupation) -> (group == g && occupation == o) ? 1.0 : 0.0
    return define_moments!(hh;
        skilled_adv = at_end(integrand = in_occ(:advantaged, :skilled),    reduce = sum),
        skilled_dis = at_end(integrand = in_occ(:disadvantaged, :skilled), reduce = sum),
        home_adv    = at_end(integrand = in_occ(:advantaged, :home),       reduce = sum),
        home_dis    = at_end(integrand = in_occ(:disadvantaged, :home),    reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"The env consumed by the chain: occupation wages (fixed in partial equilibrium)."
hhjk_env(p = params) = (; wage = p.wage)
