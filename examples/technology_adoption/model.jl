###########################################################################
# Technology / network adoption — discrete adopt choice (LogitChoiceStage)  #
###########################################################################

# A household decides whether to ADOPT a technology whose payoff rises with
# the economy-wide ADOPTION SHARE (a network externality): the more others
# adopt, the more valuable adoption is. The discrete adopt choice is a clean
# stage; the externality — payoff depending on the cross-sectional adoption
# share — is a distribution-dependent equilibrium the caller closes in the
# OUTER loop, exactly like Krusell–Smith's aggregate state. The within-period
# household block is a pure `∘`-composition, in time order:
#
#     Flow ∘ Discount ∘ AdoptChoice
#
# Library stages (NO bespoke household stage in this file):
#   Flow        — `UtilityStage`: flow payoff by technology state — `0` if not
#                 adopted, `θ · adoption_share` if adopted (the network
#                 benefit), with the share read from `env`.
#   Discount    — `TimeDiscountingStage`: the contraction.
#   AdoptChoice — `LogitChoiceStage` on the :technology axis ∈ {not, adopted}.
#                 The cost matrix charges the one-time adoption cost `κ` to
#                 switch `not → adopted` (free to stay in either state, free to
#                 abandon). The payoff is NOT a kwarg — it is the Flow above.
#
# So `V(t) = u(t; share) + β·logsumexp_{t'}[ −C[t,t'] + V(t') ]`. Because the
# adopted payoff `θ·share` depends on the aggregate adoption share, and the
# share is the stationary mass of adopters, the model can exhibit MULTIPLE
# equilibria (a low-adoption trap and a high-adoption equilibrium) — the
# driver finds a fixed point of `share = mass(adopted)`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct TechAdoptParams
    β :: Float64 = 0.85
    technologies :: Vector{Symbol} = [:not, :adopted]
    θ :: Float64 = 1.0     # network benefit per unit adoption share (adopted flow = θ·share)
    κ :: Float64 = 2.0     # one-time adoption cost (utils) to switch not → adopted
    ε :: Float64 = 0.08    # Gumbel scale of the adopt logit
    # The indifference share is x* ≈ κ(1−β)/θ; here ≈ 0.30, so a low-adoption
    # trap (share → 0) and a high-adoption equilibrium (share → 1) coexist.
end

Base.Broadcast.broadcastable(p::TechAdoptParams) = Ref(p)

const params = TechAdoptParams()


# Adoption-cost matrix (plain primitive, not a stage) #
#-----------------------------------------------------#

"""
The `2×2` adoption-cost matrix `C[from, to]` over {not, adopted}: switching
`not → adopted` costs `κ`; staying put (either state) and abandoning are free.
Plain data handed to `LogitChoiceStage`.
"""
adoption_cost_matrix(p = params) = [0.0  p.κ;      # from :not     → (:not, :adopted)
                                    0.0  0.0]      # from :adopted → (:not, :adopted)


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached network-adoption household block
`Flow ∘ Discount ∘ AdoptChoice` over the :technology axis. The adoption share
is read from `env`, so the driver can iterate the network-externality fixed
point without rebuilding the chain. Moment: the adoption share.
"""
function tech_adopt_household(p = params)
    layout = GriddedLayout(:technology => Discrete(p.technologies))

    flow = UtilityStage(layout;
        utility = (; technology, env) -> technology == :adopted ? p.θ * env.share : 0.0)
    discount = TimeDiscountingStage(layout; β = p.β)
    choice = LogitChoiceStage(layout;
        axis        = :technology,
        cost_matrix = adoption_cost_matrix(p),
        ε           = p.ε)

    hh = flow ∘ discount ∘ choice

    return define_moments!(hh;
        adoption_share = at_end(integrand = (; technology) -> technology == :adopted ? 1.0 : 0.0,
                                reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"The env consumed by the chain: the aggregate adoption share (the externality)."
tech_adopt_env(share::Real) = (; share = share)
