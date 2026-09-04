################################################################
# Persistent + transitory income (Storesletten-Telmer-Yaron 2004,#
# Kaplan-Violante 2010)                                          #
################################################################
#
# The standard two-component earnings process: log income is the sum of a
# PERSISTENT (AR(1)-like) component and a TRANSITORY (iid) component,
#
#     y_t = z_t · ν_t,   z_t persistent,   ν_t transitory iid.
#
# This is the workhorse income process of the consumption-savings
# literature (Storesletten-Telmer-Yaron 2004; Kaplan-Violante 2010). In
# the package it is a STRAIGHT concatenation of two independent
# `MarkovStage`s on two separate income axes — no new machinery:
#
#   MarkovStage(:persistent) ∘ MarkovStage(:transitory) ∘ IncomeStage ∘ ConsumptionSavingsStage
#
# The two shocks resolve independently (each its own row-stochastic chain
# on its own axis); the budget then reads BOTH axes via a dep closure,
# `(1+r) wealth + w · persistent · transitory`. The transitory chain has
# IDENTICAL ROWS (`P(ν' | ν)` independent of `ν`) — that is exactly what
# "iid" means as a degenerate Markov chain.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct PersistentTransitoryParams
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0
    r   :: Float64 = 0.03                      # fixed, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0

    # Persistent component: a 3-state moderately-persistent chain (mean ≈ 1).
    z_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_z    :: Matrix{Float64} = [0.85 0.13 0.02;
                                 0.10 0.80 0.10;
                                 0.02 0.13 0.85]

    # Transitory component: iid ⇒ a degenerate Markov chain with IDENTICAL
    # ROWS. The next draw's distribution does not depend on the current one.
    ν_grid :: Vector{Float64} = [0.7, 1.0, 1.3]
    p_ν    :: Vector{Float64} = [0.25, 0.50, 0.25]   # the common row

    N_w   :: Int     = 250
    w_min :: Float64 = 0.0
    w_max :: Float64 = 80.0
end

Base.Broadcast.broadcastable(p::PersistentTransitoryParams) = Ref(p)

const persistent_transitory_params = PersistentTransitoryParams()


# Household chain assembly #
#--------------------------#

"""
Build the persistent+transitory household block as a plain concatenation of
two independent `MarkovStage`s on two income axes:

    MarkovStage(:persistent) ∘ MarkovStage(:transitory) ∘ IncomeStage ∘ ConsumptionSavingsStage.

The transitory chain `P_ν` is iid — every row equals the common `p_ν`. The
budget dep closure reads BOTH income axes: `(1+r) wealth + w · z · ν`.
`A_mean = ∫ wealth dΛ` is the attached buffer-stock moment.
"""
function persistent_transitory_household(p = persistent_transitory_params)
    P_ν = repeat(p.p_ν', length(p.ν_grid))     # iid ⇒ identical rows

    layout = GriddedLayout(
        :wealth     => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :persistent => Discrete(p.z_grid),
        :transitory => Discrete(p.ν_grid),
    )

    shock_z = MarkovStage(layout; axis = :persistent, transition_matrix = p.P_z)
    shock_ν = MarkovStage(layout; axis = :transitory, transition_matrix = P_ν)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, persistent, transitory, env) ->
            (1 + env.r) * wealth + env.w * persistent * transitory,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )

    hh = shock_z ∘ shock_ν ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean = at_end(integrand = :wealth, reduce = sum),
    )
end

"Env for the fixed-r partial-equilibrium experiment: return `r`, wage `w`."
persistent_transitory_env(p = persistent_transitory_params) = (; r = p.r, w = p.w)
