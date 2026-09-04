###################################################################
# Entry / Birth Demonstrator (EntryStage) — Household Block        #
###################################################################

# The minimal demonstrator that birth ≠ death moves the population. A
# bare consumption–savings core is wrapped in the demographics composite:
# households die at hazard `1−s` (`ExogenousExit`) and newborns arrive as
# an additive source `g` (`EntryStage`). The chain, in time order:
#
#     IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ ExogenousExit ∘ Entry
#
# Forward, the mass map collapses to an AFFINE recursion in the total
# population:  `M_{t+1} = s·M_t + Σg`. Its fixed point is `M* = Σg/(1−s)`,
# so the stationary population is a pure function of the birth/death
# balance:
#
#   • replacement   `Σg = 1−s`   ⇒  `M* = 1`              (the usual normalization)
#   • birth-heavy   `Σg > 1−s`   ⇒  `M* = Σg/(1−s) > 1`   (a larger population)
#
# Because `EntryStage` is a FIXED (mass-independent) source, the dynamics
# are affine, not geometric: a birth-heavy regime converges to a higher
# stationary LEVEL rather than growing without bound. Genuine exponential
# population growth would require a source proportional to the current
# mass, which a fixed additive `g` deliberately does not provide. The
# driver in `steady_state.jl` exhibits both the level effect and the
# transient mass trajectory.
#
# Household policies here do not depend on the aggregate population, so
# the savings policy (hence V) is solved once and the mass dynamics are a
# linear afterthought — exactly the point of the demonstrator.
#
# The `:exiting` axis is declared at size 1; the exit composite grows it
# `1 → 2` internally.

using HouseholdStages
using LinearAlgebra


# Parameters #
#------------#

@kwdef struct EntryBirthParams
    β :: Float64       = 0.95
    σ :: Float64       = 1.0             # log utility
    s :: Float64       = 0.95            # per-period survival (death hazard 1−s)
    y_grid :: Vector{Float64} = [0.7, 1.3]
    P_y    :: Matrix{Float64} = [0.8 0.2;
                                 0.2 0.8]
    N_w   :: Int       = 120
    w_min :: Float64   = 0.0
    w_max :: Float64   = 60.0
    bequest :: Float64 = 0.0
end

Base.Broadcast.broadcastable(p::EntryBirthParams) = Ref(p)

const entry_birth_params = EntryBirthParams()


"""
Stationary (ergodic) distribution of a row-stochastic transition matrix
`P` (the `π` solving `π = π·P`, `Σπ = 1`), as the normalized left
eigenvector of the unit eigenvalue.
"""
function ergodic_distribution(P::AbstractMatrix)
    vals, vecs = eigen(collect(P'))
    k = argmin(abs.(vals .- 1))
    π = real.(vecs[:, k])
    return π ./ sum(π)
end


# Household chain assembly #
#--------------------------#

"""
Build the entry/birth household block
`IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ ExogenousExit ∘ Entry`,
with the newborn inflow scaled to total mass `birth_mass` (newborns at
zero wealth, ergodic income draw). `birth_mass = 1 − s` is replacement;
`birth_mass > 1 − s` is birth-heavy. Moments: `pop = ∫ dΛ` and
`A_total = ∫ wealth dΛ`.
"""
function entry_birth_household(p = entry_birth_params; birth_mass = 1 - p.s)
    layout = GriddedLayout(
        :wealth  => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income  => Discrete(p.y_grid),
        :exiting => Discrete([0]),
    )

    n_y = length(p.y_grid)
    π = ergodic_distribution(p.P_y)
    g = zeros(p.N_w, n_y, 1)
    g[1, :, 1] .= birth_mass .* π            # newborns: zero wealth, ergodic income

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;
        axis        = :wealth,
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
    )
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c) -> u_crra(c, Val(p.σ)),
        axis            = :wealth,
    )
    exit  = ExogenousExit(layout; survival = p.s, bequest = p.bequest)
    entry = EntryStage(layout; entry = g)

    hh = shock ∘ receipt ∘ savings ∘ exit ∘ entry
    return define_moments!(hh;
        pop     = at_end(integrand = 1.0,     reduce = sum),
        A_total = at_end(integrand = :wealth, reduce = sum),
    )
end
