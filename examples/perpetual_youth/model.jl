#############################################################
# Perpetual Youth (Blanchard 1985 / Yaari 1965) — Household  #
#############################################################

# An incomplete-markets consumption–savings core wrapped in the
# demographics composite: every period each household faces a constant
# death hazard `δ`, and the mass that dies is replaced one-for-one by
# newborns, so the population is stationary (mass-preserving rebirth).
# With actuarially-fair annuities, survivors earn a mortality premium on
# their assets: the gross return is `(1+r)/(1−δ)` rather than `(1+r)`
# (the dead forfeit their balances to the annuity pool, which pays out to
# survivors).
#
# The within-period chain, in time order (`∘` runs the LEFT stage first
# in the forward sweep):
#
#     IncomeShock ∘ AnnuityReceipt ∘ ConsumptionSavings ∘ ExogenousExit ∘ Entry
#
# - IncomeShock      — `MarkovStage` on the income axis (the `P_y` draw).
# - AnnuityReceipt   — `WealthChangeStage`, `b ↦ (1+r)/(1−δ)·b + w·y`.
#                      The mortality premium `1/(1−δ)` is the actuarially
#                      fair annuity return.
# - ConsumptionSavings — `ConsumptionSavingsStage`, pick `b' ` on the grid;
#                      implicit budget `c = b_in − b'`. The stage's own `β`
#                      composed with the exit's survival weighting yields an
#                      effective discount `β·(1−δ)` on the continuation.
# - ExogenousExit    — survival `s = 1−δ`; backward `V ↦ s·V + (1−s)·b`,
#                      forward `Λ ↦ s·Λ` (mass leaks out as deaths).
# - Entry            — `EntryStage`, additive newborn source `Λ ↦ Λ + g`.
#                      Newborns enter at zero wealth with the ergodic income
#                      draw; `Σg = δ` so the stationary population is ≈ 1.
#
# A clean Blanchard–Yaari identity falls out of this composition: the
# Euler equation discounts the continuation by `β·s` (the exit weighting)
# while the annuity grosses assets up by `1/s`, so the effective
# return–discount product is `β·s·(1+r)/s = β·(1+r)` — exactly the
# standard Aiyagari condition `β(1+r) < 1` for bounded wealth. The
# mortality premium and the survival discount cancel.
#
# The `:exiting` axis is declared at size 1; the exit composite grows it
# `1 → 2` internally and collapses it back, so the household V/Λ carry a
# trailing singleton axis throughout.

using HouseholdStages
using LinearAlgebra


# Parameters #
#------------#

@kwdef struct PerpetualYouthParams
    β :: Float64       = 0.96            # patience
    σ :: Float64       = 1.0             # CRRA curvature (1.0 ⇒ log)
    δ :: Float64       = 0.05            # constant per-period death hazard
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_w   :: Int       = 120
    w_min :: Float64   = 0.0
    w_max :: Float64   = 80.0
    bequest :: Float64 = 0.0             # value of death; see README on the choice
end

Base.Broadcast.broadcastable(p::PerpetualYouthParams) = Ref(p)

const perpetual_youth_params = PerpetualYouthParams()


"""
Stationary (ergodic) distribution of a row-stochastic transition matrix
`P`, i.e. the `π` solving `π = π·P`, `Σπ = 1`. Computed as the normalized
left eigenvector of the unit eigenvalue.
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
Build the moment-attached perpetual-youth household block
`IncomeShock ∘ AnnuityReceipt ∘ ConsumptionSavings ∘ ExogenousExit ∘ Entry`.

The newborn source `g` puts mass `δ` at zero wealth split across income by
the ergodic income distribution, so the stationary total population is ≈ 1
(the fixed point of `Λ = (1−δ)·K·Λ + δ·g` with `g` normalized to mass `δ`).
Two moments are attached: `A_total = ∫ wealth dΛ` and `pop = ∫ dΛ`.
"""
function perpetual_youth_household(p = perpetual_youth_params)
    layout = GriddedLayout(
        :wealth  => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income  => Discrete(p.y_grid),
        :exiting => Discrete([0]),               # size-1; exit composite grows it 1→2
    )

    n_y = length(p.y_grid)

    # Newborn distribution: zero wealth (grid index 1), ergodic income draw,
    # total mass δ ⇒ replacement keeps the population stationary at ≈ 1.
    π = ergodic_distribution(p.P_y)
    g = zeros(p.N_w, n_y, 1)
    g[1, :, 1] .= p.δ .* π

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;
        axis        = :wealth,
        wealth_post = (; wealth, income, env) ->
            (1 + env.r) / (1 - p.δ) * wealth + env.w * income,   # actuarially-fair annuity return
    )
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        axis            = :wealth,
        monotone_search = :divide_conquer,
    )
    exit  = ExogenousExit(layout; survival = 1 - p.δ, bequest = p.bequest)
    entry = EntryStage(layout; entry = g)

    hh = shock ∘ receipt ∘ savings ∘ exit ∘ entry
    return define_moments!(hh;
        A_total = at_end(integrand = :wealth, reduce = sum),     # ∫ wealth dΛ
        pop     = at_end(integrand = 1.0,     reduce = sum),     # ∫ dΛ (population mass)
    )
end
