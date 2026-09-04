#####################################################
# Heathcote–Storesletten–Violante (2017) — Aiyagari with HSV Progressive Tax #
#####################################################

# An Aiyagari (1994) general-equilibrium economy whose only departure from
# the textbook model is the fiscal scheme: labour income is taxed by the
# HSV (2017) log-linear progressive schedule. Post-tax labour income is
#
#     T(y) = λ · (w·y)^(1−τ)
#
# where `τ ∈ [0,1)` is the progressivity parameter (τ = 0 ⇒ flat, higher τ
# ⇒ more progressive) and `λ` scales the overall level. This nests the
# linear-tax / no-tax case at τ = 0, λ = 1.
#
# The within-period problem is the canonical three-stage spine, with the
# HSV schedule folded into the receipt budget closure:
#
#     IncomeShock ∘ IncomeReceipt(HSV) ∘ ConsumptionSavingsStage
#
# `IncomeShock` (MarkovStage, axis = :income) — idiosyncratic labour-
# productivity Markov draw. `IncomeReceipt(HSV)` (IncomeStage) — receipt
# `a ↦ (1+r) a + λ·(w·y)^(1−τ)`. `ConsumptionSavingsStage` — choose next-
# period capital; implicit budget `c = a_in − a_end`, CRRA utility.
#
# Production is Cobb-Douglas with fixed labour; equilibrium `r, w` come
# from marginal products at aggregate capital `K`, cleared by tatonnement
# in steady_state.jl (copied from examples/aiyagari). `λ, τ` ride in `env`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HSVParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36
    δ :: Float64       = 0.08
    L :: Float64       = 1.0
    # HSV (2017) tax schedule  T(y) = λ·(w·y)^(1−τ).
    λ :: Float64       = 0.90    # level
    τ :: Float64       = 0.181   # progressivity (HSV's US estimate ≈ 0.18)
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w   :: Int       = 300
    w_min :: Float64   = 0.0
    w_max :: Float64   = 100.0
end

Base.Broadcast.broadcastable(p::HSVParams) = Ref(p)

const hsv_params = HSVParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached HSV household block
`IncomeShock ∘ IncomeReceipt(HSV) ∘ ConsumptionSavingsStage`. The receipt
closure applies the HSV progressive schedule `λ·(w·y)^(1−τ)` to labour
income (`λ, τ` from `env`); otherwise identical to the Aiyagari block. The
`K_supplied = ∫ wealth dΛ` moment is attached for the tatonnement.
"""
function hsv_household(p = hsv_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, env) ->
            (1 + env.r) * wealth + env.λ * (env.w * income)^(1 - env.τ),
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Production prices (plain function, no AbstractBlock) #
#------------------------------------------------------#

"Cobb-Douglas factor prices at aggregate capital `K`, fixed labour `p.L`."
function hsv_prices(K::Real, p = hsv_params)
    (; α, δ, L) = p
    r = α * (K / L)^(α - 1) - δ
    w = (1 - α) * (K / L)^α
    return (; r, w)
end
