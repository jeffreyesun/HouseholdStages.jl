#######################################################################
# Aiyagari with endogenous labor (GHH) — heterogeneous-agent steady state #
#######################################################################

# An Aiyagari (1994) economy in which the household jointly chooses hours `n`
# (intensive margin) and savings `b'`. Labor of efficiency `ε` (the productivity
# state) earns `w·ε·n`, hours carry a disutility `v(n)`, and the resulting labor
# income funds consumption and saving. References: Chang–Kim (2007),
# Pijoan-Mas (2006), Heathcote–Storesletten–Violante (2014).
#
# THE FOC SUBSTITUTION (the ✅ build). With GHH preferences
#
#     felicity = u(c − v(n)),   v(n) = ψ·n^{1+1/φ}/(1+1/φ),
#
# the intratemporal first-order condition for hours is
#
#     ∂/∂n  u((1+r)b + w·ε·n − b' − v(n))  =  u'(·)·(w·ε − v'(n)) = 0
#       ⟹   v'(n) = ψ·n^{1/φ} = w·ε   ⟹   n*(w,ε) = (w·ε / ψ)^φ.
#
# Crucially the GHH optimum `n*` depends ONLY on the effective wage `w·ε` — NOT on
# wealth, consumption, or `b'`. So the hours choice is solved in closed form and
# substituted BEFORE the savings argmax, collapsing the joint problem to the plain
# Aiyagari spine:
#
#     IncomeShock ∘ IncomeReceipt(cash = (1+r)b + w·ε·n*) ∘ ConsumptionSavings(u(c − v(n*)))
#
# `IncomeReceipt` folds the closed-form labor income `w·ε·n*(w,ε)` into cash-on-hand;
# the savings stage's felicity is `u(c − v(n*))`, the GHH composite (disutility netted
# out of consumption). Both `n*` and `v(n*)` are constants given the (w, ε) the cell
# faces, so the within-period problem is a STANDARD consumption-savings argmax over a
# shifted consumption — no extra choice axis, no coupling. This is why GHH is the clean
# ✅ build.
#
# WHY GHH AND NOT SEPARABLE u(c) − v(n). With separable CRRA the FOC is
# `ψ·n^{1/φ} = w·ε·c^{−σ}`, i.e. `n*(c) = (w·ε/ψ)^φ · c^{−σφ}` — hours depend on
# consumption, so labor income `w·ε·n*(c)` re-enters the budget that funds `c`, and the
# cash-on-hand that the savings stage consumes from would itself depend on the savings
# choice. That circularity is the ◐ case. GHH severs it (`n*` wealth/consumption-free),
# giving an EXACT closed-form substitution and the ✅.
#
# THE ◐ ROUTE-A ALTERNATIVE (not built here). The fully joint version — keeping hours as
# a genuine second choice rather than substituting the FOC — is the two_asset_hank
# auxiliary-choice-axis pattern: an `ArgmaxStage` picking `n` onto an auxiliary `:hours`
# axis, a `WealthChangeStage` reading `:hours` to set cash, a `ForgetfulSumStage`
# collapsing `:hours`, then `ConsumptionSavingsStage`. That works for ANY (e.g.
# non-separable) preferences but discretizes hours; the closed-form ✅ here is exact.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct LaborSupplyParams
    β :: Float64       = 0.96
    σ :: Float64       = 2.0      # CRRA over the GHH composite c − v(n)
    φ :: Float64       = 0.5      # Frisch elasticity of labor supply (GHH: d ln n / d ln(wε) = φ)
    ψ :: Float64       = 9.0      # hours-disutility scale (pins mean hours ≈ 1/3 at w·ε ≈ 1)
    α :: Float64       = 0.36
    δ :: Float64       = 0.08
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]   # labor-efficiency (productivity) states ε
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w   :: Int       = 300
    w_min :: Float64   = 0.0
    w_max :: Float64   = 60.0
end

Base.Broadcast.broadcastable(p::LaborSupplyParams) = Ref(p)

const labor_supply_params = LaborSupplyParams()


# Closed-form hours and disutility (the FOC substitution) #
#--------------------------------------------------------#

"""
GHH intensive-margin hours `n*(w·ε) = (effective_wage / ψ)^φ`, the closed-form solution
of the intratemporal FOC `ψ·n^{1/φ} = w·ε`. Depends only on the effective wage `w·ε`,
not on wealth or consumption — the property that makes the FOC substitution exact.
"""
n_star(effective_wage, p = labor_supply_params) = (effective_wage / p.ψ)^p.φ

"""
Hours-disutility `v(n) = ψ·n^{1+1/φ}/(1+1/φ)` netted out of consumption in the GHH
composite `c − v(n)`.
"""
labor_disutility(n, p = labor_supply_params) = p.ψ * n^(1 + 1 / p.φ) / (1 + 1 / p.φ)


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Aiyagari-with-endogenous-labor household block
`IncomeShock ∘ IncomeReceipt(GHH) ∘ ConsumptionSavings(GHH)`. The receipt closure folds
the closed-form labor income `w·ε·n*(w·ε)` into cash-on-hand; the savings felicity is the
GHH composite `u(c − v(n*))`, reading the productivity axis via `utility_axes = (:income,)`.
Attaches `K_supplied = ∫ wealth`, `L_supplied = ∫ ε·n*` (effective labor), and
`hours = ∫ n*` (aggregate hours, divide by mass for the mean).
"""
function labor_supply_household(p = labor_supply_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout;                                  # cash = (1+r)b + w·ε·n*(w·ε)
        wealth_post = (; wealth, income, env) ->
            (1 + env.r) * wealth + env.w * income * n_star(env.w * income, p),
        axis = :wealth)
    savings = ConsumptionSavingsStage(layout;
        β            = p.β,
        utility      = (cell, c; env) ->                           # u(c − v(n*)): the GHH composite
            u_crra(c - labor_disutility(n_star(env.w * cell.income, p), p), Val(p.σ)),
        utility_axes = (:income,),
        axis         = :wealth)

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
        L_supplied = at_end(integrand = (; income, env) -> income * n_star(env.w * income, p),
                            reduce = sum),
        hours      = at_end(integrand = (; income, env) -> n_star(env.w * income, p),
                            reduce = sum),
    )
end


# Production prices (plain function, no AbstractBlock) #
#------------------------------------------------------#

"""
Cobb-Douglas factor prices at capital-labor ratio `κ = K/L`: `r = α·κ^{α−1} − δ`,
`w = (1−α)·κ^α`. With endogenous labor the firm's `κ` is the single sufficient statistic
for both prices, so the steady state reduces to a 1-D fixed point in `κ` (see
`steady_state.jl`).
"""
function labor_supply_prices(κ::Real, p = labor_supply_params)
    (; α, δ) = p
    r = α * κ^(α - 1) - δ
    w = (1 - α) * κ^α
    return (; r, w)
end
