#####################################################################
# Carroll (1997) buffer-stock saving — permanent + transitory income #
#####################################################################
#
# Impatient consumers with PERMANENT (unit-root) and TRANSITORY income shocks
# accumulate a target buffer stock of wealth. The defining obstruction (catalog
# §0 ◐) is that the unit-root permanent component is NOT a stationary
# `MarkovStage` on a fixed grid. Carroll's own solution NORMALIZES every level
# variable by permanent income `P_t`; in ratio-to-permanent-income units the
# problem is stationary and lives on a fixed grid.
#
# Normalized recursion (lower-case = level / permanent income, ρ = CRRA σ):
#
#   v(m) = max_c  u(c) + β · E_{ψ',ξ'}[ (G ψ')^{1-ρ} · v(m') ],
#   a    = m − c,                       (end-of-period assets, normalized)
#   m'   = (R / (G ψ')) · a + ξ',       (next cash-on-hand, RE-normalized by Gψ')
#
# with G the deterministic permanent-income growth factor, ψ' the PERMANENT
# shock (E ψ = 1), ξ' the TRANSITORY shock (E ξ = 1). The unit root is gone:
# only the RATIO m is a state, on a fixed grid. Two normalization artefacts
# remain, and BOTH map onto existing stages:
#
#   (1) the budget divides assets by `G·ψ'`  →  a `WealthChangeStage` closure
#       reading the permanent-shock axis;
#   (2) the continuation is RE-WEIGHTED by `(G ψ')^{1-ρ}`, a factor that varies
#       across the permanent-shock axis and must scale V (backward) but NOT the
#       distribution Λ (forward). That is EXACTLY the asymmetric two-sided
#       `PointwiseScaleStage(backward = (Gψ')^{1-ρ}, forward = 1)`. The scale is
#       a full-layout array (varying only along `:psi`), supplied through `env`.
#
# Household block (time order = forward / distribution order):
#
#   MarkovStage(:psi) ∘ PointwiseScaleStage((Gψ)^{1-ρ}|1) ∘ MarkovStage(:xi)
#     ∘ WealthChangeStage(m = R·a/(Gψ) + ξ) ∘ ConsumptionSavingsStage(:wealth)
#
# Backward order reads right-to-left: savings (β·V_end), receipt relabels cash,
# `MarkovStage(:xi)` takes E_ξ, `PointwiseScaleStage` multiplies by (Gψ)^{1-ρ},
# `MarkovStage(:psi)` takes E_ψ — reproducing v = β E_ψ[(Gψ)^{1-ρ} E_ξ[v(m')]].
# The permanent and transitory shocks are iid ⇒ degenerate Markov chains with
# IDENTICAL ROWS. The whole thing is existing stages only — no bespoke stage.
#
# The normalized cash-on-hand `m` is STATIONARY (the buffer-stock target);
# the LEVEL distribution of wealth is non-stationary (grows with P), which is
# exactly why the normalization is the thing that makes this expressible.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CarrollParams
    β :: Float64 = 0.95
    σ :: Float64 = 2.0                 # CRRA ρ; the (Gψ)^{1-ρ} reweight vanishes only at ρ = 1
    R :: Float64 = 1.04                # gross return; return-impatience βR = 0.988 < 1
    G :: Float64 = 1.00                # permanent-income growth factor

    # Permanent shock ψ (iid, E ψ ≈ 1): a small 3-point symmetric spread.
    ψ_grid :: Vector{Float64} = [0.90, 1.00, 1.111111]   # ≈ {1/1.111, 1, 1.111}
    p_ψ    :: Vector{Float64} = [0.25, 0.50, 0.25]

    # Transitory shock ξ (iid, E ξ ≈ 1): 3-point spread.
    ξ_grid :: Vector{Float64} = [0.70, 1.00, 1.30]
    p_ξ    :: Vector{Float64} = [0.25, 0.50, 0.25]

    # Normalized cash-on-hand grid m (log-spaced; floor a bit above 0 so that
    # m = R·a/(Gψ) + ξ with the smallest ξ stays inside the grid).
    N_m   :: Int     = 300
    m_min :: Float64 = 0.0
    m_max :: Float64 = 60.0
end

Base.Broadcast.broadcastable(p::CarrollParams) = Ref(p)

const carroll_params = CarrollParams()


# Household chain assembly #
#--------------------------#

"""
Build the normalized-units Carroll buffer-stock block

    MarkovStage(:psi) ∘ PointwiseScaleStage((Gψ)^{1-σ}|1)
      ∘ MarkovStage(:xi) ∘ WealthChangeStage(m = R·a/(Gψ)+ξ)
      ∘ ConsumptionSavingsStage(:wealth)

— existing stages only. The permanent-income normalization leaves two
artefacts: the budget division by `G·ψ` (in the receipt closure) and the
continuation reweight `(G·ψ)^{1-σ}` (the asymmetric `PointwiseScaleStage`,
`forward = 1` so the distribution Λ is untouched). `mean_m = ∫ m dΛ` (the
buffer-stock target in normalized units) is attached.
"""
function carroll_household(p = carroll_params)
    P_ψ = repeat(p.p_ψ', length(p.ψ_grid))     # iid permanent shock ⇒ identical rows
    P_ξ = repeat(p.p_ξ', length(p.ξ_grid))     # iid transitory shock ⇒ identical rows

    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.m_min, p.m_max, p.N_m; spacing = :log),
        :psi    => Discrete(p.ψ_grid),
        :xi     => Discrete(p.ξ_grid),
    )

    shock_ψ = MarkovStage(layout; axis = :psi, transition_matrix = P_ψ)
    reweight = PointwiseScaleStage(layout;
        backward = FromEnv(:psi_reweight),     # (G·ψ)^{1-σ} array, varies along :psi
        forward  = 1.0)                        # leave the distribution untouched
    shock_ξ = MarkovStage(layout; axis = :xi, transition_matrix = P_ξ)
    receipt = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; wealth, psi, xi, env) ->
            (env.R / (env.G * psi)) * wealth + xi)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
        axis    = :wealth)

    hh = shock_ψ ∘ reweight ∘ shock_ξ ∘ receipt ∘ savings
    return define_moments!(hh;
        mean_m = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (offline pre-computations) #
#--------------------------------#

"""
Env for the normalized Carroll solve: prices `R, G` and the precomputed
permanent-income reweight array `(G·ψ)^{1-σ}`, broadcast to the full layout
(it varies only along the `:psi` axis). This array is the analytic
change-of-variables artefact — an offline parameter pre-computation — fed to
the asymmetric `PointwiseScaleStage`.
"""
function carroll_env(p = carroll_params)
    psi_reweight = [(p.G * ψ)^(1 - p.σ) for _ in 1:p.N_m, ψ in p.ψ_grid, _ in p.ξ_grid]
    return (; p.R, p.G, psi_reweight)
end
