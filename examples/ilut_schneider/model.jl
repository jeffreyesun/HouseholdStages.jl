################################################################
# Ilut–Schneider (2014) — ambiguous business cycles             #
################################################################
#
# A household that is AMBIGUITY-AVERSE about the AGGREGATE business cycle. It
# does not trust the aggregate transition `P_z` over the persistent cycle
# state `z ∈ {boom, recession}`: it acts on a WORST-CASE mean drawn from an
# entropy-constrained ambiguity set around `P_z` (max-min utility, Gilboa–
# Schmeidler / Hansen–Sargent multiplier form). The smoothed/entropic version
# of that worst-case operator is
#
#     V_start[z] = −θ · log Σ_z′ P_z[z,z′] · exp(−V_end[z′]/θ),
#
# the literal set-based max-min being the θ → 0⁺ (ε → 0⁻) limit. Idiosyncratic
# income risk is CYCLE-DEPENDENT (recessions are sticky-low, à la
# countercyclical risk), so pessimism about the cycle is pessimism about one's
# own future income — the Ilut–Schneider amplification channel.
#
# This differs from `examples/hansen_sargent` (which tilts the IDIOSYNCRATIC
# income transition directly): here the ambiguity acts on the PERSISTENT
# AGGREGATE cycle transition, and idiosyncratic income is an ordinary (trusted)
# `MarkovStage` whose law is selected by the cycle. The block mirrors
# `examples/regime_switching`, with the aggregate `MarkovStage(:z)` swapped for
# the ε<0 `LogitChoiceStage(:z)` ambiguity operator.
#
# The chain is four existing library stages, no bespoke stage rolled here —
#
#     AmbiguityTilt ∘ IncomeDraw ∘ Receipt ∘ ConsumptionSavings
#
# `AmbiguityTilt` — `LogitChoiceStage(:z, ε = −θ, cost = −ε·log P_z)`: the
#                   worst-case entropic tilt of the aggregate cycle transition.
# `IncomeDraw`    — `MarkovStage(:income; transition = (; z) -> T_income(z))`:
#                   ordinary idiosyncratic income, cycle-dependent law.
# `Receipt`       — `IncomeStage`: `a ↦ (1+r)·a + w·y`.
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period wealth.
#
# Returns are exogenous (partial equilibrium): a single inner V/Λ solve. The
# `LogitChoiceStage.forward!` pushes Λ through the seated WORST-CASE cycle
# kernel, so the stationary cycle distribution is the household's PESSIMISTIC
# one — here that is the point: ambiguity tilts the ergodic measure toward
# recessions (more recession mass than the reference `P_z` ergodic share),
# which is the "ambiguous business cycles" amplification we report.

using HouseholdStages


# Parameters #
#------------#

const Z_BOOM      = 1.0
const Z_RECESSION = 2.0

@kwdef struct IlutSchneiderParams
    β :: Float64 = 0.96
    σ :: Float64 = 2.0
    r :: Float64 = 0.03                                # fixed (< 1/β − 1 ≈ 0.0417)
    w :: Float64 = 1.0

    # Aggregate business-cycle state: persistent, expansions longer than
    # recessions. STRICTLY POSITIVE (no log 0) — the ambiguity tilt is
    # cost = −ε·log P_z.
    z_grid :: Vector{Float64} = [Z_BOOM, Z_RECESSION]
    P_z    :: Matrix{Float64} = [0.90 0.10;            # boom → mostly boom
                                 0.25 0.75]            # recession → shorter-lived

    # Idiosyncratic income (mean ≈ 1), shared grid, cycle-dependent law.
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    T_income_boom :: Matrix{Float64} = [0.50 0.40 0.10;     # favourable, upward
                                        0.10 0.60 0.30;
                                        0.05 0.25 0.70]
    T_income_recession :: Matrix{Float64} = [0.80 0.18 0.02;  # adverse, sticky-low
                                             0.40 0.50 0.10;
                                             0.15 0.45 0.40]

    N_w   :: Int     = 120
    w_min :: Float64 = 0.0
    w_max :: Float64 = 80.0
end

Base.Broadcast.broadcastable(p::IlutSchneiderParams) = Ref(p)

const ilut_schneider_params = IlutSchneiderParams()

income_law(p, z) = z == Z_BOOM ? p.T_income_boom : p.T_income_recession


# Household chain assembly #
#--------------------------#

"""
Build the ambiguous-business-cycle household block `AmbiguityTilt ∘ IncomeDraw ∘
Receipt ∘ ConsumptionSavings` at ambiguity multiplier `θ` (logit scale `ε =
−θ < 0`). The worst-case operator over the aggregate cycle transition is a
`LogitChoiceStage` on the `:z` axis with `cost_matrix = −ε·log P_z` and `ε = −θ`
(so `exp(−C/ε) = P_z`, and the logit backward becomes the entropic worst-case
`−θ·log Σ_z′ P_z[z,z′]·exp(−V_end[z′]/θ)`). Idiosyncratic income is an ordinary
`MarkovStage(:income)` with a `(; z) ->` cycle-dependent law. Four existing
stages, no bespoke household stage. `θ` large ⇒ ordinary expectation over the
cycle; `θ → 0⁺` ⇒ the literal set-based max-min.
"""
function ilut_schneider_household(p = ilut_schneider_params; θ::Float64 = 1.0)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :z      => Discrete(p.z_grid),
        :income => Discrete(p.y_grid),
    )

    ε = -θ                                                        # ε < 0 ⇒ soft-MIN / worst-case
    ambiguity = LogitChoiceStage(layout;
        axis        = :z,
        cost_matrix = -ε .* log.(p.P_z),                          # C = −ε·log P_z ⇒ exp(−C/ε) = P_z
        ε           = ε)
    income_draw = MarkovStage(layout; axis = :income,
        transition_matrix = (; z) -> income_law(p, z))
    receipt = IncomeStage(layout)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = ambiguity ∘ income_draw ∘ receipt ∘ savings
    return define_moments!(hh;
        mean_wealth     = at_end(integrand = :wealth, reduce = sum),
        recession_share = at_end(integrand = (; z) -> z == Z_RECESSION ? 1.0 : 0.0, reduce = sum))
end

"""
Reference NON-ambiguous household: the aggregate cycle is an ordinary
`MarkovStage(:z, P_z)` instead of the worst-case `LogitChoiceStage`. Supplies
the θ→∞ cross-check and the reference cycle distribution. A library-stage
composition, like the ambiguous block.
"""
function ilut_schneider_reference(p = ilut_schneider_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :z      => Discrete(p.z_grid),
        :income => Discrete(p.y_grid),
    )
    cycle = MarkovStage(layout; axis = :z, transition_matrix = p.P_z)
    income_draw = MarkovStage(layout; axis = :income,
        transition_matrix = (; z) -> income_law(p, z))
    receipt = IncomeStage(layout)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = cycle ∘ income_draw ∘ receipt ∘ savings
    return define_moments!(hh;
        mean_wealth     = at_end(integrand = :wealth, reduce = sum),
        recession_share = at_end(integrand = (; z) -> z == Z_RECESSION ? 1.0 : 0.0, reduce = sum))
end

"Env for the fixed-`r` partial-equilibrium experiment: gross return `r`, wage `w`."
ilut_schneider_env(p = ilut_schneider_params) = (; r = p.r, w = p.w)


# Policy extraction (reporting only — outside the block) #
#-------------------------------------------------------#

"""
Recover the chosen next-period wealth VALUES `b'(state)` from a solved
household — the unique `(N_w, n_z·n_income)`-shaped policy-bearing leaf (the
savings `ContinuousArgmaxStage`), whose policy holds the chosen next-wealth
values directly. Used for the distribution-free precautionary statistic in
`steady_state.jl`.
"""
function ilut_schneider_savings_policy(hh, p = ilut_schneider_params)
    n_other = length(p.z_grid) * length(p.y_grid)
    leaves = filter(s -> !(s isa HouseholdStages.ChainStage) &&
                         hasmethod(HouseholdStages.policy, Tuple{typeof(s)}),
                    collect(hh.buffer.stages))
    savings = only(filter(s -> length(HouseholdStages.policy(s)) == p.N_w * n_other, leaves))
    return HouseholdStages.policy(savings)
end
