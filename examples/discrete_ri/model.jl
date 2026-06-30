###################################################################
# Discrete rational inattention — occupation choice (Matějka–McKay) #
###################################################################

# A stationary household that re-chooses a DISCRETE occupation each period
# under a Shannon information cost λ. The point of this example: the static
# RI discrete choice IS a generalized logit, so the entire within-period
# problem is THREE existing library stages, in time order, with **no
# bespoke household stage rolled here** —
#
#     OccupationChoice ∘ Discount ∘ IncomeDraw
#
# `OccupationChoice` — `LogitUtilityStage(axis = :occupation, ε = λ)`,
#                      i.e. `LogitChoiceStage(ε = λ, cost = C) ∘ UtilityStage(u)`.
#                      The UtilityStage carries the per-occupation payoff
#                      `u(z, a) = λ·log q(a) + flow(z, a)`: the Matějka–McKay
#                      *attention prior* `λ·log q(a)` (q = the unconditional
#                      occupation share, read from `env.q`) plus the state-
#                      dependent wage payoff `flow(z, a)`. The logit's
#                      `cost_matrix = C[a, a′]` is the origin-dependent
#                      occupation-SWITCHING cost (a friction a bare payoff
#                      can't carry — exactly what the cost matrix is for).
# `Discount`        — `TimeDiscountingStage(β)`: `V_start = β·V_end`.
# `IncomeDraw`      — `MarkovStage(:income; transition = P_z(·|occupation))`:
#                      the occupation just chosen sets next period's income
#                      process, so the discrete choice carries genuine
#                      DYNAMIC value (occupations differ in income mean and
#                      persistence). The occupation-dependent transition is a
#                      row-stochastic matrix selected by the occupation axis
#                      and handed to an existing stage — data, not a stage.
#
# Why the discrete choice is genuinely solved (not a degenerate static pick).
# Occupations differ in their income dynamics, so the continuation value
# `β·E[V | a]` differs by occupation; the wage payoff `flow(z, a)` interacts
# with current income; the switching cost `C[a, a′]` makes the choice
# origin-dependent; and the Shannon cost λ smooths the argmax into the
# Matějka–McKay posterior `P(a′|z, a) ∝ q(a′)·exp((flow + V)/λ)`. The
# stationary distribution Λ over (income, occupation) is shaped by this RI
# policy, and the unconditional prior `q(a) = ∫ P(a′ = a | ·) dΛ` is the
# Matějka–McKay endogenous-prior consistency condition — converged in the
# driver's outer loop (see steady_state.jl). Returns are exogenous (partial
# equilibrium): the only fixed point that closes the model is the prior q.
#
# Literature: Matějka & McKay (2015, AER) "Rational Inattention to Discrete
# Choices: A New Foundation for the Multinomial Logit Model"; Steiner,
# Stewart & Matějka (2017, ECMA) on the dynamic logit.

using HouseholdStages


# Parameters #
#------------#

# Three occupations, five income levels. Occupations differ in their income
# process (rows of `P_by_occ`) and in a flat wage premium (`premium`). The
# switching-cost matrix `C[a, a′]` is zero on the diagonal (staying is free)
# and a flat κ off-diagonal (any switch costs κ).

@kwdef struct DiscreteRiParams
    β :: Float64       = 0.95
    λ :: Float64       = 0.30                      # Shannon information-cost scale (logit temperature)
    κ :: Float64       = 0.20                      # occupation-switching cost (off-diagonal of C)
    income_grid :: Vector{Float64} = [0.5, 0.75, 1.0, 1.5, 2.0]
    premium     :: Vector{Float64} = [0.0, 0.15, 0.35]   # flat wage premium per occupation a = 1,2,3

    # One income transition per occupation. Occupation 1 is the safe/flat job
    # (mean-reverting to the middle), occupation 2 is persistent middling,
    # occupation 3 is the high-variance "career" job (climbs but can fall far).
    P_by_occ :: Vector{Matrix{Float64}} = [
        # occupation 1 — strongly mean-reverting toward the centre
        [0.45 0.35 0.15 0.04 0.01;
         0.25 0.40 0.25 0.08 0.02;
         0.10 0.25 0.40 0.20 0.05;
         0.05 0.15 0.30 0.35 0.15;
         0.02 0.08 0.25 0.35 0.30],
        # occupation 2 — persistent (high diagonal)
        [0.70 0.20 0.07 0.02 0.01;
         0.15 0.65 0.15 0.04 0.01;
         0.05 0.15 0.60 0.15 0.05;
         0.01 0.04 0.15 0.65 0.15;
         0.01 0.02 0.07 0.20 0.70],
        # occupation 3 — climbs (mass shifts up) but with a fat lower tail
        [0.30 0.30 0.25 0.10 0.05;
         0.15 0.25 0.30 0.20 0.10;
         0.08 0.17 0.30 0.30 0.15;
         0.05 0.10 0.25 0.35 0.25;
         0.03 0.07 0.15 0.35 0.40],
    ]
end

Base.Broadcast.broadcastable(p::DiscreteRiParams) = Ref(p)

const discrete_ri_params = DiscreteRiParams()


# Per-occupation flow payoff (plain economic function, no stage) #
#----------------------------------------------------------------#

"""
The per-occupation flow payoff `flow(z, a) = log(income_z) + premium_a`: log
labour income at the current income level plus the occupation wage premium.
This is the state-dependent destination payoff the RI `UtilityStage` carries;
the attention prior `λ·log q(a)` is added on top inside the household chain.
"""
flow_payoff(income::Real, premium::Real) = log(income) + premium


# Household chain assembly #
#--------------------------#

"""
Build the discrete-RI occupation-choice block
`OccupationChoice ∘ Discount ∘ IncomeDraw`, with the occupation-share moments
attached. Three existing stages, no bespoke household stage: the choice leaf
is a `LogitUtilityStage` over the `:occupation` axis at temperature λ, whose
`UtilityStage` carries `λ·log q(a) + flow(z, a)` (the Matějka–McKay attention
prior `q` read from `env.q`) and whose `LogitChoiceStage` carries the
switching-cost matrix `C`. The income draw is occupation-dependent
(`MarkovStage` with a per-occupation transition), so the discrete choice
carries dynamic value.
"""
function discrete_ri_household(p = discrete_ri_params)
    layout = GriddedLayout(
        :income     => Discrete(p.income_grid),
        :occupation => Discrete([:safe, :persistent, :career]),
    )
    occ_index = Dict(:safe => 1, :persistent => 2, :career => 3)

    # Switching cost C[a, a′]: 0 on the diagonal, κ off-diagonal.
    n_occ = length(p.premium)
    C = [a == ap ? 0.0 : p.κ for a in 1:n_occ, ap in 1:n_occ]

    # RI occupation choice: logit at temperature λ over (flow + λ·log q), with
    # the switching-cost friction C. `flow` reads the income grid value and the
    # occupation premium; `λ·log q(a)` reads the env-carried prior. Both are
    # V-additive destination payoffs, so they ride in through the UtilityStage.
    choice = LogitUtilityStage(layout;
        axis        = :occupation,
        cost_matrix = C,
        ε           = FromEnv(:λ),       # the Shannon cost λ IS the logit temperature; track env.λ
        utility     = (; occupation, income, env) -> (a = occ_index[occupation];
                                      env.λ * log(env.q[a]) +
                                      flow_payoff(income, p.premium[a])))

    discount = TimeDiscountingStage(layout; β = p.β)

    # Occupation-dependent income draw: the occupation axis is a spectator dep
    # selecting which row-stochastic income transition applies. Constant-but-
    # occupation-indexed matrices handed to an existing MarkovStage — data.
    income_draw = MarkovStage(layout;
        axis = :income,
        transition_matrix = (; occupation, env) -> p.P_by_occ[occ_index[occupation]])

    hh = choice ∘ discount ∘ income_draw
    return define_moments!(hh;
        mean_income     = at_end(integrand = :income, reduce = sum),
        career_share    = at_end(integrand = (; occupation) -> occupation === :career    ? 1.0 : 0.0, reduce = sum),
        safe_share      = at_end(integrand = (; occupation) -> occupation === :safe       ? 1.0 : 0.0, reduce = sum),
        persistent_share= at_end(integrand = (; occupation) -> occupation === :persistent ? 1.0 : 0.0, reduce = sum),
    )
end


# RI choice probabilities (read off the seated logit kernel) #
#------------------------------------------------------------#

"""
Recover the per-state occupation choice probabilities `P(a′ | income, a)` from
the seated logit kernel after a solve. The household chain flattens
`LogitUtilityStage ∘ Discount ∘ IncomeDraw` to leaf stages
`[LogitChoice, Utility, Discount, Markov]`, so the logit is `stages[1]`; its
Gibbs operator gives `π(a′ | i, s) = eψC[i, a′]·value_weight[a′, s]/normalizer[i, s]`
(here the off-occupation state `s` indexes income). Returns an
`(income, origin-occ, dest-occ)` array.
"""
function ri_choice_probs(hh)
    k = hh.buffer.stages[1].kernel
    n_inc, n_occ = size(k.value_weight)        # value_weight is (income, occupation)-shaped
    eC = reshape(parent(k.eψC), n_occ, n_occ)  # eψC[origin-occ, dest-occ]
    return [eC[i, j] * k.value_weight[s, j] / k.normalizer[s, i]
            for s in 1:n_inc, i in 1:n_occ, j in 1:n_occ]
end
