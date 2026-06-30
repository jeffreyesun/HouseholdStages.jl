# Logit utility — the general state-dependent discrete-choice stage, built as a
# composition `LogitChoiceStage ∘ UtilityStage`. The UtilityStage adds a
# destination payoff `u(; dep…[, env])` to V_end; the LogitChoiceStage then does the
# Gumbel logit over `(V_end + u)` with the origin→destination cost matrix. All
# the work is in logit_choice.jl and utility.jl; this file is the named
# composition. See examples/logit_utility_composition.jl for the closed form.

"""
Logit discrete choice with a **state-dependent destination payoff** — a
[`LogitChoiceStage`](@ref) composed after a [`UtilityStage`](@ref): `backward!` adds
`utility(; dep…[, env])` to `V_end`, then the logit chooses over
`(−C[i,j] + u(j,s) + V_end[j,s]) / ε`. `LogitChoiceStage` has no per-action payoff closure by
design — destination-dependent payoffs are V-additive and enter through `UtilityStage`; only the
origin-dependent friction lives on `cost_matrix`.
"""
LogitUtilityStage(layout::GriddedLayout; axis::Symbol,
                  cost_matrix::AbstractMatrix, utility, ε=1.0) =
    LogitChoiceStage(layout; axis, cost_matrix, ε) ∘
    UtilityStage(layout; utility)
