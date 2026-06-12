# Logit utility — the general state-dependent discrete-choice stage, built as a
# composition `LogitChoiceStage ∘ UtilityStage`. The UtilityStage adds a
# destination payoff `u(cell; env)` to V_end; the LogitChoiceStage then does the
# Gumbel logit over `(V_end + u)` with the origin→destination cost matrix. All
# the work is in logit_choice.jl and utility.jl; this file is the named
# composition. See examples/logit_utility_composition.jl for the closed form.

"""
Logit discrete choice with a **state-dependent destination payoff** — a
[`LogitChoiceStage`](@ref) composed after a [`UtilityStage`](@ref):

    LogitUtilityStage = LogitChoiceStage(cost_matrix, ε) ∘ UtilityStage(utility)

so that `backward!` adds `utility(cell; env)` to `V_end`, then the logit chooses
over `(−C[i,j] + u(j,s) + V_end[j,s]) / ε`. `LogitChoiceStage` has no general
per-action payoff closure by design: any payoff depending on the *destination*
cell is V-additive and enters through the `UtilityStage`; only the
origin-dependent friction lives on `cost_matrix`.
"""
LogitUtilityStage(layout::GriddedLayout; choice_axis::Symbol,
                  cost_matrix::AbstractMatrix, utility, ε=1.0) =
    LogitChoiceStage(layout; choice_axis, cost_matrix, ε) ∘
    UtilityStage(layout; utility)
