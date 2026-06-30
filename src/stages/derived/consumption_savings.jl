# Consumption-savings — domain-named sugar over ArgmaxStage on the wealth (or any chosen) axis.

"""
Consumption-savings choice on the `axis` grid — an `ArgmaxStage` whose reward is the
`(after, before)` matrix `U[after, before] = utility(cell, b_in − b_end)`, consumption
`c = b_in − b_end ≤ 0` infeasible. `monotone_search` assumes the policy is non-decreasing in
`axis` (verified each `backward!` unless `assume_monotone = true`).

`utility` keeps its positional whole-cell `(cell, c; env)` signature; since the wrapper can't infer
which `cell` fields it reads, pass `utility_axes` for any state beyond `axis` it depends on (those
become the reward's deps), else `cell` carries only the `axis` coordinate.
"""
function ConsumptionSavingsStage(layout::GriddedLayout; β, utility,
                                 axis::Symbol=:wealth,
                                 utility_axes::Union{Nothing, NTuple{N, Symbol} where N}=nothing,
                                 monotone_search::Symbol=:divide_conquer,
                                 assume_monotone::Bool=false)
    # The reward is a start-and-end dep closure on the operative `axis`, lowered to a matrix source by
    # `to_matrix_source` (end-goal §4.1). Its declared deps are the extra `utility_axes` (the operative
    # axis is the matrix's two dims, never a dep); none by default (consumption-only utility).
    extra   = utility_axes === nothing ? () : Tuple(filter(!=(axis), utility_axes))
    # Start-and-end payoff forming `c = state_before − state_after` (wealth − wealth_next) and masking
    # infeasible `c ≤ 0` to `typemin` (the budget-constraint sentinel). The operative pair is POSITIONAL
    # (`b`, `a`); the extra `utility_axes` are runtime symbols, so the payoff rides a `DepClosure`
    # carrying them (plus `env`) as type params for the Sources API — `nt` is its dep/env NamedTuple.
    payoff  = DepClosure(extra, true) do b, a, nt
        c        = b - a
        cell     = merge(NamedTuple{(axis,)}((b,)), NamedTuple{extra}(nt))
        feasible = c > 0
        u        = utility(cell, feasible ? c : oneunit(c); env = nt.env)   # `oneunit` when infeasible: keeps the utility's eltype
        feasible ? u : typemin(typeof(u))                                   # mask infeasible consumption to `typemin`
    end
    # Discount is its own stage (end-goal §1): the argmax solves `max(reward + V_end)`, with `β·V_end`
    # supplied by a `TimeDiscountingStage` composed BEFORE the argmax in the value (backward) direction.
    # `∘` is time-ordered, so the backward sweep runs the discount first — reproducing the fused
    # `max(reward + β·V_end)` exactly. `β` keyword preserved for the caller.
    return ArgmaxStage(layout; reward = to_matrix_source(payoff, layout, axis), axis,
                       search = monotone_search, assume_monotone) ∘ TimeDiscountingStage(layout; β)
end
