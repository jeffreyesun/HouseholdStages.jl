# Consumption-savings — domain-named sugar over ContinuousArgmaxStage on the
# wealth axis. All the work lives in continuous_argmax.jl.

"""
Consumption-savings choice on the wealth grid — a `ContinuousArgmaxStage` with
payoff `utility(cell, b_in − b_end)`, consumption `c = b_in − b_end ≤ 0` infeasible.
`monotone_search` assumes the policy is non-decreasing in wealth (verified each
`backward!` unless `assume_monotone = true`).

`utility` keeps its positional `(cell, c; env)` signature, so the wrapper can't infer
which `cell` fields it reads: pass `utility_axes` for any state beyond wealth it
depends on, or that axis is silently filled with a dummy.
"""
function ConsumptionSavingsStage(layout::GriddedLayout; β, utility,
                                 wealth_axis::Symbol=:wealth,
                                 utility_axes::Union{Nothing, NTuple{N, Symbol} where N}=nothing,
                                 monotone_search::Symbol=:divide_conquer,
                                 assume_monotone::Bool=false)
    # The payoff declares its input-axis deps explicitly (no IR walk). The wealth
    # axis is always a dep (the budget reads `b_in`); the rest come from the user's
    # `utility_axes`, defaulting to none (consumption-only utility) — the minimal
    # U table, matching the pre-refactor IR-walk result for `u(c)` and the GPU
    # backward's consumption-only fast path.
    declared = utility_axes === nothing ? () : utility_axes
    axes     = Tuple(unique((wealth_axis, declared...)))
    payoff   = _CSPayoff(utility, wealth_axis, axes)
    return ContinuousArgmaxStage(layout; payoff, choice_axis=wealth_axis, β,
                                 monotone_search, assume_monotone)
end

"""
The kwarg-signature payoff for `ConsumptionSavingsStage`: wraps the user's positional
`utility`, computing `c = cell.<wealth_axis> − choice` and masking `c ≤ 0` to `-Inf`.
The declared axes `Axes` ride in the type parameter so the payoff is synthesised over
runtime-chosen axes without an `eval`'d signature.
"""
struct _CSPayoff{Axes, U, W}
    utility     :: U
    wealth_axis :: W
end
_CSPayoff(utility, wealth_axis::Symbol, axes::Tuple) =
    _CSPayoff{axes, typeof(utility), typeof(wealth_axis)}(utility, wealth_axis)

@inline function (p::_CSPayoff{Axes})(; choice, env = nothing, kwargs...) where {Axes}
    cell = NamedTuple{Axes}(map(a -> kwargs[a], Axes))
    c    = getfield(cell, p.wealth_axis) - choice
    return c > 0 ? p.utility(cell, c; env = env) : -Inf
end

# Dep discovery for the CS payoff: declare its kwargs from the type `Axes`, not the signature
# (the call site slurps the axis kwargs to stay generic over `Axes`). Overrides the shared
# `_closure_kwargs_raw` introspection, which would otherwise choke on the `kwargs...` slurp.
_closure_kwargs_raw(::_CSPayoff{Axes}) where {Axes} = (:choice, Axes..., :env)
