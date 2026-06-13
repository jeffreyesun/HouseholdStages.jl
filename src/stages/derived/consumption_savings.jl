# Consumption-savings — domain-named sugar over ContinuousArgmaxStage on the
# wealth axis. All the work lives in continuous_argmax.jl.

"""
Consumption-savings choice on the wealth grid — a `ContinuousArgmaxStage` with
payoff `utility(cell, b_in − b_end)`, consumption `c = b_in − b_end ≤ 0` infeasible.
`monotone_search` assumes the policy is non-decreasing in wealth (verified each
`backward!` unless `assume_monotone = true`).

`utility` keeps its positional `(cell, c)` signature, so the wrapper can't infer
which `cell` fields it reads: pass `utility_axes` for any state beyond wealth it
depends on, or that axis is silently filled with a dummy.

`utility` may optionally declare a trailing `; env` keyword. Omitting it marks the
payoff **env-independent** — the savings payoff `u(c)` usually is, since prices enter
the chain through a separate income stage — and the payoff table is then materialised
once and reused across `backward!` calls instead of rebuilt every call (a large saving
for VFI / training loops; see `ContinuousArgmaxStage.backward!`). Declare `; env` only
when the utility genuinely reads `env`.
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
struct _CSPayoff{Axes, EnvDep, U, W}
    utility     :: U
    wealth_axis :: W
end

# Does `utility` declare a `; env` kwarg? If so the payoff is env-dependent (refilled each
# backward); if not it is env-independent (table cached). Fall back to env-dependent — the
# safe always-refill default — for any utility the introspection can't read (e.g. multi-method).
_cs_env_dep(utility) = try _closure_env_dep(utility) catch; true end

_CSPayoff(utility, wealth_axis::Symbol, axes::Tuple) =
    _CSPayoff{axes, _cs_env_dep(utility), typeof(utility), typeof(wealth_axis)}(utility, wealth_axis)

@inline function (p::_CSPayoff{Axes, EnvDep})(; choice, env = nothing, kwargs...) where {Axes, EnvDep}
    cell = NamedTuple{Axes}(map(a -> kwargs[a], Axes))
    c    = getfield(cell, p.wealth_axis) - choice
    c > 0 || return -Inf
    return EnvDep ? p.utility(cell, c; env = env) : p.utility(cell, c)
end

# Dep discovery for the CS payoff: declare its kwargs from the type params, not the signature
# (the call site slurps the axis kwargs to stay generic over `Axes`). Overrides the shared
# `_closure_kwargs_raw` introspection, which would otherwise choke on the `kwargs...` slurp.
# `:env` is declared iff `utility` declares it, so env-independent payoffs report `env_dep = false`.
_closure_kwargs_raw(::_CSPayoff{Axes, EnvDep}) where {Axes, EnvDep} =
    EnvDep ? (:choice, Axes..., :env) : (:choice, Axes...)
