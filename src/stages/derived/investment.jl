"""call `f(value)`, passing `env` only if `has_env` says the closure declared it."""
_call_env(f, value, has_env::Bool, env) = has_env ? f(value; env) : f(value)

"""
Investment in a productive stock — an [`ArgmaxStage`](@ref) on the capital axis `axis` with
`reward[h', h] = production(h) − effort_cost(max(h' − (1−δ)h, 0))` at `δ = depreciation`, discounted
at `β`. `production` and `effort_cost` take one value, plus `env` if they declare it. Ben-Porath
(1967) human-capital investment is the canonical instance; Grossman (1972) health investment is the
same stage on a health stock.
"""
function CapitalInvestmentStage(start_layout::GriddedLayout,
                                end_layout::GriddedLayout = start_layout;
                                axis::Symbol=:h, β=1.0,
                                production, effort_cost, depreciation::Real=0.0)
    prod_env = _reads_env_declared(production)
    cost_env = _reads_env_declared(effort_cost)
    gross(h, h′) = max(h′ - (1 - depreciation) * h, zero(h))
    payoff = if prod_env || cost_env
        (h, h′; env) -> _call_env(production, h, prod_env, env) - _call_env(effort_cost, gross(h, h′), cost_env, env)
    else
        (h, h′) -> production(h) - effort_cost(gross(h, h′))
    end
    reward = to_matrix_source(payoff, start_layout, end_layout, axis)
    return ArgmaxStage(start_layout, end_layout; reward, axis = axis) ∘
           TimeDiscountingStage(end_layout; β)
end

"""
Smooth durable adjustment — an [`ArgmaxStage`](@ref) on a durable-stock `axis` with
`reward[d', d] = service(d') − adjustment_cost(d' − d)`, discounted at `β`.
"""
function DurableAdjustmentStage(start_layout::GriddedLayout,
                                end_layout::GriddedLayout = start_layout;
                                axis::Symbol=:durable, β=1.0,
                                service, adjustment_cost)
    serv_env = _reads_env_declared(service)
    cost_env = _reads_env_declared(adjustment_cost)
    payoff = if serv_env || cost_env
        (d, d′; env) -> _call_env(service, d′, serv_env, env) - _call_env(adjustment_cost, d′ - d, cost_env, env)
    else
        (d, d′) -> service(d′) - adjustment_cost(d′ - d)
    end
    reward = to_matrix_source(payoff, start_layout, end_layout, axis)
    return ArgmaxStage(start_layout, end_layout; reward, axis = axis) ∘
           TimeDiscountingStage(end_layout; β)
end
