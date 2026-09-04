"""build the `cell` the utility sees — the `axis` coordinate `b` plus the `extra` axes read off `nt`."""
@inline _cs_cell(::Val{axis}, ::Val{extra}, b, nt) where {axis, extra} =
    merge(NamedTuple{(axis,)}((b,)), NamedTuple{extra}(nt))

"""
Consumption-savings choice — a [`ContinuousArgmaxStage`](@ref) on the `axis` grid picking
next-period wealth `a'` to maximise `utility(cell, c) + β·V_end(a')`, with `c` this period's wealth
minus `a'` and `c ≤ 0` masked out. `utility` takes `(cell, c)` positionally, optionally with
`; env`, and must be supermodular; name any state beyond `axis` that it reads in `utility_axes`.
"""
function ConsumptionSavingsStage(start_layout::GriddedLayout,
                                 end_layout::GriddedLayout = start_layout; β, utility,
                                 axis::Symbol=:wealth,
                                 utility_axes::Union{Nothing, NTuple{N, Symbol} where N}=nothing,
                                 skip_monotonicity_check::Bool=false)
    extra    = utility_axes === nothing ? () : Tuple(filter(!=(axis), utility_axes))
    uses_env = _reads_env_declared(utility)
    axv, exv = Val(axis), Val(extra)
    payoff   = DepClosure(extra, uses_env) do b, a, nt
        c        = b - a
        feasible = c > 0
        cc       = feasible ? c : oneunit(c)                                # dummy consumption when infeasible, to keep the utility's return type
        cell     = _cs_cell(axv, exv, b, nt)
        u        = uses_env ? utility(cell, cc; env = nt.env) : utility(cell, cc)
        feasible ? u : typemin(typeof(u))
    end
    return ContinuousArgmaxStage(start_layout, end_layout; reward = payoff, axis,
                                 skip_monotonicity_check) ∘
           TimeDiscountingStage(end_layout; β)
end
