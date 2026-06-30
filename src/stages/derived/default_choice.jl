# Default / bankruptcy — domain-named sugar over ArgmaxStage (a gated discrete choice on a
# repay/default status axis). The state consequences of defaulting (wealth reset, exclusion) are
# NOT applied here — like BuyHomeStage's price, they compose as following stages (a
# WealthChangeStage resetting `b`, a MarkovStage into exclusion).

"""
Default choice (Eaton–Gersovitz / Chatterjee–Corbae–Nakajima–Ríos-Rull) — an [`ArgmaxStage`](@ref)
over a status `axis` whose levels are the repay/default options (`default_index = 2`). Choosing
`default` scores `−default_penalty` (a number, or a `(; env)` closure), repay scores `0`; the
continuation `V_end` at each status carries that branch's value (set up by the stages composed
after). `avail(before, after)` gates moves — e.g. an excluded state barred from re-defaulting; the
default admits every move.
"""
function DefaultStage(layout::GriddedLayout; axis::Symbol=:status,
                      default_index::Int=2, default_penalty=0.0,
                      avail=(before, after) -> true)
    # Canonical §4.1 lowering (matching buy/sell): a start-and-end `payoff` taking the origin/
    # destination status GRID VALUES POSITIONALLY, gating inline (`avail` infeasible → `-Inf`, else
    # default scores `−penalty`, repay `0`). `avail` keeps its PUBLIC index API `(before, after)` via a
    # thin value→index adapter (`idx`, exact on the distinct status grid).
    grid   = axis_grid(layout, axis)
    dflt   = grid[default_index]
    idx(v) = findfirst(==(v), grid)
    pen(env) = default_penalty isa Function ? default_penalty(; env) : default_penalty
    payoff = (before, after; env) ->
        avail(idx(before), idx(after)) ? (after == dflt ? -pen(env) : 0.0) : -Inf
    return ArgmaxStage(layout; axis = axis, reward = to_matrix_source(payoff, layout, axis))
end
