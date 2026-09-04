"""
Default choice (Eaton–Gersovitz; Chatterjee–Corbae–Nakajima–Ríos-Rull) — an [`ArgmaxStage`](@ref)
over a status `axis` scoring `−default_penalty` (a number
or a `(; env)` closure) at `default_index` and `0` elsewhere, with moves gated by
`avail(before, after)` on status indices.
"""
function DefaultStage(layout::GriddedLayout; axis::Symbol=:status,
                      default_index::Int=2, default_penalty=0.0,
                      avail=(before, after) -> true)
    grid   = axis_grid(layout, axis)
    dflt   = grid[default_index]
    idx(v) = findfirst(==(v), grid)
    payoff = if default_penalty isa Function && _reads_env_declared(default_penalty)
        (before, after; env) -> avail(idx(before), idx(after)) ? (after == dflt ? -default_penalty(; env) : 0.0) : -Inf
    else
        pen = default_penalty isa Function ? default_penalty() : default_penalty
        (before, after) -> avail(idx(before), idx(after)) ? (after == dflt ? -pen : 0.0) : -Inf
    end
    return ArgmaxStage(layout; axis = axis, reward = to_matrix_source(payoff, layout, layout, axis))
end
