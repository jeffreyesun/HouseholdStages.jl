# Start-and-end closures → matrix sources #
#=========================================#
# A payoff on a transition is written `payoff(state_before, state_after; deps…, env)`, and returns
# `-Inf` on an infeasible transition.

"""
Turn a `payoff` on transitions into a matrix source on `axis`: a `DepClosure` building
`M[after, before] = payoff(g_origin[before], g_dest[after]; deps…)`, `(n_dest, n_origin)`-shaped.
"""
function to_matrix_source(payoff, start_layout::GriddedLayout, end_layout::GriddedLayout, axis::Symbol)
    g_origin, g_dest = collect(axis_grid(start_layout, axis)), collect(axis_grid(end_layout, axis))
    deps    = _closure_deps(payoff, start_layout)
    env_dep = _closure_env_dep(payoff)
    return DepClosure(deps, env_dep) do nt
        [payoff(g_origin[before], g_dest[after]; nt...)
         for after in eachindex(g_dest), before in eachindex(g_origin)]
    end
end
