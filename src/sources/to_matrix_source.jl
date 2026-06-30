# Start-and-end closures → matrix sources (end-goal §4.1) #
#========================================================#
# A reward/cost on a transition depends on the operative axis at BOTH its origin and its destination.
# The payoff closure takes those two operative grid VALUES POSITIONALLY — `payoff(state_before,
# state_after; deps…, env)` — its ONLY keyword args being the genuine other-axis deps and `env` (iff
# used). `to_matrix_source` lowers it into an ordinary matrix source the standard
# `matrix_field`/`fill_field!` path consumes unchanged: it returns a matrix-valued `DepClosure` that
# sweeps the operative grid over origin (the `(out, in)` matrix's `in` columns) and destination (its
# `out` rows), declaring the payoff's other axes as its deps. The operative axis is handed to
# `to_matrix_source` explicitly, so the payoff naming it would be redundant — hence positional. Gating
# is expressed inside the payoff (`avail ? value : -Inf`); there is no separate `feasible` path.

"""
Lower a start-and-end reward/cost `payoff` to a matrix source on `axis`. `payoff(state_before,
state_after; deps…, env)` takes the operative origin/destination as POSITIONAL grid values and
declares any other-axis deps (and `env`, iff used) as keyword args. The returned matrix-valued
`DepClosure` plugs straight into the matrix-field fill path: it builds the `(n_dest, n_origin)` face
`M[after, before] = payoff(g[before], g[after]; deps…)` by sweeping the operative grid `g`, and its
declared deps are the payoff's keyword axes (`env` drives the refill policy). General over `axis`.
"""
function to_matrix_source(payoff, layout::GriddedLayout, axis::Symbol)
    g       = collect(axis_grid(layout, axis))
    deps    = _closure_deps(payoff, layout)
    env_dep = _closure_env_dep(payoff)
    return DepClosure(deps, env_dep) do nt
        [payoff(g[before], g[after]; nt...) for after in eachindex(g), before in eachindex(g)]
    end
end
