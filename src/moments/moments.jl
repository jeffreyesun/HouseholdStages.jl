# Moments — ⟨f, Λ⟩ on the Sources backbone (end-goal §7) #
#=======================================================#
# A moment is `⟨f, Λ⟩` for a state function `f`, with `f` an integrand SOURCE (§4): a dep
# closure `(; ax…[, env]) -> value`, a constant, or a `FromEnv`. Computation needs only a
# Source, a Layout, and Λ — never a stage or kernel — so it rides the same closure backbone
# (`evaluate`/`declared_deps`/`dep_combos`, sources/sources.jl) as field contents and choice costs.
#
# This file is the PURE computation layer (§7): `MomentSpec`, `at_end`, and
# `compute_moment(layout, moment, Λ, env)` — Sources/Layout/Λ only, no `ChainStage` anywhere. The
# combinator-layer ATTACHMENT MENU (`define_moment!`, `compute_moments(::ChainStage…)`) lives in
# combinators/moment_menu.jl, loaded after the stage algebra it depends on.

# Integrand source + spec #
#-------------------------#

"""
One moment to emit: an integrand SOURCE (§4) — a dep closure `(; ax…[, env]) -> value`, a
constant, or a `FromEnv` — mass-weighted by Λ and collapsed by `reduce`. Construct via `at_end`.

`reduce = sum` is the AGGREGATE `Σ x·λ` (e.g. `K = ∫ wealth dΛ`), correct under non-unit mass
since Λ need not sum to 1 (see `population.jl` header). For a per-capita mean, divide by `sum(Λ)`
yourself: `reduce = mean` divides by the cell COUNT, not `Σ(Λ)`, so it is a mean only when every
cell carries equal mass.
"""
struct MomentSpec{F, R}
    integrand :: F
    reduce    :: R
end

"""
A moment anchored at the end of the chain. A `Symbol` `integrand` is sugar for the axis
accessor (`:wealth` → the dep source `(; wealth) -> wealth`).
"""
at_end(; integrand, reduce) = MomentSpec(_wrap_integrand(integrand), reduce)

# Integrand may be a Source (closure / constant / `FromEnv`) or a `Symbol` axis shortcut. The shortcut
# `:wealth` is the dep-source equivalent of `(; wealth) -> wealth`: an env-independent `DepClosure`
# declaring the single (runtime-named) axis and evaluating to that axis's value.
_wrap_integrand(f)         = f
_wrap_integrand(s::Symbol) = DepClosure(nt -> nt[s], (s,), false)

# Pure computation — Sources + Layout + Λ, no ChainStage (§7) #
#------------------------------------------------------------#

"""
Materialize a moment integrand Source over `layout` as an array broadcastable against a
layout-shaped Λ. Mirrors a dep-closure field's fill: `evaluate` the Source at each
dep-combination (`dep_combos`), store on the compact dep grid, then reshape with singleton
dims on the non-dep axes so it broadcast-expands to the full grid. A constant / `FromEnv`
(no deps) collapses to an all-singleton array that still broadcasts to Λ's shape.
"""
function _integrand_grid(layout::GriddedLayout, source, env)
    deps      = declared_deps(source, layout)
    dep_sizes = map(a -> _axis_size(layout, a), deps)
    vals      = nothing
    for (idx, combo) in dep_combos(layout, deps)
        v = evaluate(source, combo, env)
        vals === nothing && (vals = Array{typeof(v)}(undef, dep_sizes...))
        vals[idx...] = v
    end
    names = axisnames(layout)
    full  = ntuple(i -> names[i] in deps ? _axis_size(layout, names[i]) : 1, length(names))
    return reshape(vals, full)
end

"""
Pure moment computation (§7): the moment `⟨f, Λ⟩` for integrand Source `f = spec.integrand`,
mass-weighted by `Λ` and collapsed by `spec.reduce`. Depends only on a Layout, a Source, and Λ —
no stage or kernel (see the `MomentSpec` header for the non-unit-mass semantics).
"""
compute_moment(layout::GriddedLayout, spec::MomentSpec, Λ, env) =
    spec.reduce(_integrand_grid(layout, spec.integrand, env) .* Λ)
