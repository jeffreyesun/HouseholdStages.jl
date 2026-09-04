# Moments — ⟨f, Λ⟩ on the Sources backbone #
#==========================================#
# A moment is `⟨f, Λ⟩` for a state function `f` given as an integrand Source.

"""
One moment to emit: an integrand Source — a dep closure `(; ax…[, env]) -> value`, a constant, or a
`FromEnv` — mass-weighted by Λ and collapsed by `reduce`.
"""
struct MomentSpec{F, R}
    integrand :: F
    reduce    :: R
end

"A moment anchored at the end of the chain; a `Symbol` `integrand` is sugar for the axis accessor."
at_end(; integrand, reduce) = MomentSpec(_wrap_integrand(integrand), reduce)

_wrap_integrand(f)         = f
_wrap_integrand(s::Symbol) = DepClosure(nt -> nt[s], (s,), false)

# Pure computation — Sources + Layout + Λ #
#-----------------------------------------#

"""
A moment integrand Source over the end layout: full-size on its declared deps, singleton elsewhere.
"""
_integrand_grid(end_layout::GriddedLayout, source, env) =
    _integrand_grid(end_layout, source, Val(declared_deps(source, end_layout)),
                    Val(reads_env(source)), env)

# The moment integrand's function barrier.
function _integrand_grid(end_layout::GriddedLayout, source::S, deps::Val{D}, envdep::Val, env) where {S, D}
    dep_sizes = map(a -> _axis_size(end_layout, a), D)
    vals      = nothing
    for (idx, combo) in dep_combos(end_layout, deps)
        v = evaluate(source, combo, env, envdep)
        vals === nothing && (vals = Array{typeof(v)}(undef, dep_sizes...))
        vals[idx...] = v
    end
    names = axisnames(end_layout)
    full  = ntuple(i -> names[i] in D ? _axis_size(end_layout, names[i]) : 1, length(names))
    return reshape(vals, full)
end

"""
The moment `⟨f, Λ⟩`: the integrand over the end layout, weighted by `Λ` and collapsed by `reduce`.
"""
compute_moment(end_layout::GriddedLayout, spec::MomentSpec, Λ, env) =
    spec.reduce(_integrand_grid(end_layout, spec.integrand, env) .* Λ)
