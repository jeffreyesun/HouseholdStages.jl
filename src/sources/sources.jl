# Closure dependency discovery — the shared backbone for declared-dependency
# flexibility. A user closure declares which axes it varies along by NAMING them as
# keyword arguments; `Base.kwarg_decl` reads those names off the method signature.
# A kwarg named after a layout axis means "varies along that axis"; `:env` means
# "depends on env"; any name in `reserved` is a stage-specific role (e.g. the logit
# cost's `:choice` destination value, or its choice axis read as the origin) that is
# neither a stored dep axis nor `:env`. Used by the stratified-kernel cost/transition
# closures (e.g. Logit's `cost_matrix`) for their declared deps.



"Layout query: the level/grid count along the named axis."
_axis_size(layout::GriddedLayout, a::Symbol) = axissize(layout.axes[axis_position(layout, a)])
# The coordinate values along a named axis are the public `axis_grid` (gridded_layout.jl).

"Reshape grid `v` to a rank-`N` tuple with its length at dim `k`, 1 elsewhere."
_reshape_singleton(v, k::Int, ::Val{N}) where {N} =
    reshape(v, ntuple(i -> i == k ? length(v) : 1, Val(N)))

"""
The raw declared kwargs of a single-method closure, via `Base.kwarg_decl` (no layout
validation). Errors on a multi-method closure or a `; kwargs...` slurp — both would
make the declared deps unknowable.
"""
function _closure_kwargs_raw(closure)
    ms = collect(methods(closure))
    length(ms) == 1 ||
        error("dependency closure must have a single method; got $(length(ms)).")
    kws = Base.kwarg_decl(first(ms))
    any(k -> endswith(String(k), "..."), kws) &&
        error("dependency closure slurps keyword args (`; kwargs...`); declare the explicit deps.")
    return Tuple(kws)
end

"""
Dep axes from a closure's declared `kws`: drop the `reserved` roles and `:env`,
validate the rest name layout axes (typo-catching), return them in layout order.
The kwargs are passed in (rather than re-read) so a stage with a synthesised closure
can supply them through its own `_..._declared_kwargs` override.
"""
function _closure_deps_from_kwargs(kws::Tuple, names::NTuple{N, Symbol}; reserved::Tuple = ()) where {N}
    deps = filter(k -> !(k in reserved) && k != :env, kws)
    for a in deps
        a in names ||
            error("dependency closure kwarg `$a` is not a layout axis ($(names)), `:env`, nor reserved $(reserved).")
    end
    return Tuple(sort(collect(deps); by = a -> findfirst(==(a), names)))
end

"The dep axes a closure varies along (declared kwargs minus `reserved` and `:env`), in layout order."
_closure_deps(closure, layout::GriddedLayout; reserved::Tuple = ()) =
    _closure_deps_from_kwargs(_closure_kwargs_raw(closure), axisnames(layout); reserved)

"Whether a closure declares the `:env` kwarg (so it re-materialises when env changes)."
_closure_env_dep(closure) = :env in _closure_kwargs_raw(closure)

# Dynamic-axis dep closures #
#---------------------------#

"""
A dep closure with DYNAMIC declared axes: it carries its axis names (`Axes`) and env-dependence
(`Env`) as type params so the Sources API can introspect them, wrapping an ordinary function `f` of
the evaluated `NamedTuple` (which reads `nt.<axis>` / `nt.env`). Use where the declared kwargs are
runtime symbols and a literal `(; ax…) -> …` closure can't be written — `Base.kwarg_decl` reads
nothing off such a closure, so `_closure_kwargs_raw` / `_closure_env_dep` are overridden from the
type params instead. Construct as `DepClosure(f, axes, env)`.
"""
struct DepClosure{Axes, Env, F} <: Function
    f :: F
end

DepClosure(f, axes::NTuple{N, Symbol}, env::Bool) where {N} = DepClosure{axes, env, typeof(f)}(f)

(c::DepClosure)(; kwargs...) = c.f(values(kwargs))
# Positional operative pair (start-and-end payoffs, sources/to_matrix_source.jl): the two operative
# grid values lead, the declared deps (+`env`) trail as kwargs — so `f` is `(before, after, nt) -> …`.
(c::DepClosure)(before, after; kwargs...) = c.f(before, after, values(kwargs))

_closure_kwargs_raw(::DepClosure{Axes, Env}) where {Axes, Env} = Env ? (Axes..., :env) : Axes
_closure_env_dep(::DepClosure{Axes, Env}) where {Axes, Env} = Env

# The Sources API — the single discrimination over source forms (end-goal §4) #
#----------------------------------------------------------------------------#
# A stage describes a matrix/scalar array by a *source*: a constant, a `FromEnv(:key)`, a dep
# closure `(; dep…[, env]) -> value`, or a `MappedField` (fields/matrix_field.jl) wrapping one of
# these. `evaluate`/`declared_deps`/`reads_env`/`dep_combos` are the single
# discrimination over those forms — what field contents, choice costs (logit), and moments all build
# on. (`resolve`/`FromEnv` live with the stage protocol in stages/abstract.jl; the bodies here bind
# them lazily.)

"Evaluate a source at one dep-combination — THE single discrimination over source forms: a constant returns itself and a `FromEnv` is looked up in `env` (both via `resolve`); a dep closure is called with `combo` (a `NamedTuple` axis ⇒ value) plus `env` iff it declares it. (The legacy whole-cell `(cell; env)` closure is not a form here.)"
evaluate(src, _combo, env) = resolve(src, env)
evaluate(fn::Function, combo, env) = _closure_env_dep(fn) ? fn(; combo..., env) : fn(; combo...)

"The dep axes a source varies along, in layout order — `()` for a constant / `FromEnv`, a dep closure's declared (non-`env`) axes otherwise."
declared_deps(::Any, ::GriddedLayout) = ()
declared_deps(fn::Function, layout::GriddedLayout) = _closure_deps(fn, layout)

"Whether a source reads `env` (drives the refill policy): `false` for a constant `Number`/`AbstractArray`, `true` for `FromEnv` and (conservatively) custom sources; for a closure, whether it declares `:env`."
reads_env(::Any) = true
reads_env(::Union{Number, AbstractArray}) = false
reads_env(fn::Function) = _closure_env_dep(fn)

"Iterate `(idx, combo)` over the cartesian product of the dep grids: `idx` the trailing buffer index (a tuple), `combo` the `NamedTuple` of axis ⇒ grid value. One empty pair when there are no deps."
function dep_combos(layout::GriddedLayout, deps::NTuple{D, Symbol}) where {D}
    grids = map(a -> axis_grid(layout, a), deps)
    return ((Tuple(ci), NamedTuple{deps}(map((g, i) -> g[i], grids, Tuple(ci))))
            for ci in CartesianIndices(map(length, grids)))
end
