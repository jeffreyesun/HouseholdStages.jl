# Closure dependency discovery — the shared backbone for declared-dependency
# flexibility. A user closure declares which axes it varies along by NAMING them as
# keyword arguments; `Base.kwarg_decl` reads those names off the method signature.
# A kwarg named after a layout axis means "varies along that axis"; `:env` means
# "depends on env"; any name in `reserved` is a stage-specific role (e.g. the logit
# cost's `:choice` destination value, or its choice axis read as the origin) that is
# neither a stored dep axis nor `:env`. Used by the stratified-kernel cost/transition
# closures and the argmax payoff alike.



"Layout query: the level/grid count along the named axis."
_axis_size(layout::GriddedLayout, a::Symbol) = axissize(layout.axes[axis_position(layout, a)])
"Layout query: the coordinate values (grid / levels) along the named axis."
_axis_grid(layout::GriddedLayout, a::Symbol) = axisvalues(layout.axes[axis_position(layout, a)])

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
