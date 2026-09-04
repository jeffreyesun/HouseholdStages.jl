# Primal reads #
#--------------#

"The plain value of `x`, from the bottom of a nested `Dual`; the identity on any other `Real`."
@inline _frz(x::ForwardDiff.Dual) = _frz(ForwardDiff.value(x))
@inline _frz(x::Real)             = x

"`env` with every numeric entry stripped to its plain value; a non-`NamedTuple` `env` passes through untouched."
_frz_env(env::NamedTuple) = map(_frz_entry, env)
_frz_env(env) = env

"One `env` entry stripped to its plain value, recursing into nested `NamedTuple`s and arrays."
@inline _frz_entry(v::Real)                                = _frz(v)
@inline _frz_entry(v::NamedTuple)                          = _frz_env(v)
@inline _frz_entry(v::AbstractArray{<:ForwardDiff.Dual})   = _frz.(v)
@inline _frz_entry(v)                                      = v

"Whether buffers of eltype `T` carry ForwardDiff derivatives; `false` for any other eltype."
carries_tangents(::Type)                      = false
carries_tangents(::Type{<:ForwardDiff.Dual})  = true

"""
The underlying primal float type of `T`, peeling any ForwardDiff tangent lanes: `Float64 -> Float64`,
`Float32 -> Float32`, `Dual{tag,V,N} -> primal_eltype(V)`. Coordinate grids collect at this type so
they track the model's working precision while staying tangent-free — a `Dual` grid would collide the
continuous solvers' nested-Dual probe tags.
"""
primal_eltype(::Type{T}) where {T} = T
primal_eltype(::Type{T}) where {T<:ForwardDiff.Dual} = primal_eltype(ForwardDiff.valtype(T))

"""
Refuse a field fill that would seat a tangent-carrying value (`Src`) into a buffer that cannot hold
one (`Dst`). The type-level branch folds away unless a `Dual` value reaches a plain buffer — the
signature of pushing a `Dual` through `env` into a block built at a non-`Dual` eltype. Differentiation
goes through a block rebuilt at the Dual eltype: `lift_jacobian(stage)` or `with_eltype(stage, T)`.
"""
@inline function _assert_seatable_fill(::Type{Src}, ::Type{Dst}) where {Src, Dst}
    carries_tangents(Src) && !carries_tangents(Dst) && error(
        "HouseholdStages: a field fill produced a $(Src) value for a buffer of eltype $(Dst), which " *
        "cannot carry ForwardDiff tangents. A Dual reached a block built at a non-Dual eltype — the " *
        "usual cause is feeding a Dual through `env` into a block that was built at $(Dst). To " *
        "differentiate, rebuild the block at the Dual eltype first: `lift_jacobian(stage)` or " *
        "`with_eltype(stage, T)`, then seed the tangents there.")
    return nothing
end

# Closure dependency discovery: a closure's declared keyword names are its deps — a layout axis name,
# `:env`, or a stage-specific role the caller passes in `reserved`.

"The level/grid count along the named axis."
_axis_size(layout::GriddedLayout, a::Symbol) = axissize(layout.axes[axis_position(layout, a)])

"Reshape grid `v` to a rank-`N` tuple with its length at dim `k`, 1 elsewhere."
_reshape_singleton(v, k::Int, ::Val{N}) where {N} =
    reshape(v, ntuple(i -> i == k ? length(v) : 1, Val(N)))

"The declared kwargs of a single-method closure. Errors on a multi-method closure or a `; kwargs...` slurp."
function _closure_kwargs_raw(closure)
    ms = collect(methods(closure))
    length(ms) == 1 ||
        error("dependency closure must have a single method; got $(length(ms)).")
    kws = Base.kwarg_decl(first(ms))
    any(k -> endswith(String(k), "..."), kws) &&
        error("dependency closure slurps keyword args (`; kwargs...`); declare the explicit deps.")
    return Tuple(kws)
end

"Dep axes from a closure's declared `kws`: drop the `reserved` roles and `:env`, and return the rest in layout order."
function _closure_deps_from_kwargs(kws::Tuple, names::NTuple{N, Symbol}; reserved::Tuple = ()) where {N}
    deps = filter(k -> !(k in reserved) && k != :env, kws)
    for a in deps
        a in names ||
            error("dependency closure kwarg `$a` is not a layout axis ($(names)), `:env`, nor reserved $(reserved).")
    end
    return Tuple(a for a in names if a in deps)
end

"The dep axes a closure varies along, in the order `dep_layout` names them."
_closure_deps(closure, dep_layout::GriddedLayout; reserved::Tuple = ()) =
    _closure_deps_from_kwargs(_closure_kwargs_raw(closure), axisnames(dep_layout); reserved)

"Whether a closure declares the `:env` kwarg."
_closure_env_dep(closure) = :env in _closure_kwargs_raw(closure)

"Whether `fn` declares an `env` kwarg; `false` for `nothing`, an absent optional sub-closure."
function _reads_env_declared(fn)
    fn === nothing && return false
    try
        return _closure_env_dep(fn)
    catch e
        error("HouseholdStages: could not introspect the env-dependence of a user closure " *
              "($(fn)). Write it as a SINGLE-method closure with explicit keyword args — no " *
              "`; kwargs...` slurp and no multiple methods — e.g. `(cell, c; env) -> …` (reads env) " *
              "or `(cell, c) -> …` (does not). Underlying: $(e)")
    end
end

# Refill instrumentation (opt-in) #
#---------------------------------#
# A module-level count of field fills, bumped once per `fill_field!` or `fill_scalar_field!` call.
const REFILL_COUNT = Ref(0)

"Reset the field-refill counter to zero, before a measured run."
reset_refill_count!() = (REFILL_COUNT[] = 0)

"The number of matrix and scalar field fills since the last reset."
refill_count() = REFILL_COUNT[]

# Dynamic-axis dep closures #
#---------------------------#

"""
A dep closure whose declared axes are chosen at runtime: `f` takes the evaluated `NamedTuple`
(reading `nt.<axis>` and `nt.env`), with the axis names and env-dependence in the type parameters.
"""
struct DepClosure{Axes, Env, F} <: Function
    f :: F
end

DepClosure(f, axes::NTuple{N, Symbol}, env::Bool) where {N} = DepClosure{axes, env, typeof(f)}(f)

(c::DepClosure)(; kwargs...) = c.f(values(kwargs))
# A payoff on a transition takes the operative axis's before and after grid values positionally.
(c::DepClosure)(before, after; kwargs...) = c.f(before, after, values(kwargs))

_closure_kwargs_raw(::DepClosure{Axes, Env}) where {Axes, Env} = Env ? (Axes..., :env) : Axes
_closure_env_dep(::DepClosure{Axes, Env}) where {Axes, Env} = Env

# The Sources API #
#-----------------#
# A stage describes a matrix or scalar array by a *source*: a constant, a `FromEnv(:key)`, a dep
# closure `(; dep…[, env]) -> value`, or a `MappedField` wrapping one of these. `evaluate`,
# `declared_deps`, `reads_env` and `dep_combos` are where those forms are told apart.

"""
Evaluate a source at one combination of its dep axes: a constant or `FromEnv` resolves against `env`,
a dep closure is called with `combo`, a `NamedTuple` of axis ⇒ value, plus `env` if it declares it.
"""
evaluate(src, _combo, env) = resolve(src, env)
evaluate(fn::Function, combo, env) = evaluate(fn, combo, env, Val(_closure_env_dep(fn)))

# The four-argument form carries the source's env-dependence as a `Val`, decided once per fill.
evaluate(src, combo, env, ::Val) = evaluate(src, combo, env)
evaluate(fn::Function, combo, env, ::Val{true})  = fn(; combo..., env)
evaluate(fn::Function, combo, env, ::Val{false}) = fn(; combo...)

"The dep axes a source varies along, in the order `dep_layout` names them; `()` unless the source is a dep closure."
declared_deps(::Any, ::GriddedLayout) = ()
declared_deps(fn::Function, dep_layout::GriddedLayout) = _closure_deps(fn, dep_layout)

"Whether a source reads `env`; an unrecognised source type answers `true`."
reads_env(::Any) = true
reads_env(::Union{Number, AbstractArray}) = false
reads_env(fn::Function) = _closure_env_dep(fn)

"Iterate `(idx, combo)` over the cartesian product of the dep grids: `idx` the trailing buffer index as a tuple, `combo` the `NamedTuple` of axis ⇒ grid value; one empty pair when there are no deps."
function dep_combos(dep_layout::GriddedLayout, ::Val{deps}) where {deps}
    grids = map(a -> axis_grid(dep_layout, a), deps)
    return ((Tuple(ci), NamedTuple{deps}(map(getindex, grids, Tuple(ci))))
            for ci in CartesianIndices(map(length, grids)))
end
