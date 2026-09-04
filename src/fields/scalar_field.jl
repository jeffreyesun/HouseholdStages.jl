# ScalarField — one scalar per grid cell, held in a buffer that broadcasts against the gridded V/Λ.
# The source, which lives in the stage's Spec, may be a number or array constant, a `FromEnv` lookup,
# or a `(; dep…[, env])` closure giving one scalar per combination of the axes it declares; a
# closure's values are stored compactly, with size 1 on every axis it does not vary along.

"One scalar per grid cell, broadcastable against the gridded `V`/`Λ`."
mutable struct ScalarField{A, To}
    data          :: A        # the values: a compact dep-array, a scalar, or a full array
    bshape        :: Tuple    # reshape target that makes a compact `data` broadcast; `()` ⇒ use `data` as-is
    env_dependent :: Bool     # `reads_env(source)`, computed once
    to            :: To       # where a refill lands: `identity` in place, or an adaptor onto a device
end

"A `ScalarField` whose refills land in place, on whatever backing it was built with."
ScalarField{A}(data, bshape, env_dependent, to = identity) where {A} =
    ScalarField{A, typeof(to)}(data, bshape, env_dependent, to)

"Fill a fresh buffer with `NaN`; integer buffers get zeros instead."
_nanfill!(A::AbstractArray) = fill!(A, eltype(A) <: Integer ? zero(eltype(A)) : NaN)

"""
Build a `ScalarField` from a payoff `source`: a dep closure gets a compact dep-sized buffer, while a
scalar, array, or `FromEnv` source stores its resolved value directly. A source that reads `env` is
left NaN for the stage's `backward!` to fill; any other is filled here.
"""
function ScalarField(source, dep_layout::GriddedLayout, ::Type{T} = Float64) where {T}
    env_dep = reads_env(source)
    if source isa Function
        deps  = declared_deps(source, dep_layout)
        names = axisnames(dep_layout); rank = length(names)
        buf   = Array{T}(undef, map(a -> _axis_size(dep_layout, a), deps)...)
        bsh   = ntuple(i -> names[i] in deps ? _axis_size(dep_layout, names[i]) : 1, rank)
        sf    = ScalarField{typeof(buf)}(buf, bsh, env_dep)
        env_dep ? _nanfill!(buf) : fill_scalar_field!(sf, source, dep_layout, nothing)
        return sf
    elseif env_dep                                      # FromEnv: type unknown ⇒ Any
        return ScalarField{Any}(NaN, (), true)
    else                                                # scalar / array constant
        d = resolve(source, nothing)
        return ScalarField{typeof(d)}(d, (), false)
    end
end

"The broadcastable view of a `ScalarField`: its buffer reshaped to `bshape`, or `data` as-is."
scalar_broadcastable(sf::ScalarField) = sf.bshape === () ? sf.data : reshape(sf.data, sf.bshape)

"""
Fill a `ScalarField`'s `data` from `source` at `env`: a dep closure fills its compact buffer one
combination at a time, the other source forms re-resolve through the field's `to` adaptor. Fills
unconditionally.
"""
function fill_scalar_field!(sf::ScalarField, source, dep_layout::GriddedLayout, env)
    REFILL_COUNT[] += 1                                    # instrumentation
    if source isa Function
        _fill_scalar_buffer!(sf.data, source, dep_layout, Val(declared_deps(source, dep_layout)),
                             Val(reads_env(source)), env)
    else
        sf.data = adapt(sf.to, resolve(source, env))
    end
    return sf
end

"""
Fill a dep closure's compact `buffer` with one scalar per combination of its dep axes. The `Array`
method writes in place; the generic method fills a host buffer and `copyto!`s it across.
"""
function _fill_scalar_buffer!(buffer::Array, source::S, dep_layout::GriddedLayout, deps::Val, envdep::Val, env) where {S}
    for (idx, combo) in dep_combos(dep_layout, deps)
        value = evaluate(source, combo, env, envdep)
        _assert_seatable_fill(typeof(value), eltype(buffer))
        buffer[idx...] = value
    end
    return buffer
end

function _fill_scalar_buffer!(buffer::AbstractArray, source::S, dep_layout::GriddedLayout, deps::Val, envdep::Val, env) where {S}
    host = _fill_scalar_buffer!(Array{eltype(buffer)}(undef, size(buffer)), source, dep_layout, deps, envdep, env)
    copyto!(buffer, host)
    return buffer
end

"The broadcastable form of a `ScalarField`, refilled from `source` first when the field reads `env` and `env_changed`."
function materialize_scalar!(sf::ScalarField, source, dep_layout::GriddedLayout, env; env_changed::Bool = true)
    sf.env_dependent && env_changed && fill_scalar_field!(sf, source, dep_layout, env)
    return scalar_broadcastable(sf)
end
