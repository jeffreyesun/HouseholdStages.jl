# ScalarField — the scalar-per-base-point field (end-goal §5): a PURE materialized buffer that
# broadcasts against the gridded V/Λ, the vector sibling of field.jl's MatrixField. The Source lives
# in the stage's Spec, not on the field, so a ScalarField
# is just `(data, bshape)` plus the static `env_dependent` classification driving the refill policy
# (§5.3). Operation-agnostic: the stage applies it with any broadcast op (utility/entry `.+`). The
# materialized `data` is a `Number`/array constant stored as-is, a `FromEnv` re-resolved each fill, or
# a `(; dep…[, env])` closure's scalar-per-dep-combination stored COMPACT (dep sizes, 1 else). Static
# refill (§5.3): an env-INDEPENDENT field is filled ONCE at construction; an env-DEPENDENT field is
# NaN-filled and refilled each `backward!` unless the caller asserts `env_changed = false`. No
# per-field cache record — `env_dependent` (computed once from the source) is the whole decision.

"""
A scalar-per-base-point field broadcasting against the gridded `V`/`Λ`. Pure data: the materialized
`data` (compact dep buffer / scalar / full array), the `bshape` reshape broadcasting a compact buffer
(`()` ⇒ use `data` as-is), and the static `env_dependent` refill classification (§5.3).
"""
mutable struct ScalarField{A}
    data          :: A        # materialized values: compact dep-array / scalar / full array
    bshape        :: Tuple    # reshape target broadcasting a compact `data`; `()` ⇒ use `data` as-is
    env_dependent :: Bool     # `reads_env(source)`, computed once — the whole refill decision
end

"Fresh-buffer fill: `NaN` where the eltype allows (an unfilled read then fails fast), a zero sentinel for integers."
_nanfill!(A::AbstractArray) = fill!(A, eltype(A) <: Integer ? zero(eltype(A)) : NaN)

"""
Build a `ScalarField` from a payoff `source` (scalar / array / `FromEnv` / dep closure). A dep closure
gets a compact dep-sized buffer; the other forms store the resolved value directly. Static policy: an
env-independent source is filled here, once; an env-dependent one is left NaN, seated by `backward!`.
"""
function ScalarField(source, layout::GriddedLayout, ::Type{T} = Float64) where {T}
    env_dep = reads_env(source)
    if source isa Function
        deps  = declared_deps(source, layout)
        names = axisnames(layout); rank = length(names)
        buf   = Array{T}(undef, map(a -> _axis_size(layout, a), deps)...)
        bsh   = ntuple(i -> names[i] in deps ? _axis_size(layout, names[i]) : 1, rank)
        sf    = ScalarField{typeof(buf)}(buf, bsh, env_dep)
        env_dep ? _nanfill!(buf) : fill_scalar_field!(sf, source, layout, nothing)
        return sf
    elseif env_dep                                      # FromEnv: seated each backward; type unknown ⇒ Any
        return ScalarField{Any}(NaN, (), true)
    else                                                # scalar / array constant: resolve once
        d = resolve(source, nothing)
        return ScalarField{typeof(d)}(d, (), false)
    end
end

"The broadcastable view of a `ScalarField`: the compact buffer reshaped to broadcast, or `data` as-is."
scalar_broadcastable(sf::ScalarField) = sf.bshape === () ? sf.data : reshape(sf.data, sf.bshape)

"""
Seat a `ScalarField`'s `data` from `source` at `env` — a dep closure fills its compact buffer per
dep-combination, the other forms re-resolve into `data`. Unconditional; the policy is in
[`materialize_scalar!`](@ref).
"""
function fill_scalar_field!(sf::ScalarField, source, layout::GriddedLayout, env)
    if source isa Function
        _fill_scalar_buffer!(sf.data, source, layout, env)
    else
        sf.data = resolve(source, env)
    end
    return sf
end

"""
Fill a dep closure's compact `buffer` with one scalar per dep-combination. The `Array` method writes
in place; the generic method (a device buffer, e.g. a `CuArray` after the GPU lift) stages into a host
buffer and `copyto!`s it across, since per-element `setindex!` into a device array is a scalar-indexing
error. Keeps the host path allocation-free.
"""
function _fill_scalar_buffer!(buffer::Array, source, layout::GriddedLayout, env)
    for (idx, combo) in dep_combos(layout, Tuple(declared_deps(source, layout)))
        buffer[idx...] = evaluate(source, combo, env)
    end
    return buffer
end

function _fill_scalar_buffer!(buffer::AbstractArray, source, layout::GriddedLayout, env)
    host = _fill_scalar_buffer!(Array{eltype(buffer)}(undef, size(buffer)), source, layout, env)
    copyto!(buffer, host)
    return buffer
end

"""
Materialize a `ScalarField` to a broadcastable under the static refill policy (§5.3): env-independent
⇒ already filled at construction, returned untouched; env-dependent ⇒ refilled from `source` unless
the caller asserts `env_changed = false`.
"""
function materialize_scalar!(sf::ScalarField, source, layout::GriddedLayout, env; env_changed::Bool = true)
    sf.env_dependent && env_changed && fill_scalar_field!(sf, source, layout, env)
    return scalar_broadcastable(sf)
end
