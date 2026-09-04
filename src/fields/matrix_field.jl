# Matrix-valued heterogeneity fields: the `(n_out, n_in, dep…)` arrays behind a Markov transition
# `K`, a logit cost `eψC`, and a continuous argmax's reward `U`. A stage describes such an array by a
# *source* — a constant, a lookup in `env`, or a closure of the axes it varies along — and builds it
# with `matrix_field` and `fill_field!`. The stored `(n_end, n_start)` fiber shape comes from the
# stage's two boundary layouts; one `dep_layout` argument describes the dep axes at both ends.

using LinearAlgebra: transpose, mul!

# MappedField — a per-fiber transform baked into a source #
#-------------------------------------------------------#

"A matrix source whose stored fiber is `map` applied to `src`'s fiber (Markov's `permutedims`, logit's `C -> exp(-C/ε)`)."
struct MappedField{F, S}
    map :: F
    src :: S
end
evaluate(m::MappedField, combo, env) = evaluate(m, combo, env, Val(reads_env(m)))
evaluate(m::MappedField, combo, env, envdep::Val) = m.map(evaluate(m.src, combo, env, envdep))
declared_deps(m::MappedField, dep_layout::GriddedLayout) = declared_deps(m.src, dep_layout)
reads_env(m::MappedField) = reads_env(m.src)

# MatrixField — compact matrix data + operative-axis/dep metadata #
#-----------------------------------------------------------------#

"""
A stage's matrix-valued field: the compact `(n_out, n_in, dep_sizes…)` `array`, plus the metadata a
kernel needs to slice it — which axis the `(out, in)` fiber runs along and at which layout position,
and the layout positions of the dep axes, in layout order.
"""
struct MatrixField{A<:AbstractArray, D}
    array          :: A
    operative_axis :: Symbol
    operative_dim  :: Int
    dep_dims       :: D       # NTuple{ND, Int}: layout positions of the dep axes, layout order
end

"The compact backing array of a `MatrixField` (its `(n_out, n_in, dep…)` storage)."
Base.parent(f::MatrixField) = f.array

"Allocate the compact `(n_out, n_in, dep_sizes…)` buffer a matrix field stores."
allocate_field(::Type{T}, n_out::Int, n_in::Int, dep_layout::GriddedLayout, deps) where {T} =
    zeros(T, n_out, n_in, map(a -> _axis_size(dep_layout, a), deps)...)

"Build a `MatrixField` on `axis` from a matrix `source`, NaN-filling its compact buffer at eltype `T`."
function matrix_field(::Type{T}, start_layout::GriddedLayout, end_layout::GriddedLayout,
                      axis::Symbol, source) where {T}
    n_start, n_end = operative_sizes(start_layout, end_layout, axis)
    deps  = declared_deps(source, start_layout)
    array = allocate_field(T, n_end, n_start, start_layout, deps)
    fill!(array, T <: Integer ? zero(T) : NaN)
    return MatrixField(array, axis, axis_position(start_layout, axis),
                       map(a -> axis_position(start_layout, a), deps))
end

# The generic application driver #
#--------------------------------#

"Fiber op applying a stratum's matrix to its slice as `op(dest, mat, src)`."
struct CovariantFiber{F}
    op :: F
end
(m::CovariantFiber)(dest, src, mat) = m.op(dest, mat, src)

"The transposed twin of [`CovariantFiber`](@ref) — the fiber acts as `matᵀ`."
struct ContravariantFiber{F}
    op :: F
end
(m::ContravariantFiber)(dest, src, mat) = m.op(dest, transpose(mat), src)

"Run the in-place per-slice operator `op` at every stratum, with `mat` the field's fiber there (`mode = :covariant`) or its transpose (`:contravariant`)."
function stratified_apply!(dest, op, field::MatrixField, src; mode::Symbol)
    fiber = mode === :covariant ? CovariantFiber(op) : ContravariantFiber(op)
    stratified!(fiber, dest, src, field; dims=field.operative_dim)
    return dest
end

"Copy each dep-combination's fiber, `evaluate(source, combo, env, envdep)`, into its slice of `buffer`."
function _fill_field!(buffer, source::S, dep_layout::GriddedLayout, deps::Val, envdep::Val, env) where {S}
    REFILL_COUNT[] += 1                                    # instrumentation
    for (idx, combo) in dep_combos(dep_layout, deps)
        face  = view(buffer, :, :, idx...)
        fiber = evaluate(source, combo, env, envdep)
        @assert size(fiber) == size(face) "fill_field!: source fiber $(size(fiber)) ≠ field face $(size(face)). " *
            "A source that regrids its operative axis needs both layouts stated: " *
            "`Stage(spec, start_layout, end_layout)` — or, where the stage sits inside a composite, " *
            "the boundaries and interiors that put it between them."
        _assert_seatable_fill(eltype(fiber), eltype(face))
        copyto!(face, fiber)
    end
    return buffer
end

"Fill a `MatrixField`'s compact storage from `source`; `axis` is unused, accepted to match the scalar version."
function fill_field!(field::MatrixField, source, dep_layout::GriddedLayout, axis::Symbol, env)
    _fill_field!(field.array, source, dep_layout, Val(declared_deps(source, dep_layout)),
                 Val(reads_env(source)), env)
    return field
end
