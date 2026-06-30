# Matrix-valued heterogeneity fields — THE central treatment of the `(n_out, n_in, dep…)`
# arrays that Markov (transition `K`), Logit (cost `eψC`) and the continuous argmax (reward
# `U`) all need. A stage describes its array by a *source* (the Sources API, sources/sources.jl):
#
#   • a constant `Matrix`                    — the same fiber everywhere;
#   • a `FromEnv(:key)`                      — the matrix lives in `env`, read each backward;
#   • a closure `(; dep…[, env]) -> Matrix`  — one matrix per dep-combination, the deps
#     declared as kwargs (a subset of the layout's axes); declaring `env` makes it env-varying;
#   • a `MappedField(map, src)`              — `src`'s fibers run through `map(fiber, env)`,
#     baking a stage's per-fiber transform (Markov's transpose, Logit's `exp(−·/ε)`) INTO the
#     source so the fill stays uniform.
#
# Stages never write the fill loop: they build the source and call `matrix_field` + `fill_field!`.
# The source-driven `dense_kernel` front door (kernels/dense_kernel.jl) wraps the field as the `mul`
# operator.

using LinearAlgebra: transpose, mul!

# MappedField — a per-fiber transform baked into a source #
#-------------------------------------------------------#

"A matrix source whose stored fiber is `map(raw_fiber)` — bakes a stage's per-fiber transform (Markov's `permutedims`, Logit's `C -> exp(-C/ε)`) INTO the source so `fill_field!` is uniform. `declared_deps`/`evaluate` see through to the wrapped `src`."
struct MappedField{F, S}
    map :: F
    src :: S
end
evaluate(m::MappedField, combo, env) = m.map(evaluate(m.src, combo, env))
declared_deps(m::MappedField, layout::GriddedLayout) = declared_deps(m.src, layout)

# MatrixField — compact stratified matrix data + explicit operative-axis/dep metadata #
#------------------------------------------------------------------------------------#
# A `MatrixField` is the package's matrix-valued field (end-goal §5): the compact
# `(n_out, n_in, dep_sizes…)` array PLUS the explicit identity of its operative axis (the
# `(out, in)` fiber dims) and the layout positions of the axes it varies along. Nothing here
# specialises on a concrete backing — `array` is any `AbstractArray` with the compact shape; only
# the default allocator (`allocate_field`) picks `Array`. A kernel reads `array` + the metadata
# directly and never reverse-engineers shape from a permuted view. What the data *means*
# operationally is the consuming kernel's choice, not the field's type — the same field is a
# `mul` (Markov), a softmax cost (logit), or an argmax reward.

"""
The matrix-valued field: the compact `(n_out, n_in, dep_sizes…)` `array` over a generic
`AbstractArray` backing, plus its operative-axis identity (`operative_axis`, riding the `(out, in)`
fiber) at layout position `operative_dim`, and the layout positions `dep_dims` of the axes it varies
along (layout order, matching the array's trailing dims). Pure data; the operative-axis/dep metadata
is explicit so a kernel never reverse-engineers it from a permuted view (end-goal §5.1).
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
allocate_field(::Type{T}, n_out::Int, n_in::Int, layout::GriddedLayout, deps) where {T} =
    zeros(T, n_out, n_in, map(a -> _axis_size(layout, a), deps)...)

"The `(n_out, n_in)` of a matrix field: a constant matrix gives its own size (rectangular — forget/introduce/crosswalk); a `FromEnv`/closure is square on the contracted `axis`. A `MappedField` inherits its source's shape, except `permutedims` (Markov's stored `K = Tᵀ`) transposes it."
_field_shape(source, layout::GriddedLayout, axis::Symbol) = (n = _axis_size(layout, axis); (n, n))
_field_shape(M::AbstractMatrix, ::GriddedLayout, ::Symbol) = (size(M, 1), size(M, 2))
_field_shape(m::MappedField, layout::GriddedLayout, axis::Symbol) = _field_shape(m.src, layout, axis)
_field_shape(m::MappedField{typeof(permutedims)}, layout::GriddedLayout, axis::Symbol) =
    reverse(_field_shape(m.src, layout, axis))

"""
Build a `MatrixField` from a matrix `source` (const / `FromEnv` / closure / `MappedField`): derive
its dep axes and `(n_out, n_in)` shape, allocate the compact dep-sized buffer at eltype `T`, and
record the operative-axis identity + dep layout positions. The buffer is NaN-filled (§5.3) so a read
before `fill_field!` seats it fails fast; an env-independent stage fills it once at construction.
"""
function matrix_field(::Type{T}, layout::GriddedLayout, axis::Symbol, source) where {T}
    deps        = declared_deps(source, layout)
    n_out, n_in = _field_shape(source, layout, axis)
    array       = allocate_field(T, n_out, n_in, layout, deps)
    fill!(array, T <: Integer ? zero(T) : NaN)
    return MatrixField(array, axis, axis_position(layout, axis),
                       map(a -> axis_position(layout, a), deps))
end

# The generic application driver (end-goal §5.4) #
#-----------------------------------------------#
# Apply a per-base-point operator along a field's operative axis, stratified over its deps. THE
# reference semantics AND the fallback for any non-contiguous backing — written against the
# `AbstractArray` interface only, naming no concrete array type (representation-specialised fast
# paths are a kernel concern, §8, not a field concern). Iteration is `eachslice`-style over the
# non-operative axes (a `CartesianIndices` loop, not `mapslices` — which allocates and is
# type-unstable); each non-operative coordinate is projected onto the field's dep axes to pick the
# fiber (the field is constant along its non-dep, non-operative dims).

"""
Generic stratified application (end-goal §5.4): for every coordinate over the non-operative axes,
slice `src`/`dest` along the field's operative axis, project the coordinate onto the field's deps to
select the fiber, and run the in-place per-slice operator `op(dest_slice, mat, src_slice)` with `mat`
the fiber (`mode = :covariant`) or its transpose (`:contravariant`). Backing-agnostic.
"""
function stratified_apply!(dest, op, field::MatrixField, src; mode::Symbol)
    k         = field.operative_dim
    deps      = field.dep_dims
    A         = field.array
    N         = ndims(src)
    covariant = mode === :covariant
    other     = ntuple(d -> d == k ? 1 : size(src, d), N)
    @inbounds for ci in CartesianIndices(other)
        idx     = ntuple(d -> d == k ? Colon() : ci[d], N)
        s_slice = view(src,  idx...)
        d_slice = view(dest, idx...)
        fiber   = view(A, :, :, ntuple(j -> ci[deps[j]], length(deps))...)
        op(d_slice, covariant ? fiber : transpose(fiber), s_slice)
    end
    return dest
end

"""
Buffer-driven core: fill the compact `(n_out, n_in, dep…)` buffer from a matrix `source` —
each dep-combination's fiber is `evaluate(source, combo, env)`.
"""
function _fill_field!(buffer, source, layout::GriddedLayout, deps, env)
    for (idx, combo) in dep_combos(layout, deps)
        face  = view(buffer, :, :, idx...)
        fiber = evaluate(source, combo, env)
        @assert size(fiber) == size(face) "fill_field!: source fiber $(size(fiber)) ≠ field face $(size(face))."
        copyto!(face, fiber)
    end
    return buffer
end

"""
Materialise a `MatrixField`'s compact storage from `source` — each dep-combination's
`(n_out, n_in)` fiber is `evaluate(source, combo, env)`. THE place Markov/Logit/argmax materialise
their heterogeneity arrays. (`axis` is accepted for call-site uniformity; the deps come from the
source, and the compact `array` is already shaped, so there is no reshape/parent dance.)
"""
fill_field!(field::MatrixField, source, layout::GriddedLayout, axis::Symbol, env) =
    (_fill_field!(field.array, source, layout, declared_deps(source, layout), env); field)

# Static refill policy #
#----------------------#
# A contracting stage refills its matrix field from the source every backward. In a fixed-env VFI
# inner loop the source's env inputs don't change, so the refill is pure waste — the field is bit-
# identical. The refill decision is purely STATIC + ONE runtime flag, with NO per-field cache record
# (end-goal §5.3): an env-INDEPENDENT field (`reads_env(source) == false`) is filled ONCE at
# construction (no env needed) and never refilled; an env-DEPENDENT field is NaN-filled at allocation
# (`matrix_field`) and refilled each `backward!` UNLESS the caller passes `env_changed = false` (the
# fixed-env-loop assertion that env is unchanged). The §5.3 contract — the first `backward!` after any
# env change runs `env_changed = true` — guarantees the field is seated before any read, so no
# "filled" bool is needed; the NaN-fill turns a contract violation into a fast failure. Each stage
# computes `reads_env(source)` once at construction and stores the resulting `Bool` in its `cache`;
# `backward!` is then `env_dep && env_changed && fill_field!(...)`.
