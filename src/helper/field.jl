# Matrix-valued heterogeneity fields — THE central treatment of the `(n_out, n_in, dep…)`
# arrays that Markov (transition `K`), Logit (cost `eψC`) and the continuous argmax (payoff
# `U`) all need. A stage describes its array by a *source*:
#
#   • a constant `Matrix`                    — the same fiber everywhere;
#   • a `FromEnv(:key)`                      — the matrix lives in `env`, read each backward;
#   • a closure `(; dep…[, env]) -> Matrix`  — one matrix per dep-combination, the deps
#     declared as kwargs (a subset of the layout's axes); declaring `env` makes it env-varying;
#   • a `MappedField(map, src)`              — `src`'s fibers run through `map(fiber, env)`,
#     baking a stage's per-fiber transform (Markov's transpose, Logit's `exp(−·/ε)`) INTO the
#     source so the fill stays uniform.
#
# Stages never write the fill loop: they build the source and call `allocate_field` +
# `fill_field!`. `field_deps`/`matrix_for`/`dep_combos` are the primitives those two use.

"The dep axes a matrix source varies along, in layout order — `()` for a constant matrix / `FromEnv`, the closure's declared (non-`env`) axes otherwise."
field_deps(::Any, ::GriddedLayout) = ()
field_deps(fn::Function, layout::GriddedLayout) = _closure_deps(fn, layout)

"The matrix for one dep-combination `combo` (a `NamedTuple` of axis ⇒ value): the constant / env entry (via `resolve`), or the closure evaluated at `combo` (passing `env` iff declared). The single source-discrimination site."
matrix_for(src, _combo, env) = resolve(src, env)
matrix_for(fn::Function, combo, env) = _closure_env_dep(fn) ? fn(; combo..., env) : fn(; combo...)

"Iterate `(idx, combo)` over the cartesian product of the dep grids: `idx` the trailing buffer index (a tuple), `combo` the `NamedTuple` of axis ⇒ grid value. One empty pair when there are no deps."
function dep_combos(layout::GriddedLayout, deps::NTuple{D, Symbol}) where {D}
    grids = map(a -> _axis_grid(layout, a), deps)
    return ((Tuple(ci), NamedTuple{deps}(map((g, i) -> g[i], grids, Tuple(ci))))
            for ci in CartesianIndices(map(length, grids)))
end

"A matrix source whose stored fiber is `map(raw_fiber)` — bakes a stage's per-fiber transform (Markov's `permutedims`, Logit's `C -> exp(-C/ε)`) INTO the source so `fill_field!` is uniform. `field_deps`/`matrix_for` see through to the wrapped `src`."
struct MappedField{F, S}
    map :: F
    src :: S
end
matrix_for(m::MappedField, combo, env) = m.map(matrix_for(m.src, combo, env))
field_deps(m::MappedField, layout::GriddedLayout) = field_deps(m.src, layout)

"Allocate the compact `(n_out, n_in, dep_sizes…)` buffer a matrix field stores."
allocate_field(::Type{T}, n_out::Int, n_in::Int, layout::GriddedLayout, deps) where {T} =
    zeros(T, n_out, n_in, map(a -> _axis_size(layout, a), deps)...)

"The `(n_out, n_in)` of a matrix field: a constant matrix gives its own size (the only ever-rectangular case — forget/introduce); any other source (`FromEnv`/closure/`MappedField`) is square on the contracted `axis`."
_field_shape(source, layout::GriddedLayout, axis::Symbol) = (n = _axis_size(layout, axis); (n, n))
_field_shape(M::AbstractMatrix, ::GriddedLayout, ::Symbol) = (size(M, 1), size(M, 2))
_field_shape(m::MappedField, layout::GriddedLayout, axis::Symbol) = _field_shape(m.src, layout, axis)

"""
Source-driven front door over `_dense_kernel` (kernel.jl): from a matrix-field `source` (const
/ `FromEnv` / closure / `MappedField`), derive its dep axes and shape, allocate the dep-sized
buffer at eltype `T`, and wrap it as the dense self-describing kernel. The stage's
`allocate_kernel` is then one line.
"""
function dense_kernel(::Type{T}, layout::GriddedLayout, axis::Symbol, source) where {T}
    deps        = field_deps(source, layout)
    n_out, n_in = _field_shape(source, layout, axis)
    return _dense_kernel(allocate_field(T, n_out, n_in, layout, deps), layout, axis, deps)
end

"""
Buffer-driven core: fill the compact `(n_out, n_in, dep…)` buffer from a matrix `source` —
each dep-combination's fiber is `matrix_for(source, combo, env)`.
"""
function _fill_field!(buffer, source, layout::GriddedLayout, deps, env)
    for (idx, combo) in dep_combos(layout, deps)
        face  = view(buffer, :, :, idx...)
        fiber = matrix_for(source, combo, env)
        @assert size(fiber) == size(face) "fill_field!: source fiber $(size(fiber)) ≠ field face $(size(face))."
        copyto!(face, fiber)
    end
    return buffer
end

"""
Kernel-driven front door: materialize a dense kernel's storage from `source`. Recovers the
kernel's compact parent and the source's dep axes, then fills via `_fill_field!`. THE place
Markov/Logit materialize their heterogeneity arrays — the stage passes the kernel + axis +
source, never a reshaped buffer or a dep list.
"""
function fill_field!(kernel, source, layout::GriddedLayout, axis::Symbol, env)
    deps = field_deps(source, layout)
    P    = parent(kernel)
    K    = reshape(P, size(P, 1), size(P, 2), map(a -> _axis_size(layout, a), deps)...)
    return _fill_field!(K, source, layout, deps, env)
end
