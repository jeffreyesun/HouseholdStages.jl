# Markov transition along one named axis. The kernel is a `DenseKernel` (kernel.jl) owning the
# transition `MatrixField` — its compact `(n_out, n_in, dep…)` array + metadata. The user
# supplies the ROW-stochastic `transition_matrix` `T[from, to] = P(from→to)` as a constant
# `Matrix`, a `FromEnv`, or a closure `(; dep…, env) -> T` varying it along a declared subset
# of axes (the dep set, so the kernel becomes dep-varying — dep-only storage, no replication).
# Stored fiber `K = Tᵀ` (the forward operator, kernel.jl): forward applies `K`, backward `Kᵀ`.

"""
Markov transition along one named axis. `transition_matrix` supplies the ROW-stochastic
`T[from, to] = P(from→to)` — a constant `Matrix`, a `FromEnv`, or a closure
`(; dep…, env) -> T` varying it along a declared subset of axes. Backward applies `Kᵀ`,
forward `K = Tᵀ`.
"""
struct MarkovStageSpec{Tm} <: AbstractStageSpec
    transition_matrix :: Tm     # T[from, to]: a Matrix, FromEnv, or closure (; dep…, env) -> T
    axis              :: Symbol
end

MarkovStageSpec(; axis::Symbol, transition_matrix) = MarkovStageSpec(transition_matrix, axis)

default_eltype(spec::MarkovStageSpec) =
    spec.transition_matrix isa AbstractMatrix ? eltype(spec.transition_matrix) : Float64

@definestage MarkovStage MarkovStageSpec


##########################
# Gridded implementation #
##########################
# The kernel is a `DenseKernel` owning the transition `MatrixField`. `allocate_kernel` sizes it from
# the source; `backward!` seats `K = Tᵀ` via the central `fill_field!` (the `permutedims` map);
# both verbs contract through the shared dense machinery. The gather lives in
# `scratch.kernel_scratch` (derived from the kernel by `@definestage`), so we need no
# `allocate_scratch` of our own.

# The stored kernel is `K = Tᵀ` (the `permutedims` map), so a rectangular `T` (`from ≠ to`) is
# allocated at the transposed shape. `to = 1` marginalizes the axis (this is `ForgetfulSumStage`);
# `from = to` is the ordinary square transition; `from = 1 → to = n` GROWS a declared singleton
# (introduce — the exit composite's `(s, 1−s)` hazard, §3/§12). The output layout resizes the axis to
# `to` (`n_out`): `grow_axis` for the introduce, `resize_axis` (shrink / no-op) otherwise.
allocate_kernel(spec::MarkovStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    dense_kernel(T, layout, spec.axis, MappedField(permutedims, spec.transition_matrix))

"Cache: the static env-dependence classification for `K = Tᵀ` (`reads_env`, computed once), so a fixed-env VFI loop seats the transition once (§5.3). A `MappedField` is opaque to `reads_env` ⇒ conservatively env-dependent (NaN-filled at allocation, refilled each `backward!`)."
allocate_cache(spec::MarkovStageSpec, ::Type, ::GriddedLayout) =
    (transition_env_dep = reads_env(MappedField(permutedims, spec.transition_matrix)),)

function output_layout(spec::MarkovStageSpec, layout::GriddedLayout)
    n_out, n_in = _field_shape(MappedField(permutedims, spec.transition_matrix), layout, spec.axis)
    n = _axis_size(layout, spec.axis)
    @assert n_in == n "MarkovStage: construct against the full `$(spec.axis)` axis — its size $n " *
        "must equal the transition's `from` count $n_in."
    @assert n_out == n || n_out == 1 || n == 1 "MarkovStage: rectangular transition supports " *
        "`to` ∈ {1 (marginalize), $n (square)}, or growth from a size-1 axis (introduce); got from=$n, to=$n_out."
    return n_out > n ? grow_axis(layout, spec.axis, n_out) : resize_axis(layout, spec.axis, n_out)
end

function backward!(V_start, spec::MarkovStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    cache.transition_env_dep && env_changed &&                            # seat K = Tᵀ (static refill)
        fill_field!(kernel, MappedField(permutedims, spec.transition_matrix), layout, spec.axis, env)
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)   # Kᵀ·V_end
    return (V_start, kernel)
end

# forward! (K·Λ_start) is the generic modern default (abstract.jl) — pure kernel application.


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. The phase-1 stage methods above do not dispatch on
# layout type, so this is a placeholder marking where the deriv-carrying
# representation's methods will go.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder marking where the dynamic-grid
# representation's methods will go.
