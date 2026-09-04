"""
Markov transition along one named `axis`. `transition_matrix` supplies the row-stochastic
`T[from, to] = P(from → to)` — a constant `Matrix`, a `FromEnv`, or a `(; dep…, env) -> T` closure
varying it along a declared subset of axes. The stored kernel is `K = Tᵀ`.
"""
struct MarkovStageSpec{Tm} <: AbstractStageSpec
    transition_matrix :: Tm
    axis              :: Symbol
end

MarkovStageSpec(; axis::Symbol, transition_matrix) = MarkovStageSpec(transition_matrix, axis)

default_eltype(spec::MarkovStageSpec) =
    spec.transition_matrix isa AbstractMatrix ? eltype(spec.transition_matrix) : Float64

@definestage MarkovStage MarkovStageSpec


##########################
# Gridded implementation #
##########################
# The kernel is a `DenseKernel` owning the transition `MatrixField`.

operative_axis(spec::MarkovStageSpec) = spec.axis

function allocate_kernel(spec::MarkovStageSpec, ::Type{T}, start_layout::GriddedLayout,
                         end_layout::GriddedLayout) where {T}
    src    = MappedField(permutedims, spec.transition_matrix)
    kernel = dense_kernel(T, start_layout, end_layout, spec.axis, src)
    reads_env(src) || fill_field!(kernel, src, start_layout, spec.axis, nothing)  # env-independent ⇒ fill once, here
    return kernel
end

"Cache: whether the transition depends on `env`."
allocate_cache(spec::MarkovStageSpec, ::Type, ::GriddedLayout, ::GriddedLayout) =
    (transition_env_dep = reads_env(MappedField(permutedims, spec.transition_matrix)),)

function backward!(V_start, spec::MarkovStageSpec, start_layout::GriddedLayout,
                   end_layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    cache.transition_env_dep && env_changed &&                            # refill K = Tᵀ from env
        fill_field!(kernel, MappedField(permutedims, spec.transition_matrix), start_layout, spec.axis, env)
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)   # Kᵀ·V_end
    return (V_start, kernel)
end

# forward! (K·Λ_start) is the generic default.
