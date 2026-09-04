# Logit (Gumbel) choice over a discrete axis, built from four arrays: `eψC`, the exponentiated
# negative cost, stored `(dest, origin, deps…)`; `rowmax`, the per-state shift; `value_weight`, the
# exponentiated shifted `V_end`; and `normalizer`, its cost-weighted sum over destinations.

"""
Logit (Gumbel) choice over a discrete `axis`, with origin→destination friction `cost_matrix[i, j]`:

    π(j|i,s) = softmax_j( (−C[i,j] + V_end[j,s]) / ε ).

The cost is a constant `n×n` matrix, a `FromEnv`, or a closure `(; dep…, env) -> C` varying it along
a declared subset of axes (`+Inf` for an immobile group, say); its own origin and destination are the
matrix's two dims, never kwargs.
"""
struct LogitChoiceStageSpec{Cmat, T} <: AbstractStageSpec
    axis :: Symbol
    cost_matrix :: Cmat                                    # C[origin, dest]: a Matrix, FromEnv, or closure (; dep…, env) -> C
    ε           :: T                                       # Gumbel scale (literal or FromEnv); ε < 0 gives a soft-min
end

LogitChoiceStageSpec(; axis::Symbol, cost_matrix, ε = 1.0) =
    LogitChoiceStageSpec{typeof(cost_matrix), typeof(ε)}(axis, cost_matrix, ε)

# propagate the build eltype (mirrors MarkovStage) so a Float32 model's
# migration/logit kernel is Float32, not the Float64 default — the Float64 dense
# GEMM is otherwise the dominant GPU cost. See FIX_BEFORE_PUBLIC_MERGE.md.
default_eltype(spec::LogitChoiceStageSpec) =
    spec.cost_matrix isa AbstractMatrix ? eltype(spec.cost_matrix) : Float64

@definestage LogitChoiceStage LogitChoiceStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(spec::LogitChoiceStageSpec) = spec.axis

"Source for the stored field: `C[origin, dest]` mapped through `exp(−·/ε)` and transposed into the `(destination, origin)` orientation."
_eψC_source(spec::LogitChoiceStageSpec, ε) =
    MappedField(permutedims, MappedField(C -> exp.(.- C ./ ε), spec.cost_matrix))

function allocate_kernel(spec::LogitChoiceStageSpec, ::Type{T}, start_layout::GriddedLayout,
                         end_layout::GriddedLayout) where {T}
    eψC    = dense_kernel(T, start_layout, end_layout, spec.axis, _eψC_source(spec, spec.ε))
    kernel = LogitChoiceKernel(eψC, allocate_buffer(end_layout, T),      # value_weight (dest side)
                                    allocate_buffer(start_layout, T))    # normalizer (origin side)
    if !(reads_env(spec.cost_matrix) || spec.ε isa FromEnv)              # env-independent ⇒ fill once, here
        fill_field!(eψC, _eψC_source(spec, spec.ε), start_layout, spec.axis, nothing)
    end
    return kernel
end

"Cache: whether `eψC` depends on `env`; `ε` counts, since it is baked into the stored field."
allocate_cache(spec::LogitChoiceStageSpec, ::Type, ::GriddedLayout, ::GriddedLayout) =
    (cost_env_dep = reads_env(spec.cost_matrix) || spec.ε isa FromEnv,)

"Scratch: io buffers plus `rowmax`, the end layout with the choice axis collapsed."
function allocate_scratch(spec::LogitChoiceStageSpec, ::Type{T}, start_layout::GriddedLayout,
                          end_layout::GriddedLayout) where {T}
    rowmax = zeros(T, Base.setindex(layout_size(end_layout), 1, axis_position(end_layout, spec.axis)))
    return merge(io_scratch(start_layout, end_layout, T), (rowmax = rowmax,))
end

# Backward — the log-sum-exp directly #
#-------------------------------------#

"Logit backward: fill `eψC`, `value_weight` and `normalizer`, write `V_start`, and leave the kernel ready for the forward pass."
function backward!(V_start, spec::LogitChoiceStageSpec, start_layout::GriddedLayout,
                   end_layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    ε  = resolve(spec.ε, env)
    (; eψC, value_weight, normalizer) = kernel
    cache.cost_env_dep && env_changed &&
        fill_field!(eψC, _eψC_source(spec, ε), start_layout, spec.axis, env)
    @. value_weight = V_end / ε                                              # u = V_end/ε
    maximum!(scratch.rowmax, value_weight)                                   # m = maxⱼ u, the shift
    @. value_weight = exp(value_weight - scratch.rowmax)                     # exp(u − m) ∈ (0,1]
    backward!(normalizer, eψC, value_weight; scratch = scratch.kernel_scratch) # normalizer = eψCᵀ·value_weight
    @. V_start = ε * (scratch.rowmax + log(normalizer))                      # the LSE
    return (V_start, kernel)
end

# forward! (push mass `Λ_end = K·Λ_start` through the choice probabilities) is the generic default.

# Reading the choice probabilities #
#----------------------------------#

"""
The choice probabilities `π(j | i, s)` left by the last `backward!`. Shaped like the start layout in
the origin `i` and the off-axis state `s`, with the destination `j` appended as a trailing axis.
"""
function choice_probabilities(stage::LogitChoiceStage)
    (; eψC, value_weight, normalizer) = stage.kernel
    (; operative_dim, dep_dims) = eψC.field
    weights = parent(eψC)                                             # exp(−Cᵀ/ε): (dest, origin, dep…)
    n_end   = size(value_weight, operative_dim)
    probs   = similar(normalizer, size(normalizer)..., n_end)
    for j in 1:n_end, cell in CartesianIndices(normalizer)
        state = Tuple(cell)
        probs[cell, j] = weights[j, state[operative_dim], map(d -> state[d], dep_dims)...] *
                         value_weight[Base.setindex(state, j, operative_dim)...] / normalizer[cell]
    end
    return probs
end
