# LogitChoiceKernel — the Gumbel-logit transition's data #
#=======================================================#
# The kernel half of the logit-choice stage (the stage proper lives in
# stages/primitive/logit_choice.jl). The operator is the Gibbs/Luce form
# `π(j|i,s) = eψC[i,j]·value_weight[j,s]/normalizer[i,s]`, with `eψC = exp(−C/ε)` held as a contained
# `DenseKernel` over the cost `MatrixField`, so the contraction reuses that kernel's field, plan, and
# batched-mul fast path (end-goal §8.1). The stage's backward seats `eψC`, `value_weight`, and
# `normalizer`; this file owns the data and the two verbs `forward! = K·`, `backward! = Kᵀ·`.

"""
The Gumbel-logit transition's data, applying the Gibbs/Luce operator
`π(j|i,s) = eψC[i,j]·value_weight[j,s]/normalizer[i,s]`: the frozen `eψC = exp(−C/ε)` as a contained
`DenseKernel` over its cost `MatrixField`, plus the `value_weight` and `normalizer` buffers (both
layout-shaped) seated by the stage `backward!`.
"""
struct LogitChoiceKernel{M, D<:AbstractArray}
    eψC          :: M    # exp(−C/ε): a contained DenseKernel over the cost field (origin, dest, deps…)
    value_weight :: D    # exp((V−maxⱼV)/ε)[j,s]: per-(dest, state) softmax numerator, layout-shaped
    normalizer   :: D    # (Σⱼ eψC[i,j]·value_weight[j,s])[i,s]: per-(origin, state) denominator, layout-shaped
end

"The logit kernel's apply plan: the `eψC` gather scratch + a flat `tmp` the K/Kᵀ verbs stage the diagonal scalings through (sized for the larger of origin/dest — `tmp` is `src`-shaped, viewed per call)."
kernel_scratch(k::LogitChoiceKernel, layout::GriddedLayout, ::Type{T}) where {T} =
    merge(kernel_scratch(k.eψC, layout, T), (tmp = zeros(T, max(length(k.value_weight), length(k.normalizer))),))

# The kernel's two verbs — `forward! = K·`, `backward! = Kᵀ·` — for the Gibbs operator
# `K = diag(value_weight)·eψCᵀ·diag(1/normalizer)`. The primal mass push and the lift adjoints both
# route through them; `tmp` stages the diagonal scalings.

"""
Apply `K· = value_weight ⊙ (eψCᵀ·(src ./ normalizer))` (the Gibbs mass-push operator).
"""
function forward!(dest, k::LogitChoiceKernel, src; scratch)
    tmp = reshape(view(scratch.tmp, 1:length(src)), size(src))  # src-shaped (origin side)
    @. tmp = src / k.normalizer
    backward!(dest, k.eψC, tmp; scratch = scratch)              # eψCᵀ · (src ./ normalizer)
    @. dest = k.value_weight * dest
    return dest
end

"""
Apply `Kᵀ· = (eψC·(value_weight ⊙ src)) ./ normalizer` (the value-pullback operator).
"""
function backward!(dest, k::LogitChoiceKernel, src; scratch)
    tmp = reshape(view(scratch.tmp, 1:length(src)), size(src))  # src-shaped (dest side)
    @. tmp = k.value_weight * src
    forward!(dest, k.eψC, tmp; scratch = scratch)               # eψC · (value_weight ⊙ src)
    @. dest = dest / k.normalizer
    return dest
end
