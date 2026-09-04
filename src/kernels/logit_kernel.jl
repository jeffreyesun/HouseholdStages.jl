# LogitChoiceKernel — the Gumbel-logit transition's data #
#=======================================================#
# The operator is the Gibbs/Luce form `π(j|i,s) = eψC[j,i]·value_weight[j,s]/normalizer[i,s]`, with
# the cost factor `eψC = exp(−Cᵀ/ε)` held as a contained `DenseKernel`.

"""
The Gumbel-logit transition's three factors: the cost `eψC = exp(−Cᵀ/ε)` as a contained
`DenseKernel`, plus two layout-shaped buffers the stage's `backward!` fills.
"""
struct LogitChoiceKernel{M, D<:AbstractArray}
    eψC          :: M    # exp(−Cᵀ/ε): a contained DenseKernel over the cost field (dest, origin, deps…)
    value_weight :: D    # exp((V−maxⱼV)/ε)[j,s]: per-(dest, state) softmax numerator, layout-shaped
    normalizer   :: D    # (Σⱼ eψC[j,i]·value_weight[j,s])[i,s]: per-(origin, state) denominator, layout-shaped
end

"The `eψC` gather scratch plus a flat `tmp` the two verbs stage their diagonal scalings through, viewed `src`-shaped per call."
kernel_scratch(k::LogitChoiceKernel, start_layout::GriddedLayout, end_layout::GriddedLayout, ::Type{T}) where {T} =
    merge(kernel_scratch(k.eψC, start_layout, end_layout, T),
          (tmp = zeros(T, max(length(k.value_weight), length(k.normalizer))),))

"Push mass: `K· = value_weight ⊙ (eψC·(src ./ normalizer))`."
function forward!(dest, k::LogitChoiceKernel, src; scratch)
    tmp = reshape(view(scratch.tmp, 1:length(src)), size(src))  # origin-shaped
    @. tmp = src / k.normalizer
    forward!(dest, k.eψC, tmp; scratch = scratch)
    @. dest = k.value_weight * dest
    return dest
end

"Pull value back: `Kᵀ· = (eψCᵀ·(value_weight ⊙ src)) ./ normalizer`."
function backward!(dest, k::LogitChoiceKernel, src; scratch)
    tmp = reshape(view(scratch.tmp, 1:length(src)), size(src))  # destination-shaped
    @. tmp = k.value_weight * src
    backward!(dest, k.eψC, tmp; scratch = scratch)
    @. dest = dest / k.normalizer
    return dest
end
