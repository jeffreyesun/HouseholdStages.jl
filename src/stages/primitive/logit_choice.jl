# Logit (Gumbel) choice. The backward computes the LSE directly,
#
#     V_start[i,s] = ε·log Σ_j exp((−C[i,j] + V_end[j,s])/ε)
#                  = rm[s] + ε·log Σ_j eψC[i,j]·value_weight[j,s],
#
# with `eψC = exp(−C/ε)`, `rm[s] = maxⱼ V_end[j,s]`, `value_weight[j,s] = exp((V_end[j,s]−rm[s])/ε)`
# and `normalizer[i,s] = Σⱼ eψC[i,j]·value_weight[j,s]`. The choice axis is the destination `j`;
# `eψC` is the dense self-describing kernel (kernel.jl) over `(origin, dest, declared deps…)`,
# and `value_weight`/`normalizer` are full layout-shaped — so every elementwise step (max, exp,
# log) runs in layout order and only the `eψC` contraction gathers internally. A destination
# payoff shifter is V-additive, so it composes as a `UtilityStage` before the logit, never a kwarg.
#
# The cost `C[origin, dest]` is a constant `n×n` matrix, a `FromEnv`, or a closure
# `(; dep…, env) -> C` varying it along a declared subset of axes (its own origin/dest are the
# two positional matrix dims, not kwargs); `eψC` is stored only over the declared deps —
# constant cost ⇒ one matmul, dep-varying ⇒ a batched matmul over the dep batch.

"""
Logit (Gumbel) choice over a discrete `choice_axis`, with origin→destination
friction `cost_matrix[i, j]`:

    π(j|i,s) = softmax_j( (−C[i,j] + V_end[j,s]) / ε ).

The cost is a constant `n×n` matrix, a `FromEnv`, or a closure `(; dep…, env) -> C`
varying it along a declared subset of axes (e.g. `+Inf` for an immobile group). A
destination-dependent payoff is V-additive and belongs in a `UtilityStage` composed
before this one; only the origin-dependent cost lives here.
"""
# The cost field is left UNCONSTRAINED (not `<:AbstractMatrix`) so it can be a `FromEnv(:key)`
# resolved each backward OR a dep-declaring closure. The choice-axis size (from the layout, not
# the cost) sizes the kernel, so allocation works even when the cost is env-supplied / closure-built.
struct LogitChoiceStageSpec{Cmat, T} <: AbstractStageSpec
    choice_axis :: Symbol
    cost_matrix :: Cmat                                    # C[origin, dest]: a Matrix, FromEnv, or closure (; dep…, env) -> C
    ε           :: T                                       # Gumbel scale (literal or FromEnv)
end

LogitChoiceStageSpec(; choice_axis::Symbol, cost_matrix, ε = 1.0) =
    LogitChoiceStageSpec{typeof(cost_matrix), typeof(ε)}(choice_axis, cost_matrix, ε)

@definestage LogitChoiceStage LogitChoiceStageSpec


##########################
# Gridded implementation #
##########################

# LogitChoiceKernel — the Gumbel-logit transition's data #
#-------------------------------------------------------#
# The kernel IS the frozen `eψC = exp(−C/ε)` (a dense self-describing kernel) and the per-state
# `value_weight` (w) and `normalizer` (Z) buffers — nothing else. The transition is the
# Gibbs/Luce operator `π(j|i,s) = eψC[i,j]·w[j,s]/Z[i,s]`; its two verbs reuse the dense
# contraction (`forward!`/`backward!` on `eψC`) over the dep batch, never the dense origin×dest
# tensor. `backward!` seats all three; `forward!` and the Jacobian adjoints reuse them.

"""
The Gumbel-logit transition's data: the frozen `eψC = exp(−C/ε)` (a dense self-describing
kernel over `(origin, dest, deps…)`) and the per-(dest, state)/(origin, state) `value_weight`
and `normalizer` buffers — both full layout-shaped — seated by the stage `backward!`. The
operator is the Gibbs/Luce form `π(j|i,s) = eψC[i,j]·value_weight[j,s]/normalizer[i,s]`.
"""
struct LogitChoiceKernel{M<:AbstractArray, D<:AbstractArray}
    eψC          :: M    # exp(−C/ε): dense kernel over (origin, dest, deps…)
    value_weight :: D    # exp((V−maxⱼV)/ε)[j,s]: per-(dest, state) softmax numerator, layout-shaped
    normalizer   :: D    # (Σⱼ eψC[i,j]·value_weight[j,s])[i,s]: per-(origin, state) denominator, layout-shaped
end

function allocate_kernel(spec::LogitChoiceStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    eψC = dense_kernel(T, layout, spec.choice_axis, spec.cost_matrix)
    return LogitChoiceKernel(eψC, allocate_buffer(layout, T), allocate_buffer(layout, T))
end

"Scratch: the io buffers + the choice-axis row-max `rowmax` (a backward-seating buffer). The K/Kᵀ apply plan — the `eψC` gather scratch + `tmp` — is the kernel's `kernel_scratch`, merged in by `@definestage`."
function allocate_scratch(spec::LogitChoiceStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    rowmax = zeros(T, Base.setindex(layout_size(layout), 1, axis_position(layout, spec.choice_axis)))
    return merge(io_scratch(spec, layout, T), (rowmax = rowmax,))
end

"The logit kernel's apply plan: the `eψC` dense kernel's gather scratch + the `tmp` buffer the composite K/Kᵀ verbs stage the diagonal scalings through."
kernel_scratch(k::LogitChoiceKernel, layout::GriddedLayout, ::Type{T}) where {T} =
    merge(kernel_scratch(k.eψC, layout, T), (tmp = allocate_buffer(layout, T),))

# The Gibbs/Luce operator `K = diag(value_weight)·eψCᵀ·diag(1/normalizer)` and its transpose, as
# the kernel's two verbs — `forward! = K·`, `backward! = Kᵀ·`. The primal mass push AND the lift
# adjoints both route through these (the seated `value_weight`/`normalizer` make K the frozen
# choice-probability operator, envelope theorem). The `eψC` contraction gathers internally; `tmp`
# (the apply plan) stages the diagonal scalings.

"""
Apply `K· = value_weight ⊙ (eψCᵀ·(src ./ normalizer))` (the Gibbs mass-push operator).
"""
function forward!(dest, k::LogitChoiceKernel, src; scratch)
    @. scratch.tmp = src / k.normalizer
    backward!(dest, k.eψC, scratch.tmp; scratch = scratch)      # eψCᵀ · (src ./ normalizer)
    @. dest = k.value_weight * dest
    return dest
end

"""
Apply `Kᵀ· = (eψC·(value_weight ⊙ src)) ./ normalizer` (the value-pullback operator).
"""
function backward!(dest, k::LogitChoiceKernel, src; scratch)
    @. scratch.tmp = k.value_weight * src
    forward!(dest, k.eψC, scratch.tmp; scratch = scratch)       # eψC · (value_weight ⊙ src)
    @. dest = dest / k.normalizer
    return dest
end

# Backward / forward — the LSE directly (no Gibbs reward) #
#--------------------------------------------------------#
# The LSE factorises: `V_start[i,s] = rm[s] + ε·log(normalizer[i,s])`, so backward builds the
# kernel pieces the forward pass and adjoint reuse AND yields `V_start` directly — no separate
# `r + KᵀV` split, no Gibbs reward. A non-finite cost (infeasible move) gives `exp(−Inf/ε) = 0`.
#
# Unlike Markov, this backward does NOT call the kernel's `backward!` (Kᵀ·): the logit value
# recursion is the nonlinear LSE, not a linear `Kᵀ·V_end` (that would be the envelope-theorem
# linearization `Σⱼ π(j|i)·V_end[j]` — the adjoint, not the primal value). The job here is to
# compute the LSE and SEAT the frozen K; the `Kᵀ` verb is for `forward!` and the adjoints.

"""
Logit backward: materialise `eψC = exp(−C/ε)`, the per-state `value_weight`/`normalizer`, and
write `V_start = rm + ε·log(normalizer)` (the LSE) in layout order, seating the kernel for forward.
"""
function backward!(V_start, spec::LogitChoiceStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache)
    ε  = resolve(spec.ε, env)
    (; eψC, value_weight, normalizer) = kernel
    fill_field!(eψC, MappedField(C -> exp.(.- C ./ ε), spec.cost_matrix), layout, spec.choice_axis, env)
    maximum!(scratch.rowmax, V_end)                                          # rm[s] = maxⱼ V_end[j,s] (reduces the size-1 choice axis)
    @. value_weight = exp((V_end - scratch.rowmax) / ε)                       # softmax numerator
    forward!(normalizer, eψC, value_weight; scratch = scratch.kernel_scratch) # normalizer = eψC·value_weight
    @. V_start = scratch.rowmax + ε * log(normalizer)                         # the LSE, directly
    return (V_start, kernel)
end

# forward! (push mass `Λ_end = K·Λ_start` through the seated Gibbs operator) is the generic
# modern default (abstract.jl) — pure kernel application, no seating.

# Domain-named wrappers live in their own files (migration.jl, sector_switching.jl)
# to show how cheaply a `LogitChoiceStage` specialises into a named stage.


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
