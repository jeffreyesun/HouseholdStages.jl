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
Logit (Gumbel) choice over a discrete `axis`, with origin→destination
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
    axis :: Symbol
    cost_matrix :: Cmat                                    # C[origin, dest]: a Matrix, FromEnv, or closure (; dep…, env) -> C
    ε           :: T                                       # Gumbel scale (literal or FromEnv). ε < 0 ⇒ soft-MIN (robust)
end

LogitChoiceStageSpec(; axis::Symbol, cost_matrix, ε = 1.0) =
    LogitChoiceStageSpec{typeof(cost_matrix), typeof(ε)}(axis, cost_matrix, ε)

@definestage LogitChoiceStage LogitChoiceStageSpec


##########################
# Gridded implementation #
##########################

# The Gumbel-logit transition's data lives in `LogitChoiceKernel` (kernels/logit_kernel.jl); the
# stage methods below build, seat, and read it.

# Rectangular cost (origin ≠ dest, e.g. origin = 1 = the log-sum-exp collapse): the origin/`V_start`
# side resizes to the cost's origin dim. General — square is the no-op `n_origin == n_dest`.
input_layout(spec::LogitChoiceStageSpec, layout::GriddedLayout) =
    resize_axis(layout, spec.axis, _field_shape(spec.cost_matrix, layout, spec.axis)[1])

function allocate_kernel(spec::LogitChoiceStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    eψC    = dense_kernel(T, layout, spec.axis, spec.cost_matrix)
    kernel = LogitChoiceKernel(eψC, allocate_buffer(layout, T),                       # value_weight (dest side)
                                    allocate_buffer(input_layout(spec, layout), T))   # normalizer (origin side)
    if !(reads_env(spec.cost_matrix) || spec.ε isa FromEnv)                           # env-independent: seat eψC once
        ε = spec.ε
        fill_field!(eψC, MappedField(C -> exp.(.- C ./ ε), spec.cost_matrix), layout, spec.axis, nothing)
    end
    return kernel
end

"Cache: the static env-dependence classification for the cost field `eψC = exp(−C/ε)` (computed once),
so a fixed-env VFI loop seats it once (§5.3). Env-dependent iff the cost source reads env OR `ε` is
`FromEnv` (the map bakes `ε` into the field); an env-independent field is seated at construction."
allocate_cache(spec::LogitChoiceStageSpec, ::Type, ::GriddedLayout) =
    (cost_env_dep = reads_env(spec.cost_matrix) || spec.ε isa FromEnv,)

"Scratch: the io buffers + the choice-axis row-max `rowmax` (a backward-seating buffer)."
function allocate_scratch(spec::LogitChoiceStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    rowmax = zeros(T, Base.setindex(layout_size(layout), 1, axis_position(layout, spec.axis)))
    return merge(io_scratch(spec, layout, T), (rowmax = rowmax,))
end

# The kernel's apply plan (`kernel_scratch`) and its two verbs (`forward! = K·`, `backward! = Kᵀ·`,
# the Gibbs operator) live with `LogitChoiceKernel` in kernels/logit_kernel.jl.

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
# Robust soft-MIN for free: a NEGATIVE `ε` gives the entropic-risk CE `ε·log Σⱼ eψC·exp(V_end/ε)`.
# Shifting by `maxⱼ(V_end/ε)` is the right stability shift for either sign (the division flips the
# order when `ε < 0`), so no branch is needed.

function backward!(V_start, spec::LogitChoiceStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    ε  = resolve(spec.ε, env)
    (; eψC, value_weight, normalizer) = kernel
    cache.cost_env_dep && env_changed &&                                     # seat eψC (static refill)
        fill_field!(eψC, MappedField(C -> exp.(.- C ./ ε), spec.cost_matrix), layout, spec.axis, env)
    @. value_weight = V_end / ε                                              # scaled continuation u = V_end/ε
    maximum!(scratch.rowmax, value_weight)                                   # m = maxⱼ u — the sign-correct stability shift
    @. value_weight = exp(value_weight - scratch.rowmax)                     # exp(u − m) ∈ (0,1] for either sign of ε
    forward!(normalizer, eψC, value_weight; scratch = scratch.kernel_scratch) # normalizer = eψC·value_weight
    @. V_start = ε * (scratch.rowmax + log(normalizer))                      # ε·(m + log Z): the LSE / entropic-risk CE
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
