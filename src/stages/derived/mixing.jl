# Mixing / retention (closed-form rung a) — derived sugar over TWO MarkovStage applies + a
# pointwise conjugate, NOT a new primitive. The household chooses a blend `θ ∈ [0,1]` of two
# x-transitions `K_θ = θ·K_A + (1−θ)·K_B` at convex cost `c(θ)`. Because `K_θ V = θ a + (1−θ) b`
# is LINEAR in θ (`a = K_A V`, `b = K_B V`), the θ-argmax has the closed form
#
#     V(x) = max_{θ∈[0,1]} [ θ(a−b) + b − c(θ) ] = b + c*(a − b),   θ*(x) = (c')⁻¹(a − b),
#
# where `c*` is the Fenchel conjugate of `c` (CHOICE_STAGE_CATALOG rung a). No θ-axis grid: two
# Markov backward applies and a pointwise `(conjugate, policy)`. Once θ* is frozen, the stage is a
# per-state mixture `K_{θ*}` with reward `−c(θ*)`, so the forward pushes mass through that mixture
# and the frozen-policy duality holds. **Retention** ("pay not to transition") is `K_A = I`:
# `V = (exit value) + c*(W − exit value)`.

"Quadratic-cost conjugate pair `c(θ) = θ²/(2κ)` on `θ ∈ [0,1]`: policy `θ*(y)=clamp(κy,0,1)` and conjugate `c*(y)=θ*·y − θ*²/(2κ)`."
_quad_policy(κ)     = y -> clamp(κ * y, zero(y), one(y))
_quad_conjugate(κ)  = y -> (t = clamp(κ * y, zero(y), one(y)); t * y - t^2 / (2κ))
_identity_matrix(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

struct MixingStageSpec{Cj, Ct, Pl} <: AbstractStageSpec
    K_A       :: Matrix{Float64}     # row-stochastic x-transition for the θ=1 corner
    K_B       :: Matrix{Float64}     # row-stochastic x-transition for the θ=0 corner (the default)
    axis    :: Symbol
    conjugate :: Cj                  # c*(y): the Fenchel conjugate of the cost
    cost      :: Ct                  # c(θ): the cost, evaluated at θ* for the reward/duality term
    policy    :: Pl                  # θ*(y) = (c')⁻¹(y), clamped to [0,1]
end

"Buffer: the two bundled `MarkovStage`s (each holds its own V/Λ scratch), the seated policy `θ*`, a mass-split scratch, and the fused V/Λ outputs."
# The scratch tensors ride a free `A<:AbstractArray` type parameter (not a concrete `Array{T,N}`) so
# the GPU lift `to_device(::MixingStage, …)` can rebuild this buffer with device-resident scratch —
# the convention the modern-stage/`ProductStageBuffer` lifts document (lifts/gpu.jl, direct_sum.jl).
# On the host `A` infers to `Array{Float64,N}`, leaving CPU behavior unchanged.
struct MixingStageBuffer{MA, MB, A<:AbstractArray, L} <: AbstractStageBuffer
    markA         :: MA
    markB         :: MB
    policy        :: A
    mass_share    :: A
    V_start       :: A
    Λ_end         :: A
    input_layout  :: L
    output_layout :: L
end

struct MixingStage{Spec<:MixingStageSpec, Buffer<:MixingStageBuffer} <: AbstractLegacyStage
    spec   :: Spec
    buffer :: Buffer
end

"""
Mixing choice over a blend `θ∈[0,1]` of two `axis` transitions `K_A` (θ=1) and `K_B` (θ=0, the
default), at convex cost: backward `V = b + c*(a−b)` (closed-form, two Markov applies + a pointwise
conjugate — no θ grid). The cost defaults to quadratic `θ²/(2·cost_curvature)`; pass `conjugate`/
`policy`/`cost` to override. See [`RetentionStage`](@ref) for the `K_A = I` ("pay not to move") case.
"""
function MixingStage(layout::GriddedLayout; axis::Symbol, K_A, K_B,
                     cost_curvature::Real = 1.0,
                     conjugate = _quad_conjugate(cost_curvature),
                     policy    = _quad_policy(cost_curvature),
                     cost      = θ -> θ^2 / (2 * cost_curvature))
    spec = MixingStageSpec{typeof(conjugate), typeof(cost), typeof(policy)}(
        Matrix{Float64}(K_A), Matrix{Float64}(K_B), axis, conjugate, cost, policy)
    return MixingStage(spec, layout)
end

"""
Retention — pay convex cost to STAY (`θ` fraction via identity) rather than transition (`1−θ` via
`exit_kernel`): a [`MixingStage`](@ref) with `K_A = I`. `V = (exit value) + c*(W − exit value)`.
"""
RetentionStage(layout::GriddedLayout; axis::Symbol, exit_kernel, kwargs...) =
    MixingStage(layout; axis, K_A = _identity_matrix(_axis_size(layout, axis)),
                K_B = exit_kernel, kwargs...)

# Allocate / bundle #
#-------------------#

function allocate(spec::MixingStageSpec, layout::GriddedLayout, ::Type{T}=Float64) where {T}
    markA = MarkovStage(layout; axis = spec.axis, transition_matrix = spec.K_A)
    markB = MarkovStage(layout; axis = spec.axis, transition_matrix = spec.K_B)
    z()   = zeros(T, layout_size(layout))
    return MixingStageBuffer(markA, markB, z(), z(), z(), z(), layout, layout)
end

MixingStage(spec::MixingStageSpec, layout::GriddedLayout, ::Type{T}=Float64) where {T} =
    MixingStage(spec, allocate(spec, layout, T))
bundle(spec::MixingStageSpec, layout::GriddedLayout) = MixingStage(spec, layout)
bundle(spec::MixingStageSpec, layout::GriddedLayout, ::Type{T}) where {T} = MixingStage(spec, layout, T)

effective_env_slice(spec::MixingStageSpec) =
    _union_env_slices((MarkovStageSpec(; axis = spec.axis, transition_matrix = spec.K_A),
                       MarkovStageSpec(; axis = spec.axis, transition_matrix = spec.K_B)))

# Backward / forward #
#--------------------#

function backward!(buffer::MixingStageBuffer, spec::MixingStageSpec, V_end, env;
                   env_changed::Bool = true)
    a = backward!(buffer.markA, V_end, env; env_changed)              # K_A · V_end
    b = backward!(buffer.markB, V_end, env; env_changed)              # K_B · V_end
    @. buffer.policy  = a - b                            # y = a − b (reuse policy as scratch)
    @. buffer.V_start = b + spec.conjugate(buffer.policy)  # V = b + c*(y)
    @. buffer.policy  = spec.policy(buffer.policy)       # θ*(x) = (c')⁻¹(y) ∈ [0,1]
    return buffer.V_start
end

function forward!(buffer::MixingStageBuffer, spec::MixingStageSpec, Λ_start)
    @. buffer.mass_share = buffer.policy * Λ_start
    ΛA = forward!(buffer.markA, buffer.mass_share)       # θ* share pushed through K_A
    @. buffer.mass_share = (1 - buffer.policy) * Λ_start
    ΛB = forward!(buffer.markB, buffer.mass_share)       # (1−θ*) share through K_B
    @. buffer.Λ_end = ΛA + ΛB
    return buffer.Λ_end
end

"The seated mixing weight `θ*(x)` (∈ [0,1]) from the last `backward!` — the fraction routed through `K_A`."
policy(stage::MixingStage) = stage.buffer.policy

"The seated flow reward `−c(θ*(x))` (the cost paid on the chosen mix) — the term in the frozen-policy duality `⟨V_start,Λ⟩ = ⟨reward,Λ⟩ + ⟨V_end,Λ_end⟩`."
reward(stage::MixingStage) = .-stage.spec.cost.(stage.buffer.policy)
