###########################################
# lift_jacobian (forward-mode + adjoints) #
###########################################
#
# Forward-mode AD rebuilds a stage's buffer with eltype
# `ForwardDiff.Dual{Tag, T, N}`; running the existing `backward!`/`forward!`
# on Dual-typed inputs propagates tangents alongside primals.
#
# Reverse-mode is exposed via `backward_adjoint!`/`forward_adjoint!`. Per §13: a kernel that
# satisfies the linearization property needs NO bespoke adjoint — the generic
# `AbstractModernStage` methods below read both adjoints straight off the seated kernel
# (`forward_adjoint! = Kᵀ·` via `backward!`, `backward_adjoint! = K·` via `forward!`). The
# earlier "manual-per-stage-adjoint" debt is RESOLVED by that generic path: every
# linearization-satisfying kernel (Dense/Scatter/Interp/MPS/MeanVariance, plus ForgetfulSum,
# Argmax, LogitChoice, DeterministicContinuous) rides it with no override. The only surviving
# hand-written overrides are the legitimate exceptions, not debt — the asymmetric
# `PointwiseScaleStage` (a ≠ f, two unpaired diagonals) and the affine `EntryStage` (additive
# source ⇒ identity adjoints, in `stages/primitive/entry.jl`).

"""
Return a new stage whose buffer eltype is `T`, by rebundling the same Spec
against the same layout at the new `T`.
"""
with_eltype(stage::AbstractStage, ::Type{T}) where {T} =
    bundle(stage.spec, input_layout(stage), T)

# lift_jacobian #
#---------------#

"""
Forward-mode (`mode=:forward`): rebuild `stage` with
`ForwardDiff.Dual{tag, primal_eltype, n_dual}`-typed buffers.
Reverse-mode (`mode=:reverse`) returns the stage unchanged; reverse
is exposed via the per-stage adjoint methods below.
"""
function lift_jacobian(stage::AbstractStage;
                       mode::Symbol=:forward,
                       n_dual::Int=1,
                       tag::Type=Nothing,
                       primal_eltype::Type=Float64)
    if mode === :forward
        return with_eltype(stage, ForwardDiff.Dual{tag, primal_eltype, n_dual})
    elseif mode === :reverse
        return stage
    else
        error("lift_jacobian: unknown mode :$mode")
    end
end

##########################
# Reverse-mode adjoints #
##########################
#
# Conventions. K is the stage's K-operator (linear on measures /
# functions).
#
#   forward (primal)        Λ_end   = K · Λ_start
#   backward (primal)       V_start = K^T · V_end  (+ flow payoff r)
#   forward_adjoint (VJP)   dΛ_start = K^T · dΛ_end
#   backward_adjoint (VJP)  dV_end   = K · dV_start

"""
VJP of `backward!` w.r.t. `V_end`: given `dV_start = ∂L/∂V_start`, returns
`dV_end = ∂L/∂V_end`.
"""
backward_adjoint!(spec::AbstractStageSpec, dV_start, buffer) =
    error("backward_adjoint! not implemented for $(typeof(spec))")
backward_adjoint!(stage::AbstractLegacyStage, dV_start) =
    backward_adjoint!(stage.spec, dV_start, stage.buffer)

"""
VJP of `forward!` w.r.t. `Λ_start`: given `dΛ_end = ∂L/∂Λ_end`, returns
`dΛ_start = ∂L/∂Λ_start`.
"""
forward_adjoint!(spec::AbstractStageSpec, dΛ_end, buffer) =
    error("forward_adjoint! not implemented for $(typeof(spec))")
forward_adjoint!(stage::AbstractLegacyStage, dΛ_end) =
    forward_adjoint!(stage.spec, dΛ_end, stage.buffer)

# Modern stages: the adjoint of a linear, adjoint-paired transition is the kernel applied the
# OTHER way — `forward_adjoint! = Kᵀ·` (the backward verb), `backward_adjoint! = K·` (the forward
# verb) — read straight off `stage.kernel`, threading the kernel's plan. The adjoint output is
# allocated at the stage's input/output layout (NOT `similar(cotangent)`), so a SHAPE-CHANGING
# kernel — ForgetfulSum's rectangular ones-row, or any introduce/crosswalk where input and output
# sizes differ — is handled too; for a square kernel the layout size equals the cotangent's, so
# it is identical. Only a non-paired kernel (the two-sided `PointwiseScale` with `a ≠ f`) overrides below.
# NB: the adjoint output is sized from the SPEC-resized layout (`input_layout(spec, layout)`), not
# the `input_layout(stage)` accessor — the latter returns the construction layout (which `product.jl`
# relies on), so for a shape-changing kernel (rectangular logit collapse, forget) it would mis-size.
forward_adjoint!(stage::AbstractModernStage, dΛ_end) =
    backward!(similar(dΛ_end, layout_size(input_layout(stage.spec, stage.layout))), stage.kernel, dΛ_end;
              scratch = stage.scratch.kernel_scratch)   # Kᵀ · dΛ_end → input shape
backward_adjoint!(stage::AbstractModernStage, dV_start) =
    forward!(similar(dV_start, layout_size(output_layout(stage.spec, stage.layout))), stage.kernel, dV_start;
             scratch = stage.scratch.kernel_scratch)    # K · dV_start → output shape

# PointwiseScaleStage (discount / reproduction / renorm) #
#-------------------------------------------------------#
# A `PointwiseScale` is an adjoint pair only when its two scales agree, so override the generic
# modern adjoint with the per-direction diagonal: backward K is a·Id ⇒ VJP dV_end = a·dV_start;
# forward K is f·Id ⇒ dΛ_start = f·dΛ_end. Both scales read off the seated kernel `Ref`s (tracking
# a FromEnv/AD-shocked scale too). TimeDiscountingStage rides this as `a = β`, `f = 1`.

backward_adjoint!(stage::PointwiseScaleStage, dV_start) = _scale(stage.kernel) .* dV_start
forward_adjoint!(stage::PointwiseScaleStage, dΛ_end)    = _forward_scale(stage.kernel) .* dΛ_end

# ForgetfulSumStage rides the generic adjoint above: its rectangular ones-row K is a genuine
# (shape-changing) transition, and the generic now allocates at the input/output layout, so
# `forward_adjoint! = Kᵀ·` (broadcast across the forget axis) and `backward_adjoint! = K·` (sum
# along it) fall out with no override.

# ChainStage #
#------------#

# The chain buffer holds sub-STAGES; drive each through the stage-level adjoint
# (dispatches legacy/modern per component).
function backward_adjoint!(spec::ChainStageSpec, dV_start, buffer::ChainStageBuffer)
    dV = dV_start
    for s in buffer.stages
        dV = backward_adjoint!(s, dV)
    end
    return dV
end

function forward_adjoint!(spec::ChainStageSpec, dΛ_end, buffer::ChainStageBuffer)
    dΛ = dΛ_end
    for i in length(buffer.stages):-1:1
        dΛ = forward_adjoint!(buffer.stages[i], dΛ)
    end
    return dΛ
end

# Argmax / LogitChoice / DeterministicContinuous all ride the generic
# AbstractModernStage adjoint above. Each kernel is a genuine transition whose `forward!`/
# `backward!` verbs apply K/Kᵀ — the single-destination scatter/clip-gather, and the logit Gibbs
# operator — frozen at the solved point by the envelope theorem. So `forward_adjoint! = Kᵀ·` and
# `backward_adjoint! = K·` fall out with no override.
# DeterministicContinuous's `backward_adjoint! = K·` is an exact transpose at BOTH ends: the
# both-ends off-grid clamp (a3a3409) makes forward/backward an exact K/Kᵀ pair for all destinations,
# interior and off-grid (end-goal §8/§13; asserted to 1e-14 by the off-grid transpose test).
