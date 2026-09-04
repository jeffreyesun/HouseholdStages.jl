###########################################
# lift_jacobian (forward-mode + adjoints) #
###########################################
#
# Forward-mode AD by rebuilding a stage's buffer at a Dual eltype, and reverse-mode VJPs.

"Rebundle a stage's spec against its own two boundary layouts at buffer eltype `T`."
with_eltype(stage::AbstractPrimitiveStage, ::Type{T}) where {T} =
    bundle(stage.spec, start_layout(stage), end_layout(stage), T)

with_eltype(chain::ChainStage, ::Type{T}) where {T} =
    ChainStage(chain.spec, boundaries(chain), interiors(chain), T)

with_eltype(s::ProductStage, ::Type{T}) where {T} =
    ProductStage(s.spec, start_layout(s), end_layout(s), interiors(s), T)

# lift_jacobian #
#---------------#

"""
The default ForwardDiff tag for package-built forward lifts.
"""
struct HhsLift end
const HhsLiftTag = typeof(ForwardDiff.Tag(HhsLift(), Float64))

"Rebuild `stage` with `ForwardDiff.Dual{tag, primal_eltype, n_dual}`-typed buffers, carrying tangents."
lift_jacobian(stage::AbstractStage; n_dual::Int=1, tag::Type=HhsLiftTag,
              primal_eltype::Type=Float64) =
    with_eltype(stage, ForwardDiff.Dual{tag, primal_eltype, n_dual})

# Reverse-mode adjoints #
#-----------------------#
#
#   forward (primal)        Λ_end   = K · Λ_start
#   backward (primal)       V_start = K^T · V_end  (+ flow payoff r)
#   forward_adjoint (VJP)   dΛ_start = K^T · dΛ_end
#   backward_adjoint (VJP)  dV_end   = K · dV_start

"VJP of `backward!` w.r.t. `V_end`: `dV_start = ∂L/∂V_start ↦ dV_end = ∂L/∂V_end`."
backward_adjoint!(spec::AbstractStageSpec, dV_start, buffer) =
    error("backward_adjoint! not implemented for $(typeof(spec))")
backward_adjoint!(stage::AbstractCompositeStage, dV_start) =
    backward_adjoint!(stage.spec, dV_start, stage.buffer)

"VJP of `forward!` w.r.t. `Λ_start`: `dΛ_end = ∂L/∂Λ_end ↦ dΛ_start = ∂L/∂Λ_start`."
forward_adjoint!(spec::AbstractStageSpec, dΛ_end, buffer) =
    error("forward_adjoint! not implemented for $(typeof(spec))")
forward_adjoint!(stage::AbstractCompositeStage, dΛ_end) =
    forward_adjoint!(stage.spec, dΛ_end, stage.buffer)

# Primitive stages: `stage.kernel` applied the other way, allocated at the stage's far layout.
forward_adjoint!(stage::AbstractPrimitiveStage, dΛ_end) =
    backward!(similar(dΛ_end, layout_size(start_layout(stage))), stage.kernel, dΛ_end;
              scratch = stage.scratch.kernel_scratch)   # Kᵀ · dΛ_end → start shape
backward_adjoint!(stage::AbstractPrimitiveStage, dV_start) =
    forward!(similar(dV_start, layout_size(end_layout(stage))), stage.kernel, dV_start;
             scratch = stage.scratch.kernel_scratch)    # K · dV_start → end shape

# PointwiseScaleStage (discount / reproduction / renorm) #
#-------------------------------------------------------#

backward_adjoint!(stage::PointwiseScaleStage, dV_start) = _scale(stage.kernel) .* dV_start
forward_adjoint!(stage::PointwiseScaleStage, dΛ_end)    = _forward_scale(stage.kernel) .* dΛ_end

# ChainStage #
#------------#

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

# ProductStage #
#--------------#

function backward_adjoint!(spec::ProductStageSpec, dV_start, buffer::ProductStageBuffer)
    pdim   = axis_position(buffer.start_layout, spec.axis)
    dV_end = similar(dV_start, layout_size(buffer.end_layout))
    for (i, s) in enumerate(buffer.components)
        selectdim(dV_end, pdim, i:i) .= backward_adjoint!(s, selectdim(dV_start, pdim, i:i))
    end
    return dV_end
end

function forward_adjoint!(spec::ProductStageSpec, dΛ_end, buffer::ProductStageBuffer)
    pdim     = axis_position(buffer.start_layout, spec.axis)
    dΛ_start = similar(dΛ_end, layout_size(buffer.start_layout))
    for (i, s) in enumerate(buffer.components)
        selectdim(dΛ_start, pdim, i:i) .= forward_adjoint!(s, selectdim(dΛ_end, pdim, i:i))
    end
    return dΛ_start
end
