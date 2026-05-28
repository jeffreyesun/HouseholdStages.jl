###########################################
# lift_jacobian (forward-mode + adjoints) #
###########################################
#
# Forward-mode AD rebuilds a stage's buffer with eltype
# `ForwardDiff.Dual{Tag, T, N}`; running the existing
# `backward!` / `forward!` methods on Dual-typed inputs propagates
# tangents alongside primals.
#
# Reverse-mode is exposed via per-stage `backward_adjoint!` /
# `forward_adjoint!` methods. The user has flagged this manual-
# per-stage-adjoint pattern as architectural debt to be redesigned
# separately (see memory: project_manual_adjoints_concern.md). The
# methods here are ported mechanically through the 2026-05-25
# protocol refactor.

"""
    with_eltype(stage, T::Type) -> rebuilt

Return a new stage whose buffer eltype is `T`. Under the
2026-05-25 refactor the Spec is layout/T-free, so this is a single
generic operation: rebundle the same Spec against the same layout
at the new T.
"""
with_eltype(stage::AbstractStage, ::Type{T}) where {T} =
    bundle(stage.spec, stage.buffer.input_layout, T)

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
    backward_adjoint!(stage, dV_start) -> dV_end
    backward_adjoint!(spec, dV_start, buffer) -> dV_end

VJP of `backward!` w.r.t. `V_end`: given `dV_start = ∂L/∂V_start`,
returns `dV_end = ∂L/∂V_end`.
"""
backward_adjoint!(spec::AbstractStageSpec, dV_start, buffer) =
    error("backward_adjoint! not implemented for $(typeof(spec))")
backward_adjoint!(stage::AbstractStage, dV_start) =
    backward_adjoint!(stage.spec, dV_start, stage.buffer)

"""
    forward_adjoint!(stage, dΛ_end) -> dΛ_start
    forward_adjoint!(spec, dΛ_end, buffer) -> dΛ_start

VJP of `forward!` w.r.t. `Λ_start`: given `dΛ_end = ∂L/∂Λ_end`,
returns `dΛ_start = ∂L/∂Λ_start`.
"""
forward_adjoint!(spec::AbstractStageSpec, dΛ_end, buffer) =
    error("forward_adjoint! not implemented for $(typeof(spec))")
forward_adjoint!(stage::AbstractStage, dΛ_end) =
    forward_adjoint!(stage.spec, dΛ_end, stage.buffer)

# MarkovStage #
#-------------#

function backward_adjoint!(spec::MarkovStageSpec, dV_start, buffer)
    dV_end = similar(dV_start)
    _markov_apply!(dV_end, dV_start, spec.transition',
                   buffer.input_layout, spec.axis,
                   buffer.scratch.perm_in, buffer.scratch.perm_out)
    return dV_end
end

function forward_adjoint!(spec::MarkovStageSpec, dΛ_end, buffer)
    dΛ_start = similar(dΛ_end)
    _markov_apply!(dΛ_start, dΛ_end, spec.transition,
                   buffer.input_layout, spec.axis,
                   buffer.scratch.perm_in, buffer.scratch.perm_out)
    return dΛ_start
end

# IdentityStage #
#---------------#

backward_adjoint!(::IdentityStageSpec, dV_start::AbstractArray, _) = copy(dV_start)
forward_adjoint!(::IdentityStageSpec, dΛ_end::AbstractArray, _)    = copy(dΛ_end)

# ForgetfulSumStage #
#-------------------#
# Primal forward: Λ_end[c]  = Σ_{c': drop c'→c} Λ_start[c']  (sum along forget axis)
# Primal backward: V_start[c'] = V_end[c]                     (broadcast)
# Adjoint of forward: dΛ_start = K^T · dΛ_end (broadcast along forget axis)
# Adjoint of backward: dV_end = K · dV_start (sum along forget axis)

function forward_adjoint!(spec::ForgetfulSumStageSpec, dΛ_end, buffer)
    forget_dim = axis_position(buffer.input_layout, spec.forget_axis)
    dims_in    = layout_size(buffer.input_layout)
    dims_out   = layout_size(buffer.output_layout)
    @assert size(dΛ_end) == dims_out
    dΛ_start = zeros(eltype(dΛ_end), dims_in)
    shape = _insert_singleton(dims_out, forget_dim)
    dΛ_start .= reshape(dΛ_end, shape)
    return dΛ_start
end

function backward_adjoint!(spec::ForgetfulSumStageSpec, dV_start, buffer)
    forget_dim = axis_position(buffer.input_layout, spec.forget_axis)
    dims_in    = layout_size(buffer.input_layout)
    dims_out   = layout_size(buffer.output_layout)
    @assert size(dV_start) == dims_in
    dV_end = zeros(eltype(dV_start), dims_out)
    shape  = _insert_singleton(dims_out, forget_dim)
    sum!(reshape(dV_end, shape), dV_start)
    return dV_end
end

# ChainStage #
#------------#

function backward_adjoint!(spec::ChainStageSpec, dV_start, buffer::ChainStageBuffer)
    dV = dV_start
    for i in eachindex(spec.stages)
        dV = backward_adjoint!(spec.stages[i], dV, buffer.stages[i])
    end
    return dV
end

function forward_adjoint!(spec::ChainStageSpec, dΛ_end, buffer::ChainStageBuffer)
    dΛ = dΛ_end
    for i in length(spec.stages):-1:1
        dΛ = forward_adjoint!(spec.stages[i], dΛ, buffer.stages[i])
    end
    return dΛ
end

# ArgmaxStage #
#-------------#

function backward_adjoint!(spec::ArgmaxStageSpec, dV_in, buffer)
    layout     = buffer.input_layout
    choice_dim = axis_position(layout, spec.choice_axis)
    policy     = buffer.kernel.policy
    actions    = axisvalues(layout.axes[choice_dim])
    T          = eltype(dV_in)
    dV_out     = zeros(T, size(dV_in))

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        action  = actions[policy[ci_in]]
        next_axis_i = spec.next_state_idx(cell, action)
        out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
        dV_out[CartesianIndex(out_idxs)] += dV_in[ci_in]
    end
    return dV_out
end

function forward_adjoint!(spec::ArgmaxStageSpec, dΛ_end, buffer)
    layout     = buffer.input_layout
    choice_dim = axis_position(layout, spec.choice_axis)
    policy     = buffer.kernel.policy
    actions    = axisvalues(layout.axes[choice_dim])
    dΛ_start   = similar(dΛ_end)

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        action  = actions[policy[ci_in]]
        next_axis_i = spec.next_state_idx(cell, action)
        out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
        dΛ_start[ci_in] = dΛ_end[CartesianIndex(out_idxs)]
    end
    return dΛ_start
end

# LogitChoiceStage #
#------------------#

function backward_adjoint!(spec::LogitChoiceStageSpec, dV_in, buffer)
    layout     = buffer.input_layout
    choice_dim = axis_position(layout, spec.choice_axis)
    actions    = axisvalues(layout.axes[choice_dim])
    prob       = buffer.kernel.choice_prob
    T          = eltype(dV_in)
    dV_out     = zeros(T, size(dV_in))

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        d       = dV_in[ci_in]
        iszero(d) && continue
        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = spec.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            dV_out[CartesianIndex(out_idxs)] += d * p
        end
    end
    return dV_out
end

function forward_adjoint!(spec::LogitChoiceStageSpec, dΛ_end, buffer)
    layout     = buffer.input_layout
    choice_dim = axis_position(layout, spec.choice_axis)
    actions    = axisvalues(layout.axes[choice_dim])
    prob       = buffer.kernel.choice_prob
    T          = eltype(dΛ_end)
    dΛ_start   = zeros(T, size(dΛ_end))

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        acc = zero(T)
        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = spec.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            acc += p * dΛ_end[CartesianIndex(out_idxs)]
        end
        dΛ_start[ci_in] = acc
    end
    return dΛ_start
end

# MigrationStage #
#----------------#

function backward_adjoint!(spec::MigrationStageSpec, dV_in, buffer)
    layout       = buffer.input_layout
    location_dim = axis_position(layout, spec.location_axis)
    n_loc        = axissize(layout.axes[location_dim])
    prob         = buffer.kernel.choice_prob
    T            = eltype(dV_in)
    dV_out       = zeros(T, size(dV_in))
    dims         = layout_size(layout)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        d = dV_in[ci]
        iszero(d) && continue
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, location_dim)
            dV_out[CartesianIndex(out_idxs)] += d * p
        end
    end
    return dV_out
end

function forward_adjoint!(spec::MigrationStageSpec, dΛ_end, buffer)
    layout       = buffer.input_layout
    location_dim = axis_position(layout, spec.location_axis)
    n_loc        = axissize(layout.axes[location_dim])
    prob         = buffer.kernel.choice_prob
    T            = eltype(dΛ_end)
    dΛ_start     = zeros(T, size(dΛ_end))
    dims         = layout_size(layout)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        acc = zero(T)
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, location_dim)
            acc += p * dΛ_end[CartesianIndex(out_idxs)]
        end
        dΛ_start[ci] = acc
    end
    return dΛ_start
end

# ConsumptionSavingsStage #
#-------------------------#
# Primal forward: K is the sparse permutation defined by the policy.

function forward_adjoint!(spec::ConsumptionSavingsStageSpec, dΛ_end, buffer)
    layout     = buffer.input_layout
    wealth_dim = axis_position(layout, spec.wealth_axis)
    policy     = buffer.kernel.policy
    dΛ_start   = similar(dΛ_end)

    for (idx, _) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        a1_i    = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a1_i, wealth_dim)
        dΛ_start[ci_in] = dΛ_end[CartesianIndex(out_idxs)]
    end
    return dΛ_start
end

function backward_adjoint!(spec::ConsumptionSavingsStageSpec, dV_in, buffer)
    layout     = buffer.input_layout
    wealth_dim = axis_position(layout, spec.wealth_axis)
    policy     = buffer.kernel.policy
    T          = eltype(dV_in)
    dV_out     = zeros(T, size(dV_in))

    for (idx, _) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        a1_i    = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a1_i, wealth_dim)
        dV_out[CartesianIndex(out_idxs)] += dV_in[ci_in]
    end
    return dV_out
end

# WealthChangeStage #
#-------------------#
# Primal forward: K is the share-based linear redistribution of mass at
# per-cell source positions `wpost` onto the canonical wgrid. The
# K-transpose at a source cell is the share-weighted gather from
# wgrid's two neighbors of `wpost[s]`.

"""
K-transpose gather for [`WealthChangeStage`](@ref): linear interpolation
of `dΛ_end` at each source position in `wpost_slice`. Underflow clips
to the left endpoint; overflow to the right (matching
`convert_distribution!`'s policy).
"""
function _share_gather!(dΛ_start_slice::AbstractVector{T},
                        dΛ_end::AbstractVector{T},
                        wpost_slice::AbstractVector{T},
                        wgrid::AbstractVector{T}) where {T}
    n_w  = length(wgrid)
    w_lo = wgrid[1]
    w_hi = wgrid[end]
    j = 1
    @inbounds for i in eachindex(dΛ_start_slice)
        wi = wpost_slice[i]
        if wi < w_lo
            dΛ_start_slice[i] = dΛ_end[1]
        elseif wi >= w_hi
            dΛ_start_slice[i] = dΛ_end[end]
        else
            while wi >= wgrid[j+1]
                j += 1
                j == n_w - 1 && break
            end
            while j > 1 && wi < wgrid[j]
                j -= 1
            end
            left_share = (wgrid[j+1] - wi) / (wgrid[j+1] - wgrid[j])
            dΛ_start_slice[i] =
                left_share * dΛ_end[j] + (1 - left_share) * dΛ_end[j+1]
        end
    end
    return dΛ_start_slice
end

function forward_adjoint!(spec::WealthChangeStageSpec, dΛ_end, buffer)
    layout     = buffer.input_layout
    wealth_dim = axis_position(layout, spec.wealth_axis)
    T          = eltype(dΛ_end)
    wgrid      = collect(T, axisvalues(layout.axes[wealth_dim]))
    wpost      = buffer.kernel.wealth_post
    N          = ndims(dΛ_end)

    dΛ_start = similar(dΛ_end)
    dims     = size(dΛ_end)
    other    = ntuple(i -> i == wealth_dim ? 1 : dims[i], N)

    if wealth_dim == 1
        for other_ci in CartesianIndices(other)
            tail = other_ci.I[2:end]
            _share_gather!(view(dΛ_start, :, tail...),
                           view(dΛ_end,   :, tail...),
                           view(wpost,    :, tail...), wgrid)
        end
    else
        perm     = _bring_dim_first(N, wealth_dim)
        inv_perm = invperm(perm)
        dΛ_end_p   = permutedims(dΛ_end, perm)
        wpost_p    = permutedims(wpost,  perm)
        dΛ_start_p = similar(dΛ_end_p)
        permdims = ntuple(i -> dims[perm[i]], N)
        other_p  = ntuple(i -> i == 1 ? 1 : permdims[i], N)
        for other_ci in CartesianIndices(other_p)
            tail = other_ci.I[2:end]
            _share_gather!(view(dΛ_start_p, :, tail...),
                           view(dΛ_end_p,   :, tail...),
                           view(wpost_p,    :, tail...), wgrid)
        end
        dΛ_start .= permutedims(dΛ_start_p, inv_perm)
    end
    return dΛ_start
end

# WealthChangeStage backward_adjoint!: not yet implemented. SSJ's
# expectation_vectors uses only forward_adjoint!.
function backward_adjoint!(::WealthChangeStageSpec, dV_start, buffer)
    error("WealthChangeStage.backward_adjoint! not implemented")
end
