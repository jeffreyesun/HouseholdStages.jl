# Search-and-matching — a dedicated stage (SAM_RI_STAGE_PROPOSAL.md §3, Route B).
#
#   V_unemp(x) = max_e [ −cost(e) + p(e,θ)·V_emp′(x) + (1−p(e,θ))·V_unemp′(x) ]
#   V_emp(x)   = (1−δ)·V_emp′(x) + δ·V_unemp′(x)
#
# Why a dedicated stage, not a composition over an effort axis: the backward value
# recursion needs the per-effort continuation before the max (matching runs before
# the collapse), but the forward mass recursion needs the effort axis still alive
# when matching pushes mass (collapse before matching). Forward and backward want
# the effort axis alive in opposite temporal positions relative to the collapse, so
# no two-stage `∘` chain satisfies both.
#
# Re-confirmed 2026-06-08 (backlog B8) against the StratifiedKernel refactor's NEW
# rectangular introduce/forget kernels (column-introduce, ones-row forget). They do
# NOT dissolve the bind: the matching row varies over effort, so the time order is
# forced to `Introduce → EffMatch → Collapse` in both sweeps, putting the
# eff-reduction in one shared temporal slot (last in the backward walk, first in
# the forward walk). Backward needs that slot to be `max_e Q` (after Q is built);
# forward needs it to scatter mass onto the chosen `e*(x)`. The rectangular ops are
# LINEAR and adjoint-paired (introduce's backward is a SUM, not an argmax-gather), so
# composing them cannot supply the nonlinear max-collapse-with-policy-replay that
# slot demands — which is exactly THIS stage. See SAM_RI_STAGE_PROPOSAL.md §6.6.
#
# The dedicated stage keeps effort internal — a grid the stage sweeps, never a state
# axis. Backward maxes over the grid and stores the integer effort policy `e*(x)`;
# forward applies the policy-selected Bernoulli matching row plus the separation row.
# Env-dependent tightness rides the `FromEnv` contract (`tightness = FromEnv(:θ)`).
#
#TODO Effort regime is hard (max-collapse); a smooth (softmax) variant would replace
# `findmax` with a softmax over the same Q grid for SSJ Euler-smoothness.

"""
Search-and-matching over a two-level labor axis (`labor_axis`: 1 = unemployed,
2 = employed). The unemployed choose search effort from `efforts`; effort `e` costs
`cost(e)` and finds a job with probability `job_finding(e, θ)`, with tightness `θ`
from `env` via `tightness`. The employed separate at rate `separation`. Backward
solves the unemployed value by a hard max over effort and stores the effort policy;
forward replays it. `efforts` is internal — effort is not a state axis (see the
file header).
"""
struct SearchMatchingStageSpec{F, J, Sep, Tθ} <: AbstractStageSpec
    labor_axis  :: Symbol
    efforts     :: Vector{Float64}
    cost        :: F                       # cost(e) -> Real
    job_finding :: J                       # job_finding(e, θ) -> p ∈ [0,1]
    separation  :: Sep                     # δ scalar or FromEnv
    tightness   :: Tθ                      # θ scalar or FromEnv(:θ)
end

SearchMatchingStageSpec(; labor_axis::Symbol = :emp, efforts, cost, job_finding,
                          separation, tightness = FromEnv(:θ)) =
    SearchMatchingStageSpec(labor_axis, collect(Float64, efforts), cost, job_finding,
                            separation, tightness)

"""
Kernel: the integer effort policy `policy` and the per-`x` job-finding probability
`p` at the chosen effort (both shaped like the non-labor state), cached on backward
so forward replays the same matching row.
"""
struct SearchMatchingKernel{P<:AbstractArray{Int}, R<:AbstractArray}
    policy :: P                            # argmax effort index, shape x
    p      :: R                            # p(e*(x), θ), shape x
    δ      :: Base.RefValue{Float64}       # separation rate at the last backward's env (forward replays it)
end

# The labor axis must be exactly two levels (unemployed, employed); `x` is
# everything else.
_sam_x_layout(spec::SearchMatchingStageSpec, layout::GriddedLayout) =
    drop_axis(layout, spec.labor_axis)

function allocate_kernel(spec::SearchMatchingStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    n_l = axissize(layout.axes[axis_position(layout, spec.labor_axis)])
    @assert n_l == 2 "SearchMatchingStage: labor_axis `$(spec.labor_axis)` must have 2 levels (unemployed, employed), got $n_l"
    xsz = layout_size(_sam_x_layout(spec, layout))
    return SearchMatchingKernel(zeros(Int, xsz), zeros(T, xsz), Ref(0.0))
end

"Scratch: the io buffers + the `(x…, n_effort)` per-effort post-matching continuation `Q`, max-reduced over effort on backward."
function allocate_scratch(spec::SearchMatchingStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    xsz = layout_size(_sam_x_layout(spec, layout))
    return merge(io_scratch(spec, layout, T),
                 (Q = zeros(T, xsz..., length(spec.efforts)),))
end

# Backward: max over the effort grid for the unemployed value (store the policy);
# the employed value is separation-only.

function backward!(V_start, spec::SearchMatchingStageSpec, layout::GriddedLayout, V_out;
                   env, kernel, scratch, cache)
    (; labor_axis, efforts, cost, job_finding) = spec
    (; policy, p) = kernel
    Q  = scratch.Q
    θ  = resolve(spec.tightness, env)
    δ  = resolve(spec.separation, env)

    Vu = fix(V_out, layout, labor_axis => 1)   # unemployed continuation, on x
    Ve = fix(V_out, layout, labor_axis => 2)   # employed continuation, on x

    # Q[x, e] = −cost(e) + p(e,θ)·Ve(x) + (1−p(e,θ))·Vu(x), one effort slice at a time.
    for (k, e) in enumerate(efforts)
        pe = job_finding(e, θ)
        Qk = selectdim(Q, ndims(Q), k)
        @. Qk = -cost(e) + pe * Ve + (1 - pe) * Vu
    end

    # Hard choice: max over effort (trailing axis), keep the winning effort index.
    edim = ndims(Q)
    best_v, best_ci = findmax(Q; dims = edim)
    Vu_new = fix(V_start, layout, labor_axis => 1)
    Ve_new = fix(V_start, layout, labor_axis => 2)
    Vu_new .= reshape(best_v, size(Vu_new))
    policy .= getindex.(reshape(best_ci, size(policy)), edim)

    # Cache p(e*(x), θ) at the chosen effort so forward replays the same row.
    for ci in CartesianIndices(policy)
        p[ci] = job_finding(efforts[policy[ci]], θ)
    end

    # Employed: separation only (no choice). V_emp = (1−δ)·Ve + δ·Vu.
    @. Ve_new = (1 - δ) * Ve + δ * Vu
    kernel.δ[] = δ                                 # cache δ so forward replays the env's rate
    return (V_start, kernel)
end

# Forward: replay the effort policy — unemployed mass finds a job w.p. p(x), employed
# mass separates w.p. δ. Each labor column is stochastic, so mass is conserved.

function forward!(Λ_end, spec::SearchMatchingStageSpec, layout::GriddedLayout, Λ_start;
                  kernel, scratch)
    p = kernel.p
    δ = kernel.δ[]                                  # the separation rate backward last saw
    labor_axis = spec.labor_axis

    Λu = fix(Λ_start, layout, labor_axis => 1)
    Λe = fix(Λ_start, layout, labor_axis => 2)
    Λu_end = fix(Λ_end, layout, labor_axis => 1)
    Λe_end = fix(Λ_end, layout, labor_axis => 2)

    @. Λu_end = (1 - p) * Λu + δ * Λe            # stay unemployed + separations in
    @. Λe_end = p * Λu + (1 - δ) * Λe            # job finders + employed stayers
    return Λ_end
end

# Wrapper #
#---------#

@definestage SearchMatchingStage SearchMatchingStageSpec
