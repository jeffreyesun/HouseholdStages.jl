# Kernel choice (gridded θ) — derived sugar, NOT a new primitive. The household picks θ in a
# one-parameter family of x-transitions `{K_θ}` at cost `c(θ)`; the backward value is the
# `(max,+)` / `(logsumexp,+)` contraction over θ
#
#     V_start[x] = ⊕_θ [ −c(θ) + (K_θ V_end)(x) ],     ⊕ ∈ {max (hard), logsumexp (soft)},
#
# realised as the three-stage sandwich (left-to-right = time order; the choice axis is grown
# 1 → nθ internally and collapsed back to 1, so θ is a singleton at both block ends — the
# fixed-layout invariant):
#
#     Collapse[θ, n_start=1]  ∘  Markov[x | θ-dep]  ∘  ForgetfulSum[θ]
#       └ argmax/logit over θ    └ K_θ per θ-slice   └ broadcast V across θ (backward)
#
# This is rung (d) of the G1 ladder (CHOICE_STAGE_CATALOG): the general kernel choice with no
# closed-form shortcut. Rung (a), linear mixing, has the cheaper conjugate form — see MixingStage.

"""
The θ-dependent x-transition source for `KernelChoiceStage`: declares the choice axis `CA` as its
lone dep and returns `kernels[θ]`, the row-stochastic x-transition for the θ-th family member. A
plain matrix `M`-source, so `MarkovStage` materialises it through the usual `fill_field!` machinery.
"""
struct _KernelChoiceTransition{CA, Ks}
    kernels :: Ks                                          # kernels[θ] = K_θ[from, to], row-stochastic
end
declared_deps(::_KernelChoiceTransition{CA}, ::GriddedLayout) where {CA} = (CA,)
evaluate(t::_KernelChoiceTransition{CA}, combo, _env) where {CA} = t.kernels[combo[CA]]

"""
Gridded kernel choice over a one-parameter family of `x_axis` transitions `kernels[θ]` at cost
`cost[θ]`: backward `V(x) = ⊕_θ [ −cost[θ] + (kernels[θ] · V_end)(x) ]`, `⊕ = max` (`soft=false`)
or `logsumexp` at scale `ε` (`soft=true`). Derived as `Collapse ∘ Markov[θ-dep] ∘ ForgetfulSum`
(no new primitive). `choice_axis` must be a size-1 singleton in the block layout — it is grown to
`length(kernels)` internally and collapsed back, so it stays singleton at the block boundary.
"""
function KernelChoiceStage(layout::GriddedLayout; choice_axis::Symbol, x_axis::Symbol,
                           kernels::AbstractVector, cost::AbstractVector,
                           soft::Bool=false, ε=1.0)
    @assert _axis_size(layout, choice_axis) == 1 "KernelChoiceStage: `$choice_axis` must be a " *
        "size-1 singleton in the block layout (grown to $(length(kernels)) internally — the " *
        "fixed-layout invariant; declare it as a singleton, the stage never introduces an axis)."
    nθ = length(kernels)
    @assert length(cost) == nθ "KernelChoiceStage: cost ($(length(cost))) and kernels ($nθ) must match."
    full       = grow_axis(layout, choice_axis, nθ)
    transition = _KernelChoiceTransition{choice_axis, typeof(kernels)}(kernels)
    collapse   = soft ?
        LogitChoiceStage(full; axis = choice_axis, cost_matrix = reshape(collect(cost), 1, nθ), ε) :   # C[origin=1, dest=θ]
        ArgmaxStage(full; axis = choice_axis, reward = reshape(.-collect(cost), nθ, 1))                 # M[after=θ, before=1]
    markov     = MarkovStage(full; axis = x_axis, transition_matrix = transition)
    forget     = ForgetfulSumStage(full; axis = choice_axis)
    return collapse ∘ markov ∘ forget
end

"""
Portfolio / mean-variance frontier choice — a `KernelChoiceStage` whose family `kernels[k]` is a
discretised efficient frontier (each `k` a risky share / point on the frontier, its `wealth_axis`
transition the return distribution at that share) chosen at cost `cost[k]`. Rung (d): no
closed-form shortcut, an honest grid over the frontier. The frontier kernels are supplied by the
caller (the return process is the model's, not the package's).
"""
PortfolioStage(layout::GriddedLayout; share_axis::Symbol=:share, wealth_axis::Symbol=:wealth,
               kernels::AbstractVector, cost::AbstractVector=zeros(length(kernels)), soft::Bool=false, ε=1.0) =
    KernelChoiceStage(layout; choice_axis = share_axis, x_axis = wealth_axis, kernels, cost, soft, ε)

# Scale / variance choice (rung c) — choose the DISPERSION of a mean-preserving spread #
#-------------------------------------------------------------------------------------#

"""
Young-split `mass` onto the two `grid` nodes bracketing `target` (clamping to the ends), so the
split's expected landing point is `target` — the lottery that keeps a transition mean-preserving.
"""
function _lottery!(row, grid, target, mass)
    n = length(grid)
    target ≤ grid[1] && (row[1] += mass; return)
    target ≥ grid[n] && (row[n] += mass; return)
    j = searchsortedlast(grid, target)                       # grid[j] ≤ target < grid[j+1]
    w = (target - grid[j]) / (grid[j+1] - grid[j])
    row[j] += mass * (1 - w); row[j+1] += mass * w
    return
end

"""
The mean-preserving-spread transition on `grid` at dispersion `θ`: from each node `x`, mass lands
at `x + θ·ξ_k` with probability `weights[k]` (mean-zero `shocks`), each landing Young-split onto
the grid. Interior rows preserve the mean; the spread (variance) grows with `θ`.
"""
function _mps_kernel(grid::Vector{Float64}, θ::Real, shocks, weights)
    n = length(grid); K = zeros(n, n)
    for i in 1:n, k in eachindex(shocks)
        _lottery!(view(K, i, :), grid, grid[i] + θ * shocks[k], weights[k])
    end
    return K
end

"""
Scale / variance choice (G1 rung c): pick the dispersion `θ ∈ dispersions` of a mean-preserving
spread of `x_axis` at cost `cost[θ]`, `(K_θ V)(x) = Σ_k weights[k]·V(x + θ·ξ_k)`. Concave `V` ⇒
insure (`θ↓`), convex `V` ⇒ gamble (`θ↑`) — one stage, both sides of `V`'s curvature. Builds the
mean-preserving-spread kernels (mean-zero `shocks`, `weights`) and delegates to `KernelChoiceStage`.
"""
function ScaleChoiceStage(layout::GriddedLayout; x_axis::Symbol, scale_axis::Symbol=:scale,
                          dispersions::AbstractVector, shocks::AbstractVector, weights::AbstractVector,
                          cost::AbstractVector=zeros(length(dispersions)), soft::Bool=false, ε=1.0)
    @assert isapprox(sum(weights), 1; atol = 1e-10) "ScaleChoiceStage: weights must sum to 1."
    @assert isapprox(sum(weights .* shocks), 0; atol = 1e-10) "ScaleChoiceStage: shocks must be mean-zero (mean-preserving)."
    grid    = collect(Float64, axis_grid(layout, x_axis))
    kernels = [_mps_kernel(grid, θ, shocks, weights) for θ in dispersions]
    return KernelChoiceStage(layout; choice_axis = scale_axis, x_axis, kernels, cost, soft, ε)
end
