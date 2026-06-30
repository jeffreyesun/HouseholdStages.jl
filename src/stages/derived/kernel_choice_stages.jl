# Kernel-choice (optimization) primitive stages — end-goal §10. Each is an ORDINARY stage over a
# dedicated extracted kernel (`MPSKernel` / `MeanVarianceKernel`, kernel.jl): the household picks a
# scalar θ per cell from a grid, the backward STREAMS θ one value at a time (one O(nx) gather each)
# keeping a running `(max,+)` argmax — O(nx) memory, NO θ-axis introduced — and seats the per-cell
# winner θ*(x) into the kernel's frozen float field. The stage holds NO representation logic beyond
# constructing+seating its kernel and running the solver (§10/§12); the seated stencil (Young-split
# scatter / interp gather) lives entirely in the kernel. Families differ only in the kernel's
# landing map `landing(base, θ, k)`:
#
#     AdditiveSpread → MPSKernel:           base + θ·ξ_k             (mean-preserving spread — variance choice)
#     PortfolioReturn → MeanVarianceKernel: base·(R_f + θ·excess_k)  (gross return at risky share θ — mean-variance)
#
# These are PRIMITIVE per §12: the exact operation has no axis-free finite composition of the other
# primitives. The memory-heavy gridded axis-introducing versions (`ScaleChoiceStage` / `PortfolioStage`
# in kernel_choice.jl) survive ONLY as equivalence-test references (§12), not as a structural layer.
# The per-cell float policy θ* is FROZEN (always Float64, never Dual): a forward-mode Dual rebuild
# leaves the landings real and tangents flow only through V_end / the cost — derivative parity with
# `ConsumptionSavingsStage` (§13). See KERNEL_CHOICE_PRIMITIVES.md.

###############
# Stage spec  #
###############

"""
Generic streaming univariate kernel choice over `axis`: pick θ ∈ `params` of the family `landing`
at cost `cost`, `V_start(x) = max_θ[ Σ_k weights[k]·V_end(landing(x,θ,k)) − cost(θ) ]`. Seats the
per-cell winner θ*(x) into an `MPSKernel` (`AdditiveSpread`) / `MeanVarianceKernel` (`PortfolioReturn`)
and applies it on forward. Use the [`ScaleVarianceStage`](@ref) / [`MeanVarianceStage`](@ref)
constructors rather than this directly. `cost` is a `(θ; env) -> Real` closure or a vector aligned
with `params`.
"""
struct StreamingChoiceStageSpec{L, C} <: AbstractStageSpec
    axis :: Symbol
    params      :: Vector{Float64}
    weights     :: Vector{Float64}
    landing     :: L
    cost        :: C
end

@definestage StreamingChoiceStage StreamingChoiceStageSpec

"""
Seat the extracted kernel matching the spec's landing family — `MPSKernel` (`AdditiveSpread`) or
`MeanVarianceKernel` (`PortfolioReturn`) — DIRECTLY (no wrapper), over a zeroed per-cell frozen
`θstar` field the solver overwrites each `backward!`.
"""
_seat_choice_kernel(l::AdditiveSpread, θstar, grid, weights, adim)  = MPSKernel(θstar, grid, weights, adim, l)
_seat_choice_kernel(l::PortfolioReturn, θstar, grid, weights, adim) = MeanVarianceKernel(θstar, grid, weights, adim, l)

# θ* is the FROZEN policy ⇒ always Float64 (never Dual), so a forward-mode Dual rebuild leaves the
# landings real and tangents flow only through V_end / the cost.
allocate_kernel(spec::StreamingChoiceStageSpec, ::Type, layout::GriddedLayout) =
    _seat_choice_kernel(spec.landing, zeros(Float64, layout_size(layout)),
                        collect(Float64, axis_grid(layout, spec.axis)),
                        spec.weights, axis_position(layout, spec.axis))

"Scratch: the io buffers + the running argmax value `gbest` and per-θ value `gθ`."
allocate_scratch(spec::StreamingChoiceStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    merge(io_scratch(spec, layout, T),
          (gbest = zeros(T, layout_size(layout)), gθ = zeros(T, layout_size(layout))))

_cost_at(cost, θ, idx, env)                 = cost(θ; env)
_cost_at(cost::AbstractVector, θ, idx, env) = cost[idx]

"""
Streaming `(max,+)` over the parameter grid: one O(nx) gather per θ (reusing the kernel's
`_choice_gather!` with a scalar θ), folded into the running argmax `gbest`/`θstar` with the per-θ
cost. Seats `θstar` (frozen policy) into the kernel and writes `V_start = gbest`. The per-θ costs
are pre-evaluated on the host (so the device kernel runs pure arithmetic, like `SearchMatchingStage`)
and the fold itself runs in the `_streaming_choice_backward!` CPU/GPU seam (the GPU method fuses the
gather, cost, and argmax into one device kernel; it lives in `HouseholdStagesCUDAExt`).
"""
function backward!(V_start, spec::StreamingChoiceStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    (; grid, weights, adim, landing) = kernel
    gbest, gθ, θstar = scratch.gbest, scratch.gθ, kernel.θstar
    costs = _choice_costs(spec.cost, spec.params, env)
    _streaming_choice_backward!(gbest, gθ, θstar, V_end, spec.params,
                                grid, weights, adim, landing, costs)
    V_start .= gbest
    return (V_start, kernel)
end

"""
Pre-evaluate the per-θ cost on the host into a `params`-aligned `Float64` vector — `cost[idx]` for a
vector cost, or `cost(θ; env)` for a closure. Bit-identical to evaluating `_cost_at` inline (same
scalars), but lets the device backward kernel read a numeric vector instead of running a host closure.
"""
_choice_costs(cost, params, env) =
    Float64[_cost_at(cost, θ, idx, env) for (idx, θ) in enumerate(params)]

"""
Streaming argmax fold — the CPU/GPU seam factored out of `backward!` (the GPU method lives in
`HouseholdStagesCUDAExt`, fusing the per-θ gather + cost + argmax into one device kernel). For each
θ in `params`: gather `gθ = Σ_k weights[k]·V_end(landing(x,θ,k))` (clamped interp), subtract the
pre-evaluated `costs[idx]`, and fold into the running argmax `gbest`/`θstar` with a strict-`>`
(first-θ) tie-break. Writes the winning value into `gbest` and the winning parameter into `θstar`.
"""
function _streaming_choice_backward!(gbest, gθ, θstar, V_end, params, grid, weights, adim, landing, costs)
    fill!(gbest, typemin(eltype(gbest)))
    @inbounds for (idx, θ) in enumerate(params)
        _choice_gather!(gθ, V_end, θ, grid, weights, adim, landing)
        c = costs[idx]
        for i in eachindex(gbest)
            v = gθ[i] - c
            v > gbest[i] && (gbest[i] = v; θstar[i] = θ)
        end
    end
    return gbest
end

# forward! (mass push through the seated kernel) is the generic modern default (abstract.jl).

"The seated per-cell parameter policy `θ*(x)` from the last `backward!`."
policy(stage::StreamingChoiceStage) = stage.kernel.θstar


######################
# Named constructors #
######################

"""
Scale / variance (mean-preserving-spread) choice over `axis` at dispersion cost `cost`: the
household picks θ ∈ `dispersions` of the spread `Σ_k weights[k]·V(x + θ·ξ_k)` (`shocks` `ξ` mean-zero).
Concave V ⇒ insure (θ↓), convex V ⇒ gamble (θ↑). Streaming (O(nx), no θ-axis), policy frozen
(parity with `ConsumptionSavingsStage`). Gaussian-RI: pass `cost(θ; env) = λ·KL(θ)`. Seats an
`MPSKernel`.
"""
function ScaleVarianceStage(layout::GriddedLayout; axis::Symbol, dispersions, shocks, weights,
                            cost = (θ; env) -> 0.0)
    @assert isapprox(sum(weights), 1; atol = 1e-10) "ScaleVarianceStage: weights must sum to 1."
    @assert isapprox(sum(weights .* shocks), 0; atol = 1e-10) "ScaleVarianceStage: shocks must be mean-zero."
    spec = StreamingChoiceStageSpec(axis, collect(Float64, dispersions), collect(Float64, weights),
                                    AdditiveSpread(collect(Float64, shocks)), cost)
    return StreamingChoiceStage(spec, layout)
end

"""
Mean-variance / portfolio choice over the risky share `θ ∈ shares` on the wealth `axis`: next wealth
is `w·(R_f + θ·(R_k − R_f))` for risky gross returns `risky_returns[k]` with probabilities `probs[k]`.
Higher θ raises both the mean and variance of wealth, so concave V picks an interior share (the
risk–return tradeoff). Streaming (O(nx), no share-axis), policy frozen. `cost(θ; env)` is an optional
participation/transaction cost (default none). Seats a `MeanVarianceKernel`.
"""
function MeanVarianceStage(layout::GriddedLayout; axis::Symbol = :wealth, shares,
                           risk_free, risky_returns, probs, cost = (θ; env) -> 0.0)
    @assert isapprox(sum(probs), 1; atol = 1e-10) "MeanVarianceStage: probs must sum to 1."
    excess = collect(Float64, risky_returns) .- risk_free
    spec   = StreamingChoiceStageSpec(axis, collect(Float64, shares), collect(Float64, probs),
                                      PortfolioReturn(Float64(risk_free), excess), cost)
    return StreamingChoiceStage(spec, layout)
end
