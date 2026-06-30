# Kernel-choice kernels — the per-cell parameterized stencil #
#===========================================================#
# Each owns a per-cell frozen float field `θstar(x)` of chosen parameters (a dispersion / a portfolio
# share) and applies a parameterized stencil along `adim`: a per-cell lottery that Young-splits each
# cell's mass onto per-shock landing points (`forward!`, scatter `K`) and linearly interpolates value
# back from them (`backward!`, `Kᵀ` — exact transpose, same bracketing node/weight, destinations
# clamped to the grid). They are two distinct kernels (end-goal §8.1/§8.2, the same "not one struct"
# discipline as ScatterKernel ≠ InterpKernel): `MPSKernel` (here) for the mean-preserving spread
# (variance choice), `MeanVarianceKernel` (mean_variance_kernel.jl) for the gross portfolio return.
# The per-cell float policy is differentiable and needs no auxiliary θ-axis (§13); the optimization
# that produces θstar lives in each stage's backward, so the kernel is just the seated operator. What
# they share — the per-cell Young-split / interp driver — is factored into the `_choice_scatter!` /
# `_choice_gather!` core below (the streaming sweep reuses the gather with a scalar θ).

# Landing families — `landing(base, θ, k)` is the kth shock's target along the axis at parameter θ.
struct AdditiveSpread{T}  shocks :: Vector{T} end
@inline (l::AdditiveSpread)(base, θ, k)  = base + θ * l.shocks[k]

@inline _θ_at(θ::Real, _ci)         = θ
@inline _θ_at(θ::AbstractArray, ci) = θ[ci]
@inline _setdim(ci, j, d)           = CartesianIndex(Base.setindex(Tuple(ci), j, d))

# The move is a per-cell lottery, so we can't reuse the `redistribute_along!`/`reinterpolate!` seams
# (their monotone cursor walk assumes sorted destinations — true for a uniform θ but not for the
# per-cell θ*(x)). Explicit per-cell loops: Young-split (scatter) and linear interp (gather) share the
# same bracketing node `j` and weight `w`, so they are exact transposes for any θ; the clamp keeps
# `t ∈ grid` (mass piles at the boundary, which is also what makes the boundary an exact transpose).

"""
Gather `out(x) = Σ_k w_k · V(landing(x, θ, k))` along `adim` — `θ` scalar (the streaming sweep) or
the per-cell seated policy. Linear-interpolate `V` along the axis at each clamped shock landing.
"""
function _choice_gather!(out, V, θ, grid, weights, adim, landing)
    n = length(grid); g1, gn = grid[1], grid[n]
    @inbounds for ci in CartesianIndices(out)
        base = grid[ci[adim]]; θc = _θ_at(θ, ci); acc = zero(eltype(out))
        for k in eachindex(weights)
            t = clamp(landing(base, θc, k), g1, gn)
            j = min(searchsortedlast(grid, t), n - 1)
            w = (t - grid[j]) / (grid[j+1] - grid[j])
            acc += weights[k] * ((1 - w) * V[_setdim(ci, j, adim)] + w * V[_setdim(ci, j+1, adim)])
        end
        out[ci] = acc
    end
    return out
end

"""
Scatter `out = Σ_k w_k · (mass at x → landing(x, θ*(x), k))` along `adim` — the exact transpose of
`_choice_gather!`: Young-split each cell's mass onto the same bracketing nodes/weights the gather reads.
"""
function _choice_scatter!(out, Λ, θstar, grid, weights, adim, landing)
    n = length(grid); g1, gn = grid[1], grid[n]
    fill!(out, zero(eltype(out)))
    @inbounds for ci in CartesianIndices(Λ)
        mass = Λ[ci]; iszero(mass) && continue
        base = grid[ci[adim]]; θc = θstar[ci]
        for k in eachindex(weights)
            t = clamp(landing(base, θc, k), g1, gn)
            j = min(searchsortedlast(grid, t), n - 1)
            w = (t - grid[j]) / (grid[j+1] - grid[j]); m = weights[k] * mass
            out[_setdim(ci, j, adim)]   += m * (1 - w)
            out[_setdim(ci, j+1, adim)] += m * w
        end
    end
    return out
end

"""
Kernel-choice kernel for the mean-preserving-spread (variance) stencil: owns a per-cell frozen float
field `θstar(x)` of chosen dispersions plus the mean-zero shocks `ξ` (an `AdditiveSpread`) on the
`grid` along `adim`. `forward!` Young-splits each cell's mass onto `Σ_k w_k·(x → x + θ*·ξ_k)`;
`backward!` is the transpose interp gather `Σ_k w_k·V(x + θ*·ξ_k)`, destinations clamped to the grid.
One such kernel serves both the "dispersion is valuable" and "dispersion is costly" stages (§8.1).
"""
struct MPSKernel{P<:AbstractArray, T}
    θstar   :: P
    grid    :: Vector{T}
    weights :: Vector{T}
    adim    :: Int
    landing :: AdditiveSpread{T}
end
