# MeanVarianceKernel — the portfolio-return kernel-choice stencil #
#================================================================#
# The multiplicative sibling of `MPSKernel` (mps_kernel.jl): the same per-cell Young-split / interp
# core (`_choice_scatter!` / `_choice_gather!`), but a gross-return landing `base·(R_f + θ·excess_k)`
# rather than an additive spread. Kept a distinct kernel (a multiplicative return, not an additive
# spread; §8.1/§8.2). The two share their verbs in form — a per-cell scatter / interp-gather whose
# only family-specific part is the `landing` they own — so the `forward!`/`backward!` bodies below are
# shared over the union (DRY without re-merging the types).

# Landing family — the gross portfolio return at risky share θ.
struct PortfolioReturn{T} rf :: T; excess :: Vector{T} end
@inline (l::PortfolioReturn)(base, θ, k) = base * (l.rf + θ * l.excess[k])

"""
Kernel-choice kernel for the mean-variance (portfolio) stencil: owns a per-cell frozen float field
`θstar(x)` of chosen risky shares plus the gross-return family `R_f + θ·excess_k` (a `PortfolioReturn`)
on the `grid` along `adim`. `forward!` Young-splits each cell's mass onto `Σ_k w_k·(x → x·(R_f+θ*·excess_k))`;
`backward!` is the transpose interp gather, destinations clamped to the grid.
"""
struct MeanVarianceKernel{P<:AbstractArray, T}
    θstar   :: P
    grid    :: Vector{T}
    weights :: Vector{T}
    adim    :: Int
    landing :: PortfolioReturn{T}
end

# The two kernel-choice kernels stay distinct types (§8.2) but their verbs are identical in form, so
# the `forward!`/`backward!` bodies below are shared over the union.
const KernelChoiceKernel = Union{MPSKernel, MeanVarianceKernel}
forward!(Λ_end, k::KernelChoiceKernel, Λ_start; scratch) =
    _choice_scatter!(Λ_end, Λ_start, k.θstar, k.grid, k.weights, k.adim, k.landing)
backward!(V_start, k::KernelChoiceKernel, V_end; scratch) =
    _choice_gather!(V_start, V_end, k.θstar, k.grid, k.weights, k.adim, k.landing)
