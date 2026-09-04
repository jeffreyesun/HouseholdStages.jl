# GaussianLoadingKernel — the loaded-Gaussian-increment stencil #
#===============================================================#
# The operator of `GaussianLoadingStage`: a cell at axis coordinate `x` moves to
# `t = x·(anchor + θ*·(μ + σZ))` with `Z` standard normal — a Gaussian row of mean
# `m = x·(anchor + θ*μ)` and sd `s = |x|·θ*·σ`, built by the banded row primitives `_gs_band`,
# `_gs_gather_cell` and `_gs_scatter_cell!`. `s == 0` takes their deterministic landing at `clamp(m)`.

"The per-cell chosen loadings `θstar` along axis `adim`, plus the increment parameters `(anchor, μ, σ)`, at the buffer eltype."
struct GaussianLoadingKernel{P<:AbstractArray, S<:Real}
    θstar  :: P
    adim   :: Int
    anchor :: S
    μ      :: S
    σ      :: S
end

# Both griddings of the contracted axis, read at `adim` in each layout and collected at the model's
# primal float type — tangent-free, so the nested-Dual solver reads them raw.
kernel_scratch(k::GaussianLoadingKernel, start_layout::GriddedLayout,
               end_layout::GriddedLayout, ::Type{T}) where {T} =
    (origin_grid = collect(primal_eltype(T), axisvalues(start_layout.axes[k.adim])),
     dest_grid   = collect(primal_eltype(T), axisvalues(end_layout.axes[k.adim])))

# Row evaluation #
#----------------#

"""
The gather `A(θ) = Σ_j w_j(m, s)·read(V[j])` at coordinate `x`, with `m = x·(anchor + θμ)` and
`s = |x|·θ·σ`; `s == 0` takes the deterministic two-node split. Promotes over a Dual `θ`.
"""
@inline function _gl_A(col, x, θ, anchor, μ, σ, dest_grid, read::F=identity) where {F}
    m = x * (anchor + θ * μ)
    s = abs(x) * θ * σ
    return s > 0 ? _gs_gather_cell(col, m, s, dest_grid, read) :
                   _gs_point_gather(col, m, dest_grid, read)
end

# Kernel verbs #
#--------------#

"backward fiber op — the transpose gather `V_start = Kᵀ_{θ*}·V_end` at the per-cell loading."
struct GlGatherOp{S<:Real} <: AbstractFiberOp
    anchor :: S
    μ      :: S
    σ      :: S
end
function (op::GlGatherOp)(vout, vin, θcol, origin_grid, dest_grid)
    for i in eachindex(vout)
        vout[i] = _gl_A(vin, origin_grid[i], θcol[i], op.anchor, op.μ, op.σ, dest_grid)
    end
    return vout
end

"""
Forward fiber op: Young-split each cell's mass by its loaded row (`Λ_end = K_{θ*}·Λ_start`). Zeroes
the out-fiber first.
"""
struct GlScatterOp{S<:Real} <: AbstractFiberOp
    anchor :: S
    μ      :: S
    σ      :: S
end
function (op::GlScatterOp)(λout, λin, θcol, origin_grid, dest_grid)
    fill!(λout, zero(eltype(λout)))
    for i in eachindex(λin)
        mass = λin[i]
        iszero(mass) && continue
        x = origin_grid[i]
        m = x * (op.anchor + θcol[i] * op.μ)
        s = abs(x) * θcol[i] * op.σ
        s > 0 ? _gs_scatter_cell!(λout, mass, m, s, dest_grid) :
                _gs_point_scatter!(λout, mass, m, dest_grid)
    end
    return λout
end

backward!(V_start, k::GaussianLoadingKernel, V_end; scratch) =
    stratified!(GlGatherOp(k.anchor, k.μ, k.σ), V_start, V_end, k.θstar,
                scratch.origin_grid, scratch.dest_grid; dims=k.adim)
forward!(Λ_end, k::GaussianLoadingKernel, Λ_start; scratch) =
    stratified!(GlScatterOp(k.anchor, k.μ, k.σ), Λ_end, Λ_start, k.θstar,
                scratch.origin_grid, scratch.dest_grid; dims=k.adim)
