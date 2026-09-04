# MeanPreservingSpreadKernel — the Gaussian-Young mean-preserving-spread stencil #
#================================================================================#
# The operator of `MeanPreservingSpreadStage`: per cell `x` the next state is `t̄ = clamp(t, g₁, gₙ)`
# with `t ~ N(x, θ*²)` truncated at ±ZMAX·σ, and the row weights `w_j(x, θ) = E[hat_j(t̄)]` (`hat_j`
# the linear tent at node j) are the exact Young split of t̄'s law onto the grid, evaluated by a
# rolling segment sweep that gather and scatter share. `θ = 0` is the row's limit. A cell sits at
# `origin_grid[i]` and its row is written on `dest_grid`'s nodes, the two ends gridded independently.

using SpecialFunctions: erfc

# Truncation half-width in σ units.
const ZMAX = 8.0

"standard normal CDF, device-legal."
@inline _gs_Φ(z) = erfc(-z / sqrt(2)) / 2

"standard normal pdf, device-legal."
@inline _gs_φ(z) = exp(-z^2 / 2) / sqrt(2π)

"The per-cell chosen standard deviations `θstar` along axis `adim`, at the buffer eltype."
struct MeanPreservingSpreadKernel{P<:AbstractArray}
    θstar :: P
    adim  :: Int
end

# Both griddings of the contracted axis, read at `adim` in each layout and collected at the model's
# primal float type — tangent-free, so the nested-Dual solver reads them raw.
kernel_scratch(k::MeanPreservingSpreadKernel, start_layout::GriddedLayout,
               end_layout::GriddedLayout, ::Type{T}) where {T} =
    (origin_grid = collect(primal_eltype(T), axisvalues(start_layout.axes[k.adim])),
     dest_grid   = collect(primal_eltype(T), axisvalues(end_layout.axes[k.adim])))

# Banded row primitives #
#-----------------------#

"The row's node band `[jlo, jhi]`: nodes with |z| ≤ ZMAX plus one guard node per side, clamped to `[1, n]`."
@inline function _gs_band(dest_grid, x, θ, n)
    jlo = max(searchsortedfirst(dest_grid, x - ZMAX * θ) - 1, 1)
    jhi = min(searchsortedlast(dest_grid, x + ZMAX * θ) + 1, n)
    return (jlo, jhi)
end

"The deterministic landing at `t`: a two-node clamped Young split on the destination grid."
@inline function _gs_point_gather(col, t, dest_grid, read::F=identity) where {F}
    n = length(dest_grid)
    n == 1 && return read(col[1])
    tc = clamp(t, dest_grid[1], dest_grid[n])
    j  = min(searchsortedlast(dest_grid, tc), n - 1)
    ω  = (tc - dest_grid[j]) / (dest_grid[j+1] - dest_grid[j])
    a  = read(col[j])
    acc = zero(ω * a)
    ω < 1 && (acc += (1 - ω) * a)
    ω > 0 && (acc += ω * read(col[j+1]))
    return acc
end

"mass scatter of the deterministic landing, the transpose of [`_gs_point_gather`](@ref)."
@inline function _gs_point_scatter!(out, mass, t, dest_grid)
    n = length(dest_grid)
    n == 1 && (out[1] += mass; return)
    tc = clamp(t, dest_grid[1], dest_grid[n])
    j  = min(searchsortedlast(dest_grid, tc), n - 1)
    ω  = (tc - dest_grid[j]) / (dest_grid[j+1] - dest_grid[j])
    ω < 1 && (out[j]   += (1 - ω) * mass)
    ω > 0 && (out[j+1] += ω * mass)
    return
end

"""
Banded gather `Σ_j w_j(x, θ)·read(col[j])` from the cell coordinate `x` onto the destination nodes.
`read` selects the arithmetic: `identity` for the envelope value, `_frz` for the solver's frozen
scan. Promotes over a Dual `θ`.
"""
@inline function _gs_gather_cell(col, x, θ, dest_grid, read::F=identity) where {F}
    acc = zero(θ * read(col[1]))
    θ > 0 || return acc + _gs_point_gather(col, x, dest_grid, read)
    n = length(dest_grid)
    jlo, jhi = _gs_band(dest_grid, x, θ, n)
    jlo == jhi && return acc + read(col[jlo])
    za = (dest_grid[jlo] - x) / θ; Φa = _gs_Φ(za); φa = _gs_φ(za)
    w   = Φa                                            # left tail → band-lo node
    @inbounds for j in jlo:jhi-1
        zb = (dest_grid[j+1] - x) / θ; Φb = _gs_Φ(zb); φb = _gs_φ(zb)
        dΦ = Φb - Φa; dφ = φb - φa; Δ = dest_grid[j+1] - dest_grid[j]
        w += max(((dest_grid[j+1] - x) * dΦ + θ * dφ) / Δ, 0.0)   # falling piece → node j
        w > 0 && (acc += w * read(col[j]))
        w  = max(((x - dest_grid[j]) * dΦ - θ * dφ) / Δ, 0.0)     # rising piece → node j+1
        za = zb; Φa = Φb; φa = φb
    end
    w += 1 - Φa                                         # right tail → band-hi node
    w > 0 && (acc += w * read(col[jhi]))
    return acc
end

"""
Banded Young-split scatter of `mass` from the cell coordinate `x` onto the destination nodes,
accumulating `out[j] += w·mass` — the transpose of [`_gs_gather_cell`](@ref).
"""
@inline function _gs_scatter_cell!(out, mass, x, θ, dest_grid)
    if θ > 0
        n = length(dest_grid)
        jlo, jhi = _gs_band(dest_grid, x, θ, n)
        if jlo == jhi
            @inbounds out[jlo] += mass
            return
        end
        za = (dest_grid[jlo] - x) / θ; Φa = _gs_Φ(za); φa = _gs_φ(za)
        w  = Φa                                         # left tail → band-lo node
        @inbounds for j in jlo:jhi-1
            zb = (dest_grid[j+1] - x) / θ; Φb = _gs_Φ(zb); φb = _gs_φ(zb)
            dΦ = Φb - Φa; dφ = φb - φa; Δ = dest_grid[j+1] - dest_grid[j]
            w += max(((dest_grid[j+1] - x) * dΦ + θ * dφ) / Δ, 0.0)   # falling piece → node j
            w > 0 && (out[j] += w * mass)
            w  = max(((x - dest_grid[j]) * dΦ - θ * dφ) / Δ, 0.0)     # rising piece → node j+1
            za = zb; Φa = Φb; φa = φb
        end
        w += 1 - Φa                                     # right tail → band-hi node
        w > 0 && (out[jhi] += w * mass)
    else
        _gs_point_scatter!(out, mass, x, dest_grid)
    end
    return
end

# Kernel verbs #
#--------------#

"Backward fiber op: the gather `V_start = Kᵀ_{θ*}·V_end`, one banded segment sweep per cell."
struct MpsGatherOp <: AbstractFiberOp end
function (::MpsGatherOp)(vout, vin, θcol, origin_grid, dest_grid)
    for i in eachindex(vout)
        vout[i] = _gs_gather_cell(vin, origin_grid[i], θcol[i], dest_grid)
    end
    return vout
end

"Forward fiber op: the Young-split scatter `Λ_end = K_{θ*}·Λ_start`; zeroes the out-fiber first."
struct MpsScatterOp <: AbstractFiberOp end
function (::MpsScatterOp)(λout, λin, θcol, origin_grid, dest_grid)
    fill!(λout, zero(eltype(λout)))
    for i in eachindex(λin)
        mass = λin[i]
        iszero(mass) || _gs_scatter_cell!(λout, mass, origin_grid[i], θcol[i], dest_grid)
    end
    return λout
end

backward!(V_start, k::MeanPreservingSpreadKernel, V_end; scratch) =
    stratified!(MpsGatherOp(), V_start, V_end, k.θstar,
                scratch.origin_grid, scratch.dest_grid; dims=k.adim)
forward!(Λ_end, k::MeanPreservingSpreadKernel, Λ_start; scratch) =
    stratified!(MpsScatterOp(), Λ_end, Λ_start, k.θstar,
                scratch.origin_grid, scratch.dest_grid; dims=k.adim)
