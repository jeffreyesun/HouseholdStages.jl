##############################
# Interpolation Helpers      #
##############################

# Routines for the V backward pass and the Λ forward pass under a deterministic
# wealth change or continuous savings choice. Adapted from
# `reference_materials/example_stages/helper/interpolations.jl`, with
# `convert_distribution!` modified to drop the `iszero(y[end])` precondition and to
# accumulate overflow/underflow mass into the last/first destination bin.

# Linear interpolation of V #
#---------------------------#

"""
Linearly interpolate `(x1, y1)` onto `x2`, writing into `y2`. Both grids
must be sorted. `extrap` controls off-grid behaviour. `:clip` clamps at BOTH
ends — snaps to `y1[1]` off-grid-left (`x2[i] < x1[1]`) and to `y1[end]`
off-grid-right (`x2[i] > x1[end]`) — so the gather is the EXACT transpose of
the mass-conserving forward Young-split (the clamp is the unique
nonnegative, mass-conserving boundary map; end-goal §8/§13). `:linear`
extends the boundary slope at both ends, and `-Inf` marks the point
unreachable off-grid-left (used to flag infeasible continuation states by
[`WealthChangeStage`](@ref) backward passes).
"""
function reinterpolate!(y2::AbstractVector, y1::AbstractVector,
                        x1::AbstractVector, x2::AbstractVector,
                        ::Val{extrap}) where extrap
    j = 1
    x1_j1 = x1[2]
    len_x2 = length(x2)
    len_x1m1 = length(x1) - 1

    i0 = 1
    # Left-extrapolation branch: clip or -Inf for x2[i] < x1[1].
    if extrap != :linear
        while x2[i0] < x1[1]
            y2[i0] = extrap == :clip ? y1[1] : -Inf
            i0 == len_x2 && return y2
            i0 += 1
        end
    end

    for i in i0:len_x2
        x2_i = x2[i]
        while x2_i > x1_j1
            j == len_x1m1 && break
            j += 1
            x1_j1 = x1[j+1]
        end
        x1_j = x1[j]
        y1_j = y1[j]
        y1_j1 = y1[j+1]
        # Right-clip (the transpose of the forward overflow-clamp): once `x2_i` runs past the
        # last node the bracket can no longer advance (`j == len_x1m1`, so `x1_j1 == x1[end]`),
        # and `x2_i > x1_j1` flags the overflow. Snap to the right endpoint `y1[end]` instead of
        # linearly extrapolating — the latter needs negative interp weights and breaks the K/Kᵀ
        # pair off-grid (mirrors the left clip above; end-goal §8/§13).
        if extrap == :clip && x2_i > x1_j1
            y2[i] = y1_j1
            continue
        end
        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = extrap == :clip ? max(y1_j, y1_j1) : -Inf
        else
            slope = (y1_j1 - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
    return y2
end

# Distribution conversion #
#-------------------------#

"""
Convert a non-negative weight vector `y` on grid `x` to a weight vector `y_new` on grid
`x_new`, preserving total mass by Young-splitting each source mass between the two
adjacent destination gridpoints by linear weight (`:share`). Both grids must be sorted;
underflow (`x[i] < x_new[1]`) accumulates into `y_new[1]`, overflow into `y_new[end]`.
"""
function convert_distribution!(y_new::AbstractVector, y::AbstractVector,
                               x::AbstractVector, x_new::AbstractVector,
                               ::Val{interp} = Val(:share)) where interp
    fill!(y_new, zero(eltype(y_new)))
    len_x = length(x)
    len_x_new_minus_1 = length(x_new) - 1

    # Monotone walk over destination bins.
    j = 1
    for i in 1:len_x
        x_i = x[i]
        y_i = y[i]
        iszero(y_i) && continue

        # Underflow.
        if x_i < x_new[1]
            y_new[1] += y_i
            continue
        end

        # Overflow — all subsequent x_k are also at or past x_new[end].
        if x_i >= x_new[end]
            y_new[end] += y_i
            for k in (i+1):len_x
                y_new[end] += y[k]
            end
            return y_new
        end

        # Advance the destination cursor until x_new[j] <= x_i < x_new[j+1].
        while x_i >= x_new[j+1]
            j += 1
            j == len_x_new_minus_1 && break
        end

        @assert interp == :share
        left_share = (x_new[j+1] - x_i) / (x_new[j+1] - x_new[j])
        y_new[j]   += y_i * left_share
        y_new[j+1] += y_i * (1 - left_share)
    end
    return y_new
end

# Along-axis redistribution #
#---------------------------#
# Generalise the leading-dim `*_arr!` routines above to a transition along ANY axis
# `adim`: `eachslice` over the other axes and run a per-slice `op` (one driver, both
# verbs — `op` carries its policy `Val` in its type for the CUDA-ext to dispatch on).
# These seams back the `ScatterKernel` / `InterpKernel` kernels (kernel.jl) used by
# `ArgmaxStage` (discrete) / `DeterministicContinuousStage` (continuous).

# Per-slice ops as callable structs (not closures) so the op carries its `Val` in its
# type — the CUDA extension dispatches `_along_axis` on the op type.

"""
Continuous-axis backward op: linearly interpolate `y1`-on-`x1` onto `x2`, into `y2`,
under the extrapolation policy `extrap`.
"""
struct ReinterpOp{Extrap}
    extrap :: Val{Extrap}
end
(op::ReinterpOp)(y2, y1, x1, x2) = reinterpolate!(y2, y1, x1, x2, op.extrap)

"""
Continuous-axis forward op: Young-split (`:share`) mass `y1`-on-`x1` onto `x2`, into `y2`.
"""
struct ConvertDistOp end
(::ConvertDistOp)(y2, y1, x1, x2) = convert_distribution!(y2, y1, x1, x2, Val(:share))

"""
Run `op` along axis `adim` of arbitrary-rank arrays: `eachslice` over every other axis
and apply `op` to the 1-D slices. Each `x` is either a shared 1-D grid (passes through)
or a per-cell N-D array (sliced like `y`). `Val(adim)` keeps the per-slice views
concretely typed.
"""
function _along_axis(y_out::AbstractArray{T,N}, y_in::AbstractArray{T,N},
                     x_for_y_in::AbstractArray, x_for_y_out::AbstractArray,
                     adim::Val, op) where {T, N}
    y_in_slices  = _axis_slices(y_in,  adim)
    y_out_slices = _axis_slices(y_out, adim)
    x_in  = _x_along(x_for_y_in,  adim, Val(N))
    x_out = _x_along(x_for_y_out, adim, Val(N))
    for ci in CartesianIndices(y_in_slices)
        op(y_out_slices[ci], y_in_slices[ci], _x_slice(x_in, ci), _x_slice(x_out, ci))
    end
    return y_out
end

"`eachslice` over every axis but the transition axis `AD` (a type param so the per-slice `SubArray` type stays concrete)."
function _axis_slices(A::AbstractArray, ::Val{AD}) where {AD}
    dims = ntuple(i -> i < AD ? i : i + 1, Val(ndims(A) - 1))
    return eachslice(A; dims)
end

"Resolve an `x` arg of `_along_axis`: an N-D array becomes the transition-axis slices, a 1-D vector passes through (`Val(N)`-dispatched to stay concretely typed)."
_x_along(x::AbstractArray{S,N}, adim::Val, ::Val{N}) where {S, N} = _axis_slices(x, adim)
_x_along(x::AbstractArray, ::Val, ::Val) = x

"Per-slice accessor matching `_x_along`: index the slices, or pass the 1-D vector through."
_x_slice(x::Base.Slices, ci) = x[ci]
_x_slice(x::AbstractVector, _) = x

# Both single-destination regimes push mass along the transition axis to a per-cell
# destination: the deterministic-continuous move uses off-grid `:share` (Young split);
# the argmax stages use an integer-grid `:nearest` destination, scattered by
# `_cs_forward_scatter!` (bit-identical, and keeps the GPU scatter kernel firing).

"""
Redistribute mass along the transition axis `adim`. `:share` Young-splits each cell's
continuous `destinations` between the two bracketing grid points; `:nearest` scatters
each cell's mass entirely to its integer `policy` grid point.
"""
redistribute_along!(Λ_end, Λ_start, destinations::AbstractArray, grid::AbstractArray,
                    adim::Val, ::Val{:share}) =
    _along_axis(Λ_end, Λ_start, destinations, grid, adim, ConvertDistOp())

redistribute_along!(Λ_end, Λ_start, policy::AbstractArray{<:Integer}, ::Nothing,
                    adim::Val, ::Val{:nearest}) =
    _cs_forward_scatter!(Λ_end, Λ_start, policy, adim)

"""
Integer-policy mass scatter along axis `WD`: `Λ_end[ν(s,σ(s))] += Λ_start[s]`. A loop
(distinct origins collide on one destination, so a broadcast would need atomics).
Factored out so the CUDA extension can overload it for `CuArray` args.
"""
function _cs_forward_scatter!(Λ_end::AbstractArray, Λ_start::AbstractArray,
                              policy::AbstractArray, ::Val{WD}) where {WD}
    fill!(Λ_end, zero(eltype(Λ_end)))
    @inbounds for ci_in in CartesianIndices(Λ_start)
        mass = Λ_start[ci_in]
        iszero(mass) && continue
        ci_out = CartesianIndex(Base.setindex(Tuple(ci_in), policy[ci_in], WD))
        Λ_end[ci_out] += mass
    end
    return Λ_end
end

"""
Integer-destination value gather along axis `WD`: `V_in[s] = V_out[ν(s)]`, where the
destination `ν(s) = setindex(s, destinations[s], WD)` relocates `s` along `WD`. The
transpose `Kᵀ` of the integer scatter [`_cs_forward_scatter!`](@ref) — each read is
independent (no collision). The exact-landing counterpart of the off-grid `reinterpolate!`.
"""
function _gather_along!(V_in::AbstractArray, V_out::AbstractArray,
                        destinations::AbstractArray, ::Val{WD}) where {WD}
    @inbounds for ci in CartesianIndices(V_in)
        ci_dst   = CartesianIndex(Base.setindex(Tuple(ci), Int(destinations[ci]), WD))
        V_in[ci] = V_out[ci_dst]
    end
    return V_in
end

