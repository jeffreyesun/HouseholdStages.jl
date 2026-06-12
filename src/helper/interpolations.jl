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
must be sorted. `extrap` controls left-extrapolation when `x2[i] < x1[1]`:
`:linear` extends the leftmost slope, `:clip` snaps to `y1[1]`, and
`-Inf` marks the point unreachable (used to flag infeasible continuation
states by [`WealthChangeStage`](@ref) backward passes).
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
        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = extrap == :clip ? max(y1_j, y1_j1) : -Inf
        else
            slope = (y1_j1 - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
    return y2
end

"""
Apply [`reinterpolate!`](@ref) along the leading dimension of
arbitrary-rank arrays, broadcasting `x1` / `x2` across trailing dims
when they have length 1 there.
"""
function reinterpolate_arr!(y2, y1, x1, x2, ::Val{extrap}) where extrap
    for idx in CartesianIndices(Base.tail(size(y2)))
        _tview(arr) = @view arr[:, _broadcast_index(idx, Base.tail(size(arr)))]
        reinterpolate!(_tview(y2), _tview(y1), _tview(x1), _tview(x2), Val(extrap))
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

"""
Apply [`convert_distribution!`](@ref) along the leading dimension of
arbitrary-rank arrays, broadcasting `x` / `x_new` across trailing dims
when they have length 1 there.
"""
function convert_distribution_arr!(y_new, y, x, x_new,
                                   ::Val{interp} = Val(:share)) where interp
    for idx in CartesianIndices(Base.tail(size(y_new)))
        _tview(arr) = @view arr[:, _broadcast_index(idx, Base.tail(size(arr)))]
        convert_distribution!(_tview(y_new), _tview(y),
                              _tview(x), _tview(x_new), Val(interp))
    end
    return y_new
end

# Along-axis redistribution #
#---------------------------#
# Generalise the leading-dim `*_arr!` routines above to a transition along ANY axis
# `adim`: `eachslice` over the other axes and run a per-slice `op` (one driver, both
# verbs — `op` carries its policy `Val` in its type for the CUDA-ext to dispatch on).
# These seams back the shared `SingleDestinationKernel` (kernel.jl) used by
# `DeterministicContinuousStage`/`ContinuousArgmaxStage`/`ArgmaxStage`.

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

# Monotone-policy argmax #
#------------------------#

"""
Maximize `u_slice[a, s] + V_post[a]` over `a` for each `s`, with the
monotone-policy guarantee that the optimum is non-decreasing in `s`. The
search for each `s` therefore starts where `s-1`'s search ended, so the
total work is `O(N + M)` rather than `O(N M)`.

Writes the integer argmax into `policy_slice[s]` and the resulting
maximum into `V_prec_slice[s]`. Returns `policy_slice`.

If no action is feasible (all candidates give `-Inf`), `policy_slice[s]
= 1`, `V_prec_slice[s] = -Inf`, and the monotone lower bound is *not*
advanced — that cell does not contaminate downstream cells' search.
"""
function k1_argmax_monotone!(V_prec_slice::AbstractVector{T},
                             policy_slice::AbstractVector{Int},
                             u_slice::AbstractMatrix{T},
                             V_post::AbstractVector{T}) where T
    n_s = length(V_prec_slice)
    n_a = length(V_post)
    @assert size(u_slice) == (n_a, n_s)
    @assert length(policy_slice) == n_s

    prev_a = 1
    for s in 1:n_s
        best_v = typemin(T)
        best_a = 0
        for a in prev_a:n_a
            u = u_slice[a, s]
            isfinite(u) || continue
            v = u + V_post[a]
            if v > best_v
                best_v = v
                best_a = a
            end
        end
        if best_a == 0
            V_prec_slice[s] = typemin(T)
            policy_slice[s] = 1
        else
            V_prec_slice[s] = best_v
            policy_slice[s] = best_a
            prev_a = best_a
        end
    end
    return policy_slice
end

# Monotone-policy divide-and-conquer argmax #
#-------------------------------------------#

"""
Divide-and-conquer monotone-policy argmax — same problem as `k1_argmax_monotone!`
(maximise `u_slice[a,s] + V_post[a]` with the optimum non-decreasing in `s`), but
tightening both bounds at each midpoint for `O((n_a + n_s) log n_s)` work.

Correct only if the optimum policy is non-decreasing in `s` (concave utility +
linear budget gives it; non-convex feasibility or non-concave payoffs can break it).
Prefer the sequential walk when in doubt.
"""
function k1_argmax_dc!(V_prec_slice::AbstractVector{T},
                       policy_slice::AbstractVector{Int},
                       u_slice::AbstractMatrix{T},
                       V_post::AbstractVector{T}) where T
    n_s = length(V_prec_slice)
    n_a = length(V_post)
    @assert size(u_slice) == (n_a, n_s)
    @assert length(policy_slice) == n_s
    n_s == 0 && return policy_slice
    _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                        1, n_s, 1, n_a)
    return policy_slice
end

# Fill `V_prec_slice[lo..hi]` / `policy_slice[lo..hi]` given that each
# entry's optimum is in `[lo_b..hi_b]`. The midpoint's argmax bounds
# the search range of its left and right sub-problems.
function _dc_argmax_recurse!(V_prec_slice::AbstractVector{T},
                             policy_slice::AbstractVector{Int},
                             u_slice::AbstractMatrix{T},
                             V_post::AbstractVector{T},
                             lo::Int, hi::Int,
                             lo_b::Int, hi_b::Int) where T
    lo > hi && return
    mid = (lo + hi) >> 1
    best_v = typemin(T)
    best_a = 0
    for a in lo_b:hi_b
        u = u_slice[a, mid]
        isfinite(u) || continue
        v = u + V_post[a]
        if v > best_v
            best_v = v
            best_a = a
        end
    end
    if best_a == 0
        # No feasible action at this midpoint; downstream cells still
        # respect the outer bounds (we don't shrink to a missing pivot).
        V_prec_slice[mid] = typemin(T)
        policy_slice[mid] = 1
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            lo, mid - 1, lo_b, hi_b)
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            mid + 1, hi, lo_b, hi_b)
    else
        V_prec_slice[mid] = best_v
        policy_slice[mid] = best_a
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            lo, mid - 1, lo_b, best_a)
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            mid + 1, hi, best_a, hi_b)
    end
    return
end

# Shared utility — broadcast index mapping #
#------------------------------------------#

"Collapse any dim of length 1 (or beyond `len(size)`) in `idx` to 1, mimicking broadcasting semantics."
function _broadcast_index(idx::CartesianIndex, size::Tuple)
    return CartesianIndex(ntuple(
        i -> i > length(size) ? 1 : min(idx[i], size[i]),
        Val(length(idx))))
end
