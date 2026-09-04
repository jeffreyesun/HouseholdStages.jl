##############################
# Interpolation Helpers      #
##############################

# The two-node lottery and the relocation fiber ops built on it: a cell sent to a real-valued
# position `x` splits its mass between the grid nodes bracketing `x`, with weight
# `w = (g[hi] - x)/(g[hi] - g[lo])` on the lower node.

# The two-node lottery #
#----------------------#

"The two nodes of `grid` bracketing position `x`; a position past either end returns that endpoint twice."
@inline function _interp_bracket(x, grid, n)
    x < grid[1] && return (Int32(1), Int32(1))
    x > grid[n] && return (Int32(n), Int32(n))
    j = clamp(searchsortedlast(grid, x), 1, max(n - 1, 1))
    return (Int32(j), Int32(min(j + 1, n)))
end

"The weight the pair `(lo, hi)` puts on `lo`, giving the pair mean destination `x`."
@inline _interp_share(x, grid, lo, hi) = (grid[hi] - x) / (grid[hi] - grid[lo])

# Relocation fiber ops #
#----------------------#

"Continuous-axis setup op: write into `lo` and `hi` the nodes of `g` bracketing each position in `x`."
struct SeatInterpOp <: AbstractFiberOp end
function (::SeatInterpOp)(lo, hi, x, g)
    n = length(g)
    @inbounds for i in eachindex(x)
        lo[i], hi[i] = _interp_bracket(_frz(x[i]), g, n)
    end
    return lo
end

"""
Continuous-axis forward op: split each cell's mass between its two destination nodes, `w` onto `lo`
and `1 - w` onto `hi`; a collapsed pair takes the whole mass. Zeroes the output fiber first.
"""
struct LotteryScatterOp <: AbstractFiberOp end
function (::LotteryScatterOp)(λout, λin, lo, hi, x, g)
    fill!(λout, zero(eltype(λout)))
    @inbounds for i in eachindex(λin)
        m = λin[i]
        iszero(m) && continue
        l = lo[i]; h = hi[i]
        if l == h
            λout[l] += m
        else
            w = _interp_share(x[i], g, l, h)
            λout[l] += m * w
            λout[h] += m * (1 - w)
        end
    end
    return λout
end

"""
Continuous-axis backward op: interpolate value between the same two nodes with the same weights, the
transpose `Kᵀ` of [`LotteryScatterOp`](@ref). At a `-Inf` node the position takes that node's value
if it sits on it, and the larger of the two otherwise.
"""
struct LotteryGatherOp <: AbstractFiberOp end
function (::LotteryGatherOp)(vin, vout, lo, hi, x, g)
    @inbounds for i in eachindex(vin)
        l = lo[i]; h = hi[i]
        a = vout[l]; b = vout[h]
        vin[i] = l == h                                ? a :
                 (_frz(a) == -Inf) | (_frz(b) == -Inf) ?
                     (_frz(x[i]) == g[l] ? a : _frz(x[i]) == g[h] ? b : max(a, b)) :
                 ((b - a) / (g[h] - g[l])) * (x[i] - g[l]) + a
    end
    return vin
end

"Discrete-axis forward op: add each cell's whole mass to its integer `policy` grid index; zeroes the output fiber first."
struct NearestScatterOp <: AbstractFiberOp end
function (::NearestScatterOp)(λout, λin, policy)
    fill!(λout, zero(eltype(λout)))
    for i in eachindex(λin)
        mass = λin[i]
        iszero(mass) || (λout[policy[i]] += mass)
    end
    return λout
end

"Discrete-axis backward op: read each cell's value from its integer `policy` destination, the transpose `Kᵀ` of [`NearestScatterOp`](@ref)."
struct NearestGatherOp <: AbstractFiberOp end
function (::NearestGatherOp)(vin, vout, policy)
    for i in eachindex(vin)
        vin[i] = vout[Int(policy[i])]
    end
    return vin
end
