# `ContinuousArgmaxStage`: a monotone divide-and-conquer walk over the destination nodes, a parabolic
# vertex for the off-grid policy, then a reseating of the bins the argmax jumps across as lotteries.

"""
Continuous argmax over one named continuous `axis`: per origin cell `x`, the FLOAT destination
`a' ∈ [grid_1, grid_n]` maximising `reward(x, a') + interp(V_end, a')`. `reward(x, a'; deps…[, env])`
takes the `axis` coordinate at origin and at destination as its two positional arguments and must be
supermodular unless `skip_monotonicity_check`; it carries no discount.
"""
struct ContinuousArgmaxStageSpec{R} <: AbstractStageSpec
    reward                  :: R
    axis                    :: Symbol
    skip_monotonicity_check :: Bool
end

ContinuousArgmaxStageSpec(; reward, axis::Symbol, skip_monotonicity_check::Bool = false) =
    ContinuousArgmaxStageSpec{typeof(reward)}(reward, axis, skip_monotonicity_check)

@definestage ContinuousArgmaxStage ContinuousArgmaxStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(spec::ContinuousArgmaxStageSpec) = spec.axis
tangent_grade(::ContinuousArgmaxStageSpec)     = :exact_ae

allocate_kernel(spec::ContinuousArgmaxStageSpec, ::Type{T}, start_layout::GriddedLayout, ::GriddedLayout) where {T} =
    InterpKernel(zeros(T, layout_size(start_layout)), Val(axis_position(start_layout, spec.axis)))

# Whether the reward face has increasing differences, one pass per dep combination.
function _has_increasing_differences(U::MatrixField{<:AbstractArray, NTuple{ND, Int}}) where {ND}
    A         = U.array
    nC        = size(A, 1)                                    # after (choices)
    nW        = size(A, 2)                                    # before (states)
    dep_sizes = ntuple(k -> size(A, 2 + k), Val(ND))
    @inbounds for dc in CartesianIndices(dep_sizes)
        face = view(A, :, :, dc.I...)
        for w in 1:nW-1, a in 1:nC-1
            u00 = face[a, w];   u10 = face[a+1, w]
            u01 = face[a, w+1]; u11 = face[a+1, w+1]
            (isfinite(u00) & isfinite(u10) & isfinite(u01) & isfinite(u11)) || continue
            tol = 1e-9 * (1 + max(abs(u00), abs(u10), abs(u01), abs(u11)))
            (u11 - u01) < (u10 - u00) - tol && return false
        end
    end
    return true
end

"Throwing guard on the supermodularity of the reward face; a device-resident face is checked on a host copy."
function _caC_check_supermodular(U::MatrixField)
    Uh = U.array isa Array ? U :
         MatrixField(Array(U.array), U.operative_axis, U.operative_dim, U.dep_dims)
    _has_increasing_differences(Uh) || error(
        "ContinuousArgmaxStage: the monotone node walk needs a supermodular reward (increasing " *
        "differences violated). Pass `skip_monotonicity_check = true` to skip this check.")
    return
end

"Scratch: io buffers, the reward face `U` with the source that fills it, and `bestk`, the layout-shaped node buffer the walk writes."
function allocate_scratch(spec::ContinuousArgmaxStageSpec, ::Type{T}, start_layout::GriddedLayout,
                          end_layout::GriddedLayout) where {T}
    src = to_matrix_source(spec.reward, start_layout, end_layout, spec.axis)
    U   = matrix_field(T, start_layout, end_layout, spec.axis, src)
    if !reads_env(src)
        fill_field!(U, src, start_layout, spec.axis, nothing)
        spec.skip_monotonicity_check || _caC_check_supermodular(U)
    end
    return merge(io_scratch(start_layout, end_layout, T),
                 (U = U, src = src, bestk = Array{Int}(undef, layout_size(start_layout))))
end

"Cache: the source's env-dependence and the two griddings, destination and origin, at the model's primal float type."
allocate_cache(spec::ContinuousArgmaxStageSpec, ::Type{T}, start_layout::GriddedLayout,
               end_layout::GriddedLayout) where {T} =
    (env_dep    = reads_env(to_matrix_source(spec.reward, start_layout, end_layout, spec.axis)),
     end_grid   = collect(primal_eltype(T), axis_grid(end_layout, spec.axis)),
     start_grid = collect(primal_eltype(T), axis_grid(start_layout, spec.axis)))

# Per-column continuous solver #
#------------------------------#

"""
The off-grid policy at origin `i` given the walk's best node `k`: the vertex of the parabola through
the three node objective values around `k`, clamped to `[g[k-1], g[k+1]]`, or `k` itself at a grid
edge, a non-finite neighbour or a non-concave fit.
"""
@inline function _caC_refine_vertex(face, vfib, dest_grid, k, i)
    f0    = face[k, i] + vfib[k]
    astar = oftype(f0, dest_grid[k])
    if !(k == 1 || k == length(dest_grid))
        fm1 = face[k - 1, i] + vfib[k - 1]
        fp1 = face[k + 1, i] + vfib[k + 1]
        if isfinite(_frz(fm1)) && isfinite(_frz(fp1))
            dm1   = f0 - fm1; dp1 = f0 - fp1
            dl    = dest_grid[k] - dest_grid[k - 1]; dr = dest_grid[k + 1] - dest_grid[k]
            denom = dm1 * dr + dp1 * dl
            # Concavity, decided at the primal.
            _frz(denom) > 1e-14 &&
                (astar = clamp(dest_grid[k] + (dm1 * dr^2 - dp1 * dl^2) / (2 * denom),
                               dest_grid[k - 1], dest_grid[k + 1]))
        end
    end
    return astar
end

"Whether a solved origin is fully infeasible, read at the primal."
@inline _ca_infeasible(v) = _frz(v) == -Inf

"Scan choices `lb:ub` at origin `i`, writing the best node and its value; comparison at the primal, storage in live arithmetic. Device-legal."
@inline function _ca_scan_choices!(Vs_col, pol_col, Vc_col, u_slice, i, lb::Int, ub::Int)
    @inbounds begin
        best = typemin(eltype(Vs_col)); best_p = _frz(best); ba = lb
        for a in lb:ub
            v  = u_slice[a, i] + Vc_col[a]
            vp = _frz(v)
            vp > best_p && (best_p = vp; best = v; ba = a)
        end
        pol_col[i] = ba; Vs_col[i] = best
    end
    return
end

"""
The monotone node walk, `O(n_origin log n_origin)`: solve `i = n_origin` by a full scan and `i = 1`
inside its bracket, then sweep halving segment lengths, each midpoint scanning between anchors
already solved. Origins (`Vs_col`) and choices (`Vc_col`) need not be equinumerous. Device-legal.
"""
function _ca_divide_conquer_walk!(Vs_col, pol_col, Vc_col, u_slice)
    n_choice = length(Vc_col); n_origin = length(Vs_col)
    @inbounds begin
        _ca_scan_choices!(Vs_col, pol_col, Vc_col, u_slice, n_origin, 1, n_choice)
        n_origin <= 1 && return
        ub1 = _ca_infeasible(Vs_col[n_origin]) ? n_choice : pol_col[n_origin]
        _ca_scan_choices!(Vs_col, pol_col, Vc_col, u_slice, 1, 1, ub1)
        seg = 1                                 # prevpow(2, n_origin - 1)
        while 2 * seg <= n_origin - 1
            seg *= 2
        end
        while seg >= 1
            i = 1 + seg
            while i < n_origin
                lb = _ca_infeasible(Vs_col[i - seg]) ? 1 : pol_col[i - seg]
                j  = min(i + seg, n_origin)
                ub = _ca_infeasible(Vs_col[j]) ? n_choice : pol_col[j]
                _ca_scan_choices!(Vs_col, pol_col, Vc_col, u_slice, i, lb, ub)
                i += 2 * seg
            end
            seg >>= 1
        end
    end
    return
end

"Whether the objective at origin `i` dips strictly below its value at `k2` somewhere between nodes `k1` and `k2`."
@inline function _caC_has_valley(face, vfib, k1, k2, i)
    f2 = _frz(face[k2, i] + vfib[k2])
    tol = 1e-9 * (1 + abs(f2))
    for a in (k1 + 1):(k2 - 1)
        _frz(face[a, i] + vfib[a]) < f2 - tol && return true
    end
    return false
end

"The origin bin holding `x̄` and that bin's edges; bins are the intervals between grid midpoints, truncated at the grid's ends."
@inline function _caC_bin(g, x̄, i)
    n = length(g)
    c = _frz(x̄) <= (g[i] + g[i+1]) / 2 ? i : i + 1
    return (c, c == 1 ? g[1] : (g[c-1] + g[c]) / 2, c == n ? g[n] : (g[c] + g[c+1]) / 2)
end

"""
Reseat as a two-mode lottery the bin holding a switch of the argmax between origins `i` and `i+1`,
returning the bin it claimed or `0`. The crossing `x̄` is one secant step on the mode value gap, and
the bin's mass divides by the fraction either side of it.
"""
@inline function _caC_seat_switch!(lo, hi, x, face, vfib, dest_grid, origin_grid, bestk, i)
    k1 = bestk[i]; k2 = bestk[i + 1]
    k2 > k1 + 1 || return 0
    _caC_has_valley(face, vfib, k1, k2, i) || return 0
    Φ0 = (face[k2, i]     + vfib[k2]) - (face[k1, i]     + vfib[k1])
    Φ1 = (face[k2, i + 1] + vfib[k2]) - (face[k1, i + 1] + vfib[k1])
    (_frz(Φ0) < 0) & (0 < _frz(Φ1) < Inf) || return 0
    x̄ = origin_grid[i] + (origin_grid[i+1] - origin_grid[i]) * (Φ0 / (Φ0 - Φ1))
    c, l, r = _caC_bin(origin_grid, x̄, i)
    w = (x̄ - l) / (r - l)
    lo[c] = k1; hi[c] = k2
    x[c]  = w * dest_grid[k1] + (1 - w) * dest_grid[k2]
    return c
end

"""
Backward fiber op: the node walk into `bestk`, the vertex position with its bracketing pair at every
origin, then the two-mode lottery at each bin the argmax switches inside.
"""
struct CaCSolveOp <: AbstractFiberOp end
function (::CaCSolveOp)(vsfib, lo, hi, pfib, vfib, face, dest_grid, origin_grid, bestk)
    _ca_divide_conquer_walk!(vsfib, bestk, vfib, face)
    for i in eachindex(vsfib)
        pfib[i] = _caC_refine_vertex(face, vfib, dest_grid, bestk[i], i)
    end
    SeatInterpOp()(lo, hi, pfib, dest_grid)
    for i in 1:(length(vsfib) - 1)                 # ascending: a doubly-claimed bin keeps the right-hand switch
        _caC_seat_switch!(lo, hi, pfib, face, vfib, dest_grid, origin_grid, bestk, i)
    end
    return vsfib
end

"Solve the off-grid argmax, returning `(V_start, kernel)`; `V_start` is the on-grid node value, not the value at the seated position."
function backward!(V_start, spec::ContinuousArgmaxStageSpec, start_layout::GriddedLayout,
                   end_layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    refilled = cache.env_dep && env_changed
    refilled && fill_field!(scratch.U, scratch.src, start_layout, spec.axis, env)
    refilled && !spec.skip_monotonicity_check && _caC_check_supermodular(scratch.U)
    operative_dim = axis_position(start_layout, spec.axis)
    stratified!(CaCSolveOp(), V_start, kernel.lo, kernel.hi, destinations(kernel), V_end, scratch.U,
                cache.end_grid, cache.start_grid, scratch.bestk; dims=Val(operative_dim))
    return (V_start, kernel)
end

"The solved policy: the mean destination per cell — the fitted vertex, or the barycentre of the two modes in a bin the argmax switches inside."
policy(stage::ContinuousArgmaxStage) = destinations(stage.kernel)
