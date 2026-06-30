# Continuous argmax over one named continuous axis — the off-grid sibling of `ArgmaxStage`
# (end-goal §10). For each origin cell `x`, choose a destination position `a'` on the CONTINUOUS
# interval `[grid_1, grid_n]` maximising
#
#     V_start[x] = max_{a' ∈ [grid_1, grid_n]} [ reward(x, a'; env) + interp(V_end, a') ]
#
# and write the off-grid optimiser `a'*(x)` (a FLOAT) into an `InterpKernel`'s position policy
# (forward Young-splits its mass, backward clip-interpolates value — both already in kernel.jl).
#
# Two things separate this from `ArgmaxStage` at exactly the two points §10 names:
#   • the SOLVER — a per-cell 1-D continuous maximisation (coarse grid scan → golden-section in the
#     bracketing interval), not a grid `findmax`/monotone halving;
#   • the KERNEL — an `InterpKernel` over a float position, not a `ScatterKernel` over an integer index.
# What it SHARES with `ArgmaxStage` is the start-and-end reward closure: `reward(x, a'; deps…, env)`
# takes the operative axis at its origin `x` AND its destination `a'` POSITIONALLY, evaluable at ANY
# CONTINUOUS `a'` — NOT a pre-materialised grid matrix. So this stage does NOT route through
# `to_matrix_source` (which would materialise a grid matrix); it holds the scalar `reward` directly and
# calls it at the off-grid candidate `a'` the solver proposes — precisely the evaluation it needs.
#
# Differentiability (end-goal §13). The float policy `a'*` is the deciding reason this stage needs an
# `InterpKernel`, not a `ScatterKernel`: a sequence-space Jacobian differentiates the kernel itself wrt
# `env`, and the continuous position moves smoothly with `env` where a discrete index would jump. The
# policy is FROZEN `Float64` (the golden-section optimiser is pure grid geometry, so a forward-mode
# `Dual` rebuild leaves `a'*` real); the VALUE carries the `env`-tangent through the reward term, and by
# the envelope theorem `∂V_start/∂env = ∂reward/∂env` at the frozen optimiser — the same frozen-policy /
# Dual-value contract as the kernel-choice primitives (`StreamingChoiceStage`) and `ConsumptionSavings`.
#
# No discount `β` (end-goal §1): a composed `TimeDiscountingStage` scales `V_end` before the argmax.
# The solver assumes the per-cell objective is unimodal over the grid (concave reward + a continuation
# whose linear interpolant is concave) so the coarse scan's best node brackets the continuous optimum;
# a node guard keeps `V_start` no worse than the best grid node when that assumption is violated.

"""
Continuous argmax over one named continuous `axis` — the off-grid sibling of [`ArgmaxStage`](@ref)
(end-goal §10). Per origin cell `x`, picks the FLOAT destination `a' ∈ [grid_1, grid_n]` maximising
`reward(x, a'; env) + interp(V_end, a')` and seats `a'*(x)` into an `InterpKernel` (Young-split
forward, clip-interp backward). `reward(x, a'; deps…, env)` is a start-and-end closure taking the
operative axis at its origin and destination POSITIONALLY, evaluable at ANY continuous `a'`. Carries no
discount `β` (a composed [`TimeDiscountingStage`](@ref) scales `V_end` first; end-goal §1). This is the
off-grid limit the discrete argmax approaches as its grid refines.
"""
struct ContinuousArgmaxStageSpec{R} <: AbstractStageSpec
    reward :: R
    axis   :: Symbol
end

ContinuousArgmaxStageSpec(; reward, axis::Symbol) =
    ContinuousArgmaxStageSpec{typeof(reward)}(reward, axis)

@definestage ContinuousArgmaxStage ContinuousArgmaxStageSpec


##########################
# Gridded implementation #
##########################

# Kernel: an `InterpKernel` (kernel.jl) over a FROZEN `Float64` position field (one off-grid landing
# per cell). The solver writes the float policy; forward Young-splits its mass, backward clip-interps.
# The policy is `Float64` (not `T`) so a forward-mode `Dual` rebuild leaves `a'*` real (§13).
allocate_kernel(spec::ContinuousArgmaxStageSpec, ::Type, layout::GriddedLayout) =
    InterpKernel(zeros(Float64, layout_size(layout)), Val(axis_position(layout, spec.axis)))

"""
Cache: the scalar start-and-end `payoff` (`spec.reward`, evaluated off-grid POSITIONALLY at the solved
`a'`), whether it reads `env` (`env_dep`), the operative-axis `grid`, and the payoff's dep axes — their
names (`deps`), layout positions (`dep_dims`), and grids (`dep_grids`) — for building each column's dep
`combo`. All `Float64`/static (independent of the buffer eltype `T`); the `env`-tangent enters only
through the payoff at solve time, so there is no field to refill.
"""
function allocate_cache(spec::ContinuousArgmaxStageSpec, ::Type, layout::GriddedLayout)
    payoff    = spec.reward
    deps      = _closure_deps(payoff, layout)
    env_dep   = _closure_env_dep(payoff)
    dep_dims  = map(a -> axis_position(layout, a), deps)
    dep_grids = map(a -> collect(Float64, axis_grid(layout, a)), deps)
    grid      = collect(Float64, axis_grid(layout, spec.axis))
    return (payoff = payoff, env_dep = env_dep, grid = grid,
            deps = deps, dep_dims = dep_dims, dep_grids = dep_grids)
end

# Per-cell continuous solver #
#----------------------------#

"""
Linear interpolation of the value fiber `v` on `grid` at the continuous position `a`, clamped at BOTH
ends (matching the `InterpKernel`'s `:clip` backward and the forward mass-clamp). `grid`/`a` are
`Float64`; the result inherits `v`'s eltype (so a `Dual` `V_end` flows through).
"""
@inline function _interp1d(grid::AbstractVector, v::AbstractVector, a::Real)
    n = length(grid)
    a <= grid[1] && return v[1]
    a >= grid[n] && return v[n]
    j = min(searchsortedlast(grid, a), n - 1)
    w = (a - grid[j]) / (grid[j+1] - grid[j])
    return (1 - w) * v[j] + w * v[j+1]
end

"""
Golden-section maximiser of `f` over `[lo, hi]` (`Float64` bracket). The trial points are pure
geometry of the bracket, so the returned `xstar` is `Float64` even when `f` returns a `Dual` (its
tangent never feeds back into the position — the frozen-policy / Dual-value contract, §13);
comparisons of `Dual` objective values compare primals. Returns `(xstar, f(xstar))`.
"""
function _golden_max(f, lo::Real, hi::Real; iters::Int = 45)
    invφ  = (sqrt(5.0) - 1) / 2          # 0.618…
    invφ2 = (3 - sqrt(5.0)) / 2          # 0.382…
    a, b = float(lo), float(hi)
    (b - a) <= eps(b) && return a, f(a)
    c  = a + invφ2 * (b - a); fc = f(c)
    d  = a + invφ  * (b - a); fd = f(d)
    for _ in 1:iters
        if fc > fd                       # maximise: drop the right sub-interval
            b = d; d = c; fd = fc
            c = a + invφ2 * (b - a); fc = f(c)
        else                             # drop the left sub-interval
            a = c; c = d; fc = fd
            d = a + invφ  * (b - a); fd = f(d)
        end
    end
    xstar = (a + b) / 2
    return xstar, f(xstar)
end

"""
Evaluate the scalar start-and-end `payoff` at one `(origin, dest)` pair POSITIONALLY, threading the
dep `combo` and `env` (iff the payoff declares it) as kwargs. `dest` may be off-grid (a golden-section
trial), so this is the off-grid evaluation `to_matrix_source`'s grid sweep cannot provide.
"""
@inline _caC_payoff(payoff, env_dep::Bool, origin, dest, combo, env) =
    env_dep ? payoff(origin, dest; combo..., env) : payoff(origin, dest; combo...)

"""
Solve one origin cell's continuous argmax: a coarse scan over the operative grid finds the best node
(its neighbours bracket the optimum under unimodality), golden-section refines inside that bracket,
and a node guard keeps the result no worse than the best grid node. Returns `(a'*, V_start)` — the
frozen `Float64` optimiser and the (eltype-following) optimal value.
"""
function _caC_solve_cell(payoff, env_dep::Bool, grid::AbstractVector, vfib::AbstractVector, origin, combo, env)
    n = length(grid)
    f = a -> _caC_payoff(payoff, env_dep, origin, a, combo, env) + _interp1d(grid, vfib, a)
    # Coarse scan: at a node the continuation is the exact fiber value (interp at a node = the node).
    best_v = _caC_payoff(payoff, env_dep, origin, grid[1], combo, env) + vfib[1]
    bestk  = 1
    @inbounds for j in 2:n
        v = _caC_payoff(payoff, env_dep, origin, grid[j], combo, env) + vfib[j]
        v > best_v && (best_v = v; bestk = j)
    end
    lo = grid[max(bestk - 1, 1)]
    hi = grid[min(bestk + 1, n)]
    astar, vstar = _golden_max(f, lo, hi)
    best_v >= vstar && return grid[bestk], best_v       # node guard (non-unimodal fallback)
    return astar, vstar
end

# Column loop, specialised on the operative dim `WD`, rank `N`, and dep count `ND` (a `Val` barrier)
# for concretely-typed per-column slice views — mirroring `ArgmaxStage`'s `_ca_backward_columns!`.
# Each column fixes the non-operative coordinates, so its dep `combo` is built once and reused across
# every origin (the operative index) and every objective evaluation in the column.

"The dep `combo` (axis ⇒ grid value) at a column's coordinates — `()` when the reward has no deps."
@inline _caC_combo(::Tuple{}, dep_dims, dep_grids, col) = NamedTuple()
@inline function _caC_combo(deps::NTuple{ND, Symbol}, dep_dims, dep_grids, col) where {ND}
    vals = ntuple(k -> dep_grids[k][col[dep_dims[k]]], Val(ND))
    return NamedTuple{deps}(vals)
end

function _caC_backward_columns!(::Val{WD}, payoff, env_dep::Bool, grid, deps, dep_dims, dep_grids,
                                V_end, V_start, policy, env, dims, ::Val{N}) where {WD, N}
    n           = length(grid)
    other_sizes = ntuple(i -> i == WD ? 1 : dims[i], Val(N))
    @inbounds for other_ci in CartesianIndices(other_sizes)
        col   = other_ci.I
        combo = _caC_combo(deps, dep_dims, dep_grids, col)
        vfib  = view(V_end,   ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        vsfib = view(V_start, ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        pfib  = view(policy,  ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        for i in 1:n
            astar, vstar = _caC_solve_cell(payoff, env_dep, grid, vfib, grid[i], combo, env)
            pfib[i]  = astar
            vsfib[i] = vstar
        end
    end
    return
end

function backward!(V_start, spec::ContinuousArgmaxStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    policy = destinations(kernel)                       # frozen Float64 a'*(x) = the InterpKernel position
    cdim   = axis_position(layout, spec.axis)
    _caC_backward_columns!(Val(cdim), cache.payoff, cache.env_dep, cache.grid, cache.deps,
                           cache.dep_dims, cache.dep_grids, V_end, V_start, policy, env,
                           layout_size(layout), Val(ndims(V_start)))
    return (V_start, kernel)
end

# forward! (Young-split mass scatter through the seated InterpKernel, K·Λ_start) is the generic
# modern default (abstract.jl).

"""
The solved policy of a [`ContinuousArgmaxStage`](@ref): the off-grid optimiser `a'*(x)` per cell (a
float position). It IS the `InterpKernel`'s destination — choosing where to land on the axis is the policy.
"""
policy(stage::ContinuousArgmaxStage) = destinations(stage.kernel)


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. The phase-1 stage methods above do not dispatch on
# layout type, so this is a placeholder marking where the deriv-carrying
# representation's methods will go.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder marking where the dynamic-grid
# representation's methods will go.
