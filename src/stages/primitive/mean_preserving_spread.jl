# Mean-preserving spread with a CONTINUOUS dispersion choice: per origin cell `x`, the spread sd
# θ ∈ [0, θ_max] maximising `g(θ) = A(x, θ) − cost(θ; env)`, where `A = Σ_j w_j(x, θ)·V_end[j]` sums
# the Gaussian-Young row of `MeanPreservingSpreadKernel` over the destination nodes.

#TODO Stratified (cell-dependent) costs.

"""
Continuous mean-preserving-spread (dispersion) choice over `axis`: per cell, pick the spread sd
`θ ∈ [0, θ_max]` of a clamped Gaussian-Young spread of the next state at cost `cost(θ; env)` (a
ForwardDiff-differentiable closure), seating the per-cell winner θ*(x) into a
[`MeanPreservingSpreadKernel`](@ref). `nscan` is the solver's bracketing resolution, not a choice grid.
"""
struct MeanPreservingSpreadStageSpec{C} <: AbstractStageSpec
    axis  :: Symbol
    θ_max :: Float64
    nscan :: Int
    cost  :: C
end

"Keyword constructor; `θ_max` must be positive and finite."
function MeanPreservingSpreadStageSpec(; axis::Symbol, θ_max, nscan::Int=12, cost=(θ; env) -> 0.0)
    @assert θ_max > 0 && isfinite(θ_max)
    return MeanPreservingSpreadStageSpec{typeof(cost)}(axis, Float64(θ_max), nscan, cost)
end

@definestage MeanPreservingSpreadStage MeanPreservingSpreadStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(spec::MeanPreservingSpreadStageSpec) = spec.axis
tangent_grade(::MeanPreservingSpreadStageSpec)     = :exact_ae

"Throwing guard: the buffer eltype must carry at most one lift, the solver's probes owning the other two lanes."
function _assert_single_lift(::Type{T}, stage::String) where {T}
    ForwardDiff.valtype(T) <: ForwardDiff.Dual && error(
        "$stage: a nested (Dual-of-Dual) buffer eltype is unsupported — the solver's θ-probes are " *
        "two-level Float64 duals read positionally, so a second lift would land on a probe lane.")
    return
end

function allocate_kernel(spec::MeanPreservingSpreadStageSpec, ::Type{T}, start_layout::GriddedLayout,
                         ::GriddedLayout) where {T}
    _assert_single_lift(T, "MeanPreservingSpreadStage")
    return MeanPreservingSpreadKernel(zeros(T, layout_size(start_layout)),
                                      axis_position(start_layout, spec.axis))
end

# Nested-Dual probes #
#--------------------#

struct MpsProbeInner end
struct MpsProbeOuter end
const _MPS_TAG_IN  = typeof(ForwardDiff.Tag(MpsProbeInner(), Float64))
const _MPS_TAG_OUT = typeof(ForwardDiff.Tag(MpsProbeOuter(), Float64))

"`(c, c′, c″)` of the cost at `θ` by one nested-Dual evaluation over the FROZEN env. Device-legal."
@inline function _mps_c012(cost, θ::Float64, envf)
    d = Dual{_MPS_TAG_OUT}(Dual{_MPS_TAG_IN}(θ, 1.0), Dual{_MPS_TAG_IN}(1.0, 0.0))
    r = cost(d; env=envf)
    return (ForwardDiff.value(ForwardDiff.value(r)),
            ForwardDiff.partials(ForwardDiff.value(r), 1),
            ForwardDiff.partials(ForwardDiff.partials(r, 1), 1))
end

"`(A′, A″)` of the banded objective at `θ > 0` by one nested-Dual gather, V read at its primals. Device-legal."
@inline function _mps_derivs(col, x, θ::Float64, dest_grid)
    d = Dual{_MPS_TAG_OUT}(Dual{_MPS_TAG_IN}(θ, 1.0), Dual{_MPS_TAG_IN}(1.0, 0.0))
    r = _gs_gather_cell(col, x, d, dest_grid, _frz)
    return (ForwardDiff.partials(ForwardDiff.value(r), 1),
            ForwardDiff.partials(ForwardDiff.partials(r, 1), 1))
end

"the frozen interpolant's slope across the destination interval `[j, j+1]`."
@inline _mps_slope(col, dest_grid, j) =
    (_frz(col[j+1]) - _frz(col[j])) / (dest_grid[j+1] - dest_grid[j])

"""
The interpolant's one-sided slopes at the origin coordinate `x` on the destination grid, which price
the kink `A′(0⁺) = φ(0)·(s_R − s_L)`. Both are the bracketing interval's off a node, the outward one
is zero at the end nodes, and both are zero strictly outside the grid.
"""
@inline function _mps_kink_slopes(col, x, dest_grid)
    n = length(dest_grid)
    n == 1 && return (0.0, 0.0)
    (x < dest_grid[1] || x > dest_grid[n]) && return (0.0, 0.0)   # clamped: all mass at one end node
    x == dest_grid[1] && return (0.0, _mps_slope(col, dest_grid, 1))
    x == dest_grid[n] && return (_mps_slope(col, dest_grid, n - 1), 0.0)
    j = searchsortedlast(dest_grid, x)
    x == dest_grid[j] || return (_mps_slope(col, dest_grid, j), _mps_slope(col, dest_grid, j))
    return (_mps_slope(col, dest_grid, j - 1), _mps_slope(col, dest_grid, j))
end

# Policy tangents — the IFT attach #
#----------------------------------#

"`v` at its primal, with its direction-`k` tangent seeded as a `Float64` in the outer probe lane."
@inline _ift_seed(v::Real, k) =
    Dual{_MPS_TAG_OUT}(Dual{_MPS_TAG_IN}(Float64(_frz(v)), 0.0),
                       Dual{_MPS_TAG_IN}(Float64(ForwardDiff.partials(v, k)), 0.0))

"The frozen `envf` with each floating-point entry re-seeded from `env`'s direction-`k` tangent; `Integer`, `Bool` and non-scalar entries stay frozen."
@inline _ift_env(envf::NamedTuple, env::NamedTuple, k) = map((vf, v) -> _ift_entry(vf, v, k), envf, env)
@inline _ift_env(envf, env, k) = envf

"one env entry re-seeded in direction `k`, or left at the primal where it carries no tangent."
@inline _ift_entry(vf, v::Union{AbstractFloat, ForwardDiff.Dual}, k) = _ift_seed(v, k)
@inline _ift_entry(vf, v, k) = vf

"Throwing guard, host-side once per `backward!`: no `env` entry may hide a tangent inside a non-scalar value."
function _assert_scalar_env_tangents(env, ::Type{T}, stage::String) where {T}
    hidden(v) = !(v isa Real) && _frz_entry(v) !== v
    (carries_tangents(T) && env isa NamedTuple && any(hidden, values(env))) || return
    error("$stage: env entries $(filter(k -> hidden(env[k]), keys(env))) hide a tangent inside a " *
          "non-scalar value — hoist what the cost reads into a top-level scalar env field.")
end

"the IFT numerator `n_k = ∂ε_k[g_θ]` off one probe of `A − cost`, with `A(θ_d, k)` the family's seeded banded row."
@inline function _ift_numerator(A::F, θf::Float64, cost, env, envf, k) where {F}
    d = Dual{_MPS_TAG_OUT}(Dual{_MPS_TAG_IN}(θf, 1.0), Dual{_MPS_TAG_IN}(0.0, 0.0))
    r = A(d, k) - cost(d; env = _ift_env(envf, env, k))
    return ForwardDiff.partials(ForwardDiff.partials(r, 1), 1)
end

"""
The seated policy at the primal optimum `θf`: `T(θf)` with exactly zero partials, except at a
strictly interior optimum of a tangent-carrying buffer with finite concave curvature and finite
numerators, where direction `k` takes `−numer(k)/curv()`.
"""
@inline function _ift_seat(::Type{T}, θf, lo, hi, numer::F, curv::G) where {T, F, G}
    (carries_tangents(T) && lo < θf < hi) || return T(θf)
    gθθ = curv()
    (isfinite(gθθ) && gθθ < 0) || return T(θf)
    ṫ = ntuple(k -> -numer(k) / gθθ, Val(ForwardDiff.npartials(T)))
    all(isfinite, ṫ) || return T(θf)
    return Dual{ForwardDiff.tagtype(T)}(θf, ṫ...)
end

# Per-cell continuous solver #
#----------------------------#

"""
Refine one scan bracket `kb` on the FROZEN objective: exact endpoint tests first (the analytic
interiority test at `kb = 0`; `F(θ_max) ≥ 0` at `kb = nscan`), then safeguarded Newton/bisection on
`F = A′ − c′` in the ε-floored bracket. Never worse than the sampled `(vb, θ_kb)`. Device-legal.
"""
function _mps_refine(col, x, dest_grid, cost, envf, θ_max, nscan, kb, vb)
    Δθ = θ_max / nscan
    if kb == 0
        # Interiority at θ = 0: `A′(0⁺) = φ(0)·(s_R − s_L)`, against an fp noise floor.
        sL, sR = _mps_kink_slopes(col, x, dest_grid)
        a0    = _gs_φ(0.0) * (sR - sL)
        atol0 = 1e-12 * _gs_φ(0.0) * (abs(sL) + abs(sR))
        c1 = _mps_c012(cost, 0.0, envf)[2]
        (isfinite(a0) && a0 > c1 + atol0) || return (vb, 0.0)   # not interior: exact θ* = 0
    elseif kb == nscan
        d1, _ = _mps_derivs(col, x, θ_max, dest_grid)
        F = d1 - _mps_c012(cost, θ_max, envf)[2]
        isfinite(F) && F >= 0 && return (vb, θ_max)             # still climbing: exact θ* = θ_max
    end
    lo = max((kb - 1) * Δθ, 1e-8 * θ_max)
    hi = min((kb + 1) * Δθ, θ_max)
    θb = clamp(kb * Δθ, lo, hi)                     # sampled winner, the safeguard anchor
    θ  = kb == 0 ? 0.5 * (lo + hi) : θb
    a, b = lo, hi
    for _ in 1:25
        d1, d2 = _mps_derivs(col, x, θ, dest_grid)
        _, c1, c2 = _mps_c012(cost, θ, envf)
        F = d1 - c1; Fp = d2 - c2
        if !(isfinite(F) && isfinite(Fp))
            θ = 0.5 * (θ + θb)                      # nonfinite: bisect toward the winner
            continue
        end
        F > 0 ? (a = θ) : (b = θ)
        θn = Fp < -1e-12 ? θ - F / Fp : NaN         # Newton iff locally concave…
        !isnan(θn) && abs(θn - θ) <= 1e-12 * θ_max && break   # sub-tol Newton step: θ is converged
        (isnan(θn) || θn <= a || θn >= b) && (θn = 0.5 * (a + b))   # …else bisect the sign bracket
        abs(θn - θ) <= 1e-12 * θ_max && (θ = θn; break)
        θ = θn
    end
    vθ = _gs_gather_cell(col, x, θ, dest_grid, _frz) - cost(θ; env=envf)
    return (isfinite(vθ) && vθ >= vb) ? (vθ, θ) : (vb, kb * Δθ)
end

"""
Solve one cell's continuous dispersion choice: rolling frozen scan over `{0, …, θ_max}`, refine EVERY
scan-local maximum, keep the best. Returns `(v, θ*)` at the BUFFER eltype `T`, or `(typemin(T), 0.0)`
where every scanned θ is nonfinite. Device-legal.
"""
function _mps_solve_cell(col, x, dest_grid, cost, env, envf, θ_max, nscan, ::Type{T}) where {T}
    vf = -Inf; θf = 0.0
    vL = -Inf
    vC = _gs_gather_cell(col, x, 0.0, dest_grid, _frz) - cost(0.0; env=envf)
    isfinite(vC) || (vC = -Inf)
    for k in 0:nscan
        vN = -Inf
        if k < nscan
            θn = θ_max * (k + 1) / nscan
            vN = _gs_gather_cell(col, x, θn, dest_grid, _frz) - cost(θn; env=envf)
            isfinite(vN) || (vN = -Inf)
        end
        if vC > -Inf && vC > vL && vC >= vN         # scan-local max, endpoints included
            v, θ = _mps_refine(col, x, dest_grid, cost, envf, θ_max, nscan, k, vC)
            v > vf && (vf = v; θf = θ)
        end
        vL = vC; vC = vN
    end
    vf > -Inf || return (typemin(T), T(0.0))

    # Envelope value: FULL (possibly Dual) arithmetic at the primal θ*.
    return (_gs_gather_cell(col, x, θf, dest_grid) - cost(θf; env),
            _mps_seat(T, col, x, dest_grid, cost, env, envf, θ_max, θf))
end

"Seat one cell's spread: the primal optimum `θf`, passed by argument, with its IFT tangent where interior."
@inline function _mps_seat(::Type{T}, col, x, dest_grid, cost, env, envf, θ_max, θf::Float64) where {T}
    A(d, k) = _gs_gather_cell(col, x, d, dest_grid, v -> _ift_seed(v, k))
    return _ift_seat(T, θf, 0.0, θ_max,
                     k -> _ift_numerator(A, θf, cost, env, envf, k),
                     () -> _mps_derivs(col, x, θf, dest_grid)[2] - _mps_c012(cost, θf, envf)[3])
end

"Backward fiber op: solve every cell of the fiber. The cost and env must be isbits, the closure running on the device."
struct MpsSolveOp{C, E, EF} <: AbstractFiberOp
    cost  :: C
    env   :: E
    envf  :: EF
    θ_max :: Float64
    nscan :: Int
end
function (op::MpsSolveOp)(vsf, vfib, θf, origin_grid, dest_grid)
    for i in eachindex(vsf)
        vsf[i], θf[i] = _mps_solve_cell(vfib, origin_grid[i], dest_grid, op.cost, op.env, op.envf,
                                        op.θ_max, op.nscan, eltype(θf))
    end
    return vsf
end

function backward!(V_start, spec::MeanPreservingSpreadStageSpec, start_layout::GriddedLayout,
                   ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool=true)
    (; origin_grid, dest_grid) = scratch.kernel_scratch   # the two axis griddings, tangent-free at the model's primal float type
    _assert_scalar_env_tangents(env, eltype(kernel.θstar), "MeanPreservingSpreadStage")
    envf = _frz_env(env)                            # primal env for the solver's cost derivatives
    op   = MpsSolveOp(spec.cost, env, envf, spec.θ_max, spec.nscan)
    stratified!(op, V_start, V_end, kernel.θstar, origin_grid, dest_grid;
                dims=axis_position(start_layout, spec.axis))
    return (V_start, kernel)
end

"the per-cell chosen spread sd θ*(x) — the seated kernel's parameter field."
policy(stage::MeanPreservingSpreadStage) = stage.kernel.θstar
