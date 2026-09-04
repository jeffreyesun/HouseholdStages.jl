# Gaussian loading with a CONTINUOUS choice: per cell at axis coordinate `x`, the loading
# θ ∈ [θ_lo, θ_hi] maximising `g(θ) = A(x, θ) − cost(θ; env)`, where `A = Σ_j w_j(m, s)·V_end[j]` sums
# the truncated-Gaussian Young row of `GaussianLoadingKernel` at mean `m = x·(anchor + θμ)` and sd
# `s = |x|·θ·σ`, so the next coordinate is `x·(anchor + θ·(μ + σZ))`.

"""
Continuous Gaussian-loading choice over `axis`: per cell at coordinate `x`, pick the loading
`θ ∈ [θ_lo, θ_hi]` (`loading_bounds`, default `(0, 1)`) on a truncated-Gaussian increment — next
coordinate `x·(anchor + θ·(μ + σZ))` — at cost `cost(θ; env)` (a ForwardDiff-differentiable closure),
seating the per-cell winner θ*(x) into a [`GaussianLoadingKernel`](@ref). `anchor`, `increment_mean`
and `increment_sd` are scalars or `FromEnv`; `nscan` is the solver's bracketing resolution.
"""
struct GaussianLoadingStageSpec{A, Mu, Sd, C} <: AbstractStageSpec
    axis           :: Symbol
    anchor         :: A
    increment_mean :: Mu
    increment_sd   :: Sd
    θ_lo           :: Float64
    θ_hi           :: Float64
    nscan          :: Int
    cost           :: C
end

"Keyword constructor; loadings must be nonnegative and `increment_sd` strictly positive."
function GaussianLoadingStageSpec(; axis::Symbol=:wealth, anchor, increment_mean, increment_sd,
                                  loading_bounds=(0.0, 1.0), nscan::Int=12, cost=(θ; env) -> 0.0)
    #TODO Signed loadings: needs `|x·θ|·σ` at both row sites plus the |θ| kink at 0.
    lo, hi = Float64(loading_bounds[1]), Float64(loading_bounds[2])
    @assert 0 <= lo < hi && isfinite(hi)
    increment_sd isa Real && @assert increment_sd > 0
    return GaussianLoadingStageSpec(axis, anchor, increment_mean, increment_sd, lo, hi, nscan, cost)
end

@definestage GaussianLoadingStage GaussianLoadingStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(spec::GaussianLoadingStageSpec) = spec.axis
tangent_grade(::GaussianLoadingStageSpec)     = :exact_ae

function allocate_kernel(spec::GaussianLoadingStageSpec, ::Type{T}, start_layout::GriddedLayout,
                         ::GriddedLayout) where {T}
    _assert_single_lift(T, "GaussianLoadingStage")
    prim(v) = v isa Real ? T(v) : T(NaN)
    return GaussianLoadingKernel(zeros(T, layout_size(start_layout)),
                                 axis_position(start_layout, spec.axis),
                                 prim(spec.anchor), prim(spec.increment_mean),
                                 prim(spec.increment_sd))
end

# Per-cell continuous solver #
#----------------------------#

"`(A′, A″)` of the objective at `θ` by one nested-Dual gather, V read at its primals. Requires `s(θ) > 0`. Device-legal."
@inline function _gl_derivs(col, x, θ::Float64, anchorf, μf, σf, dest_grid)
    d = Dual{_MPS_TAG_OUT}(Dual{_MPS_TAG_IN}(θ, 1.0), Dual{_MPS_TAG_IN}(1.0, 0.0))
    r = _gl_A(col, x, d, anchorf, μf, σf, dest_grid, _frz)
    return (ForwardDiff.partials(ForwardDiff.value(r), 1),
            ForwardDiff.partials(ForwardDiff.partials(r, 1), 1))
end

"""
Refine one scan bracket `kb` on the FROZEN objective: ε-probe endpoint tests first (returning exactly
θ_lo/θ_hi at the bounds), then safeguarded Newton/bisection on `F = A′ − c′` in the ε-floored bracket.
Never worse than the sampled `(vb, θ_kb)`. Device-legal.
"""
function _gl_refine(col, x, dest_grid, cost, envf, anchorf, μf, σf, θ_lo, θ_hi, nscan, kb, vb)
    span = θ_hi - θ_lo; Δθ = span / nscan
    ε = 1e-8 * span
    if kb == 0
        d1, _ = _gl_derivs(col, x, θ_lo + ε, anchorf, μf, σf, dest_grid)
        F = d1 - _mps_c012(cost, θ_lo, envf)[2]
        isfinite(F) && F <= 0 && return (vb, θ_lo)              # not interior: exact θ* = θ_lo
    elseif kb == nscan
        d1, _ = _gl_derivs(col, x, θ_hi, anchorf, μf, σf, dest_grid)
        F = d1 - _mps_c012(cost, θ_hi, envf)[2]
        isfinite(F) && F >= 0 && return (vb, θ_hi)              # still climbing: exact θ* = θ_hi
    end
    lo = max(θ_lo + (kb - 1) * Δθ, θ_lo + ε)
    hi = min(θ_lo + (kb + 1) * Δθ, θ_hi)
    θb = clamp(θ_lo + kb * Δθ, lo, hi)                  # sampled winner, the safeguard anchor
    θ  = kb == 0 ? 0.5 * (lo + hi) : θb
    a, b = lo, hi
    for _ in 1:25
        d1, d2 = _gl_derivs(col, x, θ, anchorf, μf, σf, dest_grid)
        _, c1, c2 = _mps_c012(cost, θ, envf)
        F = d1 - c1; Fp = d2 - c2
        if !(isfinite(F) && isfinite(Fp))
            θ = 0.5 * (θ + θb)                          # nonfinite: bisect toward the winner
            continue
        end
        F > 0 ? (a = θ) : (b = θ)
        θn = Fp < -1e-12 ? θ - F / Fp : NaN             # Newton iff locally concave…
        !isnan(θn) && abs(θn - θ) <= 1e-12 * span && break      # sub-tol Newton step: converged
        (isnan(θn) || θn <= a || θn >= b) && (θn = 0.5 * (a + b))   # …else bisect the sign bracket
        abs(θn - θ) <= 1e-12 * span && (θ = θn; break)
        θ = θn
    end
    vθ = _gl_A(col, x, θ, anchorf, μf, σf, dest_grid, _frz) - cost(θ; env=envf)
    return (isfinite(vθ) && vθ >= vb) ? (vθ, θ) : (vb, θ_lo + kb * Δθ)
end

"""
Solve one cell's continuous loading choice: rolling frozen scan over `{θ_lo, …, θ_hi}`, refine EVERY
scan-local maximum, keep the best; `x == 0` takes θ* = θ_lo directly. Returns `(v, θ*)` at the BUFFER
eltype `T`, or `(typemin(T), θ_lo)` where every scanned θ is nonfinite. Device-legal.
"""
function _gl_solve_cell(col, x, dest_grid, cost, env, envf, anchor, μ, σ, anchorf, μf, σf, θ_lo,
                        θ_hi, nscan, ::Type{T}) where {T}
    #TODO The x == 0 shortcut assumes the cost is minimized at θ_lo (true of every shipping
    # cost); a cost with an interior minimum would need a scan over cost alone.
    iszero(x) && return (_gl_A(col, x, θ_lo, anchor, μ, σ, dest_grid) - cost(θ_lo; env), T(θ_lo))

    vf = -Inf; θf = θ_lo
    vL = -Inf
    vC = _gl_A(col, x, θ_lo, anchorf, μf, σf, dest_grid, _frz) - cost(θ_lo; env=envf)
    isfinite(vC) || (vC = -Inf)
    for k in 0:nscan
        vN = -Inf
        if k < nscan
            θn = θ_lo + (θ_hi - θ_lo) * (k + 1) / nscan
            vN = _gl_A(col, x, θn, anchorf, μf, σf, dest_grid, _frz) - cost(θn; env=envf)
            isfinite(vN) || (vN = -Inf)
        end
        if vC > -Inf && vC > vL && vC >= vN             # scan-local max, endpoints included
            v, θ = _gl_refine(col, x, dest_grid, cost, envf, anchorf, μf, σf, θ_lo, θ_hi, nscan, k, vC)
            v > vf && (vf = v; θf = θ)
        end
        vL = vC; vC = vN
    end
    vf > -Inf || return (typemin(T), T(θ_lo))

    # Envelope value: FULL (possibly Dual) arithmetic — live increment and env — at the primal θ*.
    return (_gl_A(col, x, θf, anchor, μ, σ, dest_grid) - cost(θf; env),
            _gl_seat(T, col, x, dest_grid, cost, env, envf, anchor, μ, σ, anchorf, μf, σf, θ_lo, θ_hi, θf))
end

"Seat one cell's loading: the primal optimum `θf` with its IFT tangent where interior, the three increment primitives riding the same probe as V and the env."
@inline function _gl_seat(::Type{T}, col, x, dest_grid, cost, env, envf, anchor, μ, σ,
                          anchorf, μf, σf, θ_lo, θ_hi, θf::Float64) where {T}
    A(d, k) = _gl_A(col, x, d, _ift_seed(anchor, k), _ift_seed(μ, k), _ift_seed(σ, k), dest_grid,
                    v -> _ift_seed(v, k))
    return _ift_seat(T, θf, θ_lo, θ_hi,
                     k -> _ift_numerator(A, θf, cost, env, envf, k),
                     () -> _gl_derivs(col, x, θf, anchorf, μf, σf, dest_grid)[2] -
                           _mps_c012(cost, θf, envf)[3])
end

"Backward fiber op: solve every cell of the fiber. The cost and env must be isbits, the closure running on the device."
struct GlSolveOp{C, E, EF, A, M, S} <: AbstractFiberOp
    cost    :: C
    env     :: E
    envf    :: EF
    anchor  :: A          # live (possibly Dual) — envelope value
    μ       :: M
    σ       :: S
    anchorf :: Float64    # frozen primals — solver
    μf      :: Float64
    σf      :: Float64
    θ_lo    :: Float64
    θ_hi    :: Float64
    nscan   :: Int
end
function (op::GlSolveOp)(vsf, vfib, θf, origin_grid, dest_grid)
    for i in eachindex(vsf)
        vsf[i], θf[i] = _gl_solve_cell(vfib, origin_grid[i], dest_grid, op.cost, op.env, op.envf,
                                       op.anchor, op.μ, op.σ, op.anchorf, op.μf, op.σf,
                                       op.θ_lo, op.θ_hi, op.nscan, eltype(θf))
    end
    return vsf
end

function backward!(V_start, spec::GaussianLoadingStageSpec, start_layout::GriddedLayout,
                   ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool=true)
    (; origin_grid, dest_grid) = scratch.kernel_scratch # the two axis griddings, tangent-free at the model's primal float type
    _assert_scalar_env_tangents(env, eltype(kernel.θstar), "GaussianLoadingStage")
    anchor = resolve(spec.anchor, env)                  # live (possibly Dual) increment primitives
    μ = resolve(spec.increment_mean, env)
    σ = resolve(spec.increment_sd, env)
    anchorf = Float64(_frz(anchor)); μf = Float64(_frz(μ)); σf = Float64(_frz(σ))
    @assert σf > 0
    envf = _frz_env(env)                                # primal env for the solver's cost derivatives
    op = GlSolveOp(spec.cost, env, envf, anchor, μ, σ, anchorf, μf, σf,
                   spec.θ_lo, spec.θ_hi, spec.nscan)
    stratified!(op, V_start, V_end, kernel.θstar, origin_grid, dest_grid;
                dims=axis_position(start_layout, spec.axis))
    # Reseat with the primitives this backward resolved (immutable rebuild, shared θstar).
    S = eltype(kernel.θstar)
    return (V_start, GaussianLoadingKernel(kernel.θstar, kernel.adim, S(anchor), S(μ), S(σ)))
end

"the per-cell chosen loading θ*(x) — the seated kernel's parameter field."
policy(stage::GaussianLoadingStage) = stage.kernel.θstar
