"""
Argmax over one named choice `axis` — the tropical `(max, +)` contraction. For each origin cell,
choose a destination index `a` on the axis maximising `reward[a, origin] + V_end[a]`; the chosen
index IS the policy and the kernel's single destination (forward scatters by it, `:nearest`).
The stage carries **no discount** `β` — discounting is a separate `∘`-composed
[`TimeDiscountingStage`](@ref) acting on `V_end` before the argmax (end-goal §1: the discount is the
unique linearization-property exception, so it must be its own pointwise-scale stage).
`reward` is a matrix source `M[after, before]` on the choice axis (a `Matrix`, `FromEnv`, or closure
`(; dep…[, env]) -> Matrix`); a non-finite entry marks `(after, before)` unavailable.

`search` selects the apply:
- `:brute` (default) — general per-column `findmax`; any axis, any reward, no monotonicity assumed.
  Supports a rectangular `before`-singleton reward (`n_start = 1`) that collapses the axis by max
  (the kernel-choice merge / max-marginalize).
- `:divide_conquer` / `:sequential` — monotone-policy optimisations (`O(n log n)` segment-halving /
  resumed walk) assuming the optimal policy is non-decreasing along the axis (Topkis: increasing
  differences of the reward face). That is verified each `backward!` over the feasible region unless
  `assume_monotone = true`. The primitive behind `ConsumptionSavingsStage` (`search = :divide_conquer`).

This is the temperature-0 limit of `LogitChoiceStage`.
"""
struct ArgmaxStageSpec{R, Search} <: AbstractStageSpec
    reward          :: R
    axis            :: Symbol
    assume_monotone :: Bool
end

function ArgmaxStageSpec(; reward, axis::Symbol,
                         search::Symbol = :brute, assume_monotone::Bool = false)
    @assert search in (:brute, :sequential, :divide_conquer) "ArgmaxStage: search ∈ (:brute, :sequential, :divide_conquer)"
    return ArgmaxStageSpec{typeof(reward), Val{search}}(reward, axis, assume_monotone)
end

@definestage ArgmaxStage ArgmaxStageSpec


##########################
# Gridded implementation #
##########################

# Rectangular reward (`:brute` only): a `before`-singleton reward (`n_start = 1`) collapses the choice
# axis by max — agent (start) side carries the axis at size 1 (`input_layout`), continuation (end) side
# keeps it full. Square ⇒ both full. The construction `layout` carries the FULL axis (= reward `after`).
function input_layout(spec::ArgmaxStageSpec, layout::GriddedLayout)
    n_end, n_start = _field_shape(spec.reward, layout, spec.axis)
    @assert _axis_size(layout, spec.axis) == n_end "ArgmaxStage: construct against the full " *
        "`$(spec.axis)` axis — its size $(_axis_size(layout, spec.axis)) must equal the reward's `after` dim $n_end."
    @assert n_start == n_end || n_start == 1 "ArgmaxStage: rectangular reward supports only " *
        "n_start ∈ {1 (collapse), n_end (square)}; got (after=$n_end, before=$n_start)."
    resize_axis(layout, spec.axis, n_start)
end

# Kernel: a `ScatterKernel` (discrete axis) over the chosen choice-axis index per cell (the policy
# IS the destination); sized at the (possibly collapsed) `input_layout`. Forward scatters `:nearest`.
allocate_kernel(spec::ArgmaxStageSpec, ::Type, layout::GriddedLayout) =
    ScatterKernel(zeros(Int, layout_size(input_layout(spec, layout))),
                  Val(axis_position(layout, spec.axis)))

"Scratch: io buffers + the reward `MatrixField` `U` (compact `(after, before, dep…)`). The argmax READS `U` for the max-plus solve rather than applying it as an operator, so `U` is a bare field, not a `DenseKernel`; an env-independent reward is filled here once, an env-dependent one is NaN-filled and seated each `backward!`."
function allocate_scratch(spec::ArgmaxStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    U = matrix_field(T, layout, spec.axis, spec.reward)
    reads_env(spec.reward) || fill_field!(U, spec.reward, layout, spec.axis, nothing)
    return merge(io_scratch(spec, layout, T), (U = U,))
end

"Cache: the static env-dependence classification for the reward field `U` (`reads_env`, computed once), so a fixed-env VFI loop materialises it once (§5.3)."
allocate_cache(spec::ArgmaxStageSpec, ::Type, ::GriddedLayout) =
    (reward_env_dep = reads_env(spec.reward),)

# Per-column apply, selected by `search`. Each `_ca_table_walk!` mode shares the column loop and
# reward-face plumbing of `_ca_backward_columns!`, differing only in the innermost scan.

# :iter_dc — flat iterative segment-halving walk (allocation-free, `@simd`-friendly).
function _ca_table_walk!(::Val{:iter_dc}, Vs_col, pol_col, Vc_col, u_slice, n_c)
    T = eltype(Vs_col)
    n = n_c
    @inbounds begin
        pol_col[1] = 1
        Vs_col[1]  = u_slice[1, 1] + Vc_col[1]
        best = typemin(T); ba = 1
        for a in 1:n
            v = u_slice[a, n] + Vc_col[a]
            v > best && (best = v; ba = a)
        end
        pol_col[n] = ba; Vs_col[n] = best
        seg = div(n - 1, 2)
        while seg >= 1
            i = 1
            while i < n - 1
                lb = pol_col[i]
                i += seg
                ub = pol_col[i + seg]
                best = typemin(T); ba = lb
                for a in lb:min(ub, i)
                    v = u_slice[a, i] + Vc_col[a]
                    v > best && (best = v; ba = a)
                end
                pol_col[i] = ba
                Vs_col[i]  = u_slice[ba, i] + Vc_col[ba]
                i += seg
            end
            seg = div(seg, 2)
        end
    end
    return
end

# :rec_dc — recursive divide-and-conquer, arbitrary n_c.
function _ca_table_walk!(::Val{:rec_dc}, Vs_col, pol_col, Vc_col, u_slice, n_c)
    T = eltype(Vs_col)
    function rec!(lo::Int, hi::Int, lo_b::Int, hi_b::Int)
        lo > hi && return
        mid = (lo + hi) >> 1
        best_v = typemin(T); best_a = 0
        @inbounds for a in lo_b:hi_b
            v = u_slice[a, mid] + Vc_col[a]
            v > best_v && (best_v = v; best_a = a)
        end
        if best_a == 0
            @inbounds Vs_col[mid] = typemin(T); @inbounds pol_col[mid] = 1
            rec!(lo, mid - 1, lo_b, hi_b); rec!(mid + 1, hi, lo_b, hi_b)
        else
            @inbounds Vs_col[mid] = best_v; @inbounds pol_col[mid] = best_a
            rec!(lo, mid - 1, lo_b, best_a); rec!(mid + 1, hi, best_a, hi_b)
        end
        return
    end
    rec!(1, n_c, 1, n_c)
    return
end

# :seq — sequential monotone walk (search resumes from the previous argmax).
function _ca_table_walk!(::Val{:seq}, Vs_col, pol_col, Vc_col, u_slice, n_c)
    T = eltype(Vs_col)
    prev_a = 1
    @inbounds for s in 1:n_c
        best_v = typemin(T); best_a = 0
        for a in prev_a:s
            v = u_slice[a, s] + Vc_col[a]
            v > best_v && (best_v = v; best_a = a)
        end
        if best_a == 0
            Vs_col[s] = typemin(T); pol_col[s] = 1
        else
            Vs_col[s] = best_v; pol_col[s] = best_a; prev_a = best_a
        end
    end
    return
end

# :brute — general (max, +) over an arbitrary axis: a plain `findmax` per column, no monotonicity.
# Rectangular-aware (`Vs_col` length `n_start`, `Vc_col` length `n_end`); `n_c` ignored.
function _ca_table_walk!(::Val{:brute}, Vs_col, pol_col, Vc_col, u_slice, n_c)
    T = eltype(Vs_col)
    @inbounds for s in 1:length(Vs_col)
        best_v = typemin(T); best_a = 1
        for a in 1:length(Vc_col)
            v = u_slice[a, s] + Vc_col[a]
            v > best_v && (best_v = v; best_a = a)
        end
        Vs_col[s] = best_v; pol_col[s] = best_a
    end
    return
end

# Map `search` to an internal walk mode (consumed at the `_ca_backward_columns!` Val barrier). D&C
# uses the iterative walk when `n-1` is a power of two, else the recursive fallback.
_ca_walk_mode(::Type{Val{:brute}}, n_c)          = Val(:brute)
_ca_walk_mode(::Type{Val{:sequential}}, n_c)     = Val(:seq)
_ca_walk_mode(::Type{Val{:divide_conquer}}, n_c) = ispow2(n_c - 1) ? Val(:iter_dc) : Val(:rec_dc)

# The `(after, before)` reward face for layout column `col`: colon the compact `U.array`'s after (1)
# and before (2) dims, and project `col` onto the field's dep dims (the reward is constant along the
# non-dep, non-operative axes). A strided `view` of the compact `(after, before, dep…)` field —
# explicit dep metadata, no permuted-view shape inference.
@inline _ca_reward_face(U::MatrixField, odep_dims::NTuple{ND, Int}, col) where {ND} =
    view(U.array, :, :, ntuple(k -> col[odep_dims[k]], Val(ND))...)

# Monotone-policy guard (monotone modes only). By Topkis the optimal policy is non-decreasing when the
# objective has increasing differences in (choice, state); the `V_end[a]` term cancels, so the condition
# reduces to increasing differences of the reward face — an O(face) check over the dep combos,
# independent of the (huge) non-dep state space. A violation means a pruned walk may mis-solve, so we
# refuse. Each face is a strided view of `U` at a dep combo (after × before).
function _check_increasing_differences(U::MatrixField, odep_dims::NTuple{ND, Int}) where {ND}
    A         = U.array
    nC        = size(A, 1)                                    # after (choices)
    nW        = size(A, 2)                                    # before (states) — the n_in column
    dep_sizes = ntuple(k -> size(A, 2 + k), Val(ND))
    @inbounds for dc in CartesianIndices(dep_sizes)
        face = view(A, :, :, dc.I...)
        for w in 1:nW-1, a in 1:nC-1
            u00 = face[a, w];   u10 = face[a+1, w]
            u01 = face[a, w+1]; u11 = face[a+1, w+1]
            (isfinite(u00) & isfinite(u10) & isfinite(u01) & isfinite(u11)) || continue
            d0  = u10 - u00
            d1  = u11 - u01
            tol = 1e-9 * (1 + max(abs(u00), abs(u10), abs(u01), abs(u11)))
            d1 < d0 - tol && error(
                "ArgmaxStage: the monotone-policy assumption used by `:divide_conquer`/`:sequential` " *
                "fails — the reward is not supermodular (increasing differences violated near choice " *
                "$a, state $w). Use `search = :brute`, or pass `assume_monotone = true` to skip this check.")
        end
    end
    return
end

# Column loop, specialised on the choice dim `WD` and walk `Mode` (Val barrier) for concretely-typed
# per-column slice views. The reward face is a strided `view` of the self-describing `U` at the
# column's dep combo (`_ca_reward_face`); `odep_dims` (layout positions of the reward's dep axes)
# is the only thing the stage needs — never the kernel's internal layout.
function _ca_backward_columns!(::Val{WD}, mode::Val, U::MatrixField, odep_dims::NTuple{ND, Int}, Vc,
                               V_start, policy, n_c, dims, ::Val{N}, check::Bool) where {WD, ND, N}
    check && _check_increasing_differences(U, odep_dims)
    other_sizes = ntuple(i -> i == WD ? 1 : dims[i], Val(N))
    @inbounds for other_ci in CartesianIndices(other_sizes)
        col = other_ci.I
        u_slice = _ca_reward_face(U, odep_dims, col)
        Vc_col  = view(Vc,      ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        Vs_col  = view(V_start, ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        pol_col = view(policy,  ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        _ca_table_walk!(mode, Vs_col, pol_col, Vc_col, u_slice, n_c)
    end
    return
end

# :brute fast path — a SMALL operative axis with a dep-free reward (the gated discrete choices
# `buy_home`/`sell_home`: a tiny housing axis sitting at a non-leading dim, with a reward constant
# across every stratum). The generic `_ca_backward_columns!` pays per-stratum view-creation and a
# closure dispatch over ~millions of strata while the operative axis (≈7) is trivial, so the
# per-stratum overhead (not the operative-axis work) dominates. Here we hoist the reward out of the
# stratum loop and fuse the `(max, +)` solve into a unit-stride scan: reshape on the operative dim to
# `(pre, n, post)` and sweep destinations in an outer loop, so the innermost loop runs contiguously
# over the `pre` block (stride 1) and the compiler vectorises it.
#
# Bit-identical to `_ca_table_walk!(::Val{:brute})`: same `typemin`/default-`1` initial policy, same
# first-index (strict `>`) tie-break, same per-column max. `-Inf` reward entries are pruned — a
# `-Inf` payoff can never strictly beat the running best, so skipping it cannot change the value OR
# the index (this is also where the gate's sparsity pays off: each origin column keeps only its few
# feasible destinations). Restricted to a dep-free reward (`ND == 0`) so the single face indexes
# directly; a dep-carrying brute reward keeps the generic column path.
"""
Fused small-operative-axis `:brute` argmax over a dep-free reward `u[after, before]`: for each origin
`s` and stratum, `V_start = maxₐ u[a, s] + V_end[a]` with the maximiser written to `policy`. Reshapes
on the operative dim `cdim` to `(pre, n, post)` and scans the unit-stride `pre` block in the inner
loop; bit-identical to the per-column brute walk (same tie-break and `typemin`/`1` fallback), pruning
`-Inf` reward entries (they can never win). The hoisted-reward fast path behind buy/sell-home.
"""
function _ca_brute_smallaxis!(V_start, policy, V_end, u, pre::Int, n_start::Int, n_end::Int)
    T    = eltype(V_start)
    post = length(V_start) ÷ (pre * n_start)
    Vs = reshape(V_start, pre, n_start, post)
    Ve = reshape(V_end,   pre, n_end,   post)
    P  = reshape(policy,  pre, n_start, post)
    @inbounds for p in 1:post, s in 1:n_start
        @simd for i in 1:pre
            Vs[i, s, p] = typemin(T)
            P[i, s, p]  = 1
        end
        for a in 1:n_end
            uas = u[a, s]
            isfinite(uas) || continue             # a -Inf payoff never wins ⇒ skip, bit-identical
            @simd for i in 1:pre
                v      = uas + Ve[i, a, p]
                better = v > Vs[i, s, p]
                Vs[i, s, p] = ifelse(better, v, Vs[i, s, p])
                P[i, s, p]  = ifelse(better, a, P[i, s, p])
            end
        end
    end
    return
end

function backward!(V_start, spec::ArgmaxStageSpec{R, Search}, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true) where {R, Search}
    policy = destinations(kernel)            # chosen choice-axis index per cell = the destination
    cache.reward_env_dep && env_changed && fill_field!(scratch.U, spec.reward, layout, spec.axis, env)  # seat U (static refill)
    odeps     = declared_deps(spec.reward, layout)
    odep_dims = map(a -> axis_position(layout, a), odeps)                 # layout positions of the reward's dep axes
    cdim    = axis_position(layout, spec.axis)
    n_start = size(scratch.U.array, 2)                              # before (n_in): the compact field's column dim
    n_end   = size(scratch.U.array, 1)                             # after (n_out): the operative axis full size
    brute = Search === Val{:brute}                                  # V_end read directly: any discount is a preceding stage
    dims    = layout_size(layout)
    pre     = prod(dims[k] for k in 1:cdim-1; init = 1)            # stratum block below the operative dim
    # Small dep-free operative axis at a non-leading dim ⇒ the fused unit-stride scan (buy/sell-home).
    # Leading-dim (`pre == 1`) or dep-carrying brute rewards, and the monotone searches, keep the
    # generic column path — the fast path is a pure win for the per-stratum-overhead regime only.
    # CPU host arrays only: `:brute` has no device kernel (the CUDA ext errors on it via
    # `_ca_backward_columns!`), so a device run must fall through to keep that clean error.
    if brute && isempty(odeps) && pre > 1 && V_start isa Array
        u = reshape(scratch.U.array, n_end, n_start)               # ND == 0 ⇒ already 2-D; reshape is free
        _ca_brute_smallaxis!(V_start, policy, V_end, u, pre, n_start, n_end)
        @assert all(isfinite, V_start) "ArgmaxStage(:brute): every cell must have at least one finite-reward action"
        return (V_start, kernel)
    end
    _ca_backward_columns!(Val(cdim), _ca_walk_mode(Search, n_start), scratch.U, odep_dims, V_end,
                          V_start, policy, n_start, dims, Val(ndims(V_start)),
                          brute ? false : !spec.assume_monotone)
    brute && @assert all(isfinite, V_start) "ArgmaxStage(:brute): every cell must have at least one finite-reward action"
    return (V_start, kernel)
end

# forward! (on-grid integer destination → single-point `:nearest` scatter, K·Λ_start) is the generic
# modern default (abstract.jl).

"""
The solved policy of an [`ArgmaxStage`](@ref): the chosen `axis` index per cell. It IS the kernel's
destination — choosing where to go on the axis is the policy.
"""
policy(stage::ArgmaxStage) = destinations(stage.kernel)


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
