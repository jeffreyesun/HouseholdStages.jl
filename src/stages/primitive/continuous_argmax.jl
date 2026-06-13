"""
Continuous-grid argmax over one named choice axis. For each cell, choose a grid
point `a` on `choice_axis` to maximise

    payoff(; choice = grid[a], <input axes>…, env) + β · V_end[a].

The payoff is a kwarg closure declaring its deps (the reserved `choice` value, the
input axes it reads, optionally `env`); the declared axes size the U table, unread
axes stay singleton. A non-finite payoff marks the choice infeasible.

`monotone_search ∈ (:sequential, :divide_conquer)` assumes the optimal policy is
non-decreasing along the choice axis. That is verified each `backward!` by an
increasing-differences test on the payoff table (`O(N²)` along the choice axis, not
over the full state); pass `assume_monotone = true` to skip it.

The primitive behind `ConsumptionSavingsStage`.
"""
struct ContinuousArgmaxStageSpec{F_p, T, Search} <: AbstractStageSpec
    β               :: T
    payoff          :: F_p
    choice_axis     :: Symbol
    assume_monotone :: Bool
end

function ContinuousArgmaxStageSpec(; payoff, β=1, choice_axis::Symbol=:wealth,
                                   monotone_search::Symbol=:divide_conquer,
                                   assume_monotone::Bool=false)
    @assert monotone_search in (:sequential, :divide_conquer)
    return ContinuousArgmaxStageSpec{typeof(payoff), typeof(β), Val{monotone_search}}(
        β, payoff, choice_axis, assume_monotone,
    )
end

@definestage ContinuousArgmaxStage ContinuousArgmaxStageSpec


##########################
# Gridded implementation #
##########################

# Kernel: the shared `SingleDestinationKernel` (kernel.jl) over the chosen choice-axis
# index per cell — the policy IS the destination index (you pick a grid point to move to),
# so forward scatters exactly (`:nearest`).
allocate_kernel(spec::ContinuousArgmaxStageSpec, ::Type, layout::GriddedLayout) =
    SingleDestinationKernel(zeros(Int, layout_size(layout)), Val(axis_dim(layout, spec.choice_axis)))

# The payoff IS a matrix field over the choice axis: the action is the destination, the
# choice-axis state the origin. `PayoffField` is the converter — it presents the scalar payoff
# as a `(dest, origin)` matrix source, so the SHARED field core (helper/field.jl) allocates and
# fills it exactly as Markov's transition and Logit's cost are. The stage writes no fill loop.

"""
The continuous-argmax payoff as a matrix-field source: `matrix_for` builds the `(dest, origin)`
face `M[d, o] = payoff(; choice = grid[d], <choice_axis> = grid[o], deps…[, env])` (infeasible
→ `typemin`), and `field_deps` is the input axes the payoff reads OTHER than the choice axis
(the choice axis is the two positional matrix dims, never a stored dep — like Logit's cost).
"""
struct PayoffField{F}
    payoff      :: F
    choice_axis :: Symbol
    grid        :: Vector{Float64}
    env_dep     :: Bool
end

field_deps(s::PayoffField, layout::GriddedLayout) =
    filter(!=(s.choice_axis), _closure_deps(s.payoff, layout; reserved = (:choice,)))

function matrix_for(s::PayoffField, combo, env)
    g = s.grid; n = length(g)
    probe = s.env_dep ? s.payoff(; choice = g[1], NamedTuple{(s.choice_axis,)}((g[1],))..., combo..., env) :
                        s.payoff(; choice = g[1], NamedTuple{(s.choice_axis,)}((g[1],))..., combo...)
    M = Matrix{typeof(probe)}(undef, n, n); sentinel = typemin(eltype(M))
    @inbounds for o in 1:n
        origin = merge(NamedTuple{(s.choice_axis,)}((g[o],)), combo)
        for d in 1:n
            u = s.env_dep ? s.payoff(; choice = g[d], origin..., env) : s.payoff(; choice = g[d], origin...)
            M[d, o] = isfinite(u) ? u : sentinel
        end
    end
    return M
end

"Build the payoff converter for this stage/layout (the choice-axis grid is the dest = origin grid). Errors if the payoff omits the reserved `choice` kwarg."
function _payoff_field(spec::ContinuousArgmaxStageSpec, layout::GriddedLayout)
    kws = _closure_kwargs_raw(spec.payoff)
    :choice in kws || error("ContinuousArgmaxStage: payoff must declare the reserved `choice` " *
                            "kwarg (the choice-axis value); got $(kws).")
    return PayoffField(spec.payoff, spec.choice_axis,
                       collect(_axis_grid(layout, spec.choice_axis)), :env in kws)
end

"Scratch: the io buffers + the payoff field `U` (a dense kernel over `(dest, origin, dep…)`, filled by the shared `fill_field!` — once when the payoff is env-independent, else each backward), `vbuf` holding the pre-discounted `β · V_end`, and `filled` tracking whether `U` has been materialised (for the env-independent cache)."
function allocate_scratch(spec::ContinuousArgmaxStageSpec, ::Type{T}, layout::GriddedLayout) where {T}
    U = dense_kernel(T, layout, spec.choice_axis, _payoff_field(spec, layout))
    return merge(io_scratch(spec, layout, T),
                 (U = U, vbuf = zeros(T, layout_size(layout)), filled = Ref(false)))
end

# Monotone argmax over a precomputed `u_slice[choice, state]` and a
# pre-discounted `βV_col`. Three internal modes selected by `_ca_walk_mode`:
#   :iter_dc — reference's flat iterative segment-halving (needs ispow2(n-1))
#   :rec_dc  — recursive divide-and-conquer (arbitrary n fallback)
#   :seq     — sequential monotone walk (the `:sequential` user option)
# Infeasible choices carry `typemin` in `u_slice`, so they are never selected.

# :iter_dc — flat iterative segment-halving walk. Allocation-free and
# branch-predictable; `@simd`-friendly across columns.
function _ca_table_walk!(::Val{:iter_dc}, Vs_col, pol_col, βV_col, u_slice, n_c)
    T = eltype(Vs_col)
    n = n_c
    @inbounds begin
        pol_col[1] = 1
        Vs_col[1]  = u_slice[1, 1] + βV_col[1]
        best = typemin(T); ba = 1
        for a in 1:n
            v = u_slice[a, n] + βV_col[a]
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
                    v = u_slice[a, i] + βV_col[a]
                    v > best && (best = v; ba = a)
                end
                pol_col[i] = ba
                Vs_col[i]  = u_slice[ba, i] + βV_col[ba]
                i += seg
            end
            seg = div(seg, 2)
        end
    end
    return
end

# :rec_dc — recursive divide-and-conquer, arbitrary n_c.
function _ca_table_walk!(::Val{:rec_dc}, Vs_col, pol_col, βV_col, u_slice, n_c)
    T = eltype(Vs_col)
    function rec!(lo::Int, hi::Int, lo_b::Int, hi_b::Int)
        lo > hi && return
        mid = (lo + hi) >> 1
        best_v = typemin(T); best_a = 0
        @inbounds for a in lo_b:hi_b
            v = u_slice[a, mid] + βV_col[a]
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
function _ca_table_walk!(::Val{:seq}, Vs_col, pol_col, βV_col, u_slice, n_c)
    T = eltype(Vs_col)
    prev_a = 1
    @inbounds for s in 1:n_c
        best_v = typemin(T); best_a = 0
        # Feasible choices have c = b_in - b_end > 0 ⟺ a < s, so the search
        # never needs to pass s (a = s gives c = 0, masked to typemin).
        for a in prev_a:s
            v = u_slice[a, s] + βV_col[a]
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

# Map the user's `monotone_search` to an internal walk mode. The D&C path uses
# the reference's iterative walk when n-1 is a power of two (full parity on
# grids like 129), else the recursive fallback. The returned Val is consumed at
# the `_ca_backward_columns!` barrier, which specialises on it.
_ca_walk_mode(::Type{Val{:sequential}}, n_c)     = Val(:seq)
_ca_walk_mode(::Type{Val{:divide_conquer}}, n_c) =
    ispow2(n_c - 1) ? Val(:iter_dc) : Val(:rec_dc)

# Monotone-policy guard. Both walks assume the optimal policy is non-decreasing in the cell's
# choice-axis position; by Topkis that holds when the objective has increasing differences in
# (choice, state). The `βV[a]` term cancels in the cross-difference, so the condition reduces
# to increasing differences of the compact payoff face `Uc[choice, state, dep…]` alone — an
# O(face) check, independent of the (huge) non-dep state space. Checked over the feasible
# (finite) region; a violation means a pruned walk may mis-solve, so we refuse.
function _check_increasing_differences(Uc, ::Val{ND}) where {ND}
    nC, nW    = size(Uc, 1), size(Uc, 2)
    dep_sizes = ntuple(k -> size(Uc, k + 2), Val(ND))
    @inbounds for oc in CartesianIndices(dep_sizes)
        for w in 1:nW-1, a in 1:nC-1
            u00 = Uc[a, w, oc];   u10 = Uc[a+1, w, oc]
            u01 = Uc[a, w+1, oc]; u11 = Uc[a+1, w+1, oc]
            (isfinite(u00) & isfinite(u10) & isfinite(u01) & isfinite(u11)) || continue
            d0  = u10 - u00
            d1  = u11 - u01
            tol = 1e-9 * (1 + max(abs(u00), abs(u10), abs(u01), abs(u11)))
            d1 < d0 - tol && error(
                "ContinuousArgmaxStage: the monotone-policy assumption used by " *
                "`:divide_conquer`/`:sequential` fails — the payoff is not supermodular " *
                "(increasing differences violated near choice $a, state $w). This usually means " *
                "a non-concave payoff or a non-linear constraint. Use a brute-force solve, or " *
                "pass `assume_monotone = true` to skip this check if the policy is monotone.")
        end
    end
    return
end

# Column loop, specialised on the choice dimension `WD` and walk `Mode` (a Val barrier) so the
# per-column slice views are concretely typed. `Uc` is the COMPACT payoff face `(dest, origin,
# dep…)`; each non-choice layout column maps to its dep slice via `odep_dims` (the layout
# positions of U's dep axes) — non-dep axes don't appear in `Uc`, so they read the same face.
function _ca_backward_columns!(::Val{WD}, mode::Val, Uc, odep_dims::NTuple{ND, Int}, βV,
                               V_start, policy, n_c, dims, ::Val{N}, check::Bool) where {WD, ND, N}
    check && _check_increasing_differences(Uc, Val(ND))
    other_sizes = ntuple(i -> i == WD ? 1 : dims[i], Val(N))
    @inbounds for other_ci in CartesianIndices(other_sizes)
        col = other_ci.I
        # u_slice[choice, state] = Uc[:, :, dep-combo]: a plain (n_c, n_c) face.
        u_slice = view(Uc, :, :, ntuple(k -> col[odep_dims[k]], Val(ND))...)
        βV_col  = view(βV,      ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        Vs_col  = view(V_start, ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        pol_col = view(policy,  ntuple(d -> d == WD ? Colon() : col[d], Val(N))...)
        _ca_table_walk!(mode, Vs_col, pol_col, βV_col, u_slice, n_c)
    end
    return
end

function backward!(V_start, spec::ContinuousArgmaxStageSpec{F_p, T, Search},
                   layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache) where {F_p, T, Search}
    policy = kernel.destinations             # chosen choice-axis index per cell = the destination
    source = _payoff_field(spec, layout)
    # Env-independent payoff (no `; env` dep) ⇒ U is constant across calls: fill it once and reuse.
    # Env-dependent ⇒ refill every backward, since env may have changed since the last fill.
    if source.env_dep || !scratch.filled[]
        fill_field!(scratch.U, source, layout, spec.choice_axis, env)   # SHARED materialization of U
        scratch.filled[] = true
    end
    odeps      = field_deps(source, layout)
    odep_dims  = map(a -> axis_dim(layout, a), odeps)
    odep_sizes = map(a -> _axis_size(layout, a), odeps)
    n_c   = length(source.grid)
    Uc    = reshape(parent(scratch.U), n_c, n_c, odep_sizes...)     # compact (dest, origin, dep…)
    cdim  = axis_dim(layout, spec.choice_axis)
    @. scratch.vbuf = spec.β * V_end                               # pre-discount once (not per inner step)
    # Val(cdim) + walk-mode barrier: one dynamic dispatch buys type-stable views.
    _ca_backward_columns!(Val(cdim), _ca_walk_mode(Search, n_c), Uc, odep_dims,
                          scratch.vbuf, V_start, policy, n_c, layout_size(layout),
                          Val(ndims(V_start)), !spec.assume_monotone)
    return (V_start, kernel)
end

# Forward #
#---------#

# forward! (on-grid integer destination → single-point `:nearest` scatter, K·Λ_start; routes to
# `_cs_forward_scatter!`, CUDA-overloaded) is the generic modern default (abstract.jl).

"""
The solved policy of a [`ContinuousArgmaxStage`](@ref): the chosen `choice_axis` grid index
per cell (the choice variable). It IS the kernel's destination — choosing which grid point
to move to is the policy.
"""
policy(stage::ContinuousArgmaxStage) = stage.kernel.destinations


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
