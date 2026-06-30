###################################################################
# HouseholdStagesCUDAExt — GPU kernels for the stages whose hot    #
# path is a scalar-index loop the device cannot broadcast:         #
# ConsumptionSavingsStage, WealthChangeStage, and the discrete     #
# ArgmaxStage's `:brute` `(max, +)` argmax (buy/sell-home).        #
###################################################################
#
# This package extension loads only when CUDA is present. It defines
# `CuArray`-typed methods at the dispatch seams the `src` code exposes
# (`_cs_backward_columns!`, `_cs_forward_scatter!`, `_along_axis`),
# leaving every CPU code path byte-for-byte unchanged. The kernels are
# faithful ports of the hand-written, GPU-tested reference kernels in
# `reference_materials/example_stages/helper/interpolations.jl` and
# `.../consumption_savings.jl`, adapted to this package's
# layout-generic array conventions:
#
#   * the reference fixes the wealth axis at dimension 1 and flattens
#     the rest into a column index `(N_K, N_cols)`. This package puts
#     the wealth axis at an arbitrary `wdim`, so each method first
#     *permutes* the wealth axis to the front (a contiguous device
#     copy), runs the reference kernel on the `(n_w, n_cols)` matrix
#     view, then permutes back. A no-op `permutedims` (wdim == 1) is
#     elided.
#   * feasibility is encoded as `typemin(T)` in the utility table
#     `U`, exactly as the CPU path encodes it, so the ported argmax
#     never selects an infeasible action.
#
# Ported kernels:
#   - CS backward  : `k1_argmax_kernel!`  (monotone-savings argmax)
#   - CS forward   : `get_λ_postc_kernel!` (colliding scatter-add)
#   - WC backward  : `reinterpolate_GPU_kernel!`
#   - WC forward   : `convert_distribution_kernel!`
#   - Argmax bwd   : `_brute_argmax_kernel!` (discrete `(max, +)`, unordered axis;
#                    forward reuses the CS `_cs_forward_scatter!` integer scatter)

module HouseholdStagesCUDAExt

using HouseholdStages
using CUDA

const HS = HouseholdStages

# Bring the seam functions and op structs into scope for method extension.
import HouseholdStages: _ca_backward_columns!, _cs_forward_scatter!,
                        _gather_along!, _caC_backward_columns!,
                        _sam_backward_columns!,
                        _interp1d, _golden_max,
                        _along_axis, reinterpolate!, convert_distribution!,
                        ReinterpOp, ConvertDistOp,
                        _streaming_choice_backward!, _choice_scatter!, _choice_gather!

# ============================================================ #
# Shared layout helper: permute the wealth axis to the front   #
# ============================================================ #

# bring axis `WD` to dimension 1 (identity when WD == 1). Returns a
# fresh contiguous CuArray so the kernel can index columns of a (n_w, n_cols)
# reshape. `_unpermute_front!` writes the result back into `dst`.
@inline _front_perm(::Val{WD}, ::Val{N}) where {WD, N} =
    (WD, ntuple(i -> i < WD ? i : i + 1, Val(N - 1))...)

function _to_front(A::CuArray{T,N}, ::Val{WD}) where {T, N, WD}
    WD == 1 && return A
    return permutedims(A, _front_perm(Val(WD), Val(N)))
end

function _from_front!(dst::CuArray{T,N}, src::CuArray{T,N}, ::Val{WD}) where {T, N, WD}
    if WD == 1
        dst === src || copyto!(dst, src)
        return dst
    end
    # inverse of `_front_perm`: place dim 1 back at WD, shift the rest.
    inv = ntuple(d -> d == WD ? 1 : (d < WD ? d + 1 : d), Val(N))
    permutedims!(dst, src, inv)
    return dst
end

_ncols(A::AbstractArray) = prod(size(A)[2:end])

# Map a CartesianIndex of the trailing dims onto a target trailing size,
# collapsing length-1 (or absent) dims to 1 — broadcasting semantics. Mirrors
# the reference `broadcast_index`.
@inline function _bcast_index(idx::CartesianIndex, sz::Tuple)
    return CartesianIndex(ntuple(
        i -> i > length(sz) ? 1 : min(idx[i], sz[i]),
        Val(length(idx))))
end

# ============================================================ #
# WealthChange backward — reinterpolate_GPU_kernel! port       #
# ============================================================ #

# device kernel — one thread per leading-dim column of `y2`, each
# running the scalar `reinterpolate!` on its 1-D slice. `x1`/`x2` may be a
# shared length-n vector (broadcast across columns) or a per-cell matrix; the
# `_bcast_index` mapping handles both. Faithful port of `reinterpolate_GPU_kernel!`.
function _reinterpolate_gpu_kernel!(y2, y1, x1, x2, ::Val{extrap}) where {extrap}
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = _kernel_ncols(y2)
    if iv <= Nv
        cidx = CartesianIndices(_tail_size(y2))[iv]
        y2v = @inbounds @view y2[:, _bcast_index(cidx, _tail_size(y2))]
        y1v = @inbounds @view y1[:, _bcast_index(cidx, _tail_size(y1))]
        x1v = @inbounds @view x1[:, _bcast_index(cidx, _tail_size(x1))]
        x2v = @inbounds @view x2[:, _bcast_index(cidx, _tail_size(x2))]
        reinterpolate!(y2v, y1v, x1v, x2v, Val(extrap))
    end
    return
end

# Trailing-size / column-count helpers usable inside a device kernel (avoid
# allocating tuples from `size(A)[2:end]`; build them statically from ndims).
@inline _tail_size(A::AbstractArray{T,N}) where {T,N} =
    ntuple(i -> size(A, i + 1), Val(N - 1))
@inline _kernel_ncols(A::AbstractArray{T,N}) where {T,N} =
    prod(ntuple(i -> size(A, i + 1), Val(N - 1)))

# launch `_reinterpolate_gpu_kernel!` over all leading-dim columns.
function _reinterpolate_gpu!(y2::CuArray, y1::CuArray, x1::CuArray, x2::CuArray,
                             ::Val{extrap}; threads::Int = 256) where {extrap}
    Nv = _ncols(y2)
    blocks = cld(Nv, threads)
    @cuda threads=threads blocks=blocks _reinterpolate_gpu_kernel!(y2, y1, x1, x2, Val(extrap))
    return y2
end

# Seam method: wealth-axis backward interp on the GPU. `x_for_y_in` is the
# source wgrid (1-D, shared), `x_for_y_out` the per-cell post-wealth (N-D). The
# wgrid arrives host-side (the layout is never moved to the device), so x-args
# are device-promoted here.
function _along_axis(y_out::CuArray{T,N}, y_in::CuArray{T,N},
                     x_for_y_in::AbstractArray, x_for_y_out::AbstractArray,
                     wdim::Val{WD}, op::ReinterpOp{extrap}) where {T, N, WD, extrap}
    yin_f  = _to_front(y_in,  wdim)
    yout_f = _to_front(y_out, wdim)
    x1_f   = _x_to_front(_as_device(x_for_y_in),  wdim, Val(N))
    x2_f   = _x_to_front(_as_device(x_for_y_out), wdim, Val(N))
    _reinterpolate_gpu!(yout_f, yin_f, x1_f, x2_f, Val(extrap))
    _from_front!(y_out, yout_f, wdim)
    return y_out
end

# Promote a host array to the device (identity if already a CuArray).
_as_device(x::CuArray) = x
_as_device(x::AbstractArray) = CuArray(collect(x))

# ============================================================ #
# WealthChange forward — convert_distribution_kernel! port     #
# ============================================================ #

# device kernel — one thread per column, running the scalar
# `convert_distribution!` mass-redistribution on its 1-D slice. Faithful port
# of `convert_distribution_kernel!`.
function _convert_distribution_gpu_kernel!(y2, y1, x1, x2, ::Val{interp}) where {interp}
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = _kernel_ncols(y2)
    if iv <= Nv
        cidx = CartesianIndices(_tail_size(y2))[iv]
        y2v = @inbounds @view y2[:, _bcast_index(cidx, _tail_size(y2))]
        y1v = @inbounds @view y1[:, _bcast_index(cidx, _tail_size(y1))]
        x1v = @inbounds @view x1[:, _bcast_index(cidx, _tail_size(x1))]
        x2v = @inbounds @view x2[:, _bcast_index(cidx, _tail_size(x2))]
        convert_distribution!(y2v, y1v, x1v, x2v, Val(interp))
    end
    return
end

# launch `_convert_distribution_gpu_kernel!` over all leading-dim columns.
function _convert_distribution_gpu!(y2::CuArray, y1::CuArray, x1::CuArray, x2::CuArray,
                                    ::Val{interp}; threads::Int = 256) where {interp}
    Nv = _ncols(y2)
    blocks = cld(Nv, threads)
    @cuda threads=threads blocks=blocks _convert_distribution_gpu_kernel!(y2, y1, x1, x2, Val(interp))
    return y2
end

# Seam method: wealth-axis forward mass-conversion on the GPU. Here
# `x_for_y_in` is the per-cell post-wealth (N-D) and `x_for_y_out` the
# destination wgrid (1-D, shared) — the forward call swaps the x roles.
function _along_axis(y_out::CuArray{T,N}, y_in::CuArray{T,N},
                     x_for_y_in::AbstractArray, x_for_y_out::AbstractArray,
                     wdim::Val{WD}, op::ConvertDistOp) where {T, N, WD}
    yin_f  = _to_front(y_in,  wdim)
    yout_f = _to_front(y_out, wdim)
    x1_f   = _x_to_front(_as_device(x_for_y_in),  wdim, Val(N))
    x2_f   = _x_to_front(_as_device(x_for_y_out), wdim, Val(N))
    _convert_distribution_gpu!(yout_f, yin_f, x1_f, x2_f, Val(:share))
    _from_front!(y_out, yout_f, wdim)
    return y_out
end

# An x-argument is either the shared 1-D wgrid (passes through unchanged: its
# single dim already plays the role of "wealth-first", and `_bcast_index`
# collapses the empty trailing dims) or a per-cell N-D array (permute its
# wealth axis to the front like the y arrays).
_x_to_front(x::CuArray{S,N}, wdim::Val, ::Val{N}) where {S, N} = _to_front(x, wdim)
_x_to_front(x::CuArray, ::Val, ::Val) = x

# ============================================================ #
# ConsumptionSavings backward — utility-table fill             #
# ============================================================ #
# The reward table `U` is materialised by the shared `fill_field!` (helper/field.jl): it builds
# each `(after, before)` face on the host (the device can't run the host utility closure) and
# `copyto!`s it into the device `parent(U)` — so no device-specific fill seam is needed.

# ============================================================ #
# ConsumptionSavings backward — k1_argmax_kernel! port         #
# ============================================================ #

# device kernel — one thread per column, running the monotone-savings
# iterative divide-and-conquer argmax of the reference `k1_argmax_kernel!`.
# Requires `ispow2(n-1)` (the reference's segment-halving). `u` is the
# (n_a, n_w) flow-utility slice (shared across columns), `V` the pre-discounted
# continuation `(n_a, n_cols)`, `Vp`/`pol` the per-column outputs. Infeasible
# actions carry `typemin` in `u`, so they are never selected.
function _k1_argmax_pow2_kernel!(Vp, pol, V, u)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = size(Vp, 2)
    if iv <= Nv
        T = eltype(Vp)
        n = size(pol, 1)
        @inbounds begin
            pol[1, iv] = 1
            Vp[1, iv]  = u[1, 1] + V[1, iv]

            best = typemin(T); ba = 1
            for j in 1:n
                val = u[j, n] + V[j, iv]
                if val > best
                    best = val; ba = j
                end
            end
            pol[n, iv] = ba
            Vp[n, iv]  = u[ba, n] + V[ba, iv]

            seg = (n - 1) ÷ 2
            while seg >= 1
                i = 1
                while i < n - 1
                    lb = pol[i, iv]
                    i += seg
                    ub = pol[i + seg, iv]
                    best = typemin(T); ba = lb
                    hi = min(ub, i)
                    for j in lb:hi
                        val = u[j, i] + V[j, iv]
                        if val > best
                            best = val; ba = j
                        end
                    end
                    pol[i, iv] = ba
                    Vp[i, iv]  = u[ba, i] + V[ba, iv]
                    i += seg
                end
                seg ÷= 2
            end
        end
    end
    return
end

# device kernel — sequential monotone walk (the `:sequential` walk and
# the general-`n_w` fallback when `n-1` is not a power of two). One thread per
# column; the per-`s` search resumes from `s-1`'s argmax. Skips infeasible
# (`typemin`) actions explicitly, matching the CPU `_ca_table_walk!(::Val{:seq})`.
function _k1_argmax_seq_kernel!(Vp, pol, V, u)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = size(Vp, 2)
    if iv <= Nv
        T = eltype(Vp)
        n_s = size(pol, 1)
        n_a = size(V, 1)
        prev_a = 1
        @inbounds for s in 1:n_s
            best_v = typemin(T); best_a = 0
            for a in prev_a:n_a
                uas = u[a, s]
                isfinite(uas) || continue
                v = uas + V[a, iv]
                if v > best_v
                    best_v = v; best_a = a
                end
            end
            if best_a == 0
                Vp[s, iv] = typemin(T); pol[s, iv] = 1
            else
                Vp[s, iv] = best_v; pol[s, iv] = best_a; prev_a = best_a
            end
        end
    end
    return
end

# launch the monotone argmax over all columns. `u` is (n_a, n_w),
# `V`/`Vp`/`pol` are (n_a/n_w, n_cols). Picks the iterative-D&C reference
# kernel when `ispow2(n_w-1)`, else the sequential-walk fallback.
function _k1_argmax_gpu!(Vp::CuArray, pol::CuArray, V::CuArray, u::CuArray;
                         threads::Int = 256)
    Nv = size(Vp, 2)
    blocks = cld(Nv, threads)
    n_w = size(pol, 1)
    if ispow2(n_w - 1)
        @cuda threads=threads blocks=blocks _k1_argmax_pow2_kernel!(Vp, pol, V, u)
    else
        @cuda threads=threads blocks=blocks _k1_argmax_seq_kernel!(Vp, pol, V, u)
    end
    return Vp
end

# ============================================================ #
# Discrete ArgmaxStage backward — :brute (max, +) kernel       #
# ============================================================ #

# device kernel — the discrete `:brute` `(max, +)` argmax over an UNORDERED
# operative axis. One thread per stratum column `iv`; for each origin `s` it scans
# every destination `a` and keeps `maxₐ u[a, s] + V[a, iv]`, writing the value to
# `Vp[s, iv]` and the maximiser to `pol[s, iv]`. Rectangular-aware: `u` is
# `(n_end, n_start)`, `V` is `(n_end, nc)`, and `Vp`/`pol` are `(n_start, nc)`
# (`n_start == n_end` square, or `n_start == 1` collapse). Bit-identical to the CPU
# `_ca_brute_smallaxis!`: same `typemin`/default-`1` init, same strict-`>` first-index
# tie-break, and `-Inf` reward cells are pruned (`isfinite(u) || continue`) — a `-Inf`
# payoff can never strictly beat the running best, so skipping it changes neither the
# value NOR the index. An all-infeasible column leaves `(typemin, 1)`, which the
# caller's `all(isfinite, V_start)` assertion then rejects, exactly as on the CPU.
function _brute_argmax_kernel!(Vp, pol, V, u)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = size(Vp, 2)
    if iv <= Nv
        T = eltype(Vp)
        n_start = size(Vp, 1)
        n_end   = size(V, 1)
        @inbounds for s in 1:n_start
            best_v = typemin(T); best_a = 1
            for a in 1:n_end
                uas = u[a, s]
                isfinite(uas) || continue
                v = uas + V[a, iv]
                if v > best_v
                    best_v = v; best_a = a
                end
            end
            Vp[s, iv] = best_v; pol[s, iv] = best_a
        end
    end
    return
end

# launch the `:brute` argmax over all stratum columns. `u` is `(n_end, n_start)`,
# `V` is `(n_end, nc)`, `Vp`/`pol` are `(n_start, nc)`.
function _brute_argmax_gpu!(Vp::CuArray, pol::CuArray, V::CuArray, u::CuArray;
                            threads::Int = 256)
    nc = size(Vp, 2)
    blocks = cld(nc, threads)
    @cuda threads=threads blocks=blocks _brute_argmax_kernel!(Vp, pol, V, u)
    return Vp
end

# Seam method: CS backward column walk on the GPU. `Uc` is the COMPACT reward face
# `(after, before, dep…)` (a plain `(n_w, n_w)` matrix in the dependence-free case); `βV` the
# pre-discounted continuation; `V_start`/`policy` the outputs. Permute the wealth axis to the
# front, reshape to matrices, run the kernel, permute the outputs back. The supermodularity
# `check` runs on a host copy of the small face before the launch.
function _ca_backward_columns!(::Val{WD}, mode::Val, U::HS.MatrixField{A},
                               odep_dims::NTuple{ND, Int}, βV::CuArray, V_start::CuArray,
                               policy::CuArray, n_w, dims, ::Val{N}, check::Bool) where {WD, ND, N, A<:CuArray}
    # Discrete `:brute` `(max, +)` argmax (unordered operative axis: buy/sell-home). Permute the
    # operative axis to the front, reshape to matrices, and run the per-stratum-column brute kernel.
    # Rectangular-aware: V_start/policy carry `n_start` on the operative axis, βV (= V_end) carries
    # `n_end`. The reward `U` is the compact `(n_end, n_start)` face (ND == 0). No supermodularity
    # `check` (the caller passes `false` for brute — `:brute` assumes no monotonicity).
    if mode isa Val{:brute}
        ND == 0 || error(
            "HouseholdStagesCUDAExt: GPU `:brute` argmax supports only a dep-free reward " *
            "(no extra dep axes in U); the reward has $ND. Use the CPU backward (see GPU_SURVEY.md).")
        n_end   = size(U.array, 1)
        n_start = size(U.array, 2)
        u_mat = reshape(U.array, n_end, n_start)       # compact (after, before) face — no permute
        βV_f  = _to_front(βV,      Val(WD))            # (n_end, …)   continuation, operative-front
        Vs_f  = _to_front(V_start, Val(WD))            # (n_start, …) value out
        pol_f = _to_front(policy,  Val(WD))            # (n_start, …) policy index out
        nc = _ncols(Vs_f)
        _brute_argmax_gpu!(reshape(Vs_f, n_start, nc), reshape(pol_f, n_start, nc),
                           reshape(βV_f, n_end, nc), u_mat)
        _from_front!(V_start, Vs_f, Val(WD))
        _from_front!(policy,  pol_f, Val(WD))
        return
    end
    # State-dependent flow utility (extra dep axes in U) needs the per-column slice path; not
    # yet ported to the device. The reference shares one (n_a, n_w) `u` across all columns.
    ND == 0 || error(
        "HouseholdStagesCUDAExt: ConsumptionSavings GPU backward currently supports " *
        "consumption-only utilities (no dep axes in U). The reward has $ND extra dep " *
        "axis/axes — use the CPU backward for this case (see GPU_SURVEY.md).")
    # Supermodularity guard runs on a host copy of the compact reward face (the post-migration
    # `_check_increasing_differences` reads a `MatrixField`, so rewrap the host array).
    check && HS._check_increasing_differences(
        HS.MatrixField(Array(U.array), U.operative_axis, U.operative_dim, U.dep_dims), odep_dims)

    u_mat = reshape(U.array, n_w, n_w)     # compact (dest, origin) face — already plain (no permute)
    βV_f  = _to_front(βV,      Val(WD))
    Vs_f  = _to_front(V_start, Val(WD))
    pol_f = _to_front(policy,  Val(WD))

    nc = _ncols(Vs_f)
    _k1_argmax_gpu!(reshape(Vs_f, n_w, nc), reshape(pol_f, n_w, nc), reshape(βV_f, n_w, nc), u_mat)

    _from_front!(V_start, Vs_f, Val(WD))
    _from_front!(policy,  pol_f, Val(WD))
    return
end

# ============================================================ #
# ConsumptionSavings forward — get_λ_postc_kernel! port        #
# ============================================================ #

# device kernel — one thread per column, scattering each source mass
# `Λ_start[a, col]` onto the policy destination `Λ_end[policy[a,col], col]`.
# Because all writes for a column happen on a single thread, the colliding
# `+=` needs no atomics (one thread owns the column). Faithful port of
# `get_λ_postc_kernel!`, generalised to one flattened column index.
function _cs_scatter_kernel!(Λ_end_m, Λ_start_m, pol_m, n_w)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nc = size(Λ_end_m, 2)
    if col <= Nc
        @inbounds for a in 1:n_w
            dst = pol_m[a, col]
            Λ_end_m[dst, col] += Λ_start_m[a, col]
        end
    end
    return
end

# Seam method: CS forward scatter on the GPU. Permute wealth to the front,
# reshape to (n_w, n_cols) matrices, zero the destination, and run the
# per-column scatter (each thread owns a column → no atomics needed).
function _cs_forward_scatter!(Λ_end::CuArray, Λ_start::CuArray,
                              policy::CuArray, ::Val{WD}) where {WD}
    # The scatter axis may GROW from source to destination: choice-collapse stages
    # (EndogenousExit/KernelChoice/Portfolio) feed a size-1 choice axis into a grown
    # destination. Reshape source/policy by their OWN WD size, the destination by its
    # own — the only axis that differs is WD, so the column count `nc` is shared.
    n_w_src = size(Λ_start, WD)
    n_w_dst = size(Λ_end,   WD)

    Λs_f  = _to_front(Λ_start, Val(WD))
    pol_f = _to_front(policy,  Val(WD))
    Λe_f  = _to_front(Λ_end,   Val(WD))
    fill!(Λe_f, zero(eltype(Λe_f)))

    nc    = _ncols(Λe_f)
    Λs_m  = reshape(Λs_f,  n_w_src, nc)
    pol_m = reshape(pol_f, n_w_src, nc)
    Λe_m  = reshape(Λe_f,  n_w_dst, nc)

    threads = 256
    blocks  = cld(nc, threads)
    @cuda threads=threads blocks=blocks _cs_scatter_kernel!(Λe_m, Λs_m, pol_m, n_w_src)

    _from_front!(Λ_end, Λe_f, Val(WD))
    return Λ_end
end

# ============================================================ #
# ScatterKernel backward — integer-destination value gather    #
# ============================================================ #
# The adjoint `Kᵀ` of the integer scatter: the discrete deterministic move's `backward!`
# (DiscreteMoveStage, and any other index-policy `ScatterKernel` backward). The destination index
# is deterministic — `ν(s) = setindex(s, destinations[s], WD)` along the operative axis WD — NOT an
# argmax over the axis (that gated `(max, +)` case is the `:brute` kernel above). Each read is
# independent (distinct cells read distinct, possibly colliding, sources — the transpose of the
# scatter's colliding writes), so the gather is purely a copy: bit-identical to the CPU
# `_gather_along!` by construction (no arithmetic, no tie-break — same index source).

# device kernel — one thread per leading-dim column. Each thread copies its whole operative
# slice: `V_in[s, col] = V_out[destinations[s, col], col]`, with the operative axis WD permuted to
# the front so it is dimension 1. `n_out` may differ from `n_in` (the operative axis can grow/shrink
# between in and out), but every other axis matches, so the column index `col` is shared. Faithful
# port of the CPU `_gather_along!` loop — a pure relocation copy, no reduction.
function _gather_along_kernel!(Vin_m, Vout_m, dst_m, n_in)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nc = size(Vin_m, 2)
    if col <= Nc
        @inbounds for s in 1:n_in
            Vin_m[s, col] = Vout_m[dst_m[s, col], col]
        end
    end
    return
end

# launch the integer-destination gather over all operative-front columns. `Vin_m`/`dst_m`
# are `(n_in, nc)`, `Vout_m` is `(n_out, nc)` (the operative axis may resize; nc is shared).
function _gather_along_gpu!(Vin_m::CuArray, Vout_m::CuArray, dst_m::CuArray; threads::Int = 256)
    nc = size(Vin_m, 2)
    blocks = cld(nc, threads)
    @cuda threads=threads blocks=blocks _gather_along_kernel!(Vin_m, Vout_m, dst_m, size(Vin_m, 1))
    return Vin_m
end

# seam method — `ScatterKernel` backward gather on the GPU, the device port of the CPU
# `_gather_along!` (fields/interpolations.jl). Permute the operative axis WD to the front (a no-op
# when WD == 1), reshape to `(n, nc)` matrices, run the per-column gather, permute the result back.
# `V_in` (output) and `destinations` share a shape; `V_out` (source) matches except along WD.
function _gather_along!(V_in::CuArray{T,N}, V_out::CuArray, destinations::CuArray,
                        ::Val{WD}) where {T, N, WD}
    Vin_f  = _to_front(V_in,        Val(WD))   # (n_in,  …) output, operative-front
    Vout_f = _to_front(V_out,       Val(WD))   # (n_out, …) source
    dst_f  = _to_front(destinations, Val(WD))  # (n_in,  …) per-cell destination index along WD

    n_in  = size(Vin_f,  1)
    n_out = size(Vout_f, 1)
    nc    = _ncols(Vin_f)
    _gather_along_gpu!(reshape(Vin_f,  n_in,  nc),
                       reshape(Vout_f, n_out, nc),
                       reshape(dst_f,  n_in,  nc))

    _from_front!(V_in, Vin_f, Val(WD))
    return V_in
end

# ============================================================ #
# ContinuousArgmaxStage backward — off-grid 1-D maximiser      #
# ============================================================ #
# The off-grid sibling of the discrete `:brute` argmax: per origin cell, a continuous 1-D
# maximisation of `reward(x, a') + interp(V_end, a')` over the operative interval — a coarse grid
# scan to bracket the optimum, then a 45-iteration golden-section refine, writing the FLOAT optimiser
# `a'*` into the `InterpKernel` position policy and the optimal value into `V_start`. Unlike `:brute`
# (which indexes a pre-materialised reward face `U`), the reward here is the user payoff CLOSURE
# evaluated at off-grid `a'` — it must run ON the device, so the payoff closure must be GPU-able
# (isbits captures: it may close over scalars but not host arrays).
#
# Two adaptations of the CPU `_caC_solve_cell` (src/stages/primitive/continuous_argmax.jl) are needed
# for the device, both behaviour-preserving:
#   * `env_dep` (whether the payoff takes an `env` kwarg) is a RUNTIME `Bool` on the CPU. The CPU's
#     `env_dep ? payoff(…; env) : payoff(…)` ternary is fine when only the live branch executes, but
#     GPUCompiler statically compiles BOTH branches — and the dead one is an invalid method for a
#     payoff that lacks (or requires) `env`. So `env_dep` is lifted to a compile-time `Val{ED}` here
#     (`_caC_payoff_dev`), eliding the dead branch. Arithmetic is otherwise byte-identical: the device
#     cell solver reuses the SAME `_interp1d` and `_golden_max` as the CPU (imported from `src`), so any
#     CPU/GPU difference is float reassociation in the same algorithm, never a different optimum.
#   * the per-column dep `combo` (axis ⇒ grid value) is built on the HOST into a `(ND, nc)` matrix
#     (`_caC_dep_vals`, indexed by front-permuted column) and read per thread, rather than rebuilt from
#     a Cartesian coordinate on device.

# Val-dispatched payoff evaluation — compile-time `env_dep` so the device compiles only the
# live branch (the CPU's runtime ternary would force GPUCompiler to compile the invalid dead branch).
# Otherwise identical to the CPU `_caC_payoff`.
@inline _caC_payoff_dev(payoff, ::Val{true},  origin, dest, combo, env) = payoff(origin, dest; combo..., env)
@inline _caC_payoff_dev(payoff, ::Val{false}, origin, dest, combo, env) = payoff(origin, dest; combo...)

# the per-column dep `combo` (axis ⇒ grid value) for front-permuted column `iv`, read from the
# host-precomputed `(ND, nc)` `dep_vals` matrix. The no-dep case takes the `Val{()}` method (combo is
# an empty NamedTuple; `dep_vals` is `nothing` and never touched).
@inline _caC_combo_dev(::Val{()}, dep_vals, iv) = NamedTuple()
@inline _caC_combo_dev(::Val{NS}, dep_vals, iv) where {NS} =
    NamedTuple{NS}(ntuple(k -> @inbounds(dep_vals[k, iv]), Val(length(NS))))

# device port of the CPU `_caC_solve_cell` — a coarse scan over the operative `grid` finds the
# best node (its neighbours bracket the optimum under unimodality), `_golden_max` refines inside that
# bracket, and a node guard keeps the result no worse than the best grid node. Step-for-step identical
# to the CPU solver (same scan order, same `_interp1d`/`_golden_max`, same 45 iters, same `>=` node
# guard); only the payoff call is `Val`-dispatched. Returns `(a'*, V_start)`.
@inline function _caC_solve_cell_dev(payoff, ved::Val, grid, vfib, origin, combo, env)
    n = length(grid)
    f = a -> _caC_payoff_dev(payoff, ved, origin, a, combo, env) + _interp1d(grid, vfib, a)
    best_v = _caC_payoff_dev(payoff, ved, origin, grid[1], combo, env) + vfib[1]
    bestk  = 1
    @inbounds for j in 2:n
        v = _caC_payoff_dev(payoff, ved, origin, grid[j], combo, env) + vfib[j]
        v > best_v && (best_v = v; bestk = j)
    end
    lo = grid[max(bestk - 1, 1)]
    hi = grid[min(bestk + 1, n)]
    astar, vstar = _golden_max(f, lo, hi)
    best_v >= vstar && return grid[bestk], best_v
    return astar, vstar
end

# device kernel — one thread per non-operative column `iv` (the operative axis is permuted to
# the front and reshaped to `(n, nc)`). Each thread builds its dep `combo` once, then walks every origin
# cell `i` of its column running the SAME coarse-scan + golden-section refine as the CPU column loop,
# writing the value into `Vs_m[i, iv]` and the float policy `a'*` into `pol_m[i, iv]`. Mirrors the CPU
# `_caC_backward_columns!` inner loop exactly.
function _caC_backward_kernel!(Vs_m, pol_m, Ve_m, grid, dep_vals,
                               ved::Val, vdeps::Val, payoff, env)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nc = size(Vs_m, 2)
    if iv <= nc
        n     = size(Vs_m, 1)
        vfib  = @inbounds @view Ve_m[:, iv]
        combo = _caC_combo_dev(vdeps, dep_vals, iv)
        @inbounds for i in 1:n
            astar, vstar = _caC_solve_cell_dev(payoff, ved, grid, vfib, grid[i], combo, env)
            pol_m[i, iv] = astar
            Vs_m[i, iv]  = vstar
        end
    end
    return
end

# build the `(ND, nc)` per-column dep-value matrix in front-permuted column order — the same
# order the kernel's reshaped `(n, nc)` columns enumerate. After permuting the operative axis WD to the
# front, trailing dim `t` is original dim `t < WD ? t : t + 1`; a dep on original dim `d` (never WD)
# sits at trailing position `d < WD ? d : d - 1`, and `dep_grids[k]` (host-side) supplies its value.
function _caC_dep_vals(deps::NTuple{ND, Symbol}, dep_dims::NTuple{ND, Int}, dep_grids,
                       dims::NTuple{Nn, Int}, ::Val{WD}) where {ND, Nn, WD}
    tail_sizes = ntuple(t -> dims[t < WD ? t : t + 1], Val(Nn - 1))
    nc    = prod(tail_sizes)
    cinds = CartesianIndices(tail_sizes)
    out   = Array{Float64}(undef, ND, nc)
    @inbounds for iv in 1:nc
        c = cinds[iv]
        for k in 1:ND
            t = dep_dims[k] < WD ? dep_dims[k] : dep_dims[k] - 1
            out[k, iv] = dep_grids[k][c[t]]
        end
    end
    return out
end

# concretely-typed launch barrier. The seam resolves `env_dep`/`deps` to compile-time `Val`s
# and a concrete `dep_vals` (a `CuArray` or `nothing`) BEFORE calling here, so the `@cuda` site sees no
# `Union` types — keeping each kernel specialisation's host IR small (the un-barriered union site bloats
# codegen). One thread per non-operative column.
function _caC_launch!(Vs_m::CuArray, pol_m::CuArray, Ve_m::CuArray, grid::CuArray, dep_vals,
                      ved::Val, vdeps::Val, payoff, env)
    nc      = size(Vs_m, 2)
    threads = 256
    blocks  = cld(nc, threads)
    @cuda threads=threads blocks=blocks _caC_backward_kernel!(Vs_m, pol_m, Ve_m, grid, dep_vals,
                                                              ved, vdeps, payoff, env)
    return
end

# seam method — `ContinuousArgmaxStage` backward on the GPU. Permute the operative axis WD to
# the front, reshape to `(n, nc)` matrices, launch one thread per column, permute the value + float
# policy outputs back. `grid` arrives device-resident (moved by `to_device`); `dep_grids` stays host
# (a plain `Tuple`, untouched by the mover) so the per-column dep matrix is precomputed on the host.
# `env_dep` is lifted to `Val{ED}` so the kernel compiles only the live payoff branch; the two `Val`
# choices feed the typed `_caC_launch!` barrier.
function _caC_backward_columns!(::Val{WD}, payoff, env_dep::Bool, grid::AbstractVector, deps, dep_dims,
                                dep_grids, V_end::CuArray{T, N}, V_start, policy, env, dims,
                                ::Val{N}) where {WD, T, N}
    n      = length(grid)
    g_grid = _as_device(grid)                      # already on-device when moved by to_device
    Ve_f   = _to_front(V_end,   Val(WD))
    Vs_f   = _to_front(V_start, Val(WD))
    pol_f  = _to_front(policy,  Val(WD))
    nc     = _ncols(Vs_f)
    Ve_m   = reshape(Ve_f,  n, nc)
    Vs_m   = reshape(Vs_f,  n, nc)
    pol_m  = reshape(pol_f, n, nc)

    if length(deps) == 0
        env_dep ? _caC_launch!(Vs_m, pol_m, Ve_m, g_grid, nothing, Val(true),  Val(()), payoff, env) :
                  _caC_launch!(Vs_m, pol_m, Ve_m, g_grid, nothing, Val(false), Val(()), payoff, env)
    else
        dep_vals = CuArray(_caC_dep_vals(deps, dep_dims, dep_grids, dims, Val(WD)))
        env_dep ? _caC_launch!(Vs_m, pol_m, Ve_m, g_grid, dep_vals, Val(true),  Val(deps), payoff, env) :
                  _caC_launch!(Vs_m, pol_m, Ve_m, g_grid, dep_vals, Val(false), Val(deps), payoff, env)
    end

    _from_front!(V_start, Vs_f, Val(WD))
    _from_front!(policy,  pol_f, Val(WD))
    return
end

# ============================================================ #
# SearchMatchingStage backward — fused effort argmax kernel    #
# ============================================================ #
# The unemployed value is a discrete max over an INTERNAL effort grid (never a state axis):
#   Vu_new(x) = maxₖ −cost[k] + pe[k]·Ve(x) + (1−pe[k])·Vu(x),
# with the maximiser index `k*` stored as the effort policy and `p = pe[k*]` cached for the forward
# replay; the employed value is the separation mix `Ve_new = (1−δ)·Ve + δ·Vu`. The CPU seam
# materialises the `(x…, n_eff)` `Q` tensor and `findmax`-reduces it (the scalar-index seam was the
# per-cell `p[ci] = job_finding(…)` cache loop); on the device we FUSE the reduction into one thread
# per non-labor cell, scanning the (small) effort grid on the fly — no `Q` traffic. The host effort
# closures are pre-evaluated by the src `backward!` into the length-`n_eff` `cost`/`pe` host vectors
# (moved to the device here), so the kernel runs pure arithmetic.
#
# Bit-identical to the CPU `findmax` argmax (a discrete max over a fixed grid, like the `:brute`
# kernel): the per-effort `q` is the SAME left-associated expression `(−cost[k] + pe[k]·Ve) +
# omp[k]·Vu` (with `omp = 1 − pe` host-precomputed to match the CPU `(1 − pe)`), and the strict-`>`
# running max keeps the FIRST maximiser, exactly as `findmax` does (first-index tie-break). The labor
# axis is permuted to the front and reshaped to a `(2, nc)` matrix (row 1 = unemployed, row 2 =
# employed); `policy`/`p` are the x-shaped outputs, already in column-major x-order (the same order
# the front-permuted columns enumerate), so they need no permute and are written in place.

# device kernel — one thread per non-labor cell `iv`. Reads the cell's `(Vu, Ve)` from the
# 2-row matrix, scans the effort grid tracking `maxₖ −cost[k] + pe[k]·Ve + omp[k]·Vu`, writes the
# unemployed value + employed separation mix back into the 2-row matrix, the maximiser index into
# `polv`, and `pe[k*]` into `pv`. Mirrors the CPU `findmax` argmax step-for-step (same expression,
# same first-index tie-break).
function _sam_argmax_kernel!(Vsm, polv, pv, Vom, cost_d, pe_d, omp_d, δ)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nc = size(Vsm, 2)
    if iv <= nc
        T = eltype(Vsm)
        n_eff = length(cost_d)
        @inbounds begin
            Vu = Vom[1, iv]
            Ve = Vom[2, iv]
            best_v = typemin(T); best_k = 1
            for k in 1:n_eff
                q = -cost_d[k] + pe_d[k] * Ve + omp_d[k] * Vu
                if q > best_v
                    best_v = q; best_k = k
                end
            end
            Vsm[1, iv] = best_v
            Vsm[2, iv] = (1 - δ) * Ve + δ * Vu
            polv[iv]   = best_k
            pv[iv]     = pe_d[best_k]
        end
    end
    return
end

# launch the fused effort argmax over all non-labor columns. `Vsm`/`Vom` are `(2, nc)`;
# `polv`/`pv` are length-`nc`; the effort vectors are length `n_eff`.
function _sam_argmax_gpu!(Vsm::CuArray, polv::CuArray, pv::CuArray, Vom::CuArray,
                          cost_d::CuArray, pe_d::CuArray, omp_d::CuArray, δ; threads::Int = 256)
    nc = size(Vsm, 2)
    blocks = cld(nc, threads)
    @cuda threads=threads blocks=blocks _sam_argmax_kernel!(Vsm, polv, pv, Vom, cost_d, pe_d, omp_d, δ)
    return Vsm
end

# seam method — `SearchMatchingStage` backward on the GPU. Permute the labor axis `LD` to the
# front and reshape `V_out`/`V_start` to `(2, nc)` matrices (no-op permute when LD == 1); `policy`/`p`
# are the x-shaped outputs, reshaped to length-`nc` vectors in the SAME column-major x-order, written
# in place (no permute needed). The host effort vectors are device-promoted; `omp = 1 .- pe` is formed
# on the host to match the CPU `(1 − pe)` exactly. `Q` (the CPU `findmax` scratch) is unused here.
function _sam_backward_columns!(::Val{LD}, V_start::CuArray, V_out::CuArray, policy::CuArray,
                                p::CuArray, Q, cost_vec, pe_vec, δ, ::Val{N}) where {LD, N}
    Vo_f = _to_front(V_out,   Val(LD))
    Vs_f = _to_front(V_start, Val(LD))
    nc   = _ncols(Vs_f)
    Vo_m = reshape(Vo_f, 2, nc)
    Vs_m = reshape(Vs_f, 2, nc)
    pol_v = reshape(policy, nc)
    p_v   = reshape(p,      nc)

    cost_d = _as_device(cost_vec)
    pe_d   = _as_device(pe_vec)
    omp_d  = _as_device(1 .- pe_vec)
    _sam_argmax_gpu!(Vs_m, pol_v, p_v, Vo_m, cost_d, pe_d, omp_d, δ)

    _from_front!(V_start, Vs_f, Val(LD))
    return
end

# ============================================================ #
# Streaming kernel-choice backward / forward — MPS + MeanVar   #
# ============================================================ #
# The streaming kernel-choice family (`ScaleVarianceStage` over `MPSKernel`, `MeanVarianceStage` over
# `MeanVarianceKernel`): the household picks a per-cell scalar θ from a grid along `adim`. The two
# stages share the SAME device kernels here, differing only in the landing map `landing(base, θ, k)`
# — an additive mean-preserving spread `base + θ·ξ_k` vs a multiplicative portfolio return
# `base·(R_f + θ·excess_k)` — dispatched by a compile-time `Val{:additive}`/`Val{:portfolio}` tag so
# GPUCompiler emits only the live arithmetic. The landing's `(rf, vec)` and the host `grid`/`weights`/
# `params`/`costs` vectors are device-promoted at the seam (small, like the WealthChange wgrid); the
# per-cell policy `θstar` rides to the device via the kernel mover (src/lifts/gpu.jl).
#
# Backward (the θ choice) FUSES the CPU's per-θ gather + cost + running-argmax loop into one thread
# per non-axis column (no per-θ `gθ` traffic, exactly like the `SearchMatching` fuse): each thread
# walks its column's bases, scans the θ grid computing the SAME clamped-interp gather
# `Σ_k weights[k]·((1−w)·V[j] + w·V[j+1])` and the SAME `acc − costs[idx]` objective, and keeps the
# running max with a strict-`>` (first-θ) tie-break. The θ choice is over a FIXED grid, so the seated
# policy `θstar` is bit-identical to the CPU; the value is value-exact-to-sub-ulp (the gather's
# multiply-adds contract to FMA on device, like `SearchMatching`). Forward (replay) is the per-cell
# Young-split scatter: one thread per column owns the column, so the colliding `+=` along the axis is
# sequential and needs no atomics (like the CS forward), and the source rows scatter in the SAME
# ascending order as the CPU `CartesianIndices` walk — value-exact-to-sub-ulp (FMA on the `m·(1−w)`).

# Val-dispatched landing map — the device twin of `AdditiveSpread`/`PortfolioReturn`. The
# compile-time family tag elides the dead branch (the structs hold host `Vector`s, so the kernel reads
# the device-promoted `vec`/`rf` instead). Byte-identical arithmetic to the CPU landing functors.
@inline _kc_landing(::Val{:additive},  base, θ, k, rf, vec) = base + θ * vec[k]
@inline _kc_landing(::Val{:portfolio}, base, θ, k, rf, vec) = base * (rf + θ * vec[k])

# Landing-family accessors: the compile-time tag, the (unused-for-additive) risk-free scalar, and the
# device-bound shock/excess vector.
_kc_family(::HS.AdditiveSpread)  = Val(:additive)
_kc_family(::HS.PortfolioReturn) = Val(:portfolio)
_kc_rf(::HS.AdditiveSpread)      = 0.0
_kc_rf(l::HS.PortfolioReturn)    = l.rf
_kc_vec(l::HS.AdditiveSpread)    = l.shocks
_kc_vec(l::HS.PortfolioReturn)   = l.excess

# per-(base, θ) clamped-interp gather along the axis column `col`, step-for-step identical to
# the CPU `_choice_gather!` inner loop (same clamp, same `searchsortedlast` bracket, same node/weight,
# same accumulation order over k). Returns `Σ_k weights[k]·((1−w)·col[j] + w·col[j+1])`.
@inline function _kc_gather_col(col, base, θ, grid, weights, fam, rf, vec, n, n_w)
    T  = eltype(col)
    g1 = grid[1]; gn = grid[n]
    acc = zero(T)
    @inbounds for k in 1:n_w
        t = clamp(_kc_landing(fam, base, θ, k, rf, vec), g1, gn)
        j = min(searchsortedlast(grid, t), n - 1)
        w = (t - grid[j]) / (grid[j+1] - grid[j])
        acc += weights[k] * ((1 - w) * col[j] + w * col[j+1])
    end
    return acc
end

# device kernel — one thread per non-axis column `iv`. Walks every base `i` of the column,
# scans the θ grid keeping `maxₚ (gather(base, params[p]) − costs[p])` with a strict-`>` (first-θ)
# tie-break, and writes the winning value into `gb_m[i, iv]` and the winning parameter into
# `θs_m[i, iv]`. The argmax init reads the existing `θs_m` (the CPU only overwrites `θstar` on a strict
# improvement, so an all-`-Inf` column keeps the prior policy — matched exactly). Bit-identical θ
# choice; value sub-ulp (FMA on the gather multiply-adds).
function _streaming_choice_backward_kernel!(gb_m, θs_m, Ve_m, params, costs, grid, weights,
                                            fam, rf, vec, n, n_p, n_w)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nc = size(gb_m, 2)
    if iv <= nc
        T   = eltype(gb_m)
        col = @inbounds @view Ve_m[:, iv]
        @inbounds for i in 1:n
            base  = grid[i]
            best  = typemin(T)
            θbest = θs_m[i, iv]                       # preserve the stale policy (CPU never resets θstar)
            for p in 1:n_p
                v = _kc_gather_col(col, base, params[p], grid, weights, fam, rf, vec, n, n_w) - costs[p]
                if v > best
                    best = v; θbest = params[p]
                end
            end
            gb_m[i, iv] = best
            θs_m[i, iv] = θbest
        end
    end
    return
end

# seam method — streaming kernel-choice backward on the GPU (the θ-choice leg of
# `ScaleVarianceStage`/`MeanVarianceStage`). Permute the choice axis `adim` to the front, reshape to
# `(n, nc)` matrices, fuse the per-θ gather + cost + argmax into one thread per column, permute the
# value + θ policy back. `gθ` (the CPU per-θ scratch) is unused — the fuse needs no θ-traffic. The
# host `params`/`costs`/`grid`/`weights` and the landing's shock/excess vector are device-promoted.
function _streaming_choice_backward!(gbest::CuArray, gθ, θstar::CuArray, V_end::CuArray,
                                     params, grid, weights, adim::Int, landing, costs)
    wd   = Val(adim)
    gb_f = _to_front(gbest,  wd)
    θs_f = _to_front(θstar,  wd)
    Ve_f = _to_front(V_end,  wd)
    n    = size(Ve_f, 1)
    nc   = _ncols(Ve_f)
    gb_m = reshape(gb_f, n, nc)
    θs_m = reshape(θs_f, n, nc)
    Ve_m = reshape(Ve_f, n, nc)

    grid_d    = _as_device(grid)
    weights_d = _as_device(weights)
    params_d  = _as_device(params)
    costs_d   = _as_device(costs)
    vec_d     = _as_device(_kc_vec(landing))
    fam       = _kc_family(landing)
    rf        = _kc_rf(landing)
    n_w       = length(weights); n_p = length(params)

    threads = 256
    blocks  = cld(nc, threads)
    @cuda threads=threads blocks=blocks _streaming_choice_backward_kernel!(
        gb_m, θs_m, Ve_m, params_d, costs_d, grid_d, weights_d, fam, rf, vec_d, n, n_p, n_w)

    _from_front!(gbest, gb_f, wd)
    _from_front!(θstar, θs_f, wd)
    return gbest
end

# device kernel — the seated-operator gather `out(x) = Σ_k w_k·V(landing(x, θstar(x), k))`
# (the kernel's own `backward!`, `Kᵀ` at the frozen per-cell policy). One thread per column walks its
# bases, reading the per-cell `θs_m`. Mirrors the CPU `_choice_gather!` with a per-cell θ.
function _choice_gather_kernel!(out_m, V_m, θs_m, grid, weights, fam, rf, vec, n, n_w)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nc = size(out_m, 2)
    if iv <= nc
        col = @inbounds @view V_m[:, iv]
        @inbounds for i in 1:n
            out_m[i, iv] = _kc_gather_col(col, grid[i], θs_m[i, iv], grid, weights, fam, rf, vec, n, n_w)
        end
    end
    return
end

# seam method — the seated kernel-choice `backward!` gather on the GPU (per-cell frozen θ).
# Mirrors the CPU `_choice_gather!`; permute `adim` to the front, reshape, one thread per column.
function _choice_gather!(out::CuArray, V::CuArray, θstar::CuArray, grid, weights, adim::Int, landing)
    wd    = Val(adim)
    out_f = _to_front(out,   wd)
    V_f   = _to_front(V,     wd)
    θs_f  = _to_front(θstar, wd)
    n     = size(out_f, 1)
    nc    = _ncols(out_f)
    grid_d = _as_device(grid); weights_d = _as_device(weights); vec_d = _as_device(_kc_vec(landing))
    fam = _kc_family(landing); rf = _kc_rf(landing); n_w = length(weights)

    threads = 256
    blocks  = cld(nc, threads)
    @cuda threads=threads blocks=blocks _choice_gather_kernel!(
        reshape(out_f, n, nc), reshape(V_f, n, nc), reshape(θs_f, n, nc),
        grid_d, weights_d, fam, rf, vec_d, n, n_w)

    _from_front!(out, out_f, wd)
    return out
end

# device kernel — the per-cell Young-split scatter (the forward replay). One thread owns a
# whole column, so the colliding `+=` along the axis is sequential (no atomics), and the source rows
# scatter in ascending order — the SAME order as the CPU `CartesianIndices` walk over a column. Each
# source mass is split onto the same bracketing nodes/weights the gather reads (exact transpose).
# Step-for-step identical to the CPU `_choice_scatter!`; value sub-ulp (FMA on `m·(1−w)`).
function _choice_scatter_kernel!(out_m, Λ_m, θs_m, grid, weights, fam, rf, vec, n, n_w)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nc = size(out_m, 2)
    if iv <= nc
        g1 = grid[1]; gn = grid[n]
        @inbounds for i in 1:n
            mass = Λ_m[i, iv]
            iszero(mass) && continue
            base = grid[i]; θ = θs_m[i, iv]
            for k in 1:n_w
                t = clamp(_kc_landing(fam, base, θ, k, rf, vec), g1, gn)
                j = min(searchsortedlast(grid, t), n - 1)
                w = (t - grid[j]) / (grid[j+1] - grid[j]); m = weights[k] * mass
                out_m[j, iv]     += m * (1 - w)
                out_m[j + 1, iv] += m * w
            end
        end
    end
    return
end

# seam method — streaming kernel-choice forward scatter on the GPU (the replay leg). Permute
# `adim` to the front, reshape to `(n, nc)`, zero the destination, and run the per-column Young-split
# (one thread per column ⇒ no atomics). `out` and `Λ` share a shape (the scatter lands on the same
# grid). The host `grid`/`weights` and the landing vector are device-promoted.
function _choice_scatter!(out::CuArray, Λ::CuArray, θstar::CuArray, grid, weights, adim::Int, landing)
    wd    = Val(adim)
    out_f = _to_front(out,   wd)
    Λ_f   = _to_front(Λ,     wd)
    θs_f  = _to_front(θstar, wd)
    fill!(out_f, zero(eltype(out_f)))
    n  = size(Λ_f, 1)
    nc = _ncols(Λ_f)

    grid_d = _as_device(grid); weights_d = _as_device(weights); vec_d = _as_device(_kc_vec(landing))
    fam = _kc_family(landing); rf = _kc_rf(landing); n_w = length(weights)

    threads = 256
    blocks  = cld(nc, threads)
    @cuda threads=threads blocks=blocks _choice_scatter_kernel!(
        reshape(out_f, n, nc), reshape(Λ_f, n, nc), reshape(θs_f, n, nc),
        grid_d, weights_d, fam, rf, vec_d, n, n_w)

    _from_front!(out, out_f, wd)
    return out
end

end # module HouseholdStagesCUDAExt
