###################################################################
# HouseholdStagesCUDAExt — GPU kernels for the two compute-heavy   #
# stages whose hot path is a scalar-index loop the device cannot   #
# broadcast: ConsumptionSavingsStage and WealthChangeStage.        #
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

module HouseholdStagesCUDAExt

using HouseholdStages
using CUDA

const HS = HouseholdStages

# Bring the seam functions and op structs into scope for method extension.
import HouseholdStages: _ca_backward_columns!, _cs_forward_scatter!,
                        _along_axis, reinterpolate!, convert_distribution!,
                        ReinterpOp, ConvertDistOp

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
# The payoff table `U` is materialised by the shared `fill_field!` (helper/field.jl): it builds
# each `(dest, origin)` face on the host (the device can't run the Symbol-celled scalar payoff)
# and `copyto!`s it into the device `parent(U)` — so no device-specific fill seam is needed.

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
# (`typemin`) actions explicitly, matching the CPU `k1_argmax_monotone!`.
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

# Seam method: CS backward column walk on the GPU. `Uc` is the COMPACT payoff face
# `(dest, origin, dep…)` (a plain `(n_w, n_w)` matrix in the dependence-free case); `βV` the
# pre-discounted continuation; `V_start`/`policy` the outputs. Permute the wealth axis to the
# front, reshape to matrices, run the kernel, permute the outputs back. The supermodularity
# `check` runs on a host copy of the small face before the launch.
function _ca_backward_columns!(::Val{WD}, mode::Val, Uc::CuArray, odep_dims::NTuple{ND, Int},
                               βV::CuArray, V_start::CuArray, policy::CuArray, n_w, dims,
                               ::Val{N}, check::Bool) where {WD, ND, N}
    # State-dependent flow utility (extra dep axes in U) needs the per-column slice path; not
    # yet ported to the device. The reference shares one (n_a, n_w) `u` across all columns.
    ND == 0 || error(
        "HouseholdStagesCUDAExt: ConsumptionSavings GPU backward currently supports " *
        "consumption-only utilities (no dep axes in U). The payoff has $ND extra dep " *
        "axis/axes — use the CPU backward for this case (see GPU_SURVEY.md).")
    check && HS._check_increasing_differences(Array(Uc), Val(0))

    u_mat = reshape(Uc, n_w, n_w)          # compact (dest, origin) face — already plain (no permute)
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
    N   = ndims(Λ_end)
    n_w = size(Λ_end, WD)

    Λs_f  = _to_front(Λ_start, Val(WD))
    pol_f = _to_front(policy,  Val(WD))
    Λe_f  = _to_front(Λ_end,   Val(WD))
    fill!(Λe_f, zero(eltype(Λe_f)))

    nc    = _ncols(Λe_f)
    Λs_m  = reshape(Λs_f,  n_w, nc)
    pol_m = reshape(pol_f, n_w, nc)
    Λe_m  = reshape(Λe_f,  n_w, nc)

    threads = 256
    blocks  = cld(nc, threads)
    @cuda threads=threads blocks=blocks _cs_scatter_kernel!(Λe_m, Λs_m, pol_m, n_w)

    _from_front!(Λ_end, Λe_f, Val(WD))
    return Λ_end
end

end # module HouseholdStagesCUDAExt
