# Dense kernel — the `mul` operator over a MatrixField #
#=====================================================#
# `DenseKernel` owns a `MatrixField` (its compact `(n_out, n_in, dep…)` data + operative-axis/dep
# metadata, fields/matrix_field.jl) and is the `mul` operator: `forward! = K·`, `backward! = Kᵀ·`.
# Generically it routes through the field driver `stratified_apply!` (covariant / contravariant `mul`).
# For a contiguous-compact backing it takes a batched-multiply fast path — one `batched_mul!` (CUBLAS
# on device) over the dep batch, operative axis gathered first. This is the one sanctioned
# backing-specific site (end-goal §8.1, §15.3): assumed of the kernel, never of the `MatrixField`.
# The gather perm + work-buffers (the contraction plan) are derived from the field metadata in
# `kernel_scratch`. Rectangular n_out ≠ n_in is first-class: square = transition, a row (n_out = 1)
# forgets, a column (n_in = 1) introduces.

using LinearAlgebra: transpose, mul!
using NNlib: batched_mul!, batched_transpose

"""
The `mul` kernel: owns a `MatrixField` and applies it as a stratified matrix multiply,
`forward! = K·` (covariant) and `backward! = Kᵀ·` (contravariant).
"""
struct DenseKernel{F}
    field :: F        # a MatrixField (fields/matrix_field.jl)
end

"The compact backing array of a `DenseKernel`'s owned `MatrixField`."
Base.parent(k::DenseKernel) = k.field.array

# Source-driven front door + fill. Lives here, not in the field layer, so the dependency runs
# fields → kernels (`fields/` never references `DenseKernel`).

"""
Source-driven front door: build a `MatrixField` (fields/matrix_field.jl) and wrap it in a
`DenseKernel`. A stage that only *reads* the matrix (the argmax reward) calls `matrix_field` directly
instead, never wrapping it as an operator.
"""
dense_kernel(::Type{T}, layout::GriddedLayout, axis::Symbol, source) where {T} =
    DenseKernel(matrix_field(T, layout, axis, source))

"Materialise a `DenseKernel`'s owned `MatrixField` from `source` (delegates to the field fill)."
fill_field!(k::DenseKernel, source, layout::GriddedLayout, axis::Symbol, env) =
    (fill_field!(k.field, source, layout, axis, env); k)

"The src-gather perm bringing `src`'s dims into `(axis, nondep…, dep…)` order, plus the nondep count."
function _gather_perm(adim::Integer, deps::NTuple{D, Int}, ::Val{N}) where {D, N}
    a = Int(adim)
    nondep = Int[p for p in 1:N if p != a && !(p in deps)]
    return ((a, nondep..., deps...), length(nondep))
end

# The in-place per-slice `mul` closure for the generic driver: `dest_slice = mat · src_slice`,
# `mat` being the fiber or its transpose (the driver picks via `mode`).
_mul_slice!(dest, mat, src) = mul!(dest, mat, src)

# A contiguous-compact backing (`Array` / `CuArray` / any `DenseArray`) reshapes to the
# `(n_out, n_in, n_dep)` batch for free, so the batched-mul fast path applies; anything else
# (a lazy / structured / non-contiguous backing) degrades to the generic stratified driver.
_fast_dense(::DenseArray) = true
_fast_dense(::AbstractArray) = false

"""
Dense apply: `dest = K·src` (`transpose = false`) or `Kᵀ·src` (`true`). Takes the batched-mul fast
path (`_dense_contract!`) for a contiguous-compact field backing; otherwise the generic
`stratified_apply!` driver (the reference semantics, any backing).
"""
function _dense_apply!(dest, k::DenseKernel, src, scratch, transpose::Bool)
    A = k.field.array
    if _fast_dense(A)
        return _dense_contract!(dest, src, A, scratch.gather_perm,
                                scratch.gather_in, scratch.gather_out, transpose)
    end
    return stratified_apply!(dest, _mul_slice!, k.field, src;
                             mode = transpose ? :contravariant : :covariant)
end

forward!(dest, k::DenseKernel, src; scratch)  = _dense_apply!(dest, k, src, scratch, false)
backward!(dest, k::DenseKernel, src; scratch) = _dense_apply!(dest, k, src, scratch, true)

"""
The batched-mul fast path: contract the compact fiber array `A` `(n_out, n_in, dep…)` along the
operative axis of `src` into `dest`, varying over the deps. `A` reshapes to the `(n_out, n_in, n_dep)`
batch for free (contiguous); `src` is gathered into `(axis, nondep…, dep…)` by the precomputed `perm`.
The applied operator is `A` (forward) / `Aᵀ` (`transpose`, backward). A no-permute fast path skips the
gather when `perm` is the identity (the operative axis is already first).
"""
function _dense_contract!(dest, src, A, perm, gathered_in, gathered_out, transpose::Bool)
    n_out = size(A, 1)
    n_in  = size(A, 2)
    n_dep = length(A) ÷ (n_in * n_out)
    n_src = transpose ? n_out : n_in
    n_dst = transpose ? n_in  : n_out
    n_other = length(src) ÷ (n_src * n_dep)
    if perm == ntuple(identity, ndims(src))           # axis already first: reshape, no gather
        in_mat  = reshape(src,  n_src, n_other, n_dep)
        out_mat = reshape(dest, n_dst, n_other, n_dep)
        _batched_matmul!(out_mat, A, in_mat, n_dep, transpose)   # one batched_mul! over the dep batch
        return dest
    end
    # General case: gather src into axis-first order, contract, scatter back. The identity
    # fast path above is kept deliberately — routing it through here would add two full-array
    # permute-copies (gather + scatter) on the common axis-first contraction.
    src_g  = permutedims!(reshape(view(gathered_in, 1:length(src)),
                                  _gathered_shape(src, perm)), src, perm)
    in_mat = reshape(src_g, n_src, n_other, n_dep)
    out_g  = reshape(view(gathered_out, 1:(n_dst * n_other * n_dep)), n_dst, n_other, n_dep)
    _batched_matmul!(out_g, A, in_mat, n_dep, transpose)
    out_shape = Base.setindex(ntuple(i -> size(dest, perm[i]), Val(ndims(dest))), n_dst, 1)
    permutedims!(dest, reshape(out_g, out_shape), invperm(perm))
    return dest
end

# kernel_scratch — the gather plan, derived from the kernel's field metadata #
#---------------------------------------------------------------------------#
# The operative-axis and dep positions are explicit on the owned `MatrixField`, so the gather
# perm / work-buffers follow directly — no reverse-engineering from a permuted view.

"""
The gather scratch (src-gather perm + two work-buffers) a `DenseKernel` contraction needs, derived
from its owned field's metadata.
"""
function kernel_scratch(k::DenseKernel, layout::GriddedLayout, ::Type{T}) where {T}
    f           = k.field
    gperm, _    = _gather_perm(f.operative_dim, f.dep_dims, Val(length(layout)))
    n_out, n_in = size(f.array, 1), size(f.array, 2)
    # Both verbs reuse these buffers, so size them for the larger of input/output — a
    # rectangular kernel (forget n_out<n_in, introduce/crosswalk n_out>n_in) resizes the
    # contracted axis from n_in to n_out, scaling the total by n_out/n_in.
    insz        = prod(layout_size(layout))
    gsz         = max(insz, insz ÷ n_in * n_out)
    return (gather_perm = gperm, gather_in = zeros(T, gsz), gather_out = zeros(T, gsz))
end

# Shared dense-contraction helpers #
#----------------------------------#

"""
The shared per-dep-batch matmul: contract fiber batch `M` `(n_out, n_in, n_dep)`
against `in_mat` into `out_mat`; `transpose` applies `Mᵀ`. One `batched_mul!`
(CUBLAS on device) over the batch — a singleton dep is just a 1-batch call.
"""
function _batched_matmul!(out_mat, M, in_mat, n_dep::Int, transpose::Bool)
    Mmats  = reshape(M, size(M, 1), size(M, 2), n_dep)
    Mbatch = transpose ? batched_transpose(Mmats) : Mmats
    batched_mul!(out_mat, Mbatch, in_mat)
    return out_mat
end

"Shape of `A` permuted by `perm` (a tuple), as a tuple — for sizing gathered buffers."
_gathered_shape(A, perm) = ntuple(i -> size(A, perm[i]), Val(ndims(A)))
