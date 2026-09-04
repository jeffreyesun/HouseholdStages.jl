# Dense kernel — a MatrixField applied as a matrix multiply #
#===========================================================#
# `DenseKernel` owns a `MatrixField` and applies it: `forward! = K·`, `backward! = Kᵀ·`. A contiguous
# backing takes the batched-multiply fast path; any other backing routes through `stratified_apply!`.
# `n_out ≠ n_in` is allowed: a row (`n_out = 1`) forgets the axis, a column (`n_in = 1`) introduces it.

using LinearAlgebra: transpose, mul!
using NNlib: batched_mul!, batched_transpose

"A `MatrixField` applied as a stratified matrix multiply: `forward! = K·`, `backward! = Kᵀ·`."
struct DenseKernel{F}
    field :: F        # a MatrixField
end

"The compact backing array of a `DenseKernel`'s owned `MatrixField`."
Base.parent(k::DenseKernel) = k.field.array

"Build a `MatrixField` from `source` and wrap it as an operator."
dense_kernel(::Type{T}, start_layout::GriddedLayout, end_layout::GriddedLayout, axis::Symbol, source) where {T} =
    DenseKernel(matrix_field(T, start_layout, end_layout, axis, source))

"Materialise a `DenseKernel`'s owned `MatrixField` from `source`."
fill_field!(k::DenseKernel, source, dep_layout::GriddedLayout, axis::Symbol, env) =
    (fill_field!(k.field, source, dep_layout, axis, env); k)

"The src-gather perm bringing `src`'s dims into `(axis, nondep…, dep…)` order, plus the nondep count."
function _gather_perm(adim::Integer, deps::NTuple{D, Int}, ::Val{N}) where {D, N}
    a = Int(adim)
    nondep = Int[p for p in 1:N if p != a && !(p in deps)]
    return ((a, nondep..., deps...), length(nondep))
end

# The per-slice mul: `dest_slice = mat · src_slice`.
_mul_slice!(dest, mat, src) = mul!(dest, mat, src)

# Whether the backing reshapes to the `(n_out, n_in, n_dep)` batch for free.
_fast_dense(::DenseArray) = true
_fast_dense(::AbstractArray) = false

"Dense apply: `dest = K·src`, or `Kᵀ·src` when `transpose`."
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
Contract the compact fiber array `A` `(n_out, n_in, dep…)` — or `Aᵀ` under `transpose` — along the
operative axis of `src` into `dest`. `perm` gathers `src` into `(axis, nondep…, dep…)` order.
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
        _batched_matmul!(out_mat, A, in_mat, n_dep, transpose)
        return dest
    end
    # General case: gather src axis-first, contract, scatter back.
    src_g  = permutedims!(reshape(view(gathered_in, 1:length(src)),
                                  _gathered_shape(src, perm)), src, perm)
    in_mat = reshape(src_g, n_src, n_other, n_dep)
    out_g  = reshape(view(gathered_out, 1:(n_dst * n_other * n_dep)), n_dst, n_other, n_dep)
    _batched_matmul!(out_g, A, in_mat, n_dep, transpose)
    out_shape = Base.setindex(ntuple(i -> size(dest, perm[i]), Val(ndims(dest))), n_dst, 1)
    permutedims!(dest, reshape(out_g, out_shape), invperm(perm))
    return dest
end

# kernel_scratch — the gather plan #
#---------------------------------#

"The scratch a `DenseKernel` contraction needs: the src-gather perm and two work-buffers."
function kernel_scratch(k::DenseKernel, start_layout::GriddedLayout, end_layout::GriddedLayout,
                        ::Type{T}) where {T}
    f        = k.field
    gperm, _ = _gather_perm(f.operative_dim, f.dep_dims, Val(length(start_layout)))
    gsz = max(prod(layout_size(start_layout)), prod(layout_size(end_layout)))
    return (gather_perm = gperm, gather_in = zeros(T, gsz), gather_out = zeros(T, gsz))
end

# Shared dense-contraction helpers #
#----------------------------------#

"The leading `(rows, cols)` panel of an array whose remaining dims are singletons."
_panel(X) = reshape(X, size(X, 1), size(X, 2))

"Contract the single fiber `M` against `in_mat` into `out_mat`, applying `Mᵀ` under `transposed`."
function _single_matmul!(out_mat, M, in_mat, transposed::Bool)
    Mmat = _panel(M)
    mul!(_panel(out_mat), transposed ? transpose(Mmat) : Mmat, _panel(in_mat))
    return out_mat
end

"Contract fiber batch `M` `(n_out, n_in, n_dep)` against `in_mat` into `out_mat`, applying `Mᵀ` under `transpose`."
function _batched_matmul!(out_mat, M, in_mat, n_dep::Int, transpose::Bool)
    n_dep == 1 && return _single_matmul!(out_mat, M, in_mat, transpose)
    Mmats  = reshape(M, size(M, 1), size(M, 2), n_dep)
    Mbatch = transpose ? batched_transpose(Mmats) : Mmats
    batched_mul!(out_mat, Mbatch, in_mat)
    return out_mat
end

"Shape of `A` permuted by `perm` (a tuple), as a tuple — for sizing gathered buffers."
_gathered_shape(A, perm) = ntuple(i -> size(A, perm[i]), Val(ndims(A)))
