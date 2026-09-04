# Stratified payload slicing #
#============================#
# A stratified operation runs one fiber op independently at every *stratum* — every combination of
# coordinates on the layout's axes other than the operative one. Each argument is a payload, and its
# type says how it slices:
#
#   • a scalar, `nothing`, or anything of lower rank than the layout passes through whole;
#   • a full-rank array has the operative dim coloned and the rest indexed at the stratum;
#   • a `ScalarField` slices as its `bshape` says;
#   • a `MatrixField`, or the `FiberFace` a GPU kernel receives in its place, slices as its
#     `dep_dims` say and hands the op a whole `(n_out, n_in)` matrix.

"""
The slice of `payload` at stratum `c`, a full-rank `CartesianIndex` whose coordinate on the operative
axis `AD` is ignored. A shared payload passes through whole.
"""
_slice(payload, ::CartesianIndex, ::Val) = payload

function _slice(A::AbstractArray{T,N}, c::CartesianIndex{N}, ::Val{AD}) where {T, N, AD}
    return view(A, ntuple(d -> d == AD ? Colon() : _project(c[d], size(A, d)), Val(N))...)
end

"A `ScalarField` slices through the broadcastable form its `bshape` describes."
function _slice(sf::ScalarField, c::CartesianIndex, adim::Val{AD}) where {AD}
    b = scalar_broadcastable(sf)
    #TODO A field constant along the operative axis could reach the op either as a scalar or as a
    #     one-element fiber. No fiber op accepts a one-element fiber, so this refuses rather than
    #     choose.
    @assert !(b isa AbstractArray) || size(b, AD) != 1
    return _slice(b, c, adim)
end

function _slice(f::MatrixField{A, NTuple{ND, Int}}, c::CartesianIndex, ::Val) where {A, ND}
    return view(f.array, :, :, ntuple(k -> c[f.dep_dims[k]], Val(ND))...)
end

"""
The compact `(n_out, n_in, dep…)` array a matrix field stores, plus the layout positions of its
trailing dep dims: isbits, and slices exactly as a `MatrixField` does.
"""
struct FiberFace{A, D}
    array    :: A
    dep_dims :: D
end

FiberFace(f::MatrixField) = FiberFace(f.array, f.dep_dims)

function _slice(f::FiberFace{A, NTuple{ND, Int}}, c::CartesianIndex, ::Val) where {A, ND}
    return view(f.array, :, :, ntuple(k -> c[f.dep_dims[k]], Val(ND))...)
end

_check_payload(f::FiberFace, dims::NTuple, ::Int) =
    @assert all(k -> size(f.array, 2 + k) == dims[f.dep_dims[k]], eachindex(f.dep_dims))

"A stratum coordinate projected onto one payload axis; a size-1 axis is read at 1."
_project(i, n) = ifelse(n == 1, 1, i)

"The strata of a layout of size `dims`: every axis at full extent but the operative one, held at 1."
_strata(dims::NTuple{N,Int}, ::Val{AD}) where {N, AD} =
    CartesianIndices(ntuple(d -> d == AD ? 1 : dims[d], Val(N)))

"Check every payload's conformance to a layout of size `dims`, once, before the stratum loop."
check_payloads(dims::NTuple{N,Int}, adim::Int, payloads...) where {N} =
    (foreach(p -> _check_payload(p, dims, adim), payloads); nothing)

_check_payload(_, ::NTuple, ::Int) = nothing

function _check_payload(A::AbstractArray, dims::NTuple{N,Int}, adim::Int) where {N}
    @assert ndims(A) == N || ndims(A) == 1
    if ndims(A) == N
        @assert all(d -> d == adim || size(A, d) == dims[d] || size(A, d) == 1, 1:N)
    end
    return
end

_check_payload(sf::ScalarField, dims::NTuple, adim::Int) =
    _check_payload(scalar_broadcastable(sf), dims, adim)

_check_payload(f::MatrixField, dims::NTuple, ::Int) =
    @assert all(k -> size(f.array, 2 + k) == dims[f.dep_dims[k]], eachindex(f.dep_dims))

# The ops #
#---------#

"Supertype of the fiber ops that may run on either CPU or GPU: scalar-indexed loops over the fiber views, with no allocation, no BLAS call and no host-only intrinsic."
abstract type AbstractFiberOp end

# The driver #
#------------#

"The operative axis as a `Val`."
_val(d::Val) = d
_val(d::Integer) = Val(Int(d))

"""
Apply the fiber op `f` at every stratum of `payloads`, with `dims` the operative axis as a `Val` or
an `Integer`. The first payload is the one written to, and is returned.
"""
stratified!(f, payloads...; dims) = _stratified!(f, _val(dims), payloads...)

function _stratified!(f, adim::Val{AD}, payloads::Vararg{Any,K}) where {AD, K}
    out  = first(payloads)
    dims = size(out)
    check_payloads(dims, AD, payloads...)
    for c in _strata(dims, adim)
        f(map(p -> _slice(p, c, adim), payloads)...)
    end
    return out
end
