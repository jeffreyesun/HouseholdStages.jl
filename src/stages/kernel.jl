# Transition kernels — the linear operator a stage applies.
#
# A transition is applied via two in-place verbs — never `mul!`, because the stage
# backward is affine (`r + KᵀV`) and discount's two directions are not an adjoint pair.
#
#     forward!(dest, kernel, src, …)    # dest = K · src    (pushforward of Λ)
#     backward!(dest, kernel, src, …)   # dest = Kᵀ · src   (pullback of V)
#
# The kernel object IS the transition's data — nothing else. The dense case is a
# `PermutedDimsArray` of the self-describing shape (`notes/tensor_contraction_description.txt`)
# over a compact parent stored in `batched_mul` order; the structured cases
# (`SingleDestinationKernel`, `LogitChoiceKernel` — in their stage files) are data-only
# structs. The contraction's
# instructions (the contracted axis, the deps → a src-gather perm) are resolved once by
# the stage and live in its scratch, with the gather work-buffers. The two lightweight
# transitions `I` (identity) and `BackwardScale` (discount, the asymmetric outlier whose
# two directions are NOT a transpose pair) carry no instructions.
#
# Orientation convention (load-bearing — the duality gate): the stored kernel is **K**,
# the forward operator, so `forward! = K·src` and `backward! = Kᵀ·src`. A Markov-style
# transition that applies its `S` on backward (`S = Kᵀ`) stores `Sᵀ`, reproducing the
# duality `⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩`.

using LinearAlgebra: UniformScaling, I, transpose
using NNlib: batched_mul!, batched_transpose

# Identity & discount — the instruction-free transitions #
#-------------------------------------------------------#

# Identity transition — UniformScaling: no storage, both verbs are copies. The (ignored)
# `scratch` kwarg lets `I` share the generic stage-adjoint path uniformly with the dense kernels.
forward!(dest, ::UniformScaling, src; scratch=nothing)  = copyto!(dest, src)
backward!(dest, ::UniformScaling, src; scratch=nothing) = copyto!(dest, src)

# BackwardScale — discount (the asymmetric outlier) #
#--------------------------------------------------#
# forward! = copy (the *undiscounted* population is pushed forward); backward! = β·src.
# The two directions are deliberately not an adjoint pair (the discount asymmetry,
# MATH_CONTEXT §1).

"""
The discount transition: `forward! = copy`, `backward! = β·`. `β` is a scalar or a
`Base.RefValue` (an env-resolved β rewritten each backward). Not a transpose pair.
"""
struct BackwardScale{B}
    β :: B
end

"Read the scale, unwrapping a `Ref`-backed (env-rewritten) β."
_scale(s::BackwardScale)                  = s.β
_scale(s::BackwardScale{<:Base.RefValue}) = s.β[]

forward!(dest, ::BackwardScale, src; scratch=nothing)   = copyto!(dest, src)
backward!(dest, s::BackwardScale, src; scratch=nothing) = (β = _scale(s); @. dest = β * src; dest)

# Dense kernel — a self-describing PermutedDimsArray over a compact parent #
#------------------------------------------------------------------------#
# The kernel field PRESENTS the self-describing shape (rank N+1: dim 1 = n_out, dims 2…N+1
# axis-aligned with V's axes, the contracted axis carrying n_in, deps their sizes, non-deps
# singleton — `notes/tensor_contraction_description.txt`). Its PARENT stores the bytes in
# `batched_mul` order `(n_out, n_in, deps…, singletons)`, so `reshape(parent(M), n_out,
# n_in, n_dep)` is free — no per-call permute, math identical to the compact kernel.
# Rectangular n_out ≠ n_in is first-class: square = transition, a row (n_out=1) forgets,
# a column (n_in=1) introduces. The contraction's instructions (the src-gather perm) are
# resolved once by the stage and live in its scratch, with the gather work-buffers.

"The presentation perm: the parent dim feeding each self-describing position (dim 1 = n_out ← parent 1; the contracted axis ← n_in at parent 2; dep `j` ← parent 2+j; non-deps ← the trailing singletons)."
function _present_perm(adim::Int, deps::NTuple{D, Int}, ::Val{N}) where {D, N}
    perm = Vector{Int}(undef, N + 1)
    perm[1] = 1
    next_singleton = 2 + D
    for k in 1:N
        if k == adim
            perm[k + 1] = 2
        else
            j = findfirst(==(k), deps)
            if j === nothing
                next_singleton += 1
                perm[k + 1] = next_singleton
            else
                perm[k + 1] = 2 + j
            end
        end
    end
    return Tuple(perm)
end

"""
Buffer-driven core: wrap a compact fiber buffer `M` `(n_out, n_in, dep_sizes…)` as the dense
self-describing kernel — pad to layout rank with trailing singletons, then present it as a
`PermutedDimsArray` (`_present_perm` maps each self-describing position to its parent dim).
`axis` is the contracted axis, `deps` its dep axes (layout-named). The parent aliases `M` (a
reshape), so filling `M` in place updates the kernel. (`ForgetfulSum` calls this directly with
its prebuilt ones-row; the source-driven `dense_kernel` below is the front door for the rest.)
"""
function _dense_kernel(M::AbstractArray, layout::GriddedLayout, axis::Symbol, deps)
    N         = length(layout)
    dep_sizes = size(M)[3:end]
    parent    = reshape(M, size(M, 1), size(M, 2), dep_sizes...,
                        ntuple(_ -> 1, N - 1 - length(dep_sizes))...)
    depdims   = map(a -> axis_position(layout, a), Tuple(deps))
    return PermutedDimsArray(parent, _present_perm(axis_position(layout, axis), depdims, Val(N)))
end
# The source-driven `dense_kernel(T, layout, axis, source)` front door lives in helper/field.jl
# (it needs `field_deps`/`allocate_field`/`MappedField`, included after this file).

"The src-gather perm bringing `src`'s dims into `(axis, nondep…, dep…)` order, plus the nondep count. Built once by the stage (stored in scratch)."
function _gather_perm(adim::Integer, deps::NTuple{D, Int}, ::Val{N}) where {D, N}
    a = Int(adim)
    nondep = Int[p for p in 1:N if p != a && !(p in deps)]
    return ((a, nondep..., deps...), length(nondep))
end

"""
The dense contraction: contract `M` along its contracted axis of `src` into `dest`,
varying over the deps. `parent(M)` is the compact `(n_out, n_in, n_dep)` batch (a free
reshape — `parent` of a plain `Array` is itself); `src` is gathered into `(axis, nondep…,
dep…)` by the precomputed `perm`. The applied operator is `M` (forward) / `Mᵀ`
(`transpose`, backward). A no-permute fast path skips the gather when `perm` is the identity.
"""
function _dense_contract!(dest, src, M, perm, gathered_in, gathered_out, transpose::Bool)
    Mp    = parent(M)
    n_out = size(Mp, 1)
    n_in  = size(Mp, 2)
    n_dep = length(Mp) ÷ (n_in * n_out)
    n_src = transpose ? n_out : n_in
    n_dst = transpose ? n_in  : n_out
    n_other = length(src) ÷ (n_src * n_dep)
    if perm == ntuple(identity, ndims(src))           # axis already first: reshape, no gather
        in_mat  = reshape(src,  n_src, n_other, n_dep)
        out_mat = reshape(dest, n_dst, n_other, n_dep)
        _batched_matmul!(out_mat, Mp, in_mat, n_dep, transpose)   # one batched_mul! over the dep batch
        return dest
    end
    # General case: gather src into axis-first order, contract, scatter back. The identity
    # fast path above is kept deliberately — routing it through here would add two full-array
    # permute-copies (gather + scatter) on the common axis-first contraction.
    src_g  = permutedims!(reshape(view(gathered_in, 1:length(src)),
                                  _gathered_shape(src, perm)), src, perm)
    in_mat = reshape(src_g, n_src, n_other, n_dep)
    out_g  = reshape(view(gathered_out, 1:(n_dst * n_other * n_dep)), n_dst, n_other, n_dep)
    _batched_matmul!(out_g, Mp, in_mat, n_dep, transpose)
    out_shape = Base.setindex(ntuple(i -> size(dest, perm[i]), Val(ndims(dest))), n_dst, 1)
    permutedims!(dest, reshape(out_g, out_shape), invperm(perm))
    return dest
end

# TODO(transpose-as-type): consider making transpose-ness a property of the kernel type, so
# callers write `forward!(dest, transpose(kernel), …)` with `transpose(kernel)` returning a
# `TransposedKernel{K}` (or `K{…,true}`) — tidier and more semantic than the `transpose::Bool`.
forward!(dest, M::AbstractArray, src, perm, gathered_in, gathered_out)  =
    _dense_contract!(dest, src, M, perm, gathered_in, gathered_out, false)
backward!(dest, M::AbstractArray, src, perm, gathered_in, gathered_out) =
    _dense_contract!(dest, src, M, perm, gathered_in, gathered_out, true)

# Keyword front door: the stage passes the dense kernel + its `kernel_scratch` (the gather
# perm + work-buffers), and the contraction unpacks them — `forward!(dest, kernel, src; scratch)`.
# `dest`/`src` are typed `AbstractArray` so this 3-positional form is disjoint from the modern
# stage sugar `forward!(stage::AbstractModernStage, Λ, …)`.
forward!(dest::AbstractArray, M::AbstractArray, src::AbstractArray; scratch)  =
    forward!(dest, M, src, scratch.gather_perm, scratch.gather_in, scratch.gather_out)
backward!(dest::AbstractArray, M::AbstractArray, src::AbstractArray; scratch) =
    backward!(dest, M, src, scratch.gather_perm, scratch.gather_in, scratch.gather_out)

# kernel_scratch — the gather instructions, derived FROM the kernel #
#------------------------------------------------------------------#
# A dense kernel is a bare `PermutedDimsArray`; its presentation perm encodes the contracted
# axis (parent dim 2) and the dep axes (parent dims 3…, non-singleton), so the gather
# perm/work-buffers can be rebuilt from the kernel alone — no stage bookkeeping. A size-1 dep
# axis is degenerate and unsupported (indistinguishable from a padding singleton).

"The presentation perm of a dense kernel, read off its `PermutedDimsArray` type parameter."
_kernel_perm(::PermutedDimsArray{T, N, perm}) where {T, N, perm} = perm

"Recover `(contracted axis, dep axes)` (layout positions) from a dense kernel's presentation perm."
function _kernel_axes(kernel::PermutedDimsArray, N::Int)
    p    = _kernel_perm(kernel)
    P    = parent(kernel)
    adim = findfirst(k -> p[k + 1] == 2, 1:N)
    deps = sort!(Int[k for k in 1:N if p[k + 1] >= 3 && size(P, p[k + 1]) != 1]; by = k -> p[k + 1])
    return adim, Tuple(deps)
end

"""
The gather scratch (src-gather perm + two work-buffers) a dense-kernel contraction needs,
derived from the kernel itself. Stages with a dense kernel get this for free via the default
`allocate_scratch`. Kernels with no gather (single-destination, identity, discount) return `()`.
"""
function kernel_scratch(kernel::PermutedDimsArray, layout::GriddedLayout, ::Type{T}) where {T}
    adim, deps  = _kernel_axes(kernel, length(layout))
    gperm, _    = _gather_perm(adim, deps, Val(length(layout)))
    P           = parent(kernel)
    n_out, n_in = size(P, 1), size(P, 2)
    # Both verbs reuse these buffers, so size them for the larger of input/output — a
    # rectangular kernel (forget n_out<n_in, introduce/crosswalk n_out>n_in) resizes the
    # contracted axis from n_in to n_out, scaling the total by n_out/n_in. Derived from the
    # kernel, so no spec/output_layout is needed here.
    insz        = prod(layout_size(layout))
    gsz         = max(insz, insz ÷ n_in * n_out)
    return (gather_perm = gperm, gather_in = zeros(T, gsz), gather_out = zeros(T, gsz))
end
kernel_scratch(::Any, ::GriddedLayout, ::Type) = (;)

# Single-destination kernel — the sparse one-destination-per-cell transition #
#----------------------------------------------------------------------------#
# Each cell sends its mass to a per-cell destination on the contracted axis. The kernel IS
# that `destinations` array — shared by the deterministic-continuous move and the
# argmax-policy stages (the optimization/closure that PRODUCES the destination lives in
# each stage's backward; this is just the linear scatter-K / gather-or-interpolate-Kᵀ).
# Two regimes, dispatched on the destination element type:
#   • `Integer` — an exact grid index: mass lands on one point (`:nearest`), `Kᵀ` gathers
#     from that point. K is a 0/1 selection matrix.
#   • off-grid value — a continuous position: mass Young-splits between the two bracketing
#     points (`:share`), `Kᵀ` linearly interpolates. K is the 2-sparse interp matrix.
# The contracted `axis` (a `Val` dim) is INTRINSIC operator data, carried on the kernel; the
# ambient `grid` (continuous regime) is resolved from the layout into the kernel's plan by
# `kernel_scratch`. Both directions are the uniform `forward!/backward!(dest, k, src; scratch)`,
# routing to the seams in `helper/interpolations.jl`. Mass is conserved, so the forward always
# clamps off-grid targets to the grid endpoints — and clamp is the UNIQUE backward extrapolation
# that is its transpose (linear extrapolation needs negative mass), so the continuous backward
# clips. That makes the move a genuine transition: its adjoints fall out of the generic stage
# path (jacobian.jl), no override.

"""
The single-destination transition: a per-cell destination on the contracted `axis`, stored
sparsely as the `destinations` array (an `Integer` grid index → exact landing; an off-grid
value → Young-split / clip-interpolation). Carries its `axis` (a `Val` dim); the continuous
regime's grid is resolved into its `kernel_scratch` plan.
"""
struct SingleDestinationKernel{D<:AbstractArray, A}
    destinations :: D
    axis         :: A    # Val(dim): the contracted axis position
end

"Unwrap a `Val`-wrapped dimension."
_dim(::Val{D}) where {D} = D

# The plan: the integer regime needs nothing; the continuous regime needs the axis grid
# (ambient, resolved from the layout via the kernel's axis).
kernel_scratch(::SingleDestinationKernel{<:AbstractArray{<:Integer}}, ::GriddedLayout, ::Type) = (;)
kernel_scratch(k::SingleDestinationKernel, layout::GriddedLayout, ::Type{T}) where {T} =
    (grid = collect(T, axisvalues(layout.axes[_dim(k.axis)])),)

# Exact (integer destination): :nearest scatter forward, index gather backward.
forward!(dest, k::SingleDestinationKernel{<:AbstractArray{<:Integer}}, src; scratch) =
    redistribute_along!(dest, src, k.destinations, nothing, k.axis, Val(:nearest))
backward!(dest, k::SingleDestinationKernel{<:AbstractArray{<:Integer}}, src; scratch) =
    _gather_along!(dest, src, k.destinations, k.axis)

# Off-grid (continuous destination): :share Young-split forward, clip-interp backward. The
# integer methods above are more specific; this catches `Float`/`ForwardDiff.Dual`.
forward!(dest, k::SingleDestinationKernel, src; scratch) =
    redistribute_along!(dest, src, k.destinations, scratch.grid, k.axis, Val(:share))
backward!(dest, k::SingleDestinationKernel, src; scratch) =
    _along_axis(dest, src, scratch.grid, k.destinations, k.axis, ReinterpOp(Val(:clip)))

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
