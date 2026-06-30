# Transition kernels — the linear operator a stage applies.
#
# A transition is applied via two in-place verbs rather than `mul!`, because the stage backward is
# affine (`r + KᵀV`) and discount's two directions are not an adjoint pair:
#
#     forward!(dest, kernel, src, …)    # dest = K · src    (pushforward of Λ)
#     backward!(dest, kernel, src, …)   # dest = Kᵀ · src   (pullback of V)
#
# A kernel owns its field(s) and operator plan and carries no iteration state. The dense case is a
# `DenseKernel` (dense_kernel.jl) owning a `MatrixField`; the structured cases (`ScatterKernel` /
# `InterpKernel`, the kernel-choice and logit kernels) are data-only structs in their own files; the
# lightweight `I` (identity, here) and `PointwiseScale` (pointwise_scale.jl) carry no data.
#
# Orientation convention (the duality gate): the stored kernel is K, the forward operator, so
# `forward! = K·src` and `backward! = Kᵀ·src`. A Markov-style transition that applies its `S` on
# backward (`S = Kᵀ`) stores `Sᵀ`, reproducing the duality `⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩`.
#
# This file holds the protocol's two trivial pieces — the identity transition and the generic
# `kernel_scratch` fallback — plus the per-cell `DestinationField` the scatter and interp kernels share.

using LinearAlgebra: UniformScaling, I, transpose

# Identity & the generic scratch fallback #
#----------------------------------------#

# Identity transition — UniformScaling: no storage, both verbs are copies. The (ignored)
# `scratch` kwarg lets `I` share the generic stage-adjoint path uniformly with the dense kernels.
forward!(dest, ::UniformScaling, src; scratch=nothing)  = copyto!(dest, src)
backward!(dest, ::UniformScaling, src; scratch=nothing) = copyto!(dest, src)

# A kernel with no gather (single-destination, identity, discount) contributes an empty scratch.
kernel_scratch(::Any, ::GriddedLayout, ::Type) = (;)

# Shared per-cell destination data #
#---------------------------------#

"Unwrap a `Val`-wrapped dimension."
_dim(::Val{D}) where {D} = D

"""
Shared per-cell destination data for [`ScatterKernel`](@ref) and [`InterpKernel`](@ref): a
`destinations` array (one landing per cell on the contracted `axis`) plus that `axis` (a `Val` dim).
The representation is identical across the two; the element type distinguishes use — integer grid
indices for the discrete scatter, off-grid float positions for the continuous interpolation.
"""
struct DestinationField{D<:AbstractArray, A}
    destinations :: D
    axis         :: A    # Val(dim): the contracted axis position
end
