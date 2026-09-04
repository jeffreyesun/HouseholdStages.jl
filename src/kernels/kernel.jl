# Transition kernels — the linear operator a stage applies. A kernel owns its data and is applied
# in place: `forward!(dest, k, src)` writes `dest = K·src`, `backward!` writes `dest = Kᵀ·src`.

using LinearAlgebra: UniformScaling, I, transpose

# Identity & the generic scratch fallback #
#----------------------------------------#

# Identity transition: both verbs are copies.
forward!(dest, ::UniformScaling, src; scratch=nothing)  = copyto!(dest, src)
backward!(dest, ::UniformScaling, src; scratch=nothing) = copyto!(dest, src)

# A kernel with no gather contributes an empty scratch.
kernel_scratch(::Any, ::GriddedLayout, ::GriddedLayout, ::Type) = (;)

# Shared per-cell destination data #
#---------------------------------#

"Unwrap a `Val`-wrapped dimension."
_dim(::Val{D}) where {D} = D

"One landing per cell on the contracted `axis`: integer grid indices for a discrete scatter, float positions for a continuous interpolation."
struct DestinationField{D<:AbstractArray, A}
    destinations :: D
    axis         :: A    # Val(dim): the contracted axis position
end
