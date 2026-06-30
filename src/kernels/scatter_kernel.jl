# ScatterKernel — the discrete single-destination transition #
#===========================================================#
# Each cell sends all its mass to a per-cell destination on the contracted axis. The optimization /
# closure that produces the destination lives in each stage's backward; the kernel is just the linear
# operator. `ScatterKernel` is the discrete case: the destination is an exact integer grid index —
# mass lands on one point (`:nearest`), `Kᵀ` gathers from that point, `K` is a 0/1 selection. It
# backs the argmax-policy stages (index policy), and is kept distinct from the continuous
# `InterpKernel` (interp_kernel.jl) — different ops, axis kinds, differentiability (§8.2). What they
# share is the destination representation, factored into `DestinationField` (kernel.jl).
#
# Both directions are the uniform `forward!/backward!(dest, k, src; scratch)`. The discrete scatter
# needs no grid plan (the index lands exactly).

"""
The discrete single-destination transition: each cell sends all its mass to one integer grid index on
the contracted axis. `forward!` scatters mass to the index (a 0/1 selection `K`); `backward!` gathers
value from it (`Kᵀ`). Owns a [`DestinationField`](@ref) of integer indices that the argmax solver
writes the index policy into.
"""
struct ScatterKernel{D<:DestinationField}
    dest :: D
end

# Convenience constructor mirroring the old `(destinations, axis)` call sites.
ScatterKernel(destinations::AbstractArray, axis) = ScatterKernel(DestinationField(destinations, axis))

"The per-cell destination array a [`ScatterKernel`](@ref) owns."
destinations(k::ScatterKernel) = k.dest.destinations
"The contracted axis (a `Val` dim) of a [`ScatterKernel`](@ref)."
_kaxis(k::ScatterKernel) = k.dest.axis

# Discrete (integer destination): :nearest scatter forward, index gather backward.
forward!(dest, k::ScatterKernel, src; scratch) =
    redistribute_along!(dest, src, destinations(k), nothing, _kaxis(k), Val(:nearest))
backward!(dest, k::ScatterKernel, src; scratch) =
    _gather_along!(dest, src, destinations(k), _kaxis(k))
