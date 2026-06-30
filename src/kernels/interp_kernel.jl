# InterpKernel — the continuous single-destination transition #
#============================================================#
# The continuous sibling of `ScatterKernel` (scatter_kernel.jl): each cell's mass lands at an
# off-grid float position on the contracted axis. Mass Young-splits between the two bracketing nodes
# (`:share`), `Kᵀ` linearly interpolates, `K` is the 2-sparse interp matrix. Backs the
# deterministic-continuous move. Kept distinct from the discrete scatter — the interp weights move
# smoothly with the float position, an index does not (§8.2/§13). The shared destination
# representation is `DestinationField` (kernel.jl).
#
# The contracted `axis` (a `Val` dim) is intrinsic operator data carried on the field; the ambient
# `grid` is resolved from the layout into the kernel's plan by `kernel_scratch`. Both directions are
# the uniform `forward!/backward!(dest, k, src; scratch)`. Mass is conserved, so the continuous
# forward clamps off-grid targets to the grid endpoints — and clamp is the unique backward
# extrapolation that is its own transpose (linear extrapolation would need negative mass), so the
# backward clips at both ends too (see `reinterpolate!`). That makes the move a genuine transition:
# its adjoints fall out of the generic stage path (jacobian.jl), no override.

"""
The continuous single-destination transition on an ordered grid axis: each cell's mass lands at an
off-grid float position. `forward!` Young-splits the mass between the two bracketing nodes
(`:share`, `K`); `backward!` linearly interpolates value to the position (`Kᵀ`). Both directions
clamp off-grid targets to the grid endpoints — the unique rule that is at once a transpose pair and
mass-conserving with nonnegative weights (end-goal §8/§13). Owns a [`DestinationField`](@ref) of
float positions; the axis grid is resolved into its `kernel_scratch`.
"""
struct InterpKernel{D<:DestinationField}
    dest :: D
end

# Convenience constructor mirroring the old `(destinations, axis)` call sites.
InterpKernel(destinations::AbstractArray, axis) = InterpKernel(DestinationField(destinations, axis))

"The per-cell destination array an [`InterpKernel`](@ref) owns."
destinations(k::InterpKernel) = k.dest.destinations
"The contracted axis (a `Val` dim) of an [`InterpKernel`](@ref)."
_kaxis(k::InterpKernel) = k.dest.axis

# The plan: the continuous interp needs the axis grid (ambient, resolved from the layout via the
# kernel's axis).
kernel_scratch(k::InterpKernel, layout::GriddedLayout, ::Type{T}) where {T} =
    (grid = collect(T, axisvalues(layout.axes[_dim(_kaxis(k))])),)

# Continuous (off-grid destination): :share Young-split forward, clip-interp backward (both ends).
forward!(dest, k::InterpKernel, src; scratch) =
    redistribute_along!(dest, src, destinations(k), scratch.grid, _kaxis(k), Val(:share))
backward!(dest, k::InterpKernel, src; scratch) =
    _along_axis(dest, src, scratch.grid, destinations(k), _kaxis(k), ReinterpOp(Val(:clip)))
