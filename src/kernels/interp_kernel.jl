# InterpKernel — the continuous two-node lottery transition #
#===========================================================#
# Each cell's mass splits between two nodes of the contracted axis: `K` is the 2-sparse lottery
# matrix and `Kᵀ` its gather. Both directions clamp off-grid targets to the grid endpoints.

"""
The continuous transition on an ordered grid axis: each cell's mass splits between destination nodes
`lo` and `hi` by the Young weight of the cell's float position, which alone carries derivatives.
"""
struct InterpKernel{D<:DestinationField, I<:AbstractArray}
    dest :: D
    lo   :: I
    hi   :: I
end

InterpKernel(destinations::AbstractArray, axis) =
    InterpKernel(DestinationField(destinations, axis),
                 ones(Int32, size(destinations)), ones(Int32, size(destinations)))

"The per-cell position an [`InterpKernel`](@ref) owns."
destinations(k::InterpKernel) = k.dest.destinations
"The contracted axis (a `Val` dim) of an [`InterpKernel`](@ref)."
_kaxis(k::InterpKernel) = k.dest.axis

"The destination gridding of the contracted axis, at the working eltype."
kernel_scratch(k::InterpKernel, ::GriddedLayout, end_layout::GriddedLayout, ::Type{T}) where {T} =
    (dest_grid = collect(T, axisvalues(end_layout.axes[_dim(_kaxis(k))])),)

"Write into `lo`/`hi` the pair of `grid` nodes bracketing each stored position."
seat_interp!(k::InterpKernel, grid) =
    stratified!(SeatInterpOp(), k.lo, k.hi, destinations(k), grid; dims=_kaxis(k))

forward!(dest, k::InterpKernel, src; scratch) =
    stratified!(LotteryScatterOp(), dest, src, k.lo, k.hi, destinations(k), scratch.dest_grid; dims=_kaxis(k))
backward!(dest, k::InterpKernel, src; scratch) =
    stratified!(LotteryGatherOp(), dest, src, k.lo, k.hi, destinations(k), scratch.dest_grid; dims=_kaxis(k))
