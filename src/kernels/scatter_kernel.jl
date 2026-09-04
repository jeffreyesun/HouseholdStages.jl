# ScatterKernel — the discrete single-destination transition #
#===========================================================#
# Each cell sends all its mass to a per-cell integer grid index on the contracted axis: `K` is a 0/1
# selection and `Kᵀ` gathers from that point.

"The discrete single-destination transition: each cell sends all its mass to one integer grid index on the contracted axis."
struct ScatterKernel{D<:DestinationField}
    dest :: D
end

ScatterKernel(destinations::AbstractArray, axis) = ScatterKernel(DestinationField(destinations, axis))

"The per-cell destination array a [`ScatterKernel`](@ref) owns."
destinations(k::ScatterKernel) = k.dest.destinations
"The contracted axis (a `Val` dim) of a [`ScatterKernel`](@ref)."
_kaxis(k::ScatterKernel) = k.dest.axis

forward!(dest, k::ScatterKernel, src; scratch) =
    stratified!(NearestScatterOp(), dest, src, destinations(k); dims=_kaxis(k))
backward!(dest, k::ScatterKernel, src; scratch) =
    stratified!(NearestGatherOp(), dest, src, destinations(k); dims=_kaxis(k))
