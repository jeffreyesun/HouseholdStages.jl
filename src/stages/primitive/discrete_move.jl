"""
Deterministic transition along one named `axis`, snapped to the nearest grid index: each cell moves
to `destination(; ax…[, env])` evaluated on the `axis` grid and rounded to the closest grid point.
Backward gathers `V_end` from that index, forward puts all of the cell's mass there.
"""
struct DiscreteMoveStageSpec{F} <: AbstractStageSpec
    destination :: F
    axis        :: Symbol
end

DiscreteMoveStageSpec(; destination, axis::Symbol) =
    DiscreteMoveStageSpec{typeof(destination)}(destination, axis)

@definestage DiscreteMoveStage DiscreteMoveStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(spec::DiscreteMoveStageSpec) = spec.axis
tangent_grade(::DiscreteMoveStageSpec)     = :wrong_object

# Kernel: a `ScatterKernel` over the per-cell snapped grid index.
allocate_kernel(spec::DiscreteMoveStageSpec, ::Type, start_layout::GriddedLayout, ::GriddedLayout) =
    ScatterKernel(zeros(Int, layout_size(start_layout)), Val(axis_position(start_layout, spec.axis)))

"Cache: the destination as a materialised `ScalarField`, plus the end layout's grid for `axis` — the coordinates the snapped index names."
allocate_cache(spec::DiscreteMoveStageSpec, ::Type{T}, start_layout::GriddedLayout,
               end_layout::GriddedLayout) where {T} =
    (destination = ScalarField(spec.destination, start_layout, T),
     dest_grid   = collect(Float64, axis_grid(end_layout, spec.axis)))

"Index of the grid point nearest `x`, clamped to the endpoints; a `Dual` `x` snaps on its value and returns a plain `Int`."
@inline function _nearest_index(grid::AbstractVector, x::Real)
    n = length(grid)
    x <= grid[1] && return 1
    x >= grid[n] && return n
    j = searchsortedlast(grid, x)
    return (x - grid[j]) <= (grid[j+1] - x) ? j : j + 1
end

function backward!(V_start, spec::DiscreteMoveStageSpec, start_layout::GriddedLayout,
                   ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    dest = materialize_scalar!(cache.destination, spec.destination, start_layout, env; env_changed)
    destinations(kernel) .= _nearest_index.(Ref(cache.dest_grid), dest)
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)
    return (V_start, kernel)
end

# forward! (push each cell's mass to its snapped index) is the generic default.

"The solved policy: the snapped destination grid index per cell."
policy(stage::DiscreteMoveStage) = destinations(stage.kernel)
