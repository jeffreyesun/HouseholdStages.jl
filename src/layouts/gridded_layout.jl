# GriddedLayout — the dense histogram-on-a-grid representation. Holds the axis geometry
# and how this representation allocates the buffers a stage's core writes into.

"""
An ordered, named tuple of state axes, each continuous axis on a numeric grid.
`Names` is lifted to the type level so NamedTuple-keyed iteration (`cells`) is
type-stable. Construct via `GriddedLayout(axes...)`.
"""
struct GriddedLayout{Names, Axes<:Tuple} <: AbstractLayout
    axes :: Axes
end

function GriddedLayout(axes::StateAxis...)
    names = map(axisname, axes)
    if length(unique(names)) != length(names)
        error("GriddedLayout axis names must be unique; got $(names)")
    end
    return GriddedLayout{names, typeof(axes)}(axes)
end

# Geometry #
#----------#

"The layout's axis names, in order."
axisnames(::GriddedLayout{Names}) where {Names} = Names
axisnames(::Type{<:GriddedLayout{Names}}) where {Names} = Names

Base.length(layout::GriddedLayout) = length(layout.axes)

"Per-axis sizes, in order."
layout_size(layout::GriddedLayout) = map(axissize, layout.axes)

"The integer dimension of the axis named `name`; errors if absent."
function axis_position(layout::GriddedLayout, name::Symbol)
    names = axisnames(layout)
    idx = findfirst(==(name), names)
    idx === nothing && error("axis :$name not found in layout with names $(names)")
    return idx
end

"A copy of `layout` with the axis named `name` removed."
function drop_axis(layout::GriddedLayout, name::Symbol)
    pos = axis_position(layout, name)
    remaining = (layout.axes[1:pos-1]..., layout.axes[pos+1:end]...)
    return GriddedLayout(remaining...)
end

"""
Collapse the axis named `name` to a single coordinate, keeping the chain's ordered
`(name, kind)` tuple intact (decision 7: forget resizes an axis to one level rather than
dropping it). The surviving coordinate is the axis's *first* level/grid-point by
convention — a marginalised axis is degenerate, so only its tuple position is
load-bearing. A continuous axis keeps a 1-element grid; a discrete axis a 1-element
level vector.
"""
function resize_axis_to_one(layout::GriddedLayout, name::Symbol)
    pos  = axis_position(layout, name)
    axes = ntuple(i -> i == pos ? _collapse_axis(layout.axes[i]) : layout.axes[i],
                  length(layout.axes))
    return GriddedLayout(axes...)
end

"Rebuild a `StateAxis` keeping its name and kind but a single (first) coordinate."
_collapse_axis(a::StateAxis{Name, <:ContinuousGrid}) where {Name} =
    StateAxis(Name, continuous_grid([first(a.kind.grid)]))
_collapse_axis(a::StateAxis{Name, <:DiscreteFinite}) where {Name} =
    StateAxis(Name, discrete_finite([first(a.kind.levels)]))

# Cells iteration #
#-----------------#

# Type-stable single-cell builder: out-of-line so the closure capturing `values`
# (a heterogeneous tuple) doesn't poison inference inside the generator from `cells`.
function _cell_pair(values::V, ci::CartesianIndex{N}, ::Val{Names}) where {Names, V, N}
    idx_nt  = NamedTuple{Names}(Tuple(ci))
    cell_nt = NamedTuple{Names}(ntuple(i -> values[i][ci[i]], Val(N)))
    return (idx_nt, cell_nt)
end

"""
Iterate the layout's cells in column-major order, yielding `(idx, cell)` — an
integer-index and an axis-value NamedTuple, both keyed by axis name.
"""
function cells(layout::GriddedLayout{Names}) where {Names}
    values = map(axisvalues, layout.axes)
    sizes  = layout_size(layout)
    return (_cell_pair(values, ci, Val(Names)) for ci in CartesianIndices(sizes))
end

"""
Dense `Array{NamedTuple}` of cell values, shaped `layout_size(layout)` — a broadcast
operand for per-cell closures: `f.(cell_array(layout); env)`. The element type is isbits
when every axis stores isbits values (numeric axes; `Symbol`-celled categorical axes are
not, and force CPU broadcasts).
"""
function cell_array(layout::GriddedLayout{Names}) where {Names}
    N = length(layout.axes)
    values = map(axisvalues, layout.axes)
    sizes  = layout_size(layout)
    return [NamedTuple{Names}(ntuple(i -> values[i][cart[i]], Val(N)))
            for cart in CartesianIndices(sizes)]
end

# Allocation — how a GriddedLayout realises the stage allocation protocol #
#------------------------------------------------------------------------#
# These are the layout-keyed DEFAULTS for the stage allocators; a stage overrides the
# one(s) it needs (a real kernel, extra scratch). `spec` is left untyped because the
# layout is the dispatch dimension — the spec is consulted only for `output_layout`.

"Allocate a layout-shaped V/Λ container on this representation (a dense `zeros`)."
allocate_buffer(layout::GriddedLayout, ::Type{T}) where {T} = zeros(T, layout_size(layout))

"Input layout the spec sees. Default: the layout passed in."
input_layout(spec, layout::GriddedLayout) = layout

"Output layout the spec produces. Default: same as input. `ForgetfulSumStage` overrides to resize the marginalised axis."
output_layout(spec, layout::GriddedLayout) = layout

"""
The standard output buffers a modern stage's core writes into and returns: `V_start` on
the layout, `Λ_end` on `output_layout(spec, layout)`. A stage needing extra scratch
`merge`s this into its own NamedTuple.
"""
io_scratch(spec, layout::GriddedLayout, ::Type{T}) where {T} =
    (V_start = allocate_buffer(layout, T),
     Λ_end   = allocate_buffer(output_layout(spec, layout), T))

"Kernel storage for a stage on this layout. Default `nothing` (a stage with a transition overrides)."
allocate_kernel(spec, ::Type, ::GriddedLayout) = nothing

"Compute scratch for a stage on this layout. Default: the I/O buffers (`io_scratch`); a stage with extra scratch `merge`s into it."
allocate_scratch(spec, ::Type{T}, layout::GriddedLayout) where {T} = io_scratch(spec, layout, T)
