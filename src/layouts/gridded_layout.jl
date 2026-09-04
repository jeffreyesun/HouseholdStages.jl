# GriddedLayout — the dense histogram-on-a-grid representation: its axis geometry and its allocators.

"""
An ordered tuple of axis representations, with the axis `Names` in the type parameter. Construct via
`GriddedLayout(:name => rep, ...)`.
"""
struct GriddedLayout{Names, Axes<:Tuple} <: AbstractLayout
    axes :: Axes
end

# Internal: build from explicit names + reps.
GriddedLayout{Names}(axes::Tuple) where {Names} = GriddedLayout{Names, typeof(axes)}(axes)

function GriddedLayout(pairs::Pair{Symbol, <:AxisRep}...)
    names = map(first, pairs)
    if length(unique(names)) != length(names)
        error("GriddedLayout axis names must be unique; got $(names)")
    end
    return GriddedLayout{names}(map(last, pairs))
end

# Geometry #
#----------#

"The layout's axis names, in order."
axisnames(::GriddedLayout{Names}) where {Names} = Names
axisnames(::Type{<:GriddedLayout{Names}}) where {Names} = Names

Base.length(layout::GriddedLayout) = length(layout.axes)

"Per-axis sizes, in order."
layout_size(layout::GriddedLayout) = map(axissize, layout.axes)

"The integer dimension of the axis named `name`."
function axis_position(layout::GriddedLayout, name::Symbol)
    names = axisnames(layout)
    idx = findfirst(==(name), names)
    idx === nothing && error("axis :$name not found in layout with names $(names)")
    return idx
end

"The coordinate values (grid points or levels) of the axis named `name`."
axis_grid(layout::GriddedLayout, name::Symbol) =
    axisvalues(layout.axes[axis_position(layout, name)])

Base.:(==)(a::GriddedLayout, b::GriddedLayout) = axisnames(a) == axisnames(b) && a.axes == b.axes
Base.hash(l::GriddedLayout, h::UInt) = hash(l.axes, hash(axisnames(l), hash(:GriddedLayout, h)))

"The operative axis's sizes at the two ends of a stage, `(n_start, n_end)`."
operative_sizes(start_layout::GriddedLayout, end_layout::GriddedLayout, axis::Symbol) =
    (_axis_size(start_layout, axis), _axis_size(end_layout, axis))

"Name-keyed `selectdim`: the view of `A` with the named `axis` fixed at index `i`."
fix(A::AbstractArray, layout::GriddedLayout, (axis, i)::Pair{Symbol, <:Integer}) =
    selectdim(A, axis_position(layout, axis), i)

"A copy of `layout` with the axis named `name` removed."
function drop_axis(layout::GriddedLayout, name::Symbol)
    pos   = axis_position(layout, name)
    names = axisnames(layout)
    new_names = (names[1:pos-1]..., names[pos+1:end]...)
    new_axes  = (layout.axes[1:pos-1]..., layout.axes[pos+1:end]...)
    return GriddedLayout{new_names}(new_axes)
end

"`layout` with the axis named `name` shrunk to its first `n` coordinates; a no-op when it already has size `n`."
function resize_axis(layout::GriddedLayout, name::Symbol, n::Int)
    pos = axis_position(layout, name)
    axissize(layout.axes[pos]) == n && return layout                       # already the requested size
    axes = ntuple(i -> i == pos ? _resize_axis(layout.axes[i], n) : layout.axes[i],
                  length(layout.axes))
    return GriddedLayout{axisnames(layout)}(axes)
end

"`layout` with the axis named `name` replaced by the `n` integer levels `1:n`."
function grow_axis(layout::GriddedLayout, name::Symbol, n::Int)
    pos  = axis_position(layout, name)
    axes = ntuple(i -> i == pos ? Discrete(collect(1:n)) : layout.axes[i],
                  length(layout.axes))
    return GriddedLayout{axisnames(layout)}(axes)
end

"Rebuild an axis representation of the same kind, keeping only its first `n` coordinates."
_resize_axis(r::GriddedContinuous, n::Int) = GriddedContinuous(r.grid[1:n])
_resize_axis(r::Discrete, n::Int)          = Discrete(r.levels[1:n])

# Cells iteration #
#-----------------#

# One cell of the layout as `(idx, cell)`, both `NamedTuple`s keyed by axis name.
function _cell_pair(values::V, ci::CartesianIndex{N}, ::Val{Names}) where {Names, V, N}
    idx_nt  = NamedTuple{Names}(Tuple(ci))
    cell_nt = NamedTuple{Names}(ntuple(i -> values[i][ci[i]], Val(N)))
    return (idx_nt, cell_nt)
end

"Iterate the layout's cells in column-major order, yielding `(idx, cell)`."
function cells(layout::GriddedLayout{Names}) where {Names}
    values = map(axisvalues, layout.axes)
    sizes  = layout_size(layout)
    return (_cell_pair(values, ci, Val(Names)) for ci in CartesianIndices(sizes))
end

"Dense `Array{NamedTuple}` of cell values, shaped `layout_size(layout)`, for per-cell broadcasts."
function cell_array(layout::GriddedLayout{Names}) where {Names}
    N = length(layout.axes)
    values = map(axisvalues, layout.axes)
    sizes  = layout_size(layout)
    return [NamedTuple{Names}(ntuple(i -> values[i][cart[i]], Val(N)))
            for cart in CartesianIndices(sizes)]
end

# Allocation — how a GriddedLayout realises the stage allocation protocol #
#------------------------------------------------------------------------#
# The defaults for the stage allocators on this representation; a stage overrides whichever it needs.

"Allocate a layout-shaped V/Λ container on this representation (a dense `zeros`)."
allocate_buffer(layout::GriddedLayout, ::Type{T}) where {T} = zeros(T, layout_size(layout))

"The buffers a primitive stage's core writes into: `V_start` on its start layout, `Λ_end` on its end layout."
io_scratch(start_layout::GriddedLayout, end_layout::GriddedLayout, ::Type{T}) where {T} =
    (V_start = allocate_buffer(start_layout, T),
     Λ_end   = allocate_buffer(end_layout, T))

"Kernel storage for a stage between these layouts; `nothing` unless the stage carries a transition."
allocate_kernel(spec, ::Type, ::GriddedLayout, ::GriddedLayout) = nothing

"Compute scratch for a stage between these layouts: the I/O buffers, plus whatever else the stage declares."
allocate_scratch(spec, ::Type{T}, start_layout::GriddedLayout, end_layout::GriddedLayout) where {T} =
    io_scratch(start_layout, end_layout, T)
