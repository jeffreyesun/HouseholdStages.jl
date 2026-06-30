# GriddedLayout — the dense histogram-on-a-grid representation. Holds the axis geometry
# and how this representation allocates the buffers a stage's core writes into.

"""
An ordered tuple of nameless axis representations; the axis `Names` ride in the type
parameter (so NamedTuple-keyed iteration like `cells` is type-stable), associated
positionally with `axes`. Construct via `GriddedLayout(:name => rep, ...)`.
"""
struct GriddedLayout{Names, Axes<:Tuple} <: AbstractLayout
    axes :: Axes
end

# Internal: build from explicit names + reps (used by the pair constructor and the
# resize/grow/drop rebuilds).
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

"The integer dimension of the axis named `name`; errors if absent."
function axis_position(layout::GriddedLayout, name::Symbol)
    names = axisnames(layout)
    idx = findfirst(==(name), names)
    idx === nothing && error("axis :$name not found in layout with names $(names)")
    return idx
end

"The coordinate values (grid points or levels) of the axis named `name`."
axis_grid(layout::GriddedLayout, name::Symbol) =
    axisvalues(layout.axes[axis_position(layout, name)])

"""
Name-keyed `selectdim`: the view of `A` with the named `axis` fixed at index `i`. A thin,
allocation-free name-keyed wrapper keeping the named-axis abstraction visible in stage bodies.
"""
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

"""
Shrink the axis named `name` to its first `n` coordinates, keeping the axis in place (the
fixed-layout invariant — an axis is resized, never dropped). A **no-op when the axis already
has size `n`** (so callers need no square/rectangular branch). The one home for the
full→singleton/smaller resizes: forget (`n = 1`), the rectangular argmax/logit collapse, and
the marginalize; the singleton→full direction is `grow_axis`. A degenerate axis keeps its
*first* grid-point/level.
"""
function resize_axis(layout::GriddedLayout, name::Symbol, n::Int)
    pos = axis_position(layout, name)
    axissize(layout.axes[pos]) == n && return layout                       # no-op (e.g. the square case)
    axes = ntuple(i -> i == pos ? _resize_axis(layout.axes[i], n) : layout.axes[i],
                  length(layout.axes))
    return GriddedLayout{axisnames(layout)}(axes)
end

"""
Grow a (typically singleton) axis to `n` integer levels `1:n`, in place (same position) — the
singleton→full "introduce" direction of the fixed-layout invariant, the dual of `resize_axis`'s
shrink (which can only keep *existing* coordinates, so it cannot expand a singleton). The axis
becomes a `Discrete(1:n)` index axis; used to expand a declared singleton choice/group axis
(`ProductStage`, `KernelChoiceStage`) to its operative width.
"""
function grow_axis(layout::GriddedLayout, name::Symbol, n::Int)
    pos  = axis_position(layout, name)
    axes = ntuple(i -> i == pos ? Discrete(collect(1:n)) : layout.axes[i],
                  length(layout.axes))
    return GriddedLayout{axisnames(layout)}(axes)
end

"Rebuild an axis rep keeping its kind but only its first `n` coordinates."
_resize_axis(r::GriddedContinuous, n::Int) = GriddedContinuous(r.grid[1:n])
_resize_axis(r::Discrete, n::Int)          = Discrete(r.levels[1:n])

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
Dense `Array{NamedTuple}` of cell values, shaped `layout_size(layout)`, for whole-grid per-cell
broadcasts (`f.(cell_array(layout))`). The element type is isbits iff every axis stores isbits
values; `Symbol`-celled categorical axes are not, and force CPU broadcasts.
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

"Output layout the spec produces. Default: same as input. A stage that resizes its operative axis (rectangular `MarkovStage`/marginalize, the rectangular `ArgmaxStage` collapse) overrides this."
output_layout(spec, layout::GriddedLayout) = layout

"""
The standard output buffers a modern stage's core writes into and returns: `V_start` on
`input_layout(spec, layout)`, `Λ_end` on `output_layout(spec, layout)`. Symmetric — a stage that
resizes its operative axis (e.g. the rectangular argmax collapse) declares the resized input/output
layouts and the buffers follow. A stage needing extra scratch `merge`s this into its own NamedTuple.
"""
io_scratch(spec, layout::GriddedLayout, ::Type{T}) where {T} =
    (V_start = allocate_buffer(input_layout(spec, layout), T),
     Λ_end   = allocate_buffer(output_layout(spec, layout), T))

"Kernel storage for a stage on this layout. Default `nothing` (a stage with a transition overrides)."
allocate_kernel(spec, ::Type, ::GriddedLayout) = nothing

"Compute scratch for a stage on this layout. Default: the I/O buffers (`io_scratch`); a stage with extra scratch `merge`s into it."
allocate_scratch(spec, ::Type{T}, layout::GriddedLayout) where {T} = io_scratch(spec, layout, T)
