# State-space layouts — the representations a stage's value/distribution lives on.
# This file holds the shared vocabulary: the axis kinds, `StateAxis`, the
# `AbstractLayout` supertype, and the representation-agnostic allocation defaults.
# Each concrete representation (`GriddedLayout`, and later `DynamicGridLayout`,
# `GriddedWithDerivativesLayout`) lives in its own file and provides how it allocates
# buffers and realises the stage protocol.

# Axis kinds #
#-----------#

"Supertype for state-axis kinds — a continuous interpolation grid (`ContinuousGrid`) or a finite discrete level set (`DiscreteFinite`)."
abstract type AxisKind end

"A continuous (interpolated) state axis, backed by a numeric grid."
struct ContinuousGrid{T<:Real, V<:AbstractVector{T}} <: AxisKind
    grid :: V
end

"A discrete state axis: a finite vector of levels (`Float64`, `Int`, `Symbol`, …)."
struct DiscreteFinite{T, V<:AbstractVector{T}} <: AxisKind
    levels :: V
end

"""
Build a continuous-grid axis: `n` points on `[lo, hi]` with `:linear` or `:log`
spacing, or wrap a supplied vector directly.

`:log` maps `t ∈ [0, log((hi−lo+shift)/shift)]` through `exp(t)·shift − shift + lo`,
placing the first knot at `lo`, the last at `hi`, and concentrating points near
`lo` (the wealth-grid convention in `examples/`). `shift` defaults to `1`.
"""
function continuous_grid(lo::Real, hi::Real;
                         length::Union{Nothing, Int} = nothing,
                         size::Union{Nothing, Int}   = nothing,
                         spacing::Symbol             = :linear,
                         shift::Real                 = 1.0)
    n = @something length size error("continuous_grid: pass either `length` or `size`")
    if spacing === :linear
        return ContinuousGrid(collect(range(lo, hi; length = n)))
    elseif spacing === :log
        grid = [exp(t) * shift - shift + lo
                for t in range(0.0, log((hi - lo + shift) / shift); length = n)]
        return ContinuousGrid(grid)
    else
        error("continuous_grid: spacing must be :linear or :log, got :$spacing")
    end
end
continuous_grid(grid::AbstractVector{<:Real}) = ContinuousGrid(grid)

"Build a discrete-finite axis from a vector of levels."
discrete_finite(levels::AbstractVector{T}) where {T} =
    DiscreteFinite{T, typeof(levels)}(levels)

"Discrete axis of `Symbol` levels — sugar over `discrete_finite`."
categorical(syms::AbstractVector{Symbol}) = discrete_finite(syms)

"""
A named axis of a state layout. `Name` (a `Symbol`) rides in the type domain so
axis names participate in compile-time dispatch and NamedTuple-keyed iteration
(`cells`); `kind` carries the runtime grid or levels.
"""
struct StateAxis{Name, K<:AxisKind}
    kind :: K
end

StateAxis(name::Symbol, kind::K) where {K<:AxisKind} =
    StateAxis{name, K}(kind)

"Shortcut: a raw vector is treated as a `discrete_finite` level set. Use `continuous_grid` for interpolated axes."
StateAxis(name::Symbol, levels::AbstractVector) =
    StateAxis(name, discrete_finite(levels))

"The axis's name (its `Name` type parameter)."
axisname(::StateAxis{Name}) where {Name} = Name
axisname(::Type{<:StateAxis{Name}}) where {Name} = Name

"Number of cells along the axis."
axissize(a::StateAxis) = axissize(a.kind)
axissize(k::ContinuousGrid) = length(k.grid)
axissize(k::DiscreteFinite) = length(k.levels)

"The axis's coordinate values (grid points or levels)."
axisvalues(a::StateAxis) = axisvalues(a.kind)
axisvalues(k::ContinuousGrid) = k.grid
axisvalues(k::DiscreteFinite) = k.levels

# Layout supertype + representation stubs #
#-----------------------------------------#

"""
A *representation* of the state space — how the value function's axes are discretised.
Stage methods dispatch on the layout type, so a stage can be realised differently per
representation. Only `GriddedLayout` is implemented; the others are stubs (below).
"""
abstract type AbstractLayout end

# Phase-2 layouts, not yet implemented — declared so the stage files can carry their
# (`error`-ing) per-representation `backward!`/`forward!` sections.
"Stub: a gridded layout that also carries value derivatives at each knot."
struct GriddedWithDerivativesLayout{Names, Axes<:Tuple} <: AbstractLayout
    axes :: Axes
end
"Stub: a moving-knot grid whose knots are part of the representation."
struct DynamicGridLayout{Names, Axes<:Tuple} <: AbstractLayout
    axes :: Axes
end

# Representation-agnostic allocation default. `cache` is persistent (spec,layout)-level
# build storage (e.g. a cost tensor); most stages need none. (`spec` is left untyped —
# this is the layout-keyed default; the stage that needs a cache overrides it.)
"Allocate persistent build storage for a stage on this layout. Default `nothing`."
allocate_cache(spec, ::Type, ::AbstractLayout) = nothing
