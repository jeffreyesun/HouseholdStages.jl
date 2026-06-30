# State-space layouts — the representations a stage's value/distribution lives on.
# This file holds the shared vocabulary: the axis representations, the `AbstractLayout`
# supertype, and the representation-agnostic allocation defaults. Each concrete
# representation (`GriddedLayout`, and later `DynamicGridLayout`,
# `GriddedWithDerivativesLayout`) lives in its own file and provides how it allocates
# buffers and realises the stage protocol.

# Axis representations #
#----------------------#
# Nameless: an axis rep carries only its grid/levels. The axis NAME lives in the layout
# (its `Names` type parameter), associated positionally via the `:name => rep` pairs the
# layout constructor takes.

"Supertype for axis representations — a continuous interpolation grid (`GriddedContinuous`) or a finite discrete level set (`Discrete`)."
abstract type AxisRep end

"A continuous (interpolated) state axis, backed by a numeric grid."
struct GriddedContinuous{T<:Real, V<:AbstractVector{T}} <: AxisRep
    grid :: V
end

"A discrete state axis: a finite vector of levels (`Float64`, `Int`, `Symbol`, …)."
struct Discrete{T, V<:AbstractVector{T}} <: AxisRep
    levels :: V
end

"""
A continuous-grid axis: `n` points on `[lo, hi]` with `:linear` or `:log` spacing.

`:log` maps `t ∈ [0, log((hi−lo+shift)/shift)]` through `exp(t)·shift − shift + lo`,
placing the first knot at `lo`, the last at `hi`, and concentrating points near `lo`
(the wealth-grid convention in `examples/`). `shift` defaults to `1`.
"""
function GriddedContinuous(lo::Real, hi::Real, n::Int;
                           spacing::Symbol = :linear, shift::Real = 1.0)
    if spacing === :linear
        return GriddedContinuous(collect(range(lo, hi; length = n)))
    elseif spacing === :log
        grid = [exp(t) * shift - shift + lo
                for t in range(0.0, log((hi - lo + shift) / shift); length = n)]
        return GriddedContinuous(grid)
    else
        error("GriddedContinuous: spacing must be :linear or :log, got :$spacing")
    end
end

"Number of cells along the axis."
axissize(r::GriddedContinuous) = length(r.grid)
axissize(r::Discrete)          = length(r.levels)

"The axis's coordinate values (grid points or levels)."
axisvalues(r::GriddedContinuous) = r.grid
axisvalues(r::Discrete)          = r.levels

# Layout supertype + representation stubs #
#-----------------------------------------#

"""
A *representation* of the state space — how the value function's axes are discretised.
Stage methods dispatch on the layout type, so a stage can be realised differently per
representation.
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
