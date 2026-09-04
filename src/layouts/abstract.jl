# State-space layouts — the representations a stage's value function and distribution live on: the
# axis representations, the `AbstractLayout` supertype, and the representation-free allocation defaults.

# Axis representations #
#----------------------#
# An axis representation carries only its grid or levels; the axis name lives in the layout.

"Supertype for axis representations."
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
A continuous-grid axis: `n` points on `[lo, hi]` with `:linear` or `:log` spacing. `:log` hits both
endpoints exactly and concentrates points near `lo`, the more so the smaller `shift`.
"""
function GriddedContinuous(lo::Real, hi::Real, n::Int;
                           spacing::Symbol = :linear, shift::Real = 1.0)
    E = float(promote_type(typeof(lo), typeof(hi)))    # grid eltype follows the endpoints, not a literal
    loE, hiE, s = convert(E, lo), convert(E, hi), convert(E, shift)
    if spacing === :linear
        return GriddedContinuous(collect(range(loE, hiE; length = n)))
    elseif spacing === :log
        grid = [exp(t) * s - s + loE
                for t in range(zero(E), log((hiE - loE + s) / s); length = n)]
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

# Equality is structural: two representations built independently from the same grid compare equal.
Base.:(==)(a::GriddedContinuous, b::GriddedContinuous) = a.grid == b.grid
Base.:(==)(a::Discrete, b::Discrete)                   = a.levels == b.levels
Base.hash(r::GriddedContinuous, h::UInt) = hash(r.grid,   hash(:GriddedContinuous, h))
Base.hash(r::Discrete, h::UInt)          = hash(r.levels, hash(:Discrete, h))

# Layout supertype + representation stubs #
#-----------------------------------------#

"A representation of the state space: how the value function's axes are discretised."
abstract type AbstractLayout end

#TODO Neither representation is implemented; they are declared so stage methods can dispatch on them.
"A gridded layout that also carries value derivatives at each knot."
struct GriddedWithDerivativesLayout{Names, Axes<:Tuple} <: AbstractLayout
    axes :: Axes
end
"A moving-knot grid whose knots are part of the representation."
struct DynamicGridLayout{Names, Axes<:Tuple} <: AbstractLayout
    axes :: Axes
end

"Persistent build storage for a stage between its two boundary layouts; `nothing` unless the stage declares some."
allocate_cache(spec, ::Type, ::AbstractLayout, ::AbstractLayout) = nothing
