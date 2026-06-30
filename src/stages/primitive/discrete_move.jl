# Discrete deterministic move along one named axis — the nearest-INDEX sibling of
# `DeterministicContinuousStage` (end-goal §10). Each cell moves to the dep closure
# `destination(; ax…[, env])` evaluated on the axis, then SNAPPED to the nearest grid INDEX
# (a discrete landing), applied by a `ScatterKernel`: forward scatters all the cell's mass onto
# that one index (a 0/1 selection `K`), backward gathers value from it (`Kᵀ`). The continuous
# sibling (`DeterministicContinuousStage`) instead Young-splits between the two bracketing nodes via
# an `InterpKernel`; the two differ at exactly the two §10 points — the destination rounding
# (nearest index vs off-grid float) and the kernel (`ScatterKernel` vs `InterpKernel`) — and share
# the source-filled destination `ScalarField`.
#
# A nearest-index landing is a 0/1 selection, so `K`/`Kᵀ` are exact transposes (no interpolation
# weights) and the reverse-mode adjoints fall out of the generic stage path (jacobian.jl). The
# integer index is inherently non-differentiable in `env` — that is the point of the discrete sibling;
# a stage needing the float position's `env`-sensitivity uses the continuous move instead (§13).

"""
Deterministic transition along one named `axis`, snapped to the NEAREST grid INDEX (end-goal §10).
Each cell moves to `destination(; ax…[, env])` evaluated on the `axis` grid and rounded to the
closest grid point: backward gathers `V_end` from that index, forward scatters all the cell's mass
to it (a `ScatterKernel`). The discrete (nearest-index) sibling of [`DeterministicContinuousStage`](@ref)
(which Young-splits off-grid via an `InterpKernel`).
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

# Kernel: a `ScatterKernel` (kernel.jl) over per-cell INTEGER grid indices (the snapped destination).
# Forward scatters `:nearest` (all mass to the index), backward gathers from it. Needs no grid plan.
allocate_kernel(spec::DiscreteMoveStageSpec, ::Type, layout::GriddedLayout) =
    ScatterKernel(zeros(Int, layout_size(layout)), Val(axis_position(layout, spec.axis)))

"""
Cache: the destination as a [`ScalarField`](@ref) (the float position buffer; the Source lives in the
Spec) plus the operative-axis `grid` the position snaps against. Env-independent ⇒ the field is filled
at construction; env-dependent ⇒ NaN-filled, seated each `backward!`.
"""
allocate_cache(spec::DiscreteMoveStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    (destination = ScalarField(spec.destination, layout, T),
     grid        = collect(Float64, axis_grid(layout, spec.axis)))

"""
Index of the grid point nearest the continuous position `x`, clamped to `[1, length(grid)]` (mass
piling at the boundary off-grid). Comparisons run on the primal, so a `Dual` `x` snaps on its value
and returns a plain `Int` — the discrete policy is non-differentiable by construction.
"""
@inline function _nearest_index(grid::AbstractVector, x::Real)
    n = length(grid)
    x <= grid[1] && return 1
    x >= grid[n] && return n
    j = searchsortedlast(grid, x)
    return (x - grid[j]) <= (grid[j+1] - x) ? j : j + 1
end

function backward!(V_start, spec::DiscreteMoveStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    dest = materialize_scalar!(cache.destination, spec.destination, layout, env; env_changed)
    destinations(kernel) .= _nearest_index.(Ref(cache.grid), dest)   # snap the float position to the nearest index
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)
    return (V_start, kernel)
end

# forward! (`:nearest` integer-scatter mass push, K·Λ_start) is the generic modern default (abstract.jl).

"""
The solved policy of a [`DiscreteMoveStage`](@ref): the snapped destination grid index per cell. It IS
the `ScatterKernel`'s destination.
"""
policy(stage::DiscreteMoveStage) = destinations(stage.kernel)


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. Placeholder marking where the deriv-carrying
# representation's methods will go.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder marking where the dynamic-grid
# representation's methods will go.
