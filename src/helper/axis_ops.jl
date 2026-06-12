# Axis-management helpers shared across stages. Small, allocation-free primitives that show up
# in more than one stage body (permutation tuples, name-keyed slicing/coordinates).
# Library-internal; nothing here is exported.

# Named-axis helpers — thin name-keyed wrappers keeping the named-axis abstraction visible in
# stage bodies; none allocate.

"Integer dimension of `axis` — a pass-through to `axis_position`, named for array-dim call sites."
axis_dim(layout::GriddedLayout, axis::Symbol) = axis_position(layout, axis)

"""
Name-keyed `selectdim`: the view of `A` with the named `axis` fixed at index `i`.
"""
fix(A::AbstractArray, layout::GriddedLayout, (axis, i)::Pair{Symbol, <:Integer}) =
    selectdim(A, axis_dim(layout, axis), i)

"""
A copy of `ci` with the coordinate at the named `axis`'s position set to `j` —
used when building the `next_ci` cache (`argmax`).
"""
set_coord(ci::CartesianIndex, layout::GriddedLayout, (axis, j)::Pair{Symbol, <:Integer}) =
    CartesianIndex(Base.setindex(Tuple(ci), j, axis_dim(layout, axis)))
