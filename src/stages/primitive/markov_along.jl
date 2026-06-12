# Markov transition along one named axis. The kernel IS the transition array — a dense
# self-describing kernel (kernel.jl) over a compact `(n_out, n_in, dep…)` parent. The user
# supplies the ROW-stochastic `transition_matrix` `T[from, to] = P(from→to)` as a constant
# `Matrix`, a `FromEnv`, or a closure `(; dep…, env) -> T` varying it along a declared subset
# of axes (the dep set, so the kernel becomes dep-varying — dep-only storage, no replication).
# Stored fiber `K = Tᵀ` (the forward operator, kernel.jl): forward applies `K`, backward `Kᵀ`.

"""
Markov transition along one named axis. `transition_matrix` supplies the ROW-stochastic
`T[from, to] = P(from→to)` — a constant `Matrix`, a `FromEnv`, or a closure
`(; dep…, env) -> T` varying it along a declared subset of axes. Backward applies `Kᵀ`,
forward `K = Tᵀ`.
"""
struct MarkovStageSpec{Tm} <: AbstractStageSpec
    transition_matrix :: Tm     # T[from, to]: a Matrix, FromEnv, or closure (; dep…, env) -> T
    axis              :: Symbol
end

function MarkovStageSpec(; axis::Symbol, transition_matrix)
    transition_matrix isa AbstractMatrix &&
        @assert size(transition_matrix, 1) == size(transition_matrix, 2) "transition_matrix must be square."
    return MarkovStageSpec(transition_matrix, axis)
end

default_eltype(spec::MarkovStageSpec) =
    spec.transition_matrix isa AbstractMatrix ? eltype(spec.transition_matrix) : Float64

@definestage MarkovStage MarkovStageSpec


##########################
# Gridded implementation #
##########################
# The kernel IS the dense self-describing transition array. `allocate_kernel` sizes it from the
# source; `backward!` seats `K = Tᵀ` via the central `fill_field!` (the `permutedims` map);
# both verbs contract through the shared dense machinery. The gather lives in
# `scratch.kernel_scratch` (derived from the kernel by `@definestage`), so we need no
# `allocate_scratch` of our own.

allocate_kernel(spec::MarkovStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    dense_kernel(T, layout, spec.axis, spec.transition_matrix)

function backward!(V_start, spec::MarkovStageSpec, layout::GriddedLayout, V_end; env, kernel, scratch, cache)
    fill_field!(kernel, MappedField(permutedims, spec.transition_matrix), layout, spec.axis, env)
    backward!(V_start, kernel, V_end; scratch = scratch.kernel_scratch)   # Kᵀ·V_end
    return (V_start, kernel)
end

# forward! (K·Λ_start) is the generic modern default (abstract.jl) — pure kernel application.


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. The phase-1 stage methods above do not dispatch on
# layout type, so this is a placeholder marking where the deriv-carrying
# representation's methods will go.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder marking where the dynamic-grid
# representation's methods will go.
