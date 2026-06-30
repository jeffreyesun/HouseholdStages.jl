# Entry — the additive forward source (the catalog's gap G2, entry half). The forward push is
#
#     Λ_end = Λ_start + g          (incumbents pass through, an entering mass `g` is added),
#
# a `Λ`-independent inhomogeneity NO linear kernel can synthesise — hence a primitive, not a Markov.
# The backward is identity (entry does not change incumbents' continuation values). Forward and
# backward are NOT an adjoint pair (the source has no backward effect); the forward_adjoint is
# identity (the source is constant in `Λ_start`). The entering mass `g` is any [`ScalarField`](@ref) source —
# a scalar (uniform inflow), a `FromEnv`, a layout-shaped distribution, or a closure (targeted
# inflow) — so there is no bespoke axis code. Λ is allowed to grow (need NOT sum to 1).

"""
Entry stage: forward `Λ_end = Λ_start + g` (an additive entry source), backward identity on V.
`entry` is any [`ScalarField`](@ref) source — a scalar (uniform inflow), a `FromEnv`, a
layout-shaped entry distribution, or a `(; dep…[, env])` closure (a targeted inflow). Mass is NOT
conserved (the population grows by `Σg` each pass), by design.
"""
struct EntryStageSpec{G} <: AbstractStageSpec
    entry :: G
end

EntryStageSpec(; entry) = EntryStageSpec{typeof(entry)}(entry)

@definestage EntryStage EntryStageSpec


##########################
# Gridded implementation #
##########################
# K = I on V (backward = copy); the forward adds the materialised entry field to Λ. The entry
# field is a `ScalarField` (the materialized buffer; the Source lives in the Spec) held in the cache
# and materialised into the kernel's seated source each backward, so a FromEnv/AD-shocked entry tracks
# env. The seated source is a scalar or a broadcastable array, applied by the forward.

"""
Seated data for the entry source: the materialised entering mass `g` (a scalar or a layout-shaped
broadcastable), re-seated each `backward!` to track a `FromEnv`/AD-shocked field.
"""
mutable struct EntryKernel
    g :: Any
end

allocate_kernel(::EntryStageSpec, ::Type, ::GriddedLayout) = EntryKernel(0.0)

"Cache: the entry mass as a `ScalarField` (the materialized buffer; the Source lives in the Spec). Env-independent ⇒ filled at construction; env-dependent ⇒ NaN-filled, seated each `backward!`."
allocate_cache(spec::EntryStageSpec, ::Type{T}, layout::GriddedLayout) where {T} =
    (entry = ScalarField(spec.entry, layout, T),)

# Backward: V passes through unchanged; seat the materialised entry source for the forward.
function backward!(V_start, spec::EntryStageSpec, layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    kernel.g = materialize_scalar!(cache.entry, spec.entry, layout, env; env_changed)
    copyto!(V_start, V_end)
    return (V_start, kernel)
end

# Forward: incumbents pass through plus the additive entry source `g`.
forward!(Λ_end, ::EntryStageSpec, ::GriddedLayout, Λ_start; kernel, scratch) =
    (@. Λ_end = Λ_start + kernel.g; Λ_end)

# Adjoints: backward K is the identity ⇒ VJP dV_end = dV_start; the source `g` is constant in
# `Λ_start` ⇒ forward VJP dΛ_start = dΛ_end. Both are the identity (NOT a kernel-paired transition).
backward_adjoint!(stage::EntryStage, dV_start) = copy(dV_start)
forward_adjoint!(stage::EntryStage, dΛ_end)    = copy(dΛ_end)


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. Placeholder.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder.
