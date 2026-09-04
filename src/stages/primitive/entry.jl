"""
Entry stage: the forward push is `Λ_end = Λ_start + g`, the backward the identity on V. `entry` is
any `ScalarField` source — a scalar for a uniform inflow, a `FromEnv`, a layout-shaped entry
distribution, or a `(; dep…[, env])` closure for a targeted inflow. Mass is not conserved: the
population grows by `Σg` each pass.
"""
struct EntryStageSpec{G} <: AbstractStageSpec
    entry :: G
end

EntryStageSpec(; entry) = EntryStageSpec{typeof(entry)}(entry)

@definestage EntryStage EntryStageSpec


##########################
# Gridded implementation #
##########################

"The entering mass `g`, a scalar or a layout-shaped broadcastable, rewritten on every `backward!`."
mutable struct EntryKernel
    g :: Any
end

operative_axis(::EntryStageSpec) = nothing

allocate_kernel(::EntryStageSpec, ::Type, ::GriddedLayout, ::GriddedLayout) = EntryKernel(0.0)

"Cache: the entry mass as a materialised `ScalarField`."
allocate_cache(spec::EntryStageSpec, ::Type{T}, start_layout::GriddedLayout, ::GriddedLayout) where {T} =
    (entry = ScalarField(spec.entry, start_layout, T),)

# V passes through unchanged; the work here is storing `g` for the forward pass.
function backward!(V_start, spec::EntryStageSpec, start_layout::GriddedLayout, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    kernel.g = materialize_scalar!(cache.entry, spec.entry, start_layout, env; env_changed)
    copyto!(V_start, V_end)
    return (V_start, kernel)
end

forward!(Λ_end, ::EntryStageSpec, ::GriddedLayout, ::GriddedLayout, Λ_start; kernel, scratch) =
    (@. Λ_end = Λ_start + kernel.g; Λ_end)

# Both adjoints are the identity.
backward_adjoint!(stage::EntryStage, dV_start) = copy(dV_start)
forward_adjoint!(stage::EntryStage, dΛ_end)    = copy(dΛ_end)
