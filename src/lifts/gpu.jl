#################################################
# to_device / to_host — relocate a stage's data #
#################################################
#
# A stage relocates through `Adapt.adapt`: isbits-eltype arrays move, everything else stays put.

"""
The relocation adaptor: applies `to` to every isbits-eltype array in a stage's field graph.
"""
struct MoveTo{T}
    to :: T
end

Adapt.adapt_storage(m::MoveTo, x::AbstractArray) = isbitstype(eltype(x)) ? adapt(m.to, x) : x
Adapt.adapt_structure(::MoveTo, f::Function) = f
Adapt.adapt_structure(::MoveTo, f::Core.OpaqueClosure) = f
Adapt.adapt_structure(m::MoveTo, r::Base.RefValue{T}) where {T} = Base.RefValue{T}(adapt(m, r[]))

# Fields — the compact backing arrays relocate.
Adapt.@adapt_structure MatrixField
Adapt.@adapt_structure FiberFace

"""
Relocate a `ScalarField`: the materialised buffer moves and `to` is recorded on the field.
"""
function Adapt.adapt_structure(to, f::ScalarField{A}) where {A}
    d = adapt(to, f.data)
    return d isa AbstractArray ? ScalarField{typeof(d)}(d, f.bshape, f.env_dependent, to) :
                                 ScalarField{A}(d, f.bshape, f.env_dependent, to)
end

# Kernels — each relocates its own data.
Adapt.@adapt_structure DenseKernel
Adapt.@adapt_structure DestinationField
Adapt.@adapt_structure ScatterKernel
Adapt.@adapt_structure InterpKernel
Adapt.@adapt_structure LogitChoiceKernel
Adapt.@adapt_structure MeanPreservingSpreadKernel
Adapt.@adapt_structure GaussianLoadingKernel
Adapt.@adapt_structure MixingKernel
Adapt.@adapt_structure PointwiseScale
Adapt.@adapt_structure EntryKernel

"""
Relocate a primitive stage's kernel, scratch and cache, leaving the spec and layouts put.
"""
Adapt.adapt_structure(to, s::S) where {S<:AbstractPrimitiveStage} =
    S.name.wrapper(s.spec, s.start_layout, s.end_layout,
                   adapt(to, s.kernel), adapt(to, s.scratch), adapt(to, s.cache))

# Composites — relocate each bundled leaf, plus a product buffer's fused V/Λ, and rebundle.
Adapt.@adapt_structure ChainStageBuffer
Adapt.@adapt_structure ChainStage
Adapt.@adapt_structure ProductStageBuffer
Adapt.@adapt_structure ProductStage

Adapt.adapt_structure(to, s::AbstractStage) =
    error("to_device: no relocation rule for $(typeof(s)); add one in src/lifts/gpu.jl.")

"Relocate a constructed `stage` with an `Adapt` adaptor: `to_device(stage, CuArray)`."
to_device(stage::AbstractStage, to) = adapt(MoveTo(to), stage)

to_device(::AbstractStage, ::Function) =
    error("to_device(stage, f): pass an Adapt adaptor, not a mover function — " *
          "`to_device(stage, CuArray)` for a device, `to_host(stage)` for the way back.")

"""
Bring a relocated `stage` back to host arrays, an already host-resident one sharing its arrays.
"""
to_host(stage::AbstractStage) = to_device(stage, Array)

"Move a constructed `stage` onto a GPU: `lift_gpu(stage, CuArray)`."
lift_gpu(stage::AbstractStage, to) = to_device(stage, to)
lift_gpu(stage::AbstractStage) =
    error("lift_gpu(stage): CUDA is not a dependency of HouseholdStages — " *
          "pass the device adaptor, e.g. `lift_gpu(stage, CuArray)` or " *
          "`to_device(stage, CuArray)`.")
