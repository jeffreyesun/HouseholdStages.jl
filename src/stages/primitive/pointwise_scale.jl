"""
Two-sided diagonal scale: `V_start = backward · V_end` and `Λ_end = forward · Λ_start`. Each of
`backward`/`forward` is a `Real` or a `FromEnv` marker resolved at run time; `forward` defaults to
`backward`.
"""
@kwdef struct PointwiseScaleSpec{A, F} <: AbstractStageSpec
    backward :: A = 1.0
    forward  :: F = backward
end

@definestage PointwiseScaleStage PointwiseScaleSpec


##########################
# Gridded implementation #
##########################
# Both scales live in `Ref{Any}`s inside the `PointwiseScale` kernel, written in place.

operative_axis(::PointwiseScaleSpec) = nothing

allocate_kernel(::PointwiseScaleSpec, ::Type, ::GriddedLayout, ::GriddedLayout) =
    PointwiseScale(Base.RefValue{Any}(1.0), Base.RefValue{Any}(1.0))

function backward!(V_start, spec::PointwiseScaleSpec, ::GriddedLayout, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    kernel.a[] = resolve(spec.backward, env)
    kernel.f[] = resolve(spec.forward, env)
    backward!(V_start, kernel, V_end)    # a · V_end
    return (V_start, kernel)
end

# forward! (f · Λ_start) is the generic default.
