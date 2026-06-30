# Pointwise scale — the two-sided diagonal scale primitive. Backward scales V, forward scales Λ:
#
#     V_start = a · V_end        (backward, the value scale — discount when a = β)
#     Λ_end   = f · Λ_start      (forward,  the measure scale — survival/growth, renorm)
#
# The two scales are INDEPENDENT, so the transition is an adjoint pair iff `a == f` (the
# self-adjoint diagonal). The asymmetric specials are the discount (`f = 1`), reproduction
# (`a = 1`), and population renorm (`a = 1`, `f = 1/g`). Each scale is a `Real` or a `FromEnv`,
# resolved into a `Ref{Any}` each pass so an env-resolved / AD-shocked (incl. `Dual`) scale is
# rewritten in place without rebuilding the kernel — exactly the old `BackwardScale(β)` pattern,
# generalised to carry both directions.

"""
Two-sided diagonal scale: `V_start = backward · V_end` and `Λ_end = forward · Λ_start`. Each of
`backward`/`forward` is a `Real` or a [`FromEnv`](@ref) marker resolved at run time. `forward`
defaults to `backward` (the self-adjoint diagonal); pass `forward = 1` for the discount asymmetry,
`backward = 1` for a pure measure scale. The kernel is a [`PointwiseScale`](@ref) whose two scales
are rewritten into `Ref`s each pass, so a `FromEnv`/AD-shocked scale tracks env.
"""
@kwdef struct PointwiseScaleSpec{A, F} <: AbstractStageSpec
    backward :: A = 1.0
    forward  :: F = backward
end

@definestage PointwiseScaleStage PointwiseScaleSpec


##########################
# Gridded implementation #
##########################
# Both scales live in `Ref{Any}`s inside the `PointwiseScale` kernel so env-resolved / AD-shocked
# scales (incl. `Dual`s) are written in place each pass.

allocate_kernel(::PointwiseScaleSpec, ::Type, ::GriddedLayout) =
    PointwiseScale(Base.RefValue{Any}(1.0), Base.RefValue{Any}(1.0))

# Backward seats BOTH scales from env (so the later forward applies a fresh `f`) and applies `a`.
# No field/buffer here — the scales are scalar `resolve`s, re-evaluated each pass (always correct,
# never stale), so `env_changed` is accepted for signature uniformity but needs no skip.
function backward!(V_start, spec::PointwiseScaleSpec, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    kernel.a[] = resolve(spec.backward, env)
    kernel.f[] = resolve(spec.forward, env)
    backward!(V_start, kernel, V_end)    # a · V_end
    return (V_start, kernel)
end

# forward! (f · Λ_start, the seated forward scale) is the generic modern default (abstract.jl)
# routed through the kernel's forward verb.


#####################################################################
# Derivative-carrying representation (GriddedWithDerivativesLayout) #
#####################################################################
# Phase 2, not implemented. Placeholder.


###################################################
# Dynamic-grid representation (DynamicGridLayout) #
###################################################
# Phase 2, not implemented. Placeholder.
