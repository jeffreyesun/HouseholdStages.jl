# PointwiseScale — the two-sided diagonal scale (discount, reproduction, renorm) #
#===============================================================================#
# `backward! = a·src` (scale V), `forward! = f·src` (scale Λ). The two scales are independent, so the
# transition is an adjoint pair iff `a == f`; the discount outlier (`a = β`, `f = 1`) is the
# asymmetric special case (MATH_CONTEXT §1). Each scale is a scalar or a `Base.RefValue` (an
# env-resolved / AD-shocked scale rewritten in place each pass, incl. `Dual`s), so a `FromEnv`/AD
# scale tracks env without rebuilding the kernel.

"""
The two-sided diagonal scale: `backward! = a·` (scale V), `forward! = f·` (scale Λ), each scale a
scalar or an env-rewritten `Base.RefValue`. A transpose pair iff `a == f`; the discount outlier
(`f = 1`) is the asymmetric case.
"""
struct PointwiseScale{A, F}
    a :: A    # backward scale (V_start = a · V_end)
    f :: F    # forward scale  (Λ_end  = f · Λ_start)
end

"Unwrap a possibly-`Ref`-backed (env-rewritten) scale."
_unref(x)                = x
_unref(r::Base.RefValue) = r[]

"Read the backward scale, unwrapping a `Ref`-backed scale (the discount `β` accessor)."
_scale(s::PointwiseScale)         = _unref(s.a)
"Read the forward scale, unwrapping a `Ref`-backed scale."
_forward_scale(s::PointwiseScale) = _unref(s.f)

forward!(dest, s::PointwiseScale, src; scratch=nothing)  = (f = _forward_scale(s); @. dest = f * src; dest)
backward!(dest, s::PointwiseScale, src; scratch=nothing) = (a = _scale(s);         @. dest = a * src; dest)

# BackwardScale(β) — the discount: a `PointwiseScale` with backward `β`, forward `1` (the
# *undiscounted* population is pushed forward). Kept as a constructor so call sites/tests that
# name the discount outlier read intent; the kernel data is the general two-sided scale.
BackwardScale(β) = PointwiseScale(β, 1)
