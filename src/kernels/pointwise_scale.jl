# PointwiseScale — the two-sided diagonal scale (discount, reproduction, renorm) #
#===============================================================================#
# `backward! = a·src` (scale V), `forward! = f·src` (scale Λ), the two scales independent. Each is a
# scalar or a `Base.RefValue` rewritten in place each pass.

"The two-sided diagonal scale: `backward! = a·` (scale V), `forward! = f·` (scale Λ)."
struct PointwiseScale{A, F}
    a :: A    # backward scale (V_start = a · V_end)
    f :: F    # forward scale  (Λ_end  = f · Λ_start)
end

"Unwrap a possibly-`Ref`-backed (env-rewritten) scale."
_unref(x)                = x
_unref(r::Base.RefValue) = r[]

"Read the backward scale, unwrapping a `Ref`-backed scale."
_scale(s::PointwiseScale)         = _unref(s.a)
"Read the forward scale, unwrapping a `Ref`-backed scale."
_forward_scale(s::PointwiseScale) = _unref(s.f)

forward!(dest, s::PointwiseScale, src; scratch=nothing)  = (f = _forward_scale(s); @. dest = f * src; dest)
backward!(dest, s::PointwiseScale, src; scratch=nothing) = (a = _scale(s);         @. dest = a * src; dest)

# The discount: backward `β`, forward `1`.
BackwardScale(β) = PointwiseScale(β, 1)
