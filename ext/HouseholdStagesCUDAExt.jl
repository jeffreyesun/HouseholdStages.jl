################################################################
# HouseholdStagesCUDAExt — the stratified driver on the device #
################################################################
# One kernel serves every stage: a thread takes one stratum, slices its payloads with the package's
# own `_slice`, and runs the op body through those views.
#
#TODO libdevice and openlibm `erf` differ at ulp level, amplified through Newton (budgeted in the
# GPU-test MPS_*_ATOLs); if a CUDA version drops the SpecialFunctionsExt override, share one
# in-package Cody-rational `_erfc` across both backends.

module HouseholdStagesCUDAExt

import HouseholdStages as HS
using CUDA

# Move a host array payload to the device. Device arrays, scalars and `nothing` pass through; a
# `MatrixField` crosses as a `FiberFace`.
_as_device(x::CuArray) = x
_as_device(x::AbstractArray) = CuArray(collect(x))
_as_device(f::HS.MatrixField) = HS.FiberFace(f)
_as_device(x) = x

"""
One thread per stratum, slicing each payload where it lies, with the operative axis at any position.
The trailing payloads are splatted, so ops taking different numbers of them all reach their own
signature.
"""
function _fiber_kernel!(op, adim::Val, out, rest::Vararg{Any,M}) where {M}
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    strata = HS._strata(size(out), adim)
    if iv <= length(strata)
        @inbounds c = strata[iv]
        op(HS._slice(out, c, adim), map(p -> HS._slice(p, c, adim), rest)...)
    end
    return
end

# The op rides into the kernel by value, so its captures — a user cost closure, `env` — must be isbits.
#TODO `isbits` over the whole op is stricter than needed for an array-carrying env the cost never
# reads; split the env if that ever bites.
"GPU implementation of the package's stratified driver — one thread per stratum, in 256-thread blocks."
function HS._stratified!(op::HS.AbstractFiberOp, adim::Val, out::CuArray,
                         rest::Vararg{Any,M}) where {M}
    isbits(op) || error("$(typeof(op)) rides into the kernel by value, so its fields must be " *
                        "isbits; a cost closure or env capturing an array keeps its stage on host arrays.")
    nstrata = length(HS._strata(size(out), adim))
    @cuda threads=256 blocks=cld(nstrata, 256) _fiber_kernel!(op, adim, out,
                                                              map(_as_device, rest)...)
    return out
end

end # module HouseholdStagesCUDAExt
