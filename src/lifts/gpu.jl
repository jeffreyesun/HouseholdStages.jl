###############################################
# to_device / lift_gpu — move a stage to a GPU #
###############################################
#
# Rebuild a constructed stage so its hot-path arrays live on a device, after which
# `backward!`/`forward!` dispatch through the device's broadcast/BLAS paths. CUDA is
# not a dependency: the move is parameterised on a caller-supplied `move_fn`
# (`CUDA.cu`). A field moves iff it is an array with an isbits eltype; non-isbits
# arrays (a `Symbol`-celled `cell_array`) stay host-side.

"""
Move `x` onto the device with `move_fn` if it's an array with isbits eltype;
otherwise pass through.
"""
_to_device_field(x::AbstractArray, move_fn) =
    isbitstype(eltype(x)) ? move_fn(x) : x
_to_device_field(x, move_fn) = x

# A dense self-describing kernel is a `PermutedDimsArray`: move its compact parent and
# rewrap with the same presentation perm (the generic array path would materialise it
# dense on the device, losing the compact-parent layout the contraction relies on).
_to_device_field(M::PermutedDimsArray{T, N, perm}, move_fn) where {T, N, perm} =
    PermutedDimsArray(_to_device_field(parent(M), move_fn), perm)

"""
Rebuild a kernel/scratch struct (or `nothing`) with each field moved. Reconstructs
the unparameterised type, so array fields must be declared on an `<:AbstractArray`
type parameter to accept the device-typed fields.
"""
_struct_to_device(::Nothing, move_fn) = nothing
function _struct_to_device(s, move_fn)
    T = typeof(s)
    isstructtype(T) || return _to_device_field(s, move_fn)
    fields = ntuple(i -> _to_device_field(getfield(s, i), move_fn), fieldcount(T))
    return T.name.wrapper(fields...)
end

# Modern scratch/cache are NamedTuples — reconstruct keyed (a positional `NamedTuple(...)`
# constructor doesn't exist), moving each field. A field that is itself a NamedTuple (the
# nested `kernel_scratch` plan) recurses through `_to_device_field` below.
_struct_to_device(nt::NamedTuple, move_fn) =
    NamedTuple{keys(nt)}(map(v -> _to_device_field(v, move_fn), values(nt)))

# A nested NamedTuple (e.g. the `kernel_scratch` plan holding the device-resident grid/buffers)
# is moved field-by-field — without this the inner plan would pass through host-side.
_to_device_field(nt::NamedTuple, move_fn) = _struct_to_device(nt, move_fn)

# Each kernel type moves its own data fields (dispatched below); the dense PermutedDimsArray
# and the scalar transitions (`I`/`BackwardScale`) are handled by the methods above / the
# generic `_to_device_field` pass-through.

# The single-destination kernel is its `destinations` array + its `axis` Val (host-side) — move
# the destinations, keep the axis (the grid lives in the kernel_scratch plan, moved by the
# NamedTuple mover above).
_to_device_field(k::SingleDestinationKernel, move_fn) =
    SingleDestinationKernel(_to_device_field(k.destinations, move_fn), k.axis)

# The logit kernel is its eψC/value_weight/normalizer factors — move them (eψC is a dense
# PermutedDimsArray, routed through the parent-aware mover above; the gather work-buffers live
# in scratch, moved by the scratch NamedTuple mover).
_to_device_field(k::LogitChoiceKernel, move_fn) =
    LogitChoiceKernel(_to_device_field(k.eψC, move_fn),
                      _to_device_field(k.value_weight, move_fn),
                      _to_device_field(k.normalizer, move_fn))

# SearchMatching's bespoke kernel: move policy/p, keep the scalar δ Ref host-side.
_to_device_field(k::SearchMatchingKernel, move_fn) =
    SearchMatchingKernel(_to_device_field(k.policy, move_fn),
                         _to_device_field(k.p, move_fn), k.δ)

"""
Move a Spec's array fields to the device (Markov/Logit `mul!` them against device
data). Reconstructs the Spec through its constructor so the type params re-infer; a
Spec with no array fields, or one the repack rejects, is returned unchanged.
"""
function _spec_to_device(spec::AbstractStageSpec, move_fn)
    any(getfield(spec, i) isa AbstractArray for i in 1:nfields(spec)) || return spec
    fns      = fieldnames(typeof(spec))
    devflds  = ntuple(i -> _to_device_field(getfield(spec, i), move_fn), nfields(spec))
    wrapper  = typeof(spec).name.wrapper
    try
        return wrapper(; NamedTuple{fns}(devflds)...)
    catch
    end
    try
        return wrapper(devflds...)
    catch
    end
    return spec
end

# Modern stage: move spec/kernel/scratch/cache; `layout` (grids) stays host-side.
#TODO Untested on device until a GPU-gated modern stage lands (P5); the seam is here
# so `to_device` doesn't fall through to the legacy `.buffer` path and error.
function to_device(stage::AbstractModernStage, move_fn)
    return typeof(stage).name.wrapper(
        _spec_to_device(stage.spec, move_fn),
        stage.layout,
        _to_device_field(stage.kernel, move_fn),
        _struct_to_device(stage.scratch, move_fn),
        _struct_to_device(stage.cache, move_fn),
    )
end

"""
Move a constructed `stage` onto a GPU via a caller-supplied mover: `lift_gpu(stage, cu)`.
The no-mover form errors rather than silently no-op'ing — CUDA is not a dependency.
"""
lift_gpu(stage::AbstractStage, move_fn) = to_device(stage, move_fn)
lift_gpu(stage::AbstractStage) =
    error("lift_gpu(stage): CUDA is not a dependency of HouseholdStages — " *
          "pass the device mover, e.g. `lift_gpu(stage, CUDA.cu)` or " *
          "`to_device(stage, CUDA.cu)`.")
