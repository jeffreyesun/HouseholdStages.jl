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

# A `MatrixField` / `DenseKernel` move their compact backing array onto the device, preserving the
# explicit operative-axis/dep metadata (host-side scalars). The wrappers are NOT `AbstractArray`s,
# so the generic array path above does not catch them — these explicit methods are required.
_to_device_field(f::MatrixField, move_fn) =
    MatrixField(_to_device_field(f.array, move_fn), f.operative_axis, f.operative_dim, f.dep_dims)
_to_device_field(k::DenseKernel, move_fn) = DenseKernel(_to_device_field(k.field, move_fn))

# A `ScalarField` (the deterministic-continuous destination cache, utility/entry payoff) moves its
# materialized `data` buffer onto the device, keeping the host-side `bshape`/`env_dependent` metadata.
# A later env-dependent refill then lands on-device through `fill_scalar_field!`'s host-staged copy
# (fields/scalar_field.jl) — the wrapper is NOT an `AbstractArray`, so this explicit method is required.
_to_device_field(f::ScalarField, move_fn) =
    (d = _to_device_field(f.data, move_fn); ScalarField{typeof(d)}(d, f.bshape, f.env_dependent))

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

# Each kernel type moves its own data fields (dispatched below); the dense `DenseKernel`/`MatrixField`
# and the scalar transitions (`I`/`BackwardScale`) are handled by the methods above / the
# generic `_to_device_field` pass-through.

# The single-destination kernels own a `DestinationField` (a `destinations` array + an `axis` Val,
# host-side) — move the destinations, keep the axis (the grid, for `InterpKernel`, lives in the
# kernel_scratch plan, moved by the NamedTuple mover above). One mover serves both kernel types via
# the shared field.
_to_device_field(f::DestinationField, move_fn) =
    DestinationField(_to_device_field(f.destinations, move_fn), f.axis)
_to_device_field(k::ScatterKernel, move_fn) = ScatterKernel(_to_device_field(k.dest, move_fn))
_to_device_field(k::InterpKernel, move_fn)  = InterpKernel(_to_device_field(k.dest, move_fn))

# The logit kernel is its eψC/value_weight/normalizer factors — move them (eψC is a contained
# `DenseKernel`, routed through the DenseKernel mover above; the gather work-buffers live
# in scratch, moved by the scratch NamedTuple mover).
_to_device_field(k::LogitChoiceKernel, move_fn) =
    LogitChoiceKernel(_to_device_field(k.eψC, move_fn),
                      _to_device_field(k.value_weight, move_fn),
                      _to_device_field(k.normalizer, move_fn))

# SearchMatching's bespoke kernel: move policy/p, keep the scalar δ Ref host-side.
_to_device_field(k::SearchMatchingKernel, move_fn) =
    SearchMatchingKernel(_to_device_field(k.policy, move_fn),
                         _to_device_field(k.p, move_fn), k.δ)

# The streaming kernel-choice kernels (`MPSKernel`/`MeanVarianceKernel`) move only their per-cell
# frozen float policy `θstar` to the device; the small host-side `grid`/`weights`/`landing` vectors
# are device-promoted at the seam (`HouseholdStagesCUDAExt`, like the WealthChange wgrid) and `adim`
# is a scalar. Without these the generic struct pass-through would leave `θstar` host and the device
# backward/forward would scalar-index.
_to_device_field(k::MPSKernel, move_fn) =
    MPSKernel(_to_device_field(k.θstar, move_fn), k.grid, k.weights, k.adim, k.landing)
_to_device_field(k::MeanVarianceKernel, move_fn) =
    MeanVarianceKernel(_to_device_field(k.θstar, move_fn), k.grid, k.weights, k.adim, k.landing)

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
Move a composite (`ChainStage`) onto the device: move each leaf sub-stage and rebundle through the
`∘`-time-ordered constructor. The chain's backward/forward sweeps drive `buffer.stages` directly, so
moving those (and re-deriving the spec/layouts from them) is the whole move; the layouts stay
host-side.
"""
to_device(stage::ChainStage, move_fn) =
    ChainStage(map(s -> to_device(s, move_fn), stage.buffer.stages))

"""
Move a `ProductStage` (the `⊕` direct sum) onto the device. Mirrors the `ChainStage` lift —
recursively move each bundled component sub-stage and keep the host-side layouts — with one extra
step: a `ProductStageBuffer` *owns* its fused `V`/`Λ` tensors (a `ChainStageBuffer` owns none), and
the per-component sweep slices each component's result into them, so they must ride along to the
device too. The spec (host-side component sub-specs, read only for `axis`/count in the sweep) is
carried unchanged. The buffer's array fields are `<:AbstractArray`-typed (direct_sum.jl) so the
device-resident fused tensors land without a host round-trip.
"""
function to_device(stage::ProductStage, move_fn)
    buf   = stage.buffer
    comps = map(s -> to_device(s, move_fn), buf.components)
    dbuf  = ProductStageBuffer(comps,
                               _to_device_field(buf.V_fused, move_fn),
                               _to_device_field(buf.Λ_fused, move_fn),
                               buf.input_layout, buf.output_layout)
    return ProductStage(stage.spec, dbuf)
end

"""
Move a `MixingStage` (and its `RetentionStage` `K_A = I` special case) onto the device. The two
blended transitions live in the bundled `markA`/`markB` MarkovStages — recurse `to_device` into each
(the modern dense `mul!` path) — and the buffer's own policy/mass-split/V_start/Λ_end scratch rides
along. The spec stays host-side: its `K_A`/`K_B` matrices are read only at allocate-time (the seated
kernels live in the sub-stages), and its pointwise `conjugate`/`policy`/`cost` closures broadcast
on-device over the device arrays. The buffer's scratch fields are `<:AbstractArray`-typed
(mixing.jl) so the device tensors land without a host round-trip.
"""
function to_device(stage::MixingStage, move_fn)
    buf  = stage.buffer
    dbuf = MixingStageBuffer(
        to_device(buf.markA, move_fn),
        to_device(buf.markB, move_fn),
        _to_device_field(buf.policy, move_fn),
        _to_device_field(buf.mass_share, move_fn),
        _to_device_field(buf.V_start, move_fn),
        _to_device_field(buf.Λ_end, move_fn),
        buf.input_layout, buf.output_layout)
    return MixingStage(stage.spec, dbuf)
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
