# Mixing and retention — a chosen lottery over two transitions, at convex cost in the mixing weight θ.

#TODO Refactor this. Maybe make MixingStage a primitive or combinator if it comes with its own spec and kernel.
# RetentionStage is the true derived stage. And maybe also move the kernel to `kernels/`.

"Quadratic-cost pair on θ ∈ [0,1]: `c(θ) = θ²/(2κ)`, policy `θ*(y) = clamp(κy, 0, 1)`."
_quad_cost(κ)   = (θ; env) -> θ^2 / (2κ)
_quad_policy(κ) = (y; env) -> clamp(κ * y, 0.0, 1.0)

"""
Mixing choice over a blend `θ ∈ [0,1]` of two row-stochastic `axis` transitions, `K_A` at `θ = 1`
and `K_B` at `θ = 0`, at convex cost. `cost(θ; env)` and `policy(y; env)` default to the quadratic
pair `θ²/(2·cost_curvature)` and `clamp(cost_curvature·y, 0, 1)`; `policy` must be the argmax of
`θ·y − cost(θ)` on [0,1].
"""
struct MixingStageSpec{Ct, Pl} <: AbstractStageSpec
    axis   :: Symbol
    K_A    :: Matrix{Float64}
    K_B    :: Matrix{Float64}
    cost   :: Ct
    policy :: Pl
end

function MixingStageSpec(; axis::Symbol, K_A, K_B, cost_curvature::Real=1.0,
                         cost=_quad_cost(Float64(cost_curvature)),
                         policy=_quad_policy(Float64(cost_curvature)))
    A = Matrix{Float64}(K_A); B = Matrix{Float64}(K_B)
    @assert size(A) == size(B) "MixingStage: K_A is $(size(A)) and K_B is $(size(B)); the two " *
        "mixed kernels span one pair of ends, so they must have the same shape."
    for (nm, K) in ((:K_A, A), (:K_B, B))
        @assert all(>=(0), K) && all(<=(1 + 1e-8), sum(K; dims=2))
    end
    _mixing_validate(cost, policy)
    return MixingStageSpec(axis, A, B, cost, policy)
end

@definestage MixingStage MixingStageSpec

"""
Retention — a [`MixingStage`](@ref) with `K_A = I`: pay a convex cost to hold a `θ` share of each
cell in place instead of sending it through `exit_kernel`.
"""
RetentionStage(layout::GriddedLayout; axis::Symbol, exit_kernel, kwargs...) =
    MixingStage(layout; axis, K_A=_mixing_identity(_axis_size(layout, axis)),
                K_B=exit_kernel, kwargs...)

_mixing_identity(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

"""error unless `policy` is the argmax of `θ·y − cost(θ)` on [0,1] at a few probe points."""
function _mixing_validate(cost, policy)
    probes = try
        [(y, Float64(policy(y; env=nothing)),
             Float64(cost(Float64(policy(y; env=nothing)); env=nothing)),
             [θ * y - Float64(cost(θ; env=nothing)) for θ in 0.0:0.125:1.0])
         for y in (-0.9, -0.2, 0.0, 0.2, 0.9, 3.0)]
    catch e
        e isa Union{MethodError, KeyError, TypeError, UndefKeywordError} ||
            occursin("has no field", sprint(showerror, e)) || rethrow()
        return nothing
    end
    for (y, θs, cs, grid_vals) in probes
        0.0 <= θs <= 1.0 ||
            error("MixingStage: policy($y) = $θs ∉ [0, 1].")
        θs * y - cs >= maximum(grid_vals) - 1e-8 ||
            error("MixingStage: policy is not the argmax of θ·y − cost(θ) at y = $y " *
                  "(policy gives $(θs * y - cs), a grid θ beats it at $(maximum(grid_vals))).")
    end
    return nothing
end


##########################
# Gridded implementation #
##########################

"""the two dense transitions plus the per-cell weight `θstar`, the share of each cell routed through `kA`."""
struct MixingKernel{KA, KB, P<:AbstractArray}
    kA    :: KA
    kB    :: KB
    θstar :: P
end

"scratch for the two dense applies, plus the blend workspaces `mixA` (start-shaped) and `mixB` (end-shaped)."
kernel_scratch(k::MixingKernel, start_layout::GriddedLayout, end_layout::GriddedLayout, ::Type{T}) where {T} =
    (A = kernel_scratch(k.kA, start_layout, end_layout, T),
     B = kernel_scratch(k.kB, start_layout, end_layout, T),
     mixA = allocate_buffer(start_layout, T),
     mixB = allocate_buffer(end_layout, T))

operative_axis(spec::MixingStageSpec) = spec.axis
tangent_grade(::MixingStageSpec)     = :exact_ae

function allocate_kernel(spec::MixingStageSpec, ::Type{T}, start_layout::GriddedLayout,
                         end_layout::GriddedLayout) where {T}
    seat(K) = begin
        src = MappedField(permutedims, K)
        k   = dense_kernel(T, start_layout, end_layout, spec.axis, src)
        fill_field!(k, src, start_layout, spec.axis, nothing)
        k
    end
    return MixingKernel(seat(spec.K_A), seat(spec.K_B), zeros(T, layout_size(start_layout)))
end

function forward!(Λ_end, k::MixingKernel, Λ_start; scratch)
    θ = k.θstar
    @. scratch.mixA = θ * Λ_start
    forward!(Λ_end, k.kA, scratch.mixA; scratch=scratch.A)
    @. scratch.mixA = (1 - θ) * Λ_start
    forward!(scratch.mixB, k.kB, scratch.mixA; scratch=scratch.B)
    @. Λ_end = Λ_end + scratch.mixB
    return Λ_end
end

function backward!(V_start, k::MixingKernel, V_end; scratch)
    θ = k.θstar
    backward!(scratch.mixA, k.kA, V_end; scratch=scratch.A)
    backward!(V_start, k.kB, V_end; scratch=scratch.B)
    @. V_start = θ * scratch.mixA + (1 - θ) * V_start
    return V_start
end

function backward!(V_start, spec::MixingStageSpec, start_layout::GriddedLayout,
                   ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool=true)
    a = scratch.kernel_scratch.mixA
    backward!(a, kernel.kA, V_end; scratch=scratch.kernel_scratch.A)        # a = K_Aᵀ·V_end
    backward!(V_start, kernel.kB, V_end; scratch=scratch.kernel_scratch.B)  # b = K_Bᵀ·V_end
    polf = spec.policy; costf = spec.cost
    pol = y -> polf(y; env)
    cst = θ -> costf(θ; env)
    @. a = a - V_start                                  # y = a − b
    @. kernel.θstar = pol(a)                            # θ* = (c′)⁻¹(y), at the buffer eltype
    @. V_start = V_start + _frz(kernel.θstar) * a - cst(_frz(kernel.θstar))   # V = b + θ*y − c(θ*), envelope
    return (V_start, kernel)
end

"the mixing weight θ* ∈ [0,1] chosen by the last `backward!`, the share routed through `K_A`."
policy(stage::MixingStage) = stage.kernel.θstar
