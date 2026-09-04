"""
Argmax over one named choice `axis`: each origin cell takes the destination `a` maximising
`reward[a, origin] + V_end[a]`. `reward` is a matrix source `M[destination, origin]` — a `Matrix`, a
`FromEnv`, or a `(; dep…[, env]) -> Matrix` closure; a non-finite entry marks that action
unavailable, and an origin with no available action keeps `typemin` and policy index 1. Carries no
discount.
"""
struct ArgmaxStageSpec{R} <: AbstractStageSpec
    reward :: R
    axis   :: Symbol
end

ArgmaxStageSpec(; reward, axis::Symbol) = ArgmaxStageSpec{typeof(reward)}(reward, axis)

@definestage ArgmaxStage ArgmaxStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(spec::ArgmaxStageSpec) = spec.axis
tangent_grade(::ArgmaxStageSpec)     = :wrong_object

# Kernel: a `ScatterKernel` over the chosen choice-axis index, one per origin cell.
allocate_kernel(spec::ArgmaxStageSpec, ::Type, start_layout::GriddedLayout, ::GriddedLayout) =
    ScatterKernel(zeros(Int, layout_size(start_layout)),
                  Val(axis_position(start_layout, spec.axis)))

"Scratch: io buffers plus the reward matrix `U`, stored destination-first as `U[destination, origin]`."
function allocate_scratch(spec::ArgmaxStageSpec, ::Type{T}, start_layout::GriddedLayout,
                          end_layout::GriddedLayout) where {T}
    U = matrix_field(T, start_layout, end_layout, spec.axis, spec.reward)
    reads_env(spec.reward) || fill_field!(U, spec.reward, start_layout, spec.axis, nothing)
    return merge(io_scratch(start_layout, end_layout, T), (U = U,))
end

"Cache: whether the reward depends on `env`."
allocate_cache(spec::ArgmaxStageSpec, ::Type, ::GriddedLayout, ::GriddedLayout) =
    (reward_env_dep = reads_env(spec.reward),)

# The best-so-far comparison runs on the primal (`_frz`) while the stored value keeps its tangents.
function _brute_walk!(Vs_col, pol_col, Vc_col, u_slice)
    T = eltype(Vs_col)
    @inbounds for s in 1:length(Vs_col)
        best_v = typemin(T); best_p = _frz(best_v); best_a = 1
        for a in 1:length(Vc_col)
            v  = u_slice[a, s] + Vc_col[a]
            vp = _frz(v)
            vp > best_p && (best_p = vp; best_v = v; best_a = a)
        end
        Vs_col[s] = best_v; pol_col[s] = best_a
    end
    return
end

"Backward op: the `(max, +)` walk over one stratum's reward matrix."
struct BruteSolveOp <: AbstractFiberOp end
(::BruteSolveOp)(Vs_col, pol_col, Vc_col, u_slice) = _brute_walk!(Vs_col, pol_col, Vc_col, u_slice)

"Fused brute argmax over a dep-free reward, for a non-leading operative axis on host arrays."
function _brute_smallaxis!(V_start, policy, V_end, u, pre::Int, n_start::Int, n_end::Int)
    T    = eltype(V_start)
    post = length(V_start) ÷ (pre * n_start)
    Vs = reshape(V_start, pre, n_start, post)
    Ve = reshape(V_end,   pre, n_end,   post)
    P  = reshape(policy,  pre, n_start, post)
    @inbounds for p in 1:post, s in 1:n_start
        @simd for i in 1:pre
            Vs[i, s, p] = typemin(T)
            P[i, s, p]  = 1
        end
        for a in 1:n_end
            uas = u[a, s]
            isfinite(uas) || continue             # skip an unavailable action
            @simd for i in 1:pre
                v      = uas + Ve[i, a, p]
                better = _frz(v) > _frz(Vs[i, s, p])
                Vs[i, s, p] = ifelse(better, v, Vs[i, s, p])
                P[i, s, p]  = ifelse(better, a, P[i, s, p])
            end
        end
    end
    return
end

"Solve the argmax, returning `(V_start, kernel)` with the chosen index stored as the kernel's destination."
function backward!(V_start, spec::ArgmaxStageSpec, start_layout::GriddedLayout,
                   end_layout::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    policy = destinations(kernel)
    cache.reward_env_dep && env_changed && fill_field!(scratch.U, spec.reward, start_layout, spec.axis, env)
    odep_dims      = scratch.U.dep_dims      # layout positions of the axes the reward varies along
    operative_dim  = axis_position(start_layout, spec.axis)
    n_start, n_end = operative_sizes(start_layout, end_layout, spec.axis)
    pre = prod(layout_size(start_layout)[k] for k in 1:operative_dim-1; init = 1)  # block below the operative dim
    # Dep-free reward, host array, non-leading operative axis: the fused unit-stride scan.
    if isempty(odep_dims) && pre > 1 && V_start isa Array
        u = reshape(scratch.U.array, n_end, n_start)
        _brute_smallaxis!(V_start, policy, V_end, u, pre, n_start, n_end)
        return (V_start, kernel)
    end
    stratified!(BruteSolveOp(), V_start, policy, V_end, scratch.U; dims=Val(operative_dim))
    return (V_start, kernel)
end

# forward! (push each cell's mass to its chosen index) is the generic default.

"The solved policy: the chosen `axis` index per cell."
policy(stage::ArgmaxStage) = destinations(stage.kernel)
