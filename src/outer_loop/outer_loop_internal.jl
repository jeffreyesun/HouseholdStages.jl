###############################################################
# Outer-loop primitives — Spec/Buffer-keyed                   #
###############################################################
#
# Fixed-point solves and the transition path, taking a Spec plus a Buffer explicitly, with no
# moments, no warm-start and no copy-back.

_buffer_V_start(buffer::ChainStageBuffer)   = V_start_buffer(buffer.stages[1])
_buffer_V_start(buffer::ProductStageBuffer) = buffer.V_fused

_buffer_Λ_end(buffer::ChainStageBuffer)   = Λ_end_buffer(buffer.stages[end])
_buffer_Λ_end(buffer::ProductStageBuffer) = buffer.Λ_fused

_default_V_init(buffer::AbstractStageBuffer) = zero(_buffer_V_start(buffer))

function _default_Λ_init(buffer::AbstractStageBuffer)
    dims = layout_size(start_layout(buffer))
    T    = eltype(_buffer_V_start(buffer))
    return fill(T(inv(prod(dims))), dims)
end


"Refuse a block whose two ends are not the same layout, testing layout equality rather than size equality."
_assert_block_closes(start_lay::AbstractLayout, end_lay::AbstractLayout, who::AbstractString) =
    start_lay == end_lay ||
        error("$who: a fixed point feeds the block's output back into its input, so its two ends " *
              "must be the same layout; this one runs $(layout_size(start_lay)) → $(layout_size(end_lay)).")

_assert_block_closes(buffer::AbstractStageBuffer, who::AbstractString) =
    _assert_block_closes(start_layout(buffer), end_layout(buffer), who)

"""
Sup-norm over the finite cells, skipping the infeasible ones.
"""
_finite_sup_norm(x) = mapreduce(v -> isfinite(v) ? abs(v) : zero(v), max, x; init = zero(eltype(x)))

"Sup-norm of `new − old` over the finite cells."
_sup_norm_diff(new, old) = _finite_sup_norm(new .- old)

"""
Refuse an eltype `T` that can hold a `Dual`, erroring under the caller's name `who`.
"""
_assert_no_tangents(::Type{T}, who::AbstractString) where {T} =
    typeintersect(T, ForwardDiff.Dual) !== Union{} && error(
        "$who: a Dual-eltype iterate is refused — this is a fixed point, and its derivative is not " *
        "the derivative of the loop. Differentiate a fixed point by implicit differentiation, or " *
        "use `compute_steady_state_gradient(mode = :fd)`, which re-solves the primal at `env ± h`.")

"""
Iterate `x ↦ step!(x, iters)` to `_sup_norm_diff ≤ tol`, returning `(; x, iters, converged)`.
"""
function _iterate_to_fixpoint(step!, x0; tol::Real, maxiter::Int, who::AbstractString)
    _assert_no_tangents(eltype(x0), who)
    x     = copy(x0)
    diff  = Inf
    iters = 0
    while diff > tol
        x_new = step!(x, iters)
        diff  = _sup_norm_diff(x_new, x)
        x    .= x_new
        iters += 1
        iters == maxiter && error("$who: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (; x, iters, converged=true)
end


# Inner fixed points #
#--------------------#

"Spec/Buffer-keyed VFI: backward-iterate `V` to its fixed point at a fixed `env`."
function solve_vfi_steady_state_given_env!(spec::AbstractStageSpec, env,
                                           buffer::AbstractStageBuffer;
                                           V_init=_default_V_init(buffer),
                                           tol::Real=1e-7,
                                           maxiter::Int=4000)
    _assert_block_closes(buffer, "solve_vfi_steady_state_given_env!")
    res = _iterate_to_fixpoint(V_init; tol, maxiter,
                               who="solve_vfi_steady_state_given_env!") do V, iters
        backward!(buffer, spec, V, env; env_changed = iters == 0)
    end
    return (; V=res.x, res.iters, res.converged)
end

"Spec/Buffer-keyed Λ-iteration to the un-renormalized stationary distribution; kernels must be seated."
function solve_lambda_steady_state_given_env!(spec::AbstractStageSpec,
                                              buffer::AbstractStageBuffer;
                                              Λ_init=_default_Λ_init(buffer),
                                              tol::Real=1e-6,
                                              maxiter::Int=20_000)
    _assert_block_closes(buffer, "solve_lambda_steady_state_given_env!")
    res = _iterate_to_fixpoint(Λ_init; tol, maxiter,
                               who="solve_lambda_steady_state_given_env!") do Λ, _
        forward!(buffer, spec, Λ)
    end
    return (; Λ=res.x, res.iters, res.converged)
end


# Bundle: V then Λ at a single env #
#----------------------------------#

"Spec/Buffer-keyed bundle of the two single-env inner solves."
function solve_steady_state_given_env!(spec::AbstractStageSpec, env,
                                       buffer::AbstractStageBuffer;
                                       V_init=_default_V_init(buffer),
                                       Λ_init=_default_Λ_init(buffer),
                                       vfi_tol::Real=1e-7,
                                       vfi_maxiter::Int=4000,
                                       lambda_tol::Real=1e-6,
                                       lambda_maxiter::Int=20_000)
    vfi = solve_vfi_steady_state_given_env!(spec, env, buffer;
                                            V_init, tol=vfi_tol, maxiter=vfi_maxiter)
    lam = solve_lambda_steady_state_given_env!(spec, buffer;
                                               Λ_init, tol=lambda_tol, maxiter=lambda_maxiter)
    return (; V=vfi.V, Λ=lam.Λ, vfi_iters=vfi.iters, lambda_iters=lam.iters)
end


# Transition path #
#-----------------#

"""
Spec-keyed deterministic transition along `env_path`, one chain per period. The working eltype is
read off `V_T`.
"""
function solve_transition_given_env_path!(spec::ChainStageSpec,
                                          env_path::AbstractVector;
                                          Λ_0::AbstractArray,
                                          V_T::AbstractArray,
                                          boundaries::Tuple,
                                          interiors::Tuple=_no_interiors(spec.stages))
    T_steps = length(env_path)
    @assert T_steps >= 1
    _assert_block_closes(first(boundaries), last(boundaries), "solve_transition_given_env_path!")

    Tel     = eltype(V_T)
    hh_path = [ChainStage(spec, boundaries, interiors, Tel) for _ in 1:T_steps]

    dims_V = size(V_T)
    dims_Λ = size(Λ_0)
    V_path = [zeros(Tel, dims_V) for _ in 1:T_steps+1]
    Λ_path = [zeros(Tel, dims_Λ) for _ in 1:T_steps+1]
    copyto!(V_path[T_steps+1], V_T)
    copyto!(Λ_path[1],         Λ_0)

    for t in T_steps:-1:1
        V_t = backward!(hh_path[t], V_path[t+1], env_path[t])
        copyto!(V_path[t], V_t)
    end
    for t in 1:T_steps
        Λ_t1 = forward!(hh_path[t], Λ_path[t])
        copyto!(Λ_path[t+1], Λ_t1)
    end

    moments_path = if isempty(spec.moments)
        [(;) for _ in 1:T_steps]
    else
        [compute_moments(hh_path[t], Λ_path[t+1], env_path[t]) for t in 1:T_steps]
    end

    return (; V_path, Λ_path, moments_path)
end


# Steady-state gradient engine #
#------------------------------#

"A copy of `env` with field `name` shifted by `amount`."
function _perturb(env, name::Symbol, amount::Real)
    @assert haskey(env, name)
    return merge(env, NamedTuple{(name,)}((getproperty(env, name) + amount,)))
end

"""
Re-solve both fixed points at `env`, warm-started from `(V_init, Λ_init)`, and return the moments.
"""
function _moments_at(spec::ChainStageSpec, buffer::ChainStageBuffer, env,
                     V_init, Λ_init; kwargs...)
    res = solve_steady_state_given_env!(spec, env, buffer;
                                        V_init=copy(V_init), Λ_init=copy(Λ_init), kwargs...)
    return compute_moments(spec, end_layout(buffer), res.Λ, env)
end

"""
`mode = :fd`: central differences of the steady state re-solved at `env_ss ± h·e_input`.
"""
function _ss_gradient_fd(stage::ChainStage, env_ss, V_ss, Λ_ss, inputs::Tuple, out_names;
                         h::Real, kwargs...)
    chain = ChainStage(stage.spec, boundaries(stage), interiors(stage))
    lane(input, δ) = _moments_at(chain.spec, chain.buffer, _perturb(env_ss, input, δ), V_ss, Λ_ss; kwargs...)
    ∂m = Dict{Tuple{Symbol, Symbol}, Float64}()
    for input in inputs
        m_up, m_dn = lane(input, +h), lane(input, -h)
        for o in out_names
            ∂m[(input, o)] = (m_up[o] - m_dn[o]) / 2h
        end
    end
    return ∂m
end

