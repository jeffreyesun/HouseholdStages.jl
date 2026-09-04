###############################################################
# Outer-loop public API — Stage-keyed                          #
###############################################################
#
# Single-env solves warm-starting from the chain's buffer, the derivative services, and the
# transition delegate.

# Inner fixed points — public delegates #
#---------------------------------------#

"Stage-keyed VFI: warm-start from the chain's buffer unless `V_init` is given, and write `V` back."
function solve_vfi_steady_state_given_env!(stage::AbstractStage, env;
                                           V_init       = nothing,
                                           tol::Real    = 1e-7,
                                           maxiter::Int = 4000)
    V0 = if V_init === nothing
        V_buf = copy(V_start_buffer(stage))
        all(iszero, V_buf) ? zero(V_start_buffer(stage)) : V_buf
    else
        V_init
    end
    res = solve_vfi_steady_state_given_env!(stage.spec, env, stage.buffer;
                                            V_init = V0, tol, maxiter)
    copyto!(V_start_buffer(stage), res.V)
    return (; V = copy(res.V), iters = res.iters, converged = res.converged)
end

"Stage-keyed Λ-iteration from `Λ_init` or the uniform default, writing `Λ` back to the buffer."
function solve_lambda_steady_state_given_env!(stage::AbstractStage;
                                              Λ_init=nothing,
                                              tol::Real=1e-6,
                                              maxiter::Int=20_000)
    Λ0 = @something Λ_init _default_Λ_init(stage.buffer)
    res = solve_lambda_steady_state_given_env!(stage.spec, stage.buffer;
                                               Λ_init=Λ0, tol, maxiter)
    copyto!(Λ_end_buffer(stage), res.Λ)
    return (; Λ=copy(res.Λ), iters=res.iters, converged=res.converged)
end


# Bundle: V then Λ at a single env, with bookkeeping #
#----------------------------------------------------#

"""
Solve the household block at one `env`: `V`, then the stationary `Λ`, then the attached moments.
Warm-starts `V` from the chain's buffer and writes `V`/`Λ` back.
"""
function solve_steady_state_given_env!(stage::ChainStage, env;
                                       vfi_tol::Real       = 1e-7,
                                       vfi_maxiter::Int    = 4000,
                                       lambda_tol::Real    = 1e-6,
                                       lambda_maxiter::Int = 20_000,
                                       V_init              = nothing,
                                       Λ_init              = nothing)
    spec, buffer = stage.spec, stage.buffer

    V0 = if V_init === nothing
        V_buf = copy(V_start_buffer(stage))
        all(iszero, V_buf) ? _default_V_init(buffer) : V_buf
    else
        V_init
    end
    Λ0 = @something Λ_init _default_Λ_init(buffer)

    res = solve_steady_state_given_env!(spec, env, buffer;
                                        V_init = V0, Λ_init = Λ0,
                                        vfi_tol, vfi_maxiter,
                                        lambda_tol, lambda_maxiter)
    V, Λ = res.V, res.Λ

    copyto!(V_start_buffer(stage), V)
    copyto!(Λ_end_buffer(stage),   Λ)

    moments = isempty(spec.moments) ? (;) : compute_moments(stage, Λ, env)
    return (; V = copy(V), Λ = copy(Λ),
            moments,
            history = (vfi_iters = res.vfi_iters,
                       lambda_iters = res.lambda_iters),
            iters = res.vfi_iters + res.lambda_iters)
end


# Steady-state gradient — the permanent-shock comparative static #
#----------------------------------------------------------------#

"""
The permanent-shock comparative static `∂(steady-state moment)/∂env`, as a `Dict` keyed by
`(input, output)`. Every moment must reduce with `sum`; `mode = :fd` is the only mode.
"""
function compute_steady_state_gradient(stage::ChainStage, env_ss;
                                       inputs=nothing, outputs=nothing,
                                       mode::Symbol=:fd, h::Real=1e-5, kwargs...)
    mode === :fd || error(
        "compute_steady_state_gradient: mode = :$mode. The steady state is a fixed point, and " *
        "differentiating one is out of scope for this package — the derivative of a fixed point is " *
        "not the derivative of the loop that finds it. `:fd` re-solves at `env ± h`.")
    (; V_ss, Λ_ss, in_names, out_names) =
        _derivative_service_setup("compute_steady_state_gradient", stage, env_ss, inputs, outputs, mode)
    tols = merge((; vfi_tol = SS_PRECONDITION_TOL, lambda_tol = SS_PRECONDITION_TOL), values(kwargs))
    return _ss_gradient_fd(stage, env_ss, V_ss, Λ_ss, in_names, out_names; h, tols...)
end


# Transition path — public delegate #
#-----------------------------------#

"Stage-keyed deterministic transition along `env_path`, from initial distribution `Λ_0` to terminal value `V_T`."
solve_transition_given_env_path!(stage::ChainStage, env_path::AbstractVector;
                                 Λ_0::AbstractArray,
                                 V_T::AbstractArray) =
    solve_transition_given_env_path!(stage.spec, env_path;
                                     Λ_0, V_T,
                                     boundaries=boundaries(stage),
                                     interiors=interiors(stage))

"""
The direct method: one Dual replay of `env_path` gives `∂(moment path)/∂env_path[s][input]`, a
NamedTuple of vectors over output dates, one per moment — column `s` of the Jacobian at a flat path.
"""
function compute_direct_ssj(stage::ChainStage, env_path::AbstractVector;
                            input::Symbol, s::Int, Λ_0::AbstractArray, V_T::AbstractArray)
    @assert !isempty(stage.spec.moments) "compute_direct_ssj: the chain has no moments attached; call define_moment! first."
    @assert 1 ≤ s ≤ length(env_path) "compute_direct_ssj: shock date s = $s is outside the 1:$(length(env_path)) path."
    TD    = ForwardDiff.Dual{HhsLiftTag, Float64, 1}
    path  = [t == s ? _seed_env(env_path[t], (input,), TD) : env_path[t] for t in eachindex(env_path)]
    res   = solve_transition_given_env_path!(stage, path; Λ_0, V_T=TD.(V_T))
    names = Tuple(keys(stage.spec.moments))
    return NamedTuple{names}(map(o -> [ForwardDiff.partials(m[o], 1) for m in res.moments_path], names))
end

"Refuse a block that is not a chain by name."
compute_steady_state_gradient(stage::AbstractStage, args...; kwargs...) =
    _derivative_service_setup("compute_steady_state_gradient", stage)
compute_direct_ssj(stage::AbstractStage, args...; kwargs...) =
    _derivative_service_setup("compute_direct_ssj", stage)
