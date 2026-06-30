###############################################################
# Outer-loop public API — Stage-keyed                          #
###############################################################
#
# Public, Stage-keyed layer over the Spec/Buffer-keyed primitives in
# `outer_loop_internal.jl` (same function names; multiple dispatch routes
# the call). Most are thin delegates (V-only, Λ-only, transition, Jacobian);
# the one exception is the `solve_steady_state_given_env!` bundle, which
# absorbs the bookkeeping the primitive omits — warm-start from the chain's
# buffer, write converged V/Λ back, compute moments, return caller-safe copies.
#
# Closing-the-model logic (tâtonnement on K / r, AR(1) shock
# generation) stays with the consumer.


# Inner V fixed point — public delegate #
#---------------------------------------#

"""
Stage-keyed VFI: warm-start `V` from the buffer (or `V_init`), run the
Spec/Buffer-keyed primitive, write the converged `V` back for the next call's
warm-start, and return a caller-safe copy.
"""
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


# Inner Λ fixed point — public delegate #
#---------------------------------------#

"""
Stage-keyed Λ-iteration. `Λ_init` defaults to the uniform distribution — Λ
converges fast from uniform, and the buffer's `Λ_end` slot can hold half-iterated
state, so it makes a poor warm-start. Writes the converged Λ back; returns a copy.
"""
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
Solve the household block at one `env`. Wraps the Spec/Buffer-keyed primitive
with the bookkeeping it omits: warm-start `V` from the buffer, write converged
`V`/`Λ` back, compute attached moments, and return caller-safe copies. `history`
reports the V- and Λ-iteration counts; `iters` is their sum.
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


# Transition path — public delegate #
#-----------------------------------#

"""
Stage-keyed transition driver — a thin delegate to the Spec-keyed primitive,
which allocates per-period chains and runs the backward+forward sweep. `Λ_0` is
the initial distribution; `V_T` the terminal continuation value.
"""
solve_transition_given_env_path!(stage::ChainStage, env_path::AbstractVector;
                                 Λ_0::AbstractArray,
                                 V_T::AbstractArray,
                                 max_inner_iters::Int=1) =
    solve_transition_given_env_path!(stage.spec, env_path;
                                     Λ_0, V_T,
                                     layout=stage.buffer.input_layout,
                                     max_inner_iters)


# Direct-effect Jacobian — public delegate #
#------------------------------------------#

"""
Stage-keyed **direct-effect-only** Jacobian: the period-0 direct effect of an env
perturbation (`curlyY_0` by two-sided FD at the steady state), written on the
diagonal `s = t` with zeros off-diagonal. It does NOT compute the distribution-
mediated effects (`curlyD`, `curlyE`) — for the full sequence-space Jacobian use
`expectation_vectors`, `build_F`, and `J_from_F` (see `examples/aiyagari_mit_shock/ssj.jl`).
Treating this as a real fake-news Jacobian would produce wrong IRFs.
"""
compute_direct_jacobian!(stage::ChainStage, env_ss, T::Int;
                         inputs    = nothing,
                         outputs   = nothing,
                         eps::Real = 1e-5) =
    compute_direct_jacobian!(stage.spec, env_ss, T, stage.buffer;
                             inputs, outputs, eps)
