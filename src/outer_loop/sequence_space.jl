###################################
# Sequence-space Jacobians        #
###################################
#
# The fake-news algorithm at the steady state: anticipation sweeps, expectation vectors, assembly.

"""
Expectation vectors `ℰ_t = (𝐊ᵀ)^t f` of `integrand`, for `t = 0,…,T-1`. Requires the chain's kernels
already seated by `backward!(chain, V_terminal, env_ss)`.
"""
expectation_vectors(chain::AbstractStage, integrand::Function, T::Int) =
    expectation_vectors(chain, integrand.(cell_array(end_layout(chain))), T)

"The same recursion from a seed `ℰ₀` materialized over the end layout, converted to the chain's eltype."
function expectation_vectors(chain::AbstractStage, E0::AbstractArray, T::Int)
    @assert T ≥ 1 "expectation_vectors: T must be at least 1"
    _assert_block_closes(start_layout(chain), end_layout(chain), "expectation_vectors")
    E_prev  = eltype_from_chain(chain).(E0)
    results = [copy(E_prev)]
    for _ in 2:T
        E_next = forward_adjoint!(chain, E_prev)
        push!(results, copy(E_next))
        E_prev = E_next
    end
    return results
end

eltype_from_chain(s::AbstractStage) = eltype(V_start_buffer(s))

"Cumulate the fake-news matrix `F` into a sequence-space Jacobian along anti-diagonals."
function J_from_F(F::AbstractMatrix)
    T_lookahead, T = size(F)
    J = copy(F)
    for t in 2:T
        for s in 1:(T_lookahead - 1)
            J[s + 1, t] += J[s, t - 1]
        end
    end
    return J
end

"Assemble the fake-news matrix from `curlyY[t]`, `curlyD[t]` and the `s`-ahead `curlyE[s]`."
function build_F(curlyY::AbstractVector, curlyD::AbstractVector,
                 curlyE::AbstractVector)
    T = length(curlyY)
    T_lookahead = length(curlyE) + 1
    @assert length(curlyD) == T "curlyD must have the same length as curlyY (one per shock time)"
    F = zeros(eltype(curlyY), T_lookahead, T)
    for t in 1:T
        F[1, t] = curlyY[t]
        for s in eachindex(curlyE)
            F[s + 1, t] = sum(curlyE[s] .* curlyD[t])
        end
    end
    return F
end


# The driver #
#------------#

"""
Sequence-space Jacobians of the chain's moments in its `env` inputs: `J[t, s] = ∂y_t/∂env_s` over a
`T`-period horizon, rows output dates and columns shock dates (index 1 is date 0 on both). Every
moment must reduce with `sum`; the steady state at `env_ss` is solved here and left in the buffers.
Returns a `Dict` keyed by `(input, output)`, bare for a single pair.
"""
function compute_fake_news_ssj(stage::AbstractStage, env_ss, T::Int;
                               inputs=nothing, outputs=nothing,
                               mode::Symbol=:fd, h::Real=1e-5, n_dual::Int=4)
    (; V_ss, Λ_ss, in_names, out_names) =
        _derivative_service_setup("compute_fake_news_ssj", stage, env_ss, inputs, outputs, mode)

    curlyE = Dict(o => expectation_vectors(stage, _moment_seed(stage, o, env_ss), T) for o in out_names)

    jacs = Dict{Tuple{Symbol, Symbol}, Matrix{Float64}}()
    for chunk in Iterators.partition(in_names, mode === :dual ? n_dual : 1)
        YDs = _fake_news_YD(stage, env_ss, V_ss, Λ_ss, T, Tuple(chunk), Val(mode); h, out_names)
        for (input, (; curlyD, curlyY)) in zip(chunk, YDs), output in out_names
            jacs[(input, output)] = J_from_F(build_F(curlyY[output], curlyD, curlyE[output][2:end]))
        end
    end
    # Leave the buffers at the steady state.
    copyto!(V_start_buffer(stage), V_ss)
    copyto!(Λ_end_buffer(stage),   Λ_ss)
    return _wrap_result(jacs)
end

"""
Check the preconditions, seat the steady state of `env_ss`, and resolve the input/output name tuples.
"""
function _derivative_service_setup(who::AbstractString, stage::ChainStage, env_ss,
                                   inputs, outputs, mode::Symbol)
    moments = stage.spec.moments
    @assert !isempty(moments) "$who: the chain has no moments attached; call define_moment! first."
    for (name, mspec) in moments
        mspec.reduce === sum ||
            error("$who: moment :$name reduces with `$(mspec.reduce)`; the assembly is bilinear in " *
                  "(f, Λ), so every moment on the chain must use `reduce = sum`.")
    end
    mode in (:fd, :dual) || error("$who: mode = :$mode; the implemented modes are :fd and :dual.")
    _assert_block_closes(stage.buffer, who)
    _check_tangent_grade(who, stage.spec, mode)

    in_names  = inputs  === nothing ? Tuple(chain_env_names(stage)) : Tuple(inputs)
    out_names = outputs === nothing ? Tuple(keys(moments))          : Tuple(outputs)
    isempty(in_names) &&
        error("$who: the chain declares no env fields — it reads `env` only through user closures, " *
              "which the package does not introspect. Name the fields to differentiate, as in " *
              "`inputs = (:r,)`.")
    return (; _seat_base_point!(who, stage, env_ss)..., in_names, out_names)
end

"The tolerance the derivative services solve their own steady state to."
const SS_PRECONDITION_TOL = 1e-11

"""
Seat the steady state of `env_ss` in the chain's buffers and return it as `(; V_ss, Λ_ss)`.
"""
function _seat_base_point!(who::AbstractString, stage::ChainStage, env_ss)
    V, Λ = copy(V_start_buffer(stage)), copy(Λ_end_buffer(stage))
    seated = !all(iszero, Λ) && all(r -> r.residual ≤ r.bar,
                                    _steady_state_residuals(stage, env_ss, V, Λ))
    seated && return (; V_ss = V, Λ_ss = Λ)

    res = solve_steady_state_given_env!(stage, env_ss;
                                        vfi_tol    = SS_PRECONDITION_TOL,
                                        lambda_tol = SS_PRECONDITION_TOL,
                                        Λ_init     = all(iszero, Λ) ? nothing : Λ)
    _assert_at_steady_state(who, stage, env_ss, res.V, res.Λ)
    return (; V_ss = res.V, Λ_ss = res.Λ)
end

"Refuse a block that is not a chain."
_derivative_service_setup(who::AbstractString, stage::AbstractStage, args...) =
    error("$who: moments attach at the end of a chain, so a derivative service is keyed on " *
          "`ChainStage`; a $(nameof(typeof(stage))) carries no moment surface. Wrap it in a chain " *
          "— `IdentityStage(start_layout(block)) ∘ block` — to give it one.")

"""
How a derivative service handles a `:wrong_object` component: `:fd` warns and runs, `:dual` errors.
"""
function _check_tangent_grade(who::AbstractString, spec::AbstractStageSpec, mode::Symbol)
    hard = _wrong_object_names(spec)
    isempty(hard) && return nothing
    mode === :dual && error("$who: $hard is `:wrong_object` (see `tangent_grade`) — a hard argmax " *
        "has no tangent to seat, so no Dual-eltype chain over it exists. Use `mode = :fd` for the " *
        "h-fragile primal-difference estimate, or replace the stage by a `LogitChoiceStage` or a " *
        "`ContinuousArgmaxStage`, which do carry tangents.")
    @warn "$who: $hard is `:wrong_object` (see `tangent_grade`) — this Jacobian is h-fragile " *
        "within `h` of a switch."
    return nothing
end

"""
The type names of a spec's `:wrong_object` leaves, found by recursing through the composites.
"""
_wrong_object_names(spec::AbstractStageSpec) =
    tangent_grade(spec) === :wrong_object ? (nameof(typeof(spec)),) : ()
_wrong_object_names(spec::ChainStageSpec)   = _wrong_object_names(spec.stages)
_wrong_object_names(spec::ProductStageSpec) = _wrong_object_names(spec.components)
_wrong_object_names(specs::Tuple) =
    mapreduce(_wrong_object_names, (a, b) -> (a..., b...), specs; init=())

"Refuse a `(V_ss, Λ_ss)` outside the residual bars."
_assert_at_steady_state(who::AbstractString, stage::ChainStage, env_ss, V_ss, Λ_ss) =
    foreach(r -> _assert_fixpoint_residual(who, r),
            _steady_state_residuals(stage, env_ss, V_ss, Λ_ss))

"""
One `backward!` against `V` and one `forward!` against `Λ`, as `(what, verb, residual, bar)` each.
"""
function _steady_state_residuals(stage::ChainStage, env_ss, V, Λ)
    residual_V = _sup_norm_diff(backward!(stage, V, env_ss), V)
    copyto!(V_start_buffer(stage), V)
    residual_Λ = _sup_norm_diff(forward!(stage, Λ), Λ)
    copyto!(Λ_end_buffer(stage), Λ)
    return ((what = "value",        verb = "backward", residual = residual_V,
             bar  = 1e-6 * (1 + _finite_sup_norm(V))),
            (what = "distribution", verb = "forward",  residual = residual_Λ, bar = 1e-10))
end

"Report the failure when one of the two steady-state residuals exceeds its bar."
_assert_fixpoint_residual(who, r) =
    r.residual ≤ r.bar ||
        error("$who: solving the steady state at this env left the $(r.what) still short of a " *
              "fixed point — one `$(r.verb)!` moves it by $(r.residual), against a bar of " *
              "$(r.bar). The solve did not converge at `$(SS_PRECONDITION_TOL)`, which is a " *
              "property of the chain and this `env` rather than of the call: check that the " *
              "kernels contract (a discount factor below 1) and that the block is solvable here.")

"A single `(input, output)` pair unwraps to the bare matrix; several stay a Dict."
_wrap_result(jacs::Dict) = length(jacs) == 1 ? first(values(jacs)) : jacs

"""
The expectation recursion's seed `ℰ₀ = ∂y/∂Λ`: moment `name`'s integrand materialized full-size.
"""
function _moment_seed(stage::ChainStage, name::Symbol, env)
    out_layout = end_layout(stage)
    E0 = Array{eltype_from_chain(stage)}(undef, layout_size(out_layout))
    E0 .= _integrand_grid(out_layout, stage.spec.moments[name].integrand, env)
    return E0
end

"""
The `u = 0…T−1` anticipation loop, `env₀` then `env_ss`; `record!(u, image, moments)` must copy.
"""
function _anticipation_sweep!(record!, chain::ChainStage, env₀, env_ss, V₀, Λ_ss, T::Int)
    V   = copy(backward!(chain, V₀, env₀))
    img = forward!(chain, Λ_ss)
    record!(0, img, compute_moments(chain, img, env₀))
    for u in 1:(T - 1)
        V   = copy(backward!(chain, V, env_ss))
        img = forward!(chain, Λ_ss)
        record!(u, img, compute_moments(chain, img, env_ss))
    end
    return nothing
end

"""
One `:fd` lane: a fresh `Float64` chain swept over the `T` distances from `env₀ = env_ss + δ·e_input`.
"""
function _fd_lane(stage::ChainStage, env_ss, V_ss, Λ_ss, T::Int, input::Symbol, δ::Real)
    chain = ChainStage(stage.spec, boundaries(stage), interiors(stage))
    imgs  = Vector{typeof(Λ_ss)}(undef, T)
    moms  = Vector{NamedTuple}(undef, T)
    _anticipation_sweep!(chain, _perturb(env_ss, input, δ), env_ss, V_ss, Λ_ss, T) do u, img, m
        imgs[u + 1] = copy(img)
        moms[u + 1] = m
    end
    return (imgs, moms)
end

"""
The lanes for `mode = :fd`: two primal sweeps at `env_ss ± h`, differenced into `𝒟_u` and `𝒴_u`.
"""
_fake_news_YD(stage::ChainStage, env_ss, V_ss, Λ_ss, T::Int, inputs::Tuple, ::Val{:fd};
              h::Real, out_names) =
    map(inputs) do input
        (imgs_p, moms_p) = _fd_lane(stage, env_ss, V_ss, Λ_ss, T, input, +h)
        (imgs_m, moms_m) = _fd_lane(stage, env_ss, V_ss, Λ_ss, T, input, -h)
        (; curlyD = [(imgs_p[u] .- imgs_m[u]) ./ 2h for u in 1:T],
           curlyY = Dict(o => [(moms_p[u][o] - moms_m[u][o]) / 2h for u in 1:T] for o in out_names))
    end

"The private tag the `:dual` lane gives its Duals."
struct FakeNewsProbe end
const FakeNewsTag = typeof(ForwardDiff.Tag(FakeNewsProbe(), Float64))

"""
The lanes for `mode = :dual`: one Dual-eltype chain swept once, reading `𝒟_u` and `𝒴_u` off partials.
"""
function _fake_news_YD(stage::ChainStage, env_ss, V_ss, Λ_ss, T::Int, inputs::Tuple, ::Val{:dual};
                       h::Real, out_names)
    TD    = ForwardDiff.Dual{FakeNewsTag, Float64, length(inputs)}
    chain = ChainStage(stage.spec, boundaries(stage), interiors(stage), TD)
    YD    = map(_ -> (; curlyD = Vector{typeof(Λ_ss)}(undef, T),
                        curlyY = Dict(o => Vector{Float64}(undef, T) for o in out_names)), inputs)
    _anticipation_sweep!(chain, _seed_env(env_ss, inputs, TD), env_ss,
                         TD.(V_ss), TD.(Λ_ss), T) do u, img, moms
        for (k, yd) in enumerate(YD)
            yd.curlyD[u + 1] = ForwardDiff.partials.(img, k)
            for o in out_names
                yd.curlyY[o][u + 1] = ForwardDiff.partials(moms[o], k)
            end
        end
    end
    return YD
end

"""
`env` lifted to the Dual eltype `TD`, with a unit partial in direction `k` on `inputs[k]`.
"""
function _seed_env(env, inputs::Tuple, ::Type{TD}) where {TD}
    N    = length(inputs)
    seed = ntuple(k -> TD(Float64(env[inputs[k]]), ForwardDiff.Partials(ntuple(j -> Float64(j == k), N))), N)
    return merge(env, NamedTuple{inputs}(seed))
end
