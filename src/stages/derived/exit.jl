# Exit with bequest — death or optimal stopping. Each sugar chains a stay-vs-exit choice, a
# `UtilityStage` paying the bequest on the exit slice, and a Markov drop removing exiters' mass; all
# three require `bequest` and a `layout` carrying `:exiting` at size 1.

"""raise the missing-`bequest` error shared by the three exit sugars."""
_exit_no_bequest(stage) = error(
    "$stage: `bequest` is required (no default). With CRRA σ>1, V<0 and the bliss point is 0, so a " *
    "default bequest of 0 makes the poor prefer death — you must supply the value of death/bequest.")

"""transition source returning the row `[s  1−s]` per dependency combination, `s` read off `survival`."""
struct ExitHazardSource{S}
    survival :: S
end

declared_deps(s::ExitHazardSource, dep_layout::GriddedLayout) = declared_deps(s.survival, dep_layout)
reads_env(s::ExitHazardSource) = reads_env(s.survival)
evaluate(s::ExitHazardSource, combo, env) = evaluate(s, combo, env, Val(reads_env(s)))
evaluate(s::ExitHazardSource, combo, env, envdep::Val) =
    (sv = evaluate(s.survival, combo, env, envdep); hcat(sv, oneunit(sv) - sv))

"""payoff array carrying `0` on the stay slice of `exiting` and the bequest on the exit slice."""
function _exit_bequest_payoff(bequest, layout::GriddedLayout, exiting::Symbol, ::Type{T}) where {T}
    L0  = drop_axis(layout, exiting)
    beq = materialize_scalar!(ScalarField(bequest, L0, T), bequest, L0, NamedTuple())
    L2  = grow_axis(layout, exiting, 2)
    strat = zeros(T, layout_size(L2))
    selectdim(strat, axis_position(L2, exiting), 2) .= beq
    return strat
end

"The `2 → 1` drop transition `T[from, to]`: stay (1) keeps its mass, exit (2) loses it."
_exit_drop_matrix(::Type{T}) where {T} = reshape(T[1, 0], 2, 1)

"""chain the stay-vs-exit `choice` with the bequest `UtilityStage` and the `2 → 1` drop."""
function _exit_chain(choice, layout::GriddedLayout, bequest, exiting::Symbol, ::Type{T}) where {T}
    L2      = grow_axis(layout, exiting, 2)
    utility = UtilityStage(L2; utility = _exit_bequest_payoff(bequest, layout, exiting, T))
    drop    = MarkovStage(L2, layout; axis = exiting, transition_matrix = _exit_drop_matrix(T))
    return ChainStage((choice, utility, drop))
end


"""
Exogenous exit at survival rate `survival` — a `Real`, a [`FromEnv`](@ref), or `(; dep…[, env]) -> s`:
backward `V_start = s·V_end + (1−s)·bequest`, forward `Λ_end = s·Λ_start`.
"""
function ExogenousExit(layout::GriddedLayout; survival, bequest = nothing, exiting::Symbol = :exiting)
    bequest === nothing && _exit_no_bequest("ExogenousExit")
    T = Float64
    choice = MarkovStage(layout, grow_axis(layout, exiting, 2);
                         axis = exiting, transition_matrix = ExitHazardSource(survival))
    return _exit_chain(choice, layout, bequest, exiting, T)
end

"""
Endogenous (hard) exit — optimal stopping: backward `V_start = max(V_end, bequest)`, forward only
the cells with `V_end ≥ bequest` survive.
"""
function EndogenousExit(layout::GriddedLayout; bequest = nothing, exiting::Symbol = :exiting)
    bequest === nothing && _exit_no_bequest("EndogenousExit")
    T = Float64
    choice = ArgmaxStage(layout, grow_axis(layout, exiting, 2); axis = exiting, reward = zeros(T, 2, 1))
    return _exit_chain(choice, layout, bequest, exiting, T)
end

"""
Smooth (logit) endogenous exit at scale `ε`: backward `V_start = ε·log(e^{V_end/ε} + e^{bequest/ε})`,
forward `Λ_end = p_stay·Λ_start` at `p_stay = e^{V_end/ε}/(e^{V_end/ε} + e^{bequest/ε})`.
"""
function LogitEndogenousExit(layout::GriddedLayout; bequest = nothing, ε = 1.0, exiting::Symbol = :exiting)
    bequest === nothing && _exit_no_bequest("LogitEndogenousExit")
    T = Float64
    choice = LogitChoiceStage(layout, grow_axis(layout, exiting, 2);
                              axis = exiting, cost_matrix = zeros(T, 1, 2), ε = ε)
    return _exit_chain(choice, layout, bequest, exiting, T)
end
