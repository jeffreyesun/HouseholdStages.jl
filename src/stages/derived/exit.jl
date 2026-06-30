# Exit with bequest — death / optimal-stopping as a COMPOSITE (end-goal §12), not a fused stage.
# Each variant is `ChoiceStage ∘ UtilityStage ∘ MarkovStage` over a transient `:exiting` axis the
# block layout declares at size 1 (the fixed-layout invariant, §3) and the composite threads
# `1 → 2 → 2 → 1`:
#   • ChoiceStage (rectangular 1 → 2 on :exiting) — the stay-vs-exit decision: a MarkovStage for an
#     exogenous hazard (s, 1−s), an ArgmaxStage for hard stopping, a LogitChoiceStage for smooth;
#   • UtilityStage — the bequest as an :exiting-stratified payoff (0 on the stay slice, b on the exit);
#   • MarkovStage (rectangular 2 → 1) — drops the exit slice, so exiters' mass leaves the population.
#
# Running BACKWARD (Markov → Utility → Choice — the chain sweeps its components in reverse) the Markov
# seats `[V_end, 0]` on {stay, exit}, the utility adds `[0, b]`, and the choice collapses `{V_end, b}`:
#   s·V_end + (1−s)·b   (exogenous Markov mixture),
#   max(V_end, b)        (argmax hard stopping),
#   ε·log(e^{V_end/ε} + e^{b/ε})   (logit smooth stopping).
# FORWARD the choice splits Λ over {stay, exit} and the Markov drop keeps only the stay mass, so the
# exiting fraction LEAVES the live population (Λ NOT conserved — by design). The auxiliary axis only
# ever reaches size two, so the transient materialisation is cheap and exit needs no hand-rolled kernel.
#
# `bequest` is REQUIRED (no default): with CRRA σ>1 the felicity is negative and the bliss point is 0,
# so a default bequest of 0 would make the poor strictly prefer death — the value of death is a real
# modelling choice the caller must make. The block layout must already carry the `:exiting` axis at
# size 1 (so exit composes with the other household stages on one shared layout).

"""
The error a missing `bequest` raises — names the CRRA-σ>1 "poor prefer death" trap so the caller
supplies the value of death rather than silently inheriting a `0.0` default.
"""
_exit_no_bequest(stage) = error(
    "$stage: `bequest` is required (no default). With CRRA σ>1, V<0 and the bliss point is 0, so a " *
    "default bequest of 0 makes the poor prefer death — you must supply the value of death/bequest.")

"""
The exogenous-hazard transition source for the growing `1 → 2` choice `MarkovStage`: per
dep-combination it returns the `1×2` row `[s  1−s]` (origin = the single alive state, destinations =
{stay, exit}), carrying that rectangular fiber shape explicitly because a closure/`FromEnv` return
shape is not inferable. `survival` is a scalar, a [`FromEnv`](@ref), or a dep closure
`(; dep…[, env]) -> s` — it rides the standard Sources `evaluate`, so per-cell / env-varying mortality
falls out (Markov stores `K = Tᵀ`, i.e. the column `[s; 1−s]`).
"""
struct ExitHazardSource{S}
    survival :: S
end

_field_shape(::ExitHazardSource, ::GriddedLayout, ::Symbol) = (1, 2)   # T = [s 1−s]; permutedims ⇒ K = (2,1)
declared_deps(s::ExitHazardSource, layout::GriddedLayout) = declared_deps(s.survival, layout)
reads_env(s::ExitHazardSource) = reads_env(s.survival)
evaluate(s::ExitHazardSource, combo, env) =
    (sv = evaluate(s.survival, combo, env); hcat(sv, oneunit(sv) - sv))

"""
The `:exiting`-stratified bequest payoff for the exit composite's `UtilityStage`: a layout-shaped
array on the grown (`:exiting = 2`) layout that is `0` on the stay slice and the materialised
`bequest` on the exit slice (so the choice compares `V_end` against `b`). The bequest (scalar / array
/ `FromEnv` / dep closure) is materialised once here against the original (no-`:exiting`) layout — it
is a value-of-death modelling primitive, taken env-independent.
"""
function _exit_bequest_payoff(bequest, layout::GriddedLayout, exiting::Symbol, ::Type{T}) where {T}
    L0  = drop_axis(layout, exiting)                              # the household axes (no :exiting)
    beq = materialize_scalar!(ScalarField(bequest, L0, T), bequest, L0, NamedTuple()) # scalar or L0-broadcastable
    L2  = grow_axis(layout, exiting, 2)
    strat = zeros(T, layout_size(L2))
    selectdim(strat, axis_position(L2, exiting), 2) .= beq        # exit slice (2) ← b; stay slice (1) stays 0
    return strat
end

"The `2×1` drop transition `T[from, to]`: stay (1) keeps its mass, exit (2) sinks — the rectangular `2 → 1` that removes the exiting fraction."
_exit_drop_matrix(::Type{T}) where {T} = reshape(T[1, 0], 2, 1)

"""
Assemble the exit composite `ChainStage((choice, utility, drop))` from a stay-vs-exit `choice` leaf and
the shared bequest `UtilityStage` + `2 → 1` drop `MarkovStage`, all on the grown `:exiting = 2` layout.
"""
function _exit_chain(choice, layout::GriddedLayout, bequest, exiting::Symbol, ::Type{T}) where {T}
    L2      = grow_axis(layout, exiting, 2)
    utility = UtilityStage(L2; utility = _exit_bequest_payoff(bequest, layout, exiting, T))
    drop    = MarkovStage(L2; axis = exiting, transition_matrix = _exit_drop_matrix(T))
    return ChainStage((choice, utility, drop))
end


"""
Exogenous exit at survival rate `s`: backward `V_start = s·V_end + (1−s)·bequest`, forward
`Λ_end = s·Λ_start` (the `(1−s)` fraction leaves — mass NOT conserved). The composite of §12 — the
choice is a `MarkovStage` with the `(s, 1−s)` hazard. `survival` is a `Real`, a [`FromEnv`](@ref), or
a dep closure `(; dep…[, env]) -> s` (age/health-dependent mortality); `bequest` is a REQUIRED
[`ScalarField`](@ref) source. The `layout` must carry the `:exiting` axis at size 1.
"""
function ExogenousExit(layout::GriddedLayout; survival, bequest = nothing, exiting::Symbol = :exiting)
    bequest === nothing && _exit_no_bequest("ExogenousExit")
    T = Float64
    choice = MarkovStage(layout; axis = exiting, transition_matrix = ExitHazardSource(survival))
    return _exit_chain(choice, layout, bequest, exiting, T)
end

"""
Endogenous (hard) exit — optimal stopping: backward `V_start = max(V_end, bequest)`, forward the
survivors are the cells with `V_end ≥ bequest` (the rest LEAVES). The composite of §12 — the choice is
an `ArgmaxStage` collapsing the `:exiting` axis by max. The `ε → 0` limit of
[`LogitEndogenousExit`](@ref). `bequest` is a REQUIRED [`ScalarField`](@ref) source; the `layout` must
carry the `:exiting` axis at size 1.
"""
function EndogenousExit(layout::GriddedLayout; bequest = nothing, exiting::Symbol = :exiting)
    bequest === nothing && _exit_no_bequest("EndogenousExit")
    T = Float64
    choice = ArgmaxStage(grow_axis(layout, exiting, 2); axis = exiting, reward = zeros(T, 2, 1), search = :brute)
    return _exit_chain(choice, layout, bequest, exiting, T)
end

"""
Smooth (logit) endogenous exit at scale `ε`: backward `V_start = ε·log(e^{V_end/ε} + e^{bequest/ε})`,
forward `Λ_end = p_stay·Λ_start` with `p_stay = e^{V_end/ε}/(e^{V_end/ε}+e^{bequest/ε})` (the rest
leaves). The composite of §12 — the choice is a `LogitChoiceStage` collapsing the `:exiting` axis by
log-sum-exp. The `ε → 0` limit is [`EndogenousExit`](@ref). `bequest` is a REQUIRED
[`ScalarField`](@ref) source; the `layout` must carry the `:exiting` axis at size 1.
"""
function LogitEndogenousExit(layout::GriddedLayout; bequest = nothing, ε = 1.0, exiting::Symbol = :exiting)
    bequest === nothing && _exit_no_bequest("LogitEndogenousExit")
    T = Float64
    choice = LogitChoiceStage(grow_axis(layout, exiting, 2); axis = exiting, cost_matrix = zeros(T, 1, 2), ε = ε)
    return _exit_chain(choice, layout, bequest, exiting, T)
end
