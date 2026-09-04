using Test
using HouseholdStages

# Exit-with-bequest as a COMPOSITE (end-goal §12): `Choice ∘ Utility ∘ Markov` over a transient
# `:exiting` axis the block layout declares at size 1 and the composite threads `1 → 2 → 2 → 1`. Mass
# LEAVES the live population on exit (Λ shrinks — need NOT sum to 1). The backward combines the
# continuation with the bequest (Markov mixture / hard max / soft max), the forward scales Λ by the
# per-cell survival. Backward/forward are NOT an adjoint pair (the bequest/sink break the duality).
# `:exiting` is declared at size 1, so V/Λ carry a trailing singleton axis (the household axes × 1).

# Layout sugar: the household axes plus the declared size-1 `:exiting` axis the exit composite grows.
exit_layout(pairs...) = GriddedLayout(pairs..., :exiting => Discrete([0]))
col(v) = reshape(collect(float.(v)), length(v), 1)          # (n,) → (n,1), matching the :exiting singleton

@testset "ExogenousExit — survival-weighted mixture, mass leaves" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    s = 0.9; b = -5.0
    stage = ExogenousExit(layout; survival = s, bequest = b)

    V_end = col([4.0, 1.0, 2.0])
    @test backward!(stage, V_end, NamedTuple()) ≈ s .* V_end .+ (1 - s) * b   # V_start = s·V + (1−s)·b

    Λ_start = col([0.2, 0.5, 0.3])
    Λ_end   = forward!(stage, Λ_start)
    @test Λ_end ≈ s .* Λ_start                                                # survivors only
    @test sum(Λ_end) ≈ s * sum(Λ_start)                                       # (1−s) fraction left
    @test sum(Λ_end) < sum(Λ_start)                                           # mass is NOT conserved
end

@testset "ExogenousExit — bequest can vary along the wealth left behind" begin
    layout = exit_layout(:wealth => GriddedContinuous([0.0, 1.0, 2.0]))
    s = 0.85
    # A closure bequest: leave behind half of the wealth (the accidental-bequest motive).
    stage = ExogenousExit(layout; survival = s, bequest = (; wealth) -> 0.5 * wealth)
    V_end = col([3.0, 2.0, 1.0]); bvec = col(0.5 .* [0.0, 1.0, 2.0])
    @test backward!(stage, V_end, NamedTuple()) ≈ s .* V_end .+ (1 - s) .* bvec
end

@testset "EndogenousExit — hard optimal stopping (max), survivors keep mass" begin
    layout = exit_layout(:x => Discrete([1, 2, 3, 4]))
    b = 2.0
    stage = EndogenousExit(layout; bequest = b)

    V_end = col([3.0, 2.5, 1.0, 2.0])                             # cells 1,2 stay (V≥b); 3 exits; 4 ties (≥ ⇒ stay)
    @test backward!(stage, V_end, NamedTuple()) ≈ max.(V_end, b)  # V_start = max(V_end, b)

    Λ_start = col([0.1, 0.2, 0.3, 0.4])
    Λ_end   = forward!(stage, Λ_start)
    stay    = V_end .≥ b
    @test Λ_end ≈ stay .* Λ_start                                 # survivors {V_end ≥ b} keep mass, rest leaves
    @test Λ_end[3] == 0.0                                         # the exiter's mass left
    @test sum(Λ_end) ≈ sum(Λ_start[stay])
end

@testset "LogitEndogenousExit — smooth stopping (softmax value + p_stay mass)" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    b = 1.0; ε = 0.5
    stage = LogitEndogenousExit(layout; bequest = b, ε = ε)

    V_end = col([2.0, 1.0, 0.0])
    V_start = backward!(stage, V_end, NamedTuple())
    @test V_start ≈ @. ε * log(exp(V_end / ε) + exp(b / ε))       # soft-max certainty equivalent

    Λ_start = col([0.3, 0.3, 0.4])
    Λ_end   = forward!(stage, Λ_start)
    p_stay  = @. exp(V_end / ε) / (exp(V_end / ε) + exp(b / ε))
    @test Λ_end ≈ p_stay .* Λ_start                              # survival probability scales each cell's mass
    @test all(0 .< Λ_end .< Λ_start)                             # strictly some mass leaves everywhere (smooth)
end

@testset "LogitEndogenousExit → EndogenousExit as ε → 0" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    b = 1.5
    V_end = col([3.0, 1.5, 0.5])
    hard  = backward!(EndogenousExit(layout; bequest = b), V_end, NamedTuple())
    soft  = backward!(LogitEndogenousExit(layout; bequest = b, ε = 1e-4), V_end, NamedTuple())
    @test soft ≈ hard atol = 1e-2                                # the temperature-0 limit recovers the max
end

@testset "Exit — bequest is REQUIRED (no default), error names the CRRA trap" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    @test_throws ErrorException ExogenousExit(layout; survival = 0.9)
    @test_throws ErrorException EndogenousExit(layout)
    @test_throws ErrorException LogitEndogenousExit(layout; ε = 1.0)
    # The message names the trap so the caller supplies the value of death.
    err = try; ExogenousExit(layout; survival = 0.9); catch e; e; end
    @test occursin("prefer death", err.msg)
    @test occursin("bequest", err.msg)
end

@testset "Exit — backward/forward VJPs are the seated diagonal (duality with the sink/source)" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    dΛ, dV = randn(3, 1), randn(3, 1)

    # Per-cell mortality as a dep closure (`:x` value = index here): age/health-varying survival.
    exo = ExogenousExit(layout; survival = (; x) -> [0.95, 0.9, 0.7][x], bequest = 0.0)
    backward!(exo, col([4.0, 1.0, 2.0]), NamedTuple())
    @test sum(forward_adjoint!(exo, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(exo, dV))

    endo = EndogenousExit(layout; bequest = 1.5)
    backward!(endo, col([3.0, 1.0, 2.0]), NamedTuple())              # stay mask = [1,0,1]
    @test sum(forward_adjoint!(endo, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(endo, dV))

    logit = LogitEndogenousExit(layout; bequest = 1.0, ε = 0.5)
    backward!(logit, col([2.0, 1.0, 0.0]), NamedTuple())
    @test sum(forward_adjoint!(logit, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(logit, dV))
end

@testset "Exit — survival via FromEnv tracks env" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    stage  = ExogenousExit(layout; survival = FromEnv(:s), bequest = 0.0)
    Λ = col([0.2, 0.5, 0.3])
    backward!(stage, col([1.0, 1.0, 1.0]), (s = 0.8,))
    @test forward!(stage, Λ) ≈ 0.8 .* Λ
    backward!(stage, col([1.0, 1.0, 1.0]), (s = 0.5,))               # re-seat a fresh hazard
    @test forward!(stage, Λ) ≈ 0.5 .* Λ
end

# The composite's two ends and its interior #
#-------------------------------------------#

@testset "Exit composites report `:exiting` at size 1 at both ends, and grow it to 2 inside" begin
    layout = exit_layout(:x => Discrete([1, 2, 3, 4]))
    grown  = grow_axis(layout, :exiting, 2)
    for stage in (ExogenousExit(layout; survival = 0.9, bequest = 0.0),
                  EndogenousExit(layout; bequest = 0.0),
                  LogitEndogenousExit(layout; bequest = 0.0, ε = 0.5))
        @test start_layout(stage) == layout
        @test end_layout(stage)   == layout
        @test layout_size(start_layout(stage)) == (4, 1)
        # `Choice ∘ Utility ∘ Markov` threads `1 → 2 → 2 → 2 → 1`: the transient axis is grown by the
        # choice leaf and dropped by the trailing Markov, and is at size 1 wherever a caller can see it.
        @test boundaries(stage) == (layout, grown, grown, layout)
        @test size(V_start_buffer(stage)) == (4, 1)
        @test size(Λ_end_buffer(stage))   == (4, 1)
    end
end


# End-to-end through the outer loop #
#-----------------------------------#
# A household block containing an exit composite, solved for its `V` and stationary `Λ` by
# `solve_steady_state_given_env!`. The block is `utility ∘ discount ∘ exit ∘ shock ∘ entry`, whose
# fixed points are closed-form in the transition `P`, the discount `β`, the flow `u` and the entry
# source `g`:
#
#     V = u + β·C(P·V)        with C the exit composite's certainty equivalent,
#     Λ = Pᵀ·(p_stay ⊙ Λ) + g.
#
# `P` is asymmetric, so a transposed transition would fail both residuals. The entry source is what
# gives the sub-stochastic block a nonzero stationary distribution; without it the exiting mass would
# drain the population to zero.

exit_ss_layout = exit_layout(:x => Discrete([1, 2, 3, 4]))
exit_ss_u      = col([0.0, 10.0, 20.0, 30.0])
exit_ss_g      = col(fill(0.05, 4))
exit_ss_β      = 0.9
exit_ss_env    = (β = exit_ss_β,)                    # the discount rides `env`, so the block has a Jacobian input
exit_ss_P      = [0.80 0.15 0.03 0.02
                  0.10 0.70 0.15 0.05
                  0.05 0.15 0.70 0.10
                  0.02 0.03 0.15 0.80]

"""
A household block around an exit composite: flow utility, discounting, the exit, a persistent shock
and an entry source, with the surviving population attached as a moment.
"""
function exit_block(exit_stage)
    hh = UtilityStage(exit_ss_layout; utility = exit_ss_u) ∘
         TimeDiscountingStage(exit_ss_layout; β = FromEnv(:β)) ∘
         exit_stage ∘
         MarkovStage(exit_ss_layout; axis = :x, transition_matrix = exit_ss_P) ∘
         EntryStage(exit_ss_layout; entry = exit_ss_g)
    return define_moments!(hh; pop = at_end(integrand = 1.0, reduce = sum))
end

"""
Solve `exit_block(exit_stage)` and check the two fixed points against the closed forms: `certainty`
maps the continuation `P·V` to the exit composite's backward value, `stay` maps it to the per-cell
survival probability the forward applies.
"""
function check_exit_steady_state(exit_stage, certainty, stay)
    (;V, Λ, moments) = solve_steady_state_given_env!(exit_block(exit_stage), exit_ss_env)
    V_cont = exit_ss_P * V
    @test V ≈ exit_ss_u .+ exit_ss_β .* certainty(V_cont)          atol = 1e-6
    @test Λ ≈ exit_ss_P' * (stay(V_cont) .* Λ) .+ exit_ss_g        atol = 1e-5
    # Mass balance: `P` conserves, so the mass leaving through the exit must equal the entry flow.
    @test sum((1 .- stay(V_cont)) .* Λ) ≈ sum(exit_ss_g)           atol = 1e-5
    @test all(isfinite, V)
    @test all(Λ .> 0)
    @test moments.pop ≈ sum(Λ)
    return (;V, Λ, stay = stay(V_cont))
end

@testset "ExogenousExit — solved end to end at a fixed env" begin
    s = 0.95
    (;Λ) = check_exit_steady_state(ExogenousExit(exit_ss_layout; survival = s, bequest = 0.0),
                                   V_cont -> s .* V_cont, _ -> s)
    # Mass balance: the surviving population is the entry flow divided by the exit rate.
    @test sum(Λ) ≈ sum(exit_ss_g) / (1 - s) atol = 1e-4
end

@testset "EndogenousExit — solved end to end, with an interior stopping boundary" begin
    b = 160.0
    (;V, stay) = check_exit_steady_state(EndogenousExit(exit_ss_layout; bequest = b),
                                         V_cont -> max.(V_cont, b), V_cont -> V_cont .≥ b)
    # The bequest is chosen so the boundary is interior — some cells stop, some continue. A test where
    # everybody stayed (or everybody left) would not exercise the argmax at all.
    @test any(stay) && !all(stay)
    # A stopping cell takes the bequest, so its value is the flow plus the discounted bequest exactly.
    @test V[.!stay] ≈ exit_ss_u[.!stay] .+ exit_ss_β * b
end

@testset "LogitEndogenousExit — solved end to end, with interior survival probabilities" begin
    b, ε = 160.0, 2.0
    soft_max(V_cont) = ε .* log.(exp.(V_cont ./ ε) .+ exp(b / ε))
    p_stay(V_cont)   = exp.(V_cont ./ ε) ./ (exp.(V_cont ./ ε) .+ exp(b / ε))
    (;stay) = check_exit_steady_state(LogitEndogenousExit(exit_ss_layout; bequest = b, ε = ε),
                                      soft_max, p_stay)
    @test all(0 .< stay .< 1)                                      # smooth: everybody partly leaves
end

@testset "Exit composites drive the transition path, the expectation vectors and the Jacobian" begin
    for (stage, sensitive) in ((ExogenousExit(exit_ss_layout; survival = 0.95, bequest = 0.0), false),
                               (EndogenousExit(exit_ss_layout; bequest = 160.0), false),
                               (LogitEndogenousExit(exit_ss_layout; bequest = 160.0, ε = 2.0), true))
        hh = exit_block(stage)
        (;V, Λ) = solve_steady_state_given_env!(hh, exit_ss_env)

        # A constant env path started from the steady state stays there.
        (;V_path, Λ_path) = solve_transition_given_env_path!(hh, [exit_ss_env for _ in 1:4];
                                                            Λ_0 = Λ, V_T = V)
        @test V_path[1] ≈ V atol = 1e-5
        @test Λ_path[end] ≈ Λ atol = 1e-5

        # The SSJ expectation vectors run on the composite's own end layout.
        @test all(E -> size(E) == layout_size(end_layout(hh)), expectation_vectors(hh, _ -> 1.0, 3))

        # The sequence-space Jacobian of the surviving population in `β`. Exogenous mortality does
        # not read `β` at all, and the hard stopping set is locally constant in it, so both are
        # exactly flat; the logit's survival probability moves with the continuation, so it is not.
        J = compute_fake_news_ssj(hh, exit_ss_env, 3)
        @test size(J) == (3, 3)
        @test all(isfinite, J)
        @test (maximum(abs, J) > 1e-6) == sensitive
    end
end
