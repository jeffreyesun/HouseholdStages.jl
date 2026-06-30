using Test
using HouseholdStages

# Exit-with-bequest as a COMPOSITE (end-goal §12): `Choice ∘ Utility ∘ Markov` over a transient
# `:exiting` axis the block layout declares at size 1 and the composite threads `1 → 2 → 2 → 1`. Mass
# LEAVES the live population on exit (Λ shrinks — need NOT sum to 1). The backward combines the
# continuation with the bequest (Markov mixture / hard max / soft max), the forward scales Λ by the
# per-cell survival. Backward/forward are NOT an adjoint pair (the bequest/sink break the duality).
# These testsets pin the exact closed forms the composite must reproduce. `:exiting` is declared at
# size 1, so V/Λ carry a trailing singleton axis (the household axes × 1).

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
