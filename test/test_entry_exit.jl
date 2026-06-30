using Test
using HouseholdStages

# Entry + exit, the two halves the retired `EntryExitStage` fused. `EntryStage` is the additive
# forward source (Λ_end = Λ + g); `ExogenousExit` is the survival-weighted death COMPOSITE (forward
# Λ_end = s·Λ, backward V_start = s·V_end + (1−s)·b). Composed `EntryStage ∘ ExogenousExit`, they
# reproduce the old fused `s·Λ + M·g` / `s·V + (1−s)b` content. Forward and backward are NOT adjoint
# pairs (the source has no backward effect, the bequest no forward effect). Exit declares the `:exiting`
# axis at size 1, so the whole block carries a trailing singleton axis (household axes × 1).

exit_layout(pairs...) = GriddedLayout(pairs..., :exiting => Discrete([0]))
col(v) = reshape(collect(float.(v)), length(v), 1)

@testset "Entry+Exit — forward source & survival-weighted backward" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    g = col([0.5, 0.3, 0.2]); s = 0.9; M = 0.1
    # `exit ∘ entry`: forward applies the left stage first, so survivors are scaled by s, THEN the
    # unscaled entry source M·g is added — exactly the old fused `s·Λ + M·g`.
    exit  = ExogenousExit(layout; survival = s, bequest = 0.0)
    entry = EntryStage(layout; entry = M .* g)
    hh    = exit ∘ entry

    V_end = col([4.0, 1.0, 2.0])
    @test backward!(hh, V_end, NamedTuple()) ≈ s .* V_end             # entry is identity on V; exit ⇒ s·V (b=0)

    Λ_start = col([0.2, 0.5, 0.3])
    @test forward!(hh, Λ_start) ≈ s .* Λ_start .+ M .* g              # survival of incumbents + entry source
end

@testset "Entry+Exit — the source gives a NONZERO stationary distribution" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    g = col([0.5, 0.3, 0.2]); s = 0.9; M = 0.1
    exit  = ExogenousExit(layout; survival = s, bequest = 0.0)
    entry = EntryStage(layout; entry = M .* g)
    hh    = exit ∘ entry
    backward!(hh, col([0.0, 0.0, 0.0]), NamedTuple())               # seat s
    Λ = col([0.2, 0.5, 0.3])
    for _ in 1:600; Λ = forward!(hh, Λ); end
    @test Λ ≈ (M .* g) ./ (1 - s)                                    # fixed point of Λ = s·Λ + M·g (no collapse to 0)
    @test sum(Λ) ≈ M / (1 - s)                                       # M = 1−s with g normalized ⇒ total mass 1
end

@testset "Exit — bequest, per-cell mortality, adjoint symmetry" begin
    layout = exit_layout(:x => Discrete([1, 2, 3]))
    # Scalar bequest: V_start = s·V_end + (1−s)·b.
    bq = ExogenousExit(layout; survival = 0.8, bequest = 5.0)
    @test backward!(bq, col([4.0, 1.0, 2.0]), NamedTuple()) ≈ 0.8 .* col([4.0, 1.0, 2.0]) .+ 0.2 * 5.0

    sv = (; x) -> [0.95, 0.9, 0.7][x]                               # per-cell survival (age/health), bequest 0
    pc = ExogenousExit(layout; survival = sv, bequest = 0.0)
    backward!(pc, col([4.0, 1.0, 2.0]), NamedTuple())               # backward seats s; forward applies it
    @test forward!(pc, col([0.2, 0.5, 0.3])) ≈ [0.95, 0.9, 0.7] .* col([0.2, 0.5, 0.3])  # survivors only
    dΛ, dV = randn(3, 1), randn(3, 1)                               # both VJPs are the diagonal s⊙· (coincide)
    @test sum(forward_adjoint!(pc, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(pc, dV))
end

@testset "Entry+Exit — in a chain, the source sustains a stationary cross-section" begin
    # Mortality + replacement: without the entry source a sub-stochastic block collapses to Λ≡0;
    # the source sustains a nonzero stationary distribution. Chain: income Markov ∘ exit ∘ entry.
    ys = [1, 2]; P = [0.7 0.3; 0.3 0.7]
    layout = exit_layout(:income => Discrete(ys))
    s = 0.96
    g = col([0.5, 0.5])                                             # newborns split across income
    shock = MarkovStage(layout; axis = :income, transition_matrix = P)
    exit  = ExogenousExit(layout; survival = s, bequest = 0.0)
    entry = EntryStage(layout; entry = (1 - s) .* g)                # replacement ⇒ mass 1
    hh = shock ∘ exit ∘ entry
    hh = define_moments!(hh; pop = at_end(integrand = 1.0, reduce = sum))

    res = solve_steady_state_given_env!(hh, NamedTuple())
    @test all(res.Λ .> 0)                                           # nonzero everywhere (no collapse)
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-4)                    # replacement keeps total mass ≈ 1
    @test all(isfinite, res.V)
end
