using Test
using HouseholdStages
using HouseholdStages: expectation, GriddedPopulation

# Mass non-conservation invariant — Λ NEED NOT sum to 1 (see `population.jl` header). The
# mass-conserving stages (Markov/argmax/logit/forget/deterministic-continuous) conserve `Σ(Λ)`;
# the mass-CHANGING stages move it on purpose: `EntryStage` adds `Σg`, the exit stages remove the
# exiters' mass, `ReproductionStage` scales it. This file pins the three
# consequences a non-unit-mass Λ has to satisfy: (a) a single forward push grows/shrinks the mass
# exactly as the stage prescribes; (b) the Λ-solver reaches a stationary distribution at the
# entry/exit balance WITHOUT force-renormalizing to 1; (c) a moment (`∫ x dΛ`) is the correct
# AGGREGATE on a Λ whose mass is not 1.

# Exit declares the `:exiting` axis at size 1, so blocks containing it carry a trailing singleton axis.
exit_layout(pairs...) = GriddedLayout(pairs..., :exiting => Discrete([0]))
col(v) = reshape(collect(float.(v)), length(v), 1)

@testset "Mass non-conservation — single forward push grows / shrinks mass" begin
    layout  = exit_layout(:x => Discrete([1, 2, 3]))
    Λ_start = col([0.2, 0.5, 0.3])                             # Σ = 1.0
    m0      = sum(Λ_start)

    # EntryStage: Λ_end = Λ + g GROWS total mass by Σg (here Σg = 0.6).
    g     = col([0.1, 0.2, 0.3])
    entry = EntryStage(layout; entry = g)
    backward!(entry, col([0.0, 0.0, 0.0]), nothing)           # seat the source
    Λ_entry = forward!(entry, Λ_start)
    @test Λ_entry ≈ Λ_start .+ g
    @test sum(Λ_entry) ≈ m0 + sum(g)                           # mass grew by exactly Σg

    # ExogenousExit(s): Λ_end = s·Λ SHRINKS total mass to s·(old mass).
    s    = 0.75
    exit = ExogenousExit(layout; survival = s, bequest = 0.0)
    backward!(exit, col([0.0, 0.0, 0.0]), nothing)            # seat s
    Λ_exit = forward!(exit, Λ_start)
    @test Λ_exit ≈ s .* Λ_start
    @test sum(Λ_exit) ≈ s * m0                                 # mass shrank to s·old
end

@testset "Mass non-conservation — Λ-solver reaches the entry/exit balance, NOT forced to 1" begin
    # Chain `exit ∘ shock ∘ entry`. `forward!` applies the LEFT stage first (see `test_entry_exit.jl`),
    # so the forward map is Λ ← P·(s·Λ) + g (exit scales survivors, the income shock mixes them, the
    # entry source adds g). P is mass-preserving, so total mass solves m* = s·m* + Σg ⇒ m* = Σg/(1−s),
    # a balance set by entry vs exit — deliberately ≠ 1 here (Σg = 0.3, s = 0.9 ⇒ m* = 3.0).
    ys  = [1, 2]; P = [0.7 0.3; 0.3 0.7]
    layout = exit_layout(:income => Discrete(ys))
    s   = 0.9
    g   = col([0.15, 0.15])                                     # Σg = 0.3
    shock = MarkovStage(layout; axis = :income, transition_matrix = P)
    exit  = ExogenousExit(layout; survival = s, bequest = 0.0)
    entry = EntryStage(layout; entry = g)
    hh    = exit ∘ shock ∘ entry

    # Seat all kernels (entry source + survival) by a backward pass, then iterate Λ WITHOUT renorm.
    backward!(hh, col([0.0, 0.0]), NamedTuple())
    res = solve_lambda_steady_state_given_env!(hh; tol = 1e-10, maxiter = 100_000)

    @test res.converged
    m_expected = sum(g) / (1 - s)                              # = 0.3/0.1 = 3.0, NOT 1
    @test isapprox(sum(res.Λ), m_expected; atol = 1e-6)
    @test !isapprox(sum(res.Λ), 1.0; atol = 1e-2)              # the solver did NOT renormalize to 1
    @test all(res.Λ .> 0)                                      # nonzero everywhere (no collapse)
    # The fixed point is genuinely stationary under the (mass-changing) forward map.
    @test maximum(abs, forward!(hh, res.Λ) .- res.Λ) < 1e-6
end

@testset "Mass non-conservation — a moment is the correct AGGREGATE on a non-unit-mass Λ" begin
    # `∫ x dΛ = Σ x·λ` must be the aggregate, never an implicit (Σ x·λ)/Σλ mean. Build a Λ whose
    # mass is deliberately ≠ 1 by pushing through an EntryStage, then check the wealth aggregate.
    layout = GriddedLayout(:wealth => GriddedContinuous([1.0, 2.0, 3.0]))
    chain  = define_moments!(EntryStage(layout; entry = [0.5, 1.0, 1.5]);
        K = at_end(integrand = :wealth, reduce = sum),         # ∫ wealth dΛ — an AGGREGATE
    )
    backward!(chain, zeros(3), NamedTuple())                   # seat the source
    Λ_start = [0.2, 0.3, 0.5]                                  # Σ = 1.0
    Λ_end   = forward!(chain, Λ_start)                         # Σ = 1.0 + 3.0 = 4.0 (mass ≠ 1)
    @test sum(Λ_end) ≈ 4.0

    moments = compute_moments(chain, Λ_end, NamedTuple())
    # Λ_end = [0.7, 1.3, 2.0]; Σ wealth·λ = 1·0.7 + 2·1.3 + 3·2.0 = 9.3 (aggregate).
    @test moments.K ≈ 9.3 atol = 1e-12
    # An aggregate is NOT the (Σ x·λ)/Σλ per-capita mean (= 9.3/4 = 2.325); confirm we did not divide.
    @test !isapprox(moments.K, 9.3 / sum(Λ_end); atol = 1e-6)
    # The per-capita mean is recovered by an EXPLICIT divide by the actual mass.
    @test moments.K / sum(Λ_end) ≈ 2.325 atol = 1e-12

    # `expectation` matches the moment (both Σ f·λ), and ⟨1, Λ⟩ recovers total mass.
    @test expectation([1.0, 2.0, 3.0], GriddedPopulation(Λ_end)) ≈ 9.3 atol = 1e-12
    @test expectation(ones(3), Λ_end) ≈ sum(Λ_end)
end
