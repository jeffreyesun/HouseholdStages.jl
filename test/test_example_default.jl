using Test, HouseholdStages

# Sovereign / consumer-default example — a regression that the literature household block
# (income Markov + repay/default choice + debt discharge + persistent exclusion + savings) is
# EXISTING library stages only, and solves to a non-degenerate stationary equilibrium. Wrapped in a
# module so the example's global names (`u_crra`, `default_household`, …) don't clash with the
# sibling example tests.
module DefaultExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "default", "model.jl"))

    @testset "example: default — existing stages only" begin
        p   = default_params
        hh  = default_household(p)
        env = default_env(p)
        res = solve_steady_state_given_env!(hh, env)
        m   = compute_moments(hh, res.Λ, env)

        # Stationary distribution is a probability measure; values are finite.
        @test sum(res.Λ) ≈ 1.0 atol = 1e-8
        @test all(isfinite, res.V)

        # The exclusion mass is a genuine interior — some agents default, but not all. (Without the
        # persistent exclusion spell this degenerates to 1.)
        @test 0.0 < m.excluded_rate < 1.0

        # Mean assets sit in the grid; the borrowing motive pushes the mean negative (net debtors).
        @test p.a_min <= m.mean_assets <= p.a_max
        @test m.mean_assets < 0.0

        # The default policy (the DefaultStage's seated choice) is genuinely two-sided: at least one
        # good-standing cell repays and at least one defaults. `default` is the 2nd stage in the
        # chain `shock ∘ default ∘ reset ∘ receipt ∘ savings ∘ readmit`.
        default_pol = HouseholdStages.policy(hh.buffer.stages[2])
        @test any(==(1), default_pol)      # someone repays
        @test any(==(2), default_pol)      # someone defaults
    end
end
