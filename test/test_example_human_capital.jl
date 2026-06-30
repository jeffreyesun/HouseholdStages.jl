using Test, HouseholdStages

# Part 3: a literature model (Ben-Porath / Huggett–Ventura–Yaron life-cycle human
# capital) whose household block is ONE existing library stage — `CapitalInvestmentStage`
# on `:h` — and NO bespoke household stage. The finite-horizon backward + forward
# cohort solve is driver logic; the age-specific ability profile enters via `env`.
# Wrapped in a module so the example's global names don't clash with sibling tests.

module HumanCapitalExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "human_capital", "steady_state.jl"))

    @testset "example: human_capital — existing stages only (CapitalInvestmentStage)" begin
        p   = HumanCapitalParams(N_h = 120, N_age = 30)     # modest grid for test speed
        res = human_capital_life_cycle(p; verbosity = 0)

        # Cohort mass is conserved at every age (each age marginal is unit mass).
        @test all(isapprox.(vec(sum(res.Λ_panel; dims = 1)), 1.0; atol = 1e-8))

        # Finite value everywhere across the life cycle.
        @test all(V -> all(isfinite, V), res.V_by_age)

        # Sensible moments: human capital and earnings are strictly positive.
        @test res.lifetime_mean_h > 0
        @test all(res.mean_h_by_age .> 0)
        @test all(res.earnings_by_age .> 0)

        # The hump: human capital rises from birth to an interior peak, the
        # signature life-cycle profile — the cross-section is non-degenerate.
        @test res.mean_h_by_age[res.peak_age] > res.mean_h_by_age[1]
        @test 1 < res.peak_age < p.N_age
    end
end
