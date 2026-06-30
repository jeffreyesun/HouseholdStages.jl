using Test, HouseholdStages

# Grossman (1972) health capital & mortality, life-cycle form. The household block
# is existing library stages ONLY — `CapitalInvestmentStage(:health) ∘ MarkovStage(:alive |
# health)` (health investment + health-dependent sub-stochastic survival, dead
# absorbing). Wrapped in a module so the example's global names don't clash with
# sibling example tests sharing the suite's global scope.
module HealthExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "health", "steady_state.jl"))

    @testset "example: health — existing stages only" begin
        p   = HealthParams(N_h = 120, N_age = 40)
        res = health_life_cycle(p; verbosity = 0)

        # Total mass on the (health, alive) grid is conserved every age — the
        # absorbing dead state catches what mortality removes from the alive slice.
        total_mass = vec(sum(res.Λ_panel; dims = (1, 2)))
        @test all(m -> isapprox(m, 1.0; atol = 1e-8), total_mass)

        # All value arrays finite.
        @test all(all.(isfinite, res.V_by_age))

        # Survival curve: starts at full cohort mass and is (weakly) decreasing —
        # mortality only ever removes alive mass.
        @test res.survival_by_age[1] ≈ 1.0 atol = 1e-6
        @test all(diff(res.survival_by_age) .<= 1e-9)
        @test res.survival_by_age[end] < res.survival_by_age[1]   # some die over the life cycle
        @test res.survival_by_age[end] > 0.0                      # cohort not wiped out

        # Life expectancy is a sensible moment (>0, < N_age).
        @test 0.0 < res.life_expectancy < p.N_age

        # Mean health among the living is positive and inside the grid range.
        living = res.alive_mass .> 1e-10
        @test all(res.mean_health_by_age[living] .> 0.0)
        @test all(p.h_min - 1e-6 .<= res.mean_health_by_age[living] .<= p.h_max + 1e-6)

        # Health-dependent mortality bites: survival should be higher when health is
        # higher (the model's whole point). Survival probability is monotone in h.
        @test survival_prob(p.h_max, p) > survival_prob(p.h_min, p)
    end
end
