using Test, HouseholdStages

# Wrapped in a module so the example's global names (params, u_crra, the
# household builder) do not clash with the other example regression tests.
module DurableHousingExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "durable_housing", "model.jl"))

    @testset "example: durable_housing — existing stages only" begin
        p   = durable_housing_params
        hh  = durable_housing_household(p)
        env = durable_housing_env(p)
        res = solve_steady_state_given_env!(hh, env)

        # Stationary distribution is a probability measure.
        @test sum(res.Λ) ≈ 1.0 atol = 1e-8
        @test all(isfinite, res.V)
        @test all(>=(0.0), res.Λ)

        # Sensible moments.
        m = compute_moments(hh, res.Λ, env)
        @test m.mean_wealth > 0.0
        @test 0.0 < m.own_rate <= 1.0          # some (but not all) households own
        @test m.mean_house >= 0.0

        # Housing-size policy lands within the housing index range, and at least
        # one renter cell chooses to own (the buy choice is live).
        h_pol = HouseholdStages.policy(hh.buffer.stages[3])
        n_h   = length(p.h_sizes)
        @test all(1 .<= h_pol .<= n_h)
        @test any(h_pol .> 1)
    end
end
