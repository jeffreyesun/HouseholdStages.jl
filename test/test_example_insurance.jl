using Test, HouseholdStages

# Part 3: a literature model (convex-cost loss insurance / annuitization) whose household block is
# FOUR existing library stages and NO bespoke household stage — it must solve end-to-end through the
# standard outer loop, with a sensible interior insurance-coverage policy. Wrapped in a module so the
# example's global names (InsuranceParams, u_crra, …) don't clash with sibling example tests.

module InsuranceExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "insurance", "model.jl"))

    @testset "example: insurance — existing stages only, solves end-to-end" begin
        p   = InsuranceParams(N_w = 80)                  # smaller grid for test speed
        hh  = insurance_household(p)
        env = insurance_env(p)
        res = solve_steady_state_given_env!(hh, env)

        @test isapprox(sum(res.Λ), 1.0; atol = 1e-5)     # stationary distribution
        @test all(isfinite, res.V)
        @test compute_moments(hh, res.Λ, env).mean_wealth > 0

        θ = HouseholdStages.policy(hh.buffer.stages[2])  # the Insurance (MixingStage) leaf
        @test all(0.0 .≤ θ .≤ 1.0)                       # valid coverage shares
        @test 0.0 < sum(θ) / length(θ) ≤ 1.0             # some coverage is bought on average
    end
end
