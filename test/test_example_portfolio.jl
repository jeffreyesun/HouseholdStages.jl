using Test
using HouseholdStages
include(joinpath(@__DIR__, "..", "examples", "portfolio", "model.jl"))

# Part 3: a literature model (incomplete-markets portfolio choice) whose household block is FOUR
# existing library stages and NO bespoke household stage — it must solve end-to-end through the
# standard outer loop, with a sensible interior risky-share policy.

@testset "example: portfolio choice — existing stages only, solves end-to-end" begin
    p   = PortfolioParams(N_w = 80)                 # smaller grid for test speed
    hh  = portfolio_household(p)
    env = (; w = 1.0)
    res = solve_steady_state_given_env!(hh, env)

    @test isapprox(sum(res.Λ), 1.0; atol = 1e-5)    # stationary distribution
    @test all(isfinite, res.V)
    @test compute_moments(hh, res.Λ, env).mean_wealth > 0

    θ = HouseholdStages.policy(hh.buffer.stages[end])   # the Portfolio (MeanVarianceStage) leaf
    @test all(0.0 .≤ θ .≤ 1.0)                      # valid shares
    @test 0.0 < sum(θ) / length(θ) < 1.0            # interior on average — a real risk–return tradeoff
end
