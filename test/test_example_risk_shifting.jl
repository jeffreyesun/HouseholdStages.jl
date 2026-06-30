using Test, HouseholdStages

# Part 3: a literature model (Vereshchagina–Hopenhayn 2009 risk-shifting /
# gambling-for-resurrection) whose household block is FIVE existing library
# stages and NO bespoke household stage — it must solve end-to-end through the
# standard outer loop, with the V–H comparative static in the seated policy:
# project risk θ*(x) is HIGH near the limited-liability floor and LOW for the
# well-capitalized. Module-wrapped to avoid global-name clashes with sibling
# example tests.
module RiskShiftingExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "risk_shifting", "model.jl"))

    @testset "example: risk_shifting — existing stages only" begin
        p   = RiskShiftingParams(N_a = 150)             # converged grid for test speed
        hh  = risk_shifting_household(p)
        env = risk_shifting_env(p)
        res = solve_steady_state_given_env!(hh, env)

        @test isapprox(sum(res.Λ), 1.0; atol = 1e-4)    # stationary distribution
        @test all(isfinite, res.V)

        # Wealth is carried after the limited-liability floor ⇒ robustly ≥ a_floor > 0.
        @test compute_moments(hh, res.Λ, env).mean_wealth > p.a_floor

        θ    = HouseholdStages.policy(risk_shifting_gamble_stage(hh))   # MeanVarianceStage leaf
        @test size(θ) == (p.N_a, length(p.z_grid))
        @test all(0.0 .≤ θ .≤ 1.0)                      # valid risky shares

        # The V–H signature: poor (near the floor) gamble MORE than rich (top decile).
        grid   = GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log).grid
        fl     = findfirst(>=(p.a_floor), grid)
        poor   = fl:min(fl + 4, p.N_a)
        rich   = round(Int, 0.9 * p.N_a):p.N_a
        θ_poor = sum(@view θ[poor, :]) / (length(poor) * length(p.z_grid))
        θ_rich = sum(@view θ[rich, :]) / (length(rich) * length(p.z_grid))
        @test θ_poor > 0.2                              # the poor gamble for resurrection
        @test θ_rich < 0.1                              # the rich take the safe project
        @test θ_poor > θ_rich                           # risk-taking decreasing in net worth
    end
end
