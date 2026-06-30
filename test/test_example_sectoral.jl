using Test, HouseholdStages

# Roy-style occupational-choice example — assembled entirely from existing
# library stages (MarkovStage, SectorSwitchingStage, WealthChangeStage,
# ConsumptionSavingsStage). The model include is wrapped in a module so its
# globals (`params`, `u_crra`, …) don't clash with other example tests.
module SectoralExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "sectoral", "model.jl"))

    @testset "example: sectoral — existing stages only" begin
        p   = SectoralParams(N_w = 120)
        hh  = sectoral_household(p)
        env = sectoral_env(p)

        res     = solve_steady_state_given_env!(hh, env)
        (; V, Λ) = res
        moments = compute_moments(hh, Λ, env)

        # Stationary distribution is a probability measure.
        @test sum(Λ) ≈ 1.0 atol = 1e-6
        @test all(≥(-1e-12), Λ)

        # Value function is everywhere finite.
        @test all(isfinite, V)

        # Aggregate wealth is positive and finite.
        @test isfinite(moments.K_supplied)
        @test moments.K_supplied > 0

        # Every sector is populated (logit smoothing ⇒ no sector empties),
        # and the population shares are a partition of unity.
        pop_tot = moments.pop_ag + moments.pop_mfg + moments.pop_svc
        @test pop_tot ≈ 1.0 atol = 1e-6
        @test moments.pop_ag  > 0
        @test moments.pop_mfg > 0
        @test moments.pop_svc > 0

        # Per-sector wealth sums to the aggregate.
        @test moments.K_ag + moments.K_mfg + moments.K_svc ≈ moments.K_supplied atol = 1e-6

        # Comparative advantage: the high-wage sector (:mfg, w = 1.1) draws a
        # larger employment share than the low-wage sector (:ag, w = 0.9).
        @test moments.pop_mfg > moments.pop_ag
    end
end
