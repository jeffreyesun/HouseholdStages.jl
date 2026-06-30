using Test, HouseholdStages

# Part 3: a literature model (Gaussian/variance rational inattention — Sims 2003; Maćkowiak–
# Wiederholt) whose household block is FOUR existing library stages and NO bespoke household stage.
# It must solve end-to-end through the standard outer loop, with a sensible, interior attention
# (dispersion) policy that responds to the information cost λ. Wrapped in a module so the model's
# global names (u_crra, ...) don't clash with the other example tests' identically-named globals.

module RationalInattentionExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "rational_inattention", "model.jl"))

    @testset "example: rational_inattention — existing stages only, solves end-to-end" begin
        p   = RationalInattentionParams(N_w = 60)        # smaller grid for test speed
        hh  = rational_inattention_household(p)
        env = (; r = p.r, w = p.w, λ = p.λ)
        res = solve_steady_state_given_env!(hh, env)

        @test isapprox(sum(res.Λ), 1.0; atol = 1e-5)     # stationary distribution, mass conserved
        @test all(isfinite, res.V)                       # VFI fixed point is finite
        @test compute_moments(hh, res.Λ, env).mean_wealth > 0

        θ = HouseholdStages.policy(hh.buffer.stages[end])    # the Attention (ScaleVarianceStage) leaf
        @test all(p.dispersions[1] .≤ θ .≤ p.dispersions[end])   # θ* stays on the dispersion grid
        @test any(θ .> 0)                                 # NOT degenerate — some cells choose dispersion
        @test 0.0 < sum(θ) / length(θ) < p.dispersions[end]      # interior on average

        # The RI comparative static: more attention is chosen (less dispersion) as λ rises.
        θmean(λ) = (h = rational_inattention_household(p);
                    solve_steady_state_given_env!(h, (; r = p.r, w = p.w, λ));
                    sum(HouseholdStages.policy(h.buffer.stages[end])) /
                    length(HouseholdStages.policy(h.buffer.stages[end])))
        @test θmean(0.0) ≥ θmean(0.1)                     # higher info cost ⇒ weakly less dispersion
    end
end
