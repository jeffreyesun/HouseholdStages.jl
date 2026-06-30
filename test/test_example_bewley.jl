using Test, HouseholdStages

# Module-wrapped so the example's globals (`u_crra`, `bewley_params`, the
# stage constructors, etc.) don't clash with the other example tests sharing
# the runtests global scope.
module BewleyExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "bewley", "model.jl"))

    @testset "example: bewley — existing stages only" begin
        # A small grid keeps the test fast while exercising the real chain.
        p   = BewleyParams(N_a = 120, a_max = 80.0)
        hh  = bewley_household(p)
        env = bewley_env(p.r)

        res = solve_steady_state_given_env!(hh, env)
        m   = compute_moments(hh, res.Λ, env)

        # Stationary distribution is a probability measure.
        @test sum(res.Λ) ≈ 1.0 atol = 1e-6
        @test all(res.Λ .>= -1e-12)

        # Value function is finite and (weakly) increasing in wealth at the
        # mid income state — more cash-on-hand is never worse.
        @test all(isfinite, res.V)
        mid = (length(p.y_grid) + 1) ÷ 2
        @test issorted(res.V[:, mid])

        # The fixed return sits strictly below the impatience knife-edge, so a
        # non-degenerate precautionary distribution exists.
        @test p.r < 1 / p.β - 1

        # Precautionary buffer stock is strictly positive (agents self-insure).
        @test m.A_mean > 0.0

        # Hand-to-mouth share is a genuine fraction in (0, 1): some mass at the
        # constraint, but not all of it.
        @test 0.0 < m.frac_constrained < 1.0

        # Savings policy is monotone in wealth (next-period asset along the
        # asset grid at the mid income state is non-decreasing).
        a_policy = HouseholdStages.policy(hh)   # reaches the ConsumptionSavings argmax leaf through the chain
        @test issorted(a_policy[:, mid])
    end
end
