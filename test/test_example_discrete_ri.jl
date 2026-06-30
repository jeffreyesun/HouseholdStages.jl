using Test, HouseholdStages

# Regression test for examples/discrete_ri — the discrete rational-inattention
# occupation-choice model (Matějka–McKay). Asserts the household block is solved
# from EXISTING library stages only: it builds, the inner V/Λ fixed point
# converges, mass is conserved, the value is finite, and the RI choice/share
# moments + the comparative static in the Shannon cost λ are sensible.
module DiscreteRiExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "discrete_ri", "model.jl"))

    @testset "example: discrete_ri — existing stages only" begin
        p     = discrete_ri_params
        hh    = discrete_ri_household(p)
        n_occ = length(p.premium)
        q     = fill(1 / n_occ, n_occ)
        env   = (; λ = p.λ, q)

        res = solve_steady_state_given_env!(hh, env)

        # Stationary distribution is a probability measure; value is finite.
        @test isapprox(sum(res.Λ), 1.0; atol = 1e-8)
        @test all(>=(-eps()), res.Λ)
        @test all(isfinite, res.V)

        # Moments: occupation shares form a distribution; mean income is positive.
        m = compute_moments(hh, res.Λ, env)
        shares = (m.safe_share, m.persistent_share, m.career_share)
        @test all(>=(-1e-10), shares)
        @test isapprox(sum(shares), 1.0; atol = 1e-8)
        @test m.mean_income > 0
        @test minimum(p.income_grid) <= m.mean_income <= maximum(p.income_grid)

        # RI choice probabilities: a proper posterior over occupations per state.
        P = ri_choice_probs(hh)                       # (income, origin-occ, dest-occ)
        @test size(P) == (length(p.income_grid), n_occ, n_occ)
        @test all(>=(-1e-10), P)
        for s in axes(P, 1), i in axes(P, 2)
            @test isapprox(sum(@view P[s, i, :]), 1.0; atol = 1e-8)
        end

        # The Matějka–McKay comparative static: as the Shannon cost λ rises, the
        # posterior collapses toward the uniform prior, so the choice flattens —
        # the per-state choice spread (max − min prob) shrinks monotonically and
        # the high-payoff "career" share falls.
        spread(λ) = begin
            r = solve_steady_state_given_env!(hh, (; λ, q))
            Pλ = ri_choice_probs(hh)
            maximum(maximum(@view Pλ[s, i, :]) - minimum(@view Pλ[s, i, :])
                    for s in axes(Pλ, 1), i in axes(Pλ, 2))
        end
        @test spread(0.10) > spread(1.0) > spread(8.0)

        career_share(λ) = compute_moments(hh, solve_steady_state_given_env!(hh, (; λ, q)).Λ, (; λ, q)).career_share
        @test career_share(0.10) > career_share(8.0)

        # At a large λ the choice is essentially the uniform prior (within 0.05).
        Phi = (solve_steady_state_given_env!(hh, (; λ = 50.0, q)); ri_choice_probs(hh))
        @test all(abs.(Phi .- 1 / n_occ) .< 0.05)
    end
end
