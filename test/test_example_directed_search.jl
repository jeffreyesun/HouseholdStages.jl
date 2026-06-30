using Test, HouseholdStages

# Wrapped in a module so the example's global names (params, u_crra, …) don't
# clash with the other example tests' identically-named globals.
module DirectedSearchExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "directed_search", "model.jl"))

    @testset "example: directed_search — existing stages only" begin
        p   = directed_search_params
        hh  = directed_search_household(p)
        env = (; r = p.r, benefit = p.benefit)
        res = solve_steady_state_given_env!(hh, env)
        m   = compute_moments(hh, res.Λ, env)

        # Stationary distribution is a probability measure; V is finite everywhere.
        @test isapprox(sum(res.Λ), 1.0; atol = 1e-6)
        @test all(isfinite, res.V)

        # Sensible moments: positive wealth, an interior unemployment rate.
        @test m.mean_wealth > 0
        @test 0.0 < m.unemp_rate < 0.5

        # Mean wage of the employed lies strictly inside the posted-wage range —
        # the directed-search tradeoff is operative (workers neither all crowd the
        # top submarket nor settle for the bottom).
        emp_rate  = 1 - m.unemp_rate
        mean_wage = m.wage_bill / emp_rate
        @test first(p.wages) < mean_wage < last(p.wages)

        # The unemployed's directed-search choice is interior: the modal submarket
        # is neither the lowest- nor the highest-wage one (the fill-prob/wage
        # tradeoff binds). Λ is (wealth, employment, submarket); employment 1 = unemp.
        unemp_by_sub = vec(sum(res.Λ[:, 1, :], dims = 1))
        modal = argmax(unemp_by_sub)
        @test 1 < modal < length(p.wages)

        # The fill-probability schedule is the Moen/Menzio–Shi tradeoff: decreasing.
        fs = fill_prob.(p.wages, Ref(p))
        @test all(diff(fs) .<= 0)
    end
end
