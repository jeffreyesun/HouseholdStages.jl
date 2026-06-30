using Test, HouseholdStages

# Wrapped in a module so the example's global names (`u_crra`, params struct,
# `life_cycle_household`, …) don't clash with the other example tests' globals.
module LifeCycleExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "life_cycle", "steady_state.jl"))

    @testset "example: life_cycle — existing stages only" begin
        # A small, fast calibration (finite horizon, modest grid).
        p   = LifeCycleParams(N = 20, N_w = 60, w_max = 40.0)
        res = life_cycle_solve(p; verbosity = 0)

        nw, nε, N = p.N_w, length(p.ε_grid), p.N

        # Stacked tensors are the right shape and finite.
        @test size(res.V) == (nw, nε, N)
        @test size(res.Λ) == (nw, nε, N)
        @test all(isfinite, res.V)

        # Each age-slice is a unit-mass cohort (forward sweep conserves mass).
        per_age_mass = vec(sum(res.Λ; dims = (1, 2)))
        @test all(m -> isapprox(m, 1.0; atol = 1e-8), per_age_mass)

        # A sensible moment: cross-sectional mean wealth is strictly positive
        # and lies inside the wealth grid.
        @test res.mean_wealth > 0
        @test res.mean_wealth < p.w_max

        # The life-cycle wealth profile is a hump: zero at birth, strictly
        # positive and larger by mid-life, and the peak is interior (not at
        # the first or last age) — the Gourinchas–Parker / CGM signature.
        prof = res.age_mean_wealth
        @test prof[1] ≈ 0 atol = 1e-10           # newborns hold zero wealth
        @test maximum(prof) > prof[1]
        @test 1 < argmax(prof) < N

        # The savings policy is a valid wealth-grid index for every age.
        for pol in res.policies
            @test all(i -> 1 <= i <= nw, pol)
        end

        @info "life_cycle: mean wealth (x-section) = $(round(res.mean_wealth; digits=4)), " *
              "peak wealth $(round(maximum(prof); digits=4)) at age $(argmax(prof))/$N"
    end
end
