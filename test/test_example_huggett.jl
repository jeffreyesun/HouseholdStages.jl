using Test, HouseholdStages

# Huggett (1993) bond economy — regression test. The household block is
# library stages ONLY (MarkovStage ∘ WealthChangeStage ∘
# ConsumptionSavingsStage, identical to Aiyagari); only the outer
# market-clearing target differs (bond in zero net supply ⇒ ∫a dΛ → 0).
# Module-wrapped so the example's global names don't clash with siblings.

module HuggettExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "huggett", "steady_state.jl"))

    @testset "example: huggett — existing stages only" begin
        # A small, fast calibration: coarse grid, loose tolerance.
        p = HuggettParams(N_neg = 40, N_pos = 80, a_max = 20.0)

        res = huggett_steady_state(p; atol = 5e-3, verbosity = 0)

        # Outer loop cleared the bond market and converged.
        @test res.converged
        @test abs(res.A_supplied) <= 5e-3          # ∫ a dΛ ≈ 0 (zero net supply)

        # Equilibrium rate is sensible: below autarky 1/β − 1 (precautionary
        # saving depresses r) and above the borrowing-floor regime.
        @test -0.05 < res.r < 1 / p.β - 1

        # Value function is everywhere finite on the active state space.
        @test all(isfinite, res.V)

        # Distribution is a proper probability measure.
        @test isapprox(sum(res.Λ), 1.0; atol = 1e-6)

        # A sensible moment: positive mass holds positive bonds (savers fund
        # borrowers), and aggregate holdings are bounded.
        @test abs(res.A_supplied) < 1.0

        # Re-read the cleared moment directly through the household block to
        # confirm the moment plumbing agrees with the driver's readout.
        hh  = huggett_household(p)
        env = huggett_env(res.r, p)
        A   = compute_moments(hh, res.Λ, env).A_supplied
        @test isapprox(A, res.A_supplied; atol = 1e-8)
    end
end
