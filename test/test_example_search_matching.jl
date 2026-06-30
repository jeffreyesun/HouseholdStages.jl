using Test, HouseholdStages

# Regression test for the search-and-matching literature example. The model
# include is wrapped in a module so its global names (`u_crra`, params, helpers)
# cannot clash with other examples' identically-named definitions in the suite.
module SearchMatchingExampleTest
    using Test, HouseholdStages
    include(joinpath(@__DIR__, "..", "examples", "search_matching", "steady_state.jl"))  # includes model.jl

    @testset "example: search_matching — existing stages only" begin
        # A small grid keeps the test fast; calibration unchanged.
        p = SearchMatchingParams(N_w = 60)

        # --- Partial equilibrium at exogenous θ ---------------------------------
        pe = search_matching_pe(p; θ = 1.5, verbosity = 0)

        @test sum(pe.Λ) ≈ 1.0 atol = 1e-6           # stationary distribution: mass 1
        @test all(isfinite, pe.V)                   # finite value everywhere
        @test 0.0 < pe.employment < 1.0             # an interior employment rate
        @test pe.K > 0.0                            # positive aggregate savings

        # Tightness comparative static: a tighter market raises employment.
        emp_loose = search_matching_pe(p; θ = 0.5, verbosity = 0).employment
        emp_tight = search_matching_pe(p; θ = 4.0, verbosity = 0).employment
        @test emp_tight > emp_loose

        # --- Free-entry general equilibrium -------------------------------------
        ge = search_matching_ge(p; verbosity = 0)
        @test sum(ge.Λ) ≈ 1.0 atol = 1e-6
        @test ge.θ > 0.0                            # a positive equilibrium tightness
        @test abs(ge.residual) < 1e-4               # free-entry condition (nearly) holds
        @test 0.0 < ge.employment < 1.0
        @test ge.K > 0.0
    end
end
