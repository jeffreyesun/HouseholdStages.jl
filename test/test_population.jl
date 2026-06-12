using Test
using HouseholdStages
using HouseholdStages: masses

@testset "GriddedPopulation — wrap / unwrap / uniform" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    Λ = uniform_distribution(layout)
    @test Λ isa GriddedPopulation
    @test size(masses(Λ)) == (3, 2)
    @test isapprox(sum(Λ), 1.0; atol = 1e-12)
    @test all(isapprox.(masses(Λ), 1 / 6; atol = 1e-12))

    raw = rand(3, 2)
    @test as_population(raw) isa GriddedPopulation
    @test masses(as_population(raw)) === raw
    @test as_population(Λ) === Λ            # idempotent on a population
    @test Array(Λ) === masses(Λ)
end

@testset "forward! on a GriddedPopulation = the kernel acting on the distribution" begin
    P = [0.8 0.2; 0.3 0.7]
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :z, transition_matrix = P)
    backward!(stage, zeros(3, 2), nothing)                  # seat the kernel

    Λ0 = uniform_distribution(layout)
    Λ1 = forward!(stage, Λ0)
    @test Λ1 isa GriddedPopulation                          # flows through unchanged in kind
    @test isapprox(sum(Λ1), 1.0; atol = 1e-12)              # mass conserved
    # Same numbers as the raw-array forward (the population is just a wrapper).
    @test masses(Λ1) ≈ forward!(stage, masses(Λ0))
end

@testset "expectation ⟨f, Λ⟩ and V/Λ duality through populations" begin
    P = [0.8 0.2; 0.3 0.7]
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    stage = MarkovStage(layout; axis = :z, transition_matrix = P)

    V_end = randn(2)
    Λ_in  = GriddedPopulation([0.4, 0.6])
    V_start = backward!(stage, V_end, nothing)
    Λ_out   = forward!(stage, Λ_in)

    # ⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩, evaluated with the population inner product.
    @test isapprox(expectation(V_start, Λ_in), expectation(V_end, Λ_out); atol = 1e-12)
    @test expectation([1.0, 1.0], Λ_in) ≈ sum(Λ_in)        # total mass
end
