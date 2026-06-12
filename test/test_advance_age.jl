using Test
using HouseholdStages

@testset "AdvanceAgeStage — is a MarkovStage on the age axis" begin
    layout = GriddedLayout(
        StateAxis(:age,    discrete_finite([1, 2, 3, 4])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = AdvanceAgeStage(layout; age_axis = :age)
    @test stage isa MarkovStage
    @test stage.spec.axis === :age
    T = stage.spec.transition_matrix
    @test size(T) == (4, 4)
    # Shift-by-one with absorbing top: T[a,a+1]=1, T[end,end]=1.
    @test T[1, 2] == 1.0 && T[2, 3] == 1.0 && T[3, 4] == 1.0 && T[4, 4] == 1.0
    @test sum(T) == 4.0           # exactly one nonzero per row
end

@testset "AdvanceAgeStage — backward reads continuation at age a+1" begin
    layout = GriddedLayout(StateAxis(:age, discrete_finite([1, 2, 3, 4])))
    stage = AdvanceAgeStage(layout)

    V_end = [10.0, 20.0, 30.0, 40.0]
    V_start = backward!(stage, V_end, nothing)
    # V_start[a] = V_end[a+1]; top (absorbing) reads itself.
    @test V_start ≈ [20.0, 30.0, 40.0, 40.0]
end

@testset "AdvanceAgeStage — forward pushes mass one age up (absorbing top)" begin
    layout = GriddedLayout(StateAxis(:age, discrete_finite([1, 2, 3, 4])))
    stage = AdvanceAgeStage(layout; absorb_top = true)
    backward!(stage, zeros(4), nothing)             # seat the kernel

    Λ_start = [0.4, 0.3, 0.2, 0.1]
    Λ_end = forward!(stage, Λ_start)
    # Age-a mass moves to a+1; top cohort absorbs (0.2→4, plus its own 0.1).
    @test Λ_end ≈ [0.0, 0.4, 0.3, 0.3]
    @test isapprox(sum(Λ_end), sum(Λ_start); atol = 1e-12)   # mass conserved
end

@testset "AdvanceAgeStage — non-absorbing top drops the terminal cohort" begin
    layout = GriddedLayout(StateAxis(:age, discrete_finite([1, 2, 3, 4])))
    stage = AdvanceAgeStage(layout; absorb_top = false)

    T = stage.spec.transition_matrix
    @test T[4, 4] == 0.0 && all(T[4, :] .== 0.0)   # terminal row empties
    backward!(stage, zeros(4), nothing)             # seat the kernel

    Λ_start = [0.4, 0.3, 0.2, 0.1]
    Λ_end = forward!(stage, Λ_start)
    # Top cohort (0.1) rolls off into nonexistence; the rest shift up.
    @test Λ_end ≈ [0.0, 0.4, 0.3, 0.2]
    @test isapprox(sum(Λ_end), sum(Λ_start) - 0.1; atol = 1e-12)   # 0.1 lost
end

@testset "AdvanceAgeStage — custom age axis name and non-leading position" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0])),
        StateAxis(:cohort, discrete_finite([1, 2, 3])),
    )
    stage = AdvanceAgeStage(layout; age_axis = :cohort)
    @test stage.spec.axis === :cohort
    @test size(stage.spec.transition_matrix) == (3, 3)

    # Backward over a non-leading axis.
    V_end = [1.0 2.0 3.0; 4.0 5.0 6.0]   # (wealth, cohort)
    V_start = backward!(stage, V_end, nothing)
    @test V_start ≈ [2.0 3.0 3.0; 5.0 6.0 6.0]
end

@testset "AdvanceAgeStage — duality identity (pure Markov, r = 0)" begin
    layout = GriddedLayout(
        StateAxis(:age,    discrete_finite([1, 2, 3, 4])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = AdvanceAgeStage(layout; absorb_top = true)

    V_out = randn(4, 2)
    Λ_in  = rand(4, 2); Λ_in ./= sum(Λ_in)

    V_in  = copy(backward!(stage, V_out, nothing))
    Λ_out = copy(forward!(stage, Λ_in))
    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out); atol = 1e-12)
end
