using Test
using HouseholdStages

@testset "resolve — literal pass-through" begin
    @test resolve(0.25, (a = 1.0,)) === 0.25
    @test resolve(0.5f0, NamedTuple()) === 0.5f0
    @test resolve(3, (foo = 1,)) === 3
end

@testset "resolve — FromEnv lookup" begin
    @test resolve(FromEnv(:foo), (foo = 0.5,)) === 0.5
    @test resolve(FromEnv(:n),   (n = 7,)) === 7
end

@testset "resolve — missing env key throws" begin
    @test_throws Exception resolve(FromEnv(:missing), (other = 1.0,))
end

@testset "spec construction — literal ε / β" begin
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoiceStage(layout;
        choice_axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = 0.5,
    )
    @test stage.spec.ε === 0.5
    @test isempty(effective_env_slice(stage))
end

@testset "spec construction — FromEnv ε resolved from env" begin
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoiceStage(layout;
        choice_axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = FromEnv(:ξ),
    )
    @test stage.spec.ε === FromEnv(:ξ)
    @test :ξ in effective_env_slice(stage)
    V_start = backward!(stage, Float64[0.0, 0.0], (ξ = 0.5,))
    @test all(isfinite, V_start)
end
