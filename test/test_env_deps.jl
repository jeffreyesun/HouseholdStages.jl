using Test
using HouseholdStages

@testset "static_env_deps defaults to empty NamedTuple" begin
    @test static_env_deps(HouseholdStages.MarkovStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.ArgmaxStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.LogitChoiceStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.IdentityStageSpec) === NamedTuple()
end

@testset "effective_env_slice is empty when closures aren't introspected" begin
    # User closures read `env.bonus`, but the package no longer requires
    # `closure_deps` to be declared. effective_env_slice reflects only
    # static_env_deps + env-resolved spec fields, so a stage whose env
    # reads flow exclusively through closures has an empty slice.
    layout = GriddedLayout(:a => Discrete([1, 2]))
    stage = ArgmaxStage(layout;
        axis = :a,
        reward = (; env) -> [after == 1 ? 0.0 : env.bonus for after in 1:2, before in 1:2],
    )
    @test isempty(effective_env_slice(stage))
end

@testset "effective_env_slice picks up env-resolved Symbol field" begin
    layout = GriddedLayout(:a => Discrete([1, 2]))
    stage = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = FromEnv(:ξ),
    )
    @test :ξ in effective_env_slice(stage)
end

@testset "validate_env catches missing env keys" begin
    layout = GriddedLayout(:a => Discrete([1, 2]))
    stage = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = FromEnv(:ξ),
    )
    @test_throws ErrorException validate_env(stage, NamedTuple())
    @test validate_env(stage, (ξ = 0.5,)) === nothing
end

@testset "chain_env_names merges per-stage env-resolved slices" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = GriddedLayout(:a => Discrete([1, 2]))
    s1 = MarkovStage(layout; axis = :a, transition_matrix = P)
    s2 = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = FromEnv(:ξ),
    )
    chain = s1 ∘ s2
    @test :ξ in chain_env_names(chain)
end

@testset "calibrated vs env-resolved are discriminated by value type" begin
    # Pure-Symbol label fields (like `choice_axis = :a`) are not env-resolved;
    # the discrimination is by value type — `isa FromEnv` marks env-resolution,
    # everything else (including pure Symbols) is a literal.
    layout = GriddedLayout(:a => Discrete([1, 2]))
    calibrated = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = 0.5,
    )
    @test isempty(effective_env_slice(calibrated))

    swept = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = FromEnv(:ξ),
    )
    @test :ξ in effective_env_slice(swept)
end
