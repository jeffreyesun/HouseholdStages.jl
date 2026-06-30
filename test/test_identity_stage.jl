using Test
using HouseholdStages

@testset "IdentityStage — backward and forward are no-ops" begin
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 1.0, 2.0]),
        :z => Discrete([0.5, 1.5]),
    )
    stage = IdentityStage(layout)

    V_end = randn(3, 2)
    V_start = backward!(stage, V_end, nothing)
    @test V_start == V_end
    @test V_start !== V_end   # written into the stage's buffer, not aliased

    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test Λ_end == Λ_start
end

@testset "IdentityStage ∘ MarkovStage has same effect as MarkovStage" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 1.0, 2.0]),
        :z => Discrete([0.5, 1.5]),
    )
    s_id   = IdentityStage(layout)
    s_mark = MarkovStage(layout; axis = :z, transition_matrix = P)
    chain_pre  = s_id ∘ s_mark
    chain_post = s_mark ∘ s_id

    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    V_seed = zeros(3, 2)                       # seat each Markov K = Tᵀ before forward! (§9 contract)
    backward!(chain_pre, V_seed, nothing)
    Λ_pre = copy(forward!(chain_pre, Λ_start))

    backward!(chain_post, V_seed, nothing)
    Λ_post = copy(forward!(chain_post, Λ_start))

    # Both compositions yield the same end-distribution as a bare MarkovStage.
    s_bare = MarkovStage(layout; axis = :z, transition_matrix = P)
    backward!(s_bare, V_seed, nothing)
    Λ_bare = forward!(s_bare, Λ_start)
    @test all(isapprox.(Λ_pre,  Λ_bare; atol = 1e-12))
    @test all(isapprox.(Λ_post, Λ_bare; atol = 1e-12))
end

@testset "IdentityStage — static_env_deps" begin
    @test static_env_deps(HouseholdStages.IdentityStageSpec) === NamedTuple()
end
