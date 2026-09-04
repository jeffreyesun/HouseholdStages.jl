using Test
using HouseholdStages

@testset "HouseholdStages" begin

include("test_layout.jl")
include("test_axis_ops.jl")
include("test_closures.jl")
include("test_env_resolution.jl")
include("test_payoff_fn.jl")
include("test_scalar_field.jl")
include("test_stratified.jl")
include("test_workspace.jl")
include("test_kernel.jl")
include("test_transition_flexibility.jl")
include("test_markov_along.jl")
include("test_discrete_choice.jl")
include("test_migration.jl")
include("test_sector_switching.jl")
include("test_logit_utility.jl")
include("test_rational_inattention.jl")
include("test_buy_home.jl")
include("test_sell_home.jl")
include("test_argmax.jl")
include("test_continuous_argmax_stage.jl")
include("test_refill_policy.jl")
include("test_deterministic_continuous.jl")
include("test_identity_stage.jl")
include("test_utility_stage.jl")
include("test_pointwise_scale.jl")
include("test_time_discounting.jl")
include("test_entry_reproduction.jl")
include("test_entry_exit.jl")
include("test_exit.jl")
include("test_mass_nonconservation.jl")
include("test_income.jl")
include("test_advance_age.jl")
include("test_env_deps.jl")
include("test_moments.jl")
include("test_product.jl")
include("test_stage_accessors.jl")
include("test_nesting_matrix.jl")
include("test_mixing.jl")
include("test_search_matching.jl")
include("test_mean_preserving_spread.jl")
include("test_gaussian_loading.jl")
include("test_streaming_choice_model.jl")
include("test_domain_wrappers.jl")
include("test_population.jl")
include("test_asset_price_change.jl")
include("test_borrowing_constraint.jl")
include("test_lift_jacobian.jl")
include("test_device_move.jl")
include("test_sequence_space.jl")
include("test_outer_loop.jl")
include("test_user_helpers.jl")

@testset "ForgetfulSumStage — drops one axis, 3D → 2D" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :income => Discrete([0.5, 1.0, 1.5]),
        :taste => Discrete([:a, :b, :c, :d, :e]),
    )
    stage = ForgetfulSumStage(layout; axis = :taste)
    # Decision 7: the forget axis is RESIZED to a single level, not dropped — the
    # output keeps it at size 1 in the same tuple position.
    @test layout_size(end_layout(stage)) == (4, 3, 1)
    @test size(stage.scratch.V_start) == (4, 3, 5)
    @test size(stage.scratch.Λ_end)   == (4, 3, 1)

    V_end = reshape(Float64.(1:12), (4, 3, 1))
    V_start = backward!(stage, V_end, nothing)
    @test size(V_start) == (4, 3, 5)
    for t in 1:5
        @test V_start[:, :, t] == V_end[:, :, 1]
    end

    Λ_start = rand(Float64, 4, 3, 5); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test size(Λ_end) == (4, 3, 1)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
    expected = sum(Λ_start; dims = 3)
    @test all(isapprox.(Λ_end, expected; atol = 1e-14))
end

@testset "ForgetfulSumStage — drops a middle axis, 4D → 3D" begin
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 1.0, 2.0]),
        :z => Discrete([0.5, 1.0, 1.5, 2.0]),
        :transient => Discrete([:lo, :hi]),
        :loc => Discrete([:A, :B, :C, :D, :E]),
    )
    stage = ForgetfulSumStage(layout; axis = :transient)
    # :transient resized to size 1, kept in position 3.
    @test layout_size(end_layout(stage)) == (3, 4, 1, 5)

    V_end = randn(3, 4, 1, 5)
    V_start = backward!(stage, V_end, nothing)
    for w in 1:3, z in 1:4, t in 1:2, loc in 1:5
        @test V_start[w, z, t, loc] == V_end[w, z, 1, loc]
    end

    Λ_start = rand(3, 4, 2, 5); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test isapprox(sum(Λ_end), sum(Λ_start); atol = 1e-12)
    expected = sum(Λ_start; dims = 3)
    @test all(isapprox.(Λ_end, expected; atol = 1e-14))
end

@testset "ForgetfulSumStage — duality identity" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 0.5, 1.0, 1.5]),
        :income => Discrete([0.5, 1.0, 1.5]),
        :taste => Discrete([:a, :b, :c, :d, :e]),
    )
    stage = ForgetfulSumStage(layout; axis = :income)

    V_out  = randn(4, 1, 5)                         # :income resized to size 1
    Λ_in   = rand(4, 3, 5); Λ_in ./= sum(Λ_in)

    V_in  = backward!(stage, V_out, nothing)
    Λ_out = forward!(stage, Λ_in)

    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out); atol = 1e-12)
end

@testset "ChainStage — composition is associative and length-2 works" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P)
    chain = s1 ∘ s2
    @test chain isa ChainStage
    @test length(chain.spec.stages) == 2

    chain3 = (s1 ∘ s2) ∘ s1
    chain3b = s1 ∘ (s2 ∘ s1)
    @test length(chain3.spec.stages) == length(chain3b.spec.stages) == 3

    # End-to-end: a uniform mass under symmetric P should remain a probability vector.
    # First seed the kernel via a backward at zero V_end.
    backward!(chain, zeros(2), nothing)
    Λ = ones(2) ./ 2
    Λ_end = forward!(chain, copy(Λ))
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end

end
