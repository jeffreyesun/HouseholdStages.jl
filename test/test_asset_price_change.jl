using Test
using HouseholdStages

@testset "AssetPriceChangeStage — constructor returns a DeterministicContinuousStage" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :h => Discrete([0.0, 1.0, 2.0]),
    )
    stage = AssetPriceChangeStage(layout; holdings_axis = :h)
    @test stage isa DeterministicContinuousStage
    @test stage.spec.axis === :wealth
end

@testset "AssetPriceChangeStage — wealth_post closure matches hand-built recipe" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0, 4.0]),
        :h => Discrete([0.0, 1.0, 2.0]),
    )
    stage_sugar = AssetPriceChangeStage(layout; holdings_axis = :h)
    stage_hand  = WealthChangeStage(layout;
        wealth_post = (; wealth, h, env) -> wealth + (env.q - env.q_last) * h,
        axis = :wealth,
    )

    env = (; q = 1.10, q_last = 1.00)

    V_end = reshape(Float64.(1:15), (5, 3))
    V_s = copy(backward!(stage_sugar, V_end, env))
    V_h = copy(backward!(stage_hand,  V_end, env))
    @test all(isapprox.(V_s, V_h; atol = 1e-12))

    Λ_start = rand(5, 3); Λ_start ./= sum(Λ_start)
    Λ_s = copy(forward!(stage_sugar, Λ_start))
    Λ_h = copy(forward!(stage_hand,  Λ_start))
    @test all(isapprox.(Λ_s, Λ_h; atol = 1e-12))
end

@testset "AssetPriceChangeStage — q == q_last is identity (up to interpolation)" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :h => Discrete([0.0, 1.0]),
    )
    stage = AssetPriceChangeStage(layout; holdings_axis = :h)
    env   = (; q = 1.0, q_last = 1.0)

    V_end = reshape(Float64.(1:8), (4, 2))
    V_start = backward!(stage, V_end, env)
    @test all(isapprox.(V_start, V_end; atol = 1e-12))

    Λ_start = rand(4, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test all(isapprox.(Λ_end, Λ_start; atol = 1e-12))
end

@testset "AssetPriceChangeStage — custom field names" begin
    layout = GriddedLayout(
        :b => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :k => Discrete([0.0, 1.0, 2.0]),
    )
    stage = AssetPriceChangeStage(layout;
        holdings_axis = :k,
        wealth_axis   = :b,
        q_field       = :p_now,
        q_last_field  = :p_prev,
    )
    @test stage.spec.axis === :b

    env = (; p_now = 1.5, p_prev = 1.0)
    V_end = reshape(Float64.(1:12), (4, 3))
    @test backward!(stage, V_end, env) isa AbstractArray
end

@testset "AssetPriceChangeStage — single-asset case (holdings_axis == wealth_axis)" begin
    # Regression: `DepClosure((:wealth, :wealth), …)` → `NamedTuple{(:wealth, :wealth)}` threw
    # "duplicate field name". The declared axes must be deduped when the asset IS the wealth axis.
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]), :z => Discrete([1.0, 2.0]))
    stage = AssetPriceChangeStage(layout; holdings_axis = :wealth, wealth_axis = :wealth)
    @test stage.spec.axis === :wealth
    env = (; q = 1.5, q_last = 1.0)
    V_end = reshape(Float64.(1:8), (4, 2))
    @test backward!(stage, V_end, env) isa AbstractArray   # builds + runs (revaluation 0.5·wealth)
end
