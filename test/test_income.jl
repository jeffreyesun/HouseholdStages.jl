using Test
using HouseholdStages

# IncomeStage is a domain wrapper over WealthChangeStage with the standard cash-on-hand
# receipt `(1+r)·wealth + w·income`. These check the default matches a hand-built
# WealthChangeStage and that the `axis` override + custom `wealth_post` both work.

@testset "IncomeStage — default receipt equals the hand-built WealthChange" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0, 4.0]),
        :income => Discrete([0.6, 1.0, 1.4]),
    )
    env = (; r = 0.04, w = 1.1)

    inc = IncomeStage(layout)
    wc  = WealthChangeStage(layout;
            wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income)

    @test inc isa DeterministicContinuousStage   # WealthChange (hence Income) returns the primitive

    V_end = reshape(Float64.(1:15), (5, 3))
    @test backward!(inc, V_end, env) == backward!(wc, V_end, env)

    Λ = rand(5, 3); Λ ./= sum(Λ)
    @test forward!(inc, copy(Λ)) == forward!(wc, copy(Λ))
end

@testset "IncomeStage — custom wealth_post overrides the default" begin
    layout = GriddedLayout(
        :assets => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :income => Discrete([1.0, 2.0]),
    )
    env = (; r = 0.02, w = 1.0)
    # A pure-return receipt on a differently-named wealth axis, no labour income.
    inc = IncomeStage(layout; axis = :assets,
                      wealth_post = (; assets, env) -> (1 + env.r) * assets)
    wc  = WealthChangeStage(layout; axis = :assets,
                      wealth_post = (; assets, env) -> (1 + env.r) * assets)
    V_end = reshape(Float64.(1:8), (4, 2))
    @test backward!(inc, V_end, env) == backward!(wc, V_end, env)
end

@testset "IncomeStage — non-:income shock axis via income_axis" begin
    # Regression: the default wealth_post hard-coded the kwarg `income`; `income_axis` generalises it
    # so a shock axis named anything (`:employment`, …) works with the default budget.
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
                           :employment => Discrete([0.0, 1.0]))
    env = (; r = 0.02, w = 1.0)
    inc = IncomeStage(layout; income_axis = :employment)
    wc  = WealthChangeStage(layout; axis = :wealth,
                            wealth_post = (; wealth, employment, env) -> (1 + env.r) * wealth + env.w * employment)
    V_end = reshape(Float64.(1:8), (4, 2))
    @test backward!(inc, V_end, env) == backward!(wc, V_end, env)
end
