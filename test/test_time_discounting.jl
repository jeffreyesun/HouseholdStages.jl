using Test
using HouseholdStages

@testset "TimeDiscountingStage — backward scales V_end by β" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :income => Discrete([0.5, 1.0]),
    )
    stage = TimeDiscountingStage(layout; β = 0.96)

    V_end = reshape(Float64.(1:6), (3, 2))
    V_start = backward!(stage, V_end, nothing)
    @test all(isapprox.(V_start, 0.96 .* V_end; atol = 1e-12))
end

@testset "TimeDiscountingStage — default β = 1 is identity on V" begin
    layout = GriddedLayout(:z => Discrete([1, 2, 3]))
    stage = TimeDiscountingStage(layout)
    V_end = [10.0, 20.0, 30.0]
    @test backward!(stage, V_end, nothing) ≈ V_end
end

@testset "TimeDiscountingStage — β from env (FromEnv)" begin
    layout = GriddedLayout(:z => Discrete([1, 2, 3]))
    stage = TimeDiscountingStage(layout; β = FromEnv(:β))
    @test effective_env_slice(stage) == (:β,)

    V_end = [1.0, 2.0, 4.0]
    V_start = backward!(stage, V_end, (β = 0.5,))
    @test V_start ≈ [0.5, 1.0, 2.0]
end

@testset "TimeDiscountingStage — forward is identity on Λ" begin
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 0.5, 1.0]),
        :z => Discrete([:a, :b]),
    )
    stage = TimeDiscountingStage(layout; β = 0.9)
    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test Λ_end == Λ_start
    @test Λ_end !== Λ_start              # written into the buffer
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)   # mass conserved
end

@testset "TimeDiscountingStage — duality identity (flow r = (β-1)V_end)" begin
    # ⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩ with Λ_out = Λ_in and
    # V_in = β·V_out, so r = V_in - V_out = (β-1)·V_out.
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :income => Discrete([0.5, 1.0, 1.5]),
    )
    β = 0.94
    stage = TimeDiscountingStage(layout; β = β)

    V_out = randn(4, 3)
    Λ_in  = rand(4, 3); Λ_in ./= sum(Λ_in)

    V_in  = copy(backward!(stage, V_out, nothing))
    Λ_out = copy(forward!(stage, Λ_in))

    r = V_in .- V_out
    @test isapprox(sum(V_in .* Λ_in),
                   sum(V_out .* Λ_out) + sum(r .* Λ_in); atol = 1e-12)
end

@testset "TimeDiscountingStage — adjoint dot-product identity" begin
    # ⟨backward!(V_end), dV_start⟩ = ⟨V_end, backward_adjoint!(dV_start)⟩
    # since backward K = β·Id is linear and self-adjoint up to the scalar.
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :z => Discrete([0.5, 1.5]),
    )
    β = 0.97
    stage = TimeDiscountingStage(layout; β = β)

    V_end    = randn(3, 2)
    dV_start = randn(3, 2)
    V_start  = backward!(stage, V_end, nothing)
    dV_end   = backward_adjoint!(stage, dV_start)
    @test dV_end ≈ β .* dV_start
    @test isapprox(sum(V_start .* dV_start), sum(V_end .* dV_end); atol = 1e-12)

    # Forward adjoint of the identity K is the identity.
    dΛ_end   = randn(3, 2)
    Λ_start  = rand(3, 2); Λ_start ./= sum(Λ_start)
    Λ_end    = forward!(stage, Λ_start)
    dΛ_start = forward_adjoint!(stage, dΛ_end)
    @test dΛ_start == dΛ_end
    @test isapprox(sum(Λ_end .* dΛ_end), sum(Λ_start .* dΛ_start); atol = 1e-12)
end

@testset "TimeDiscountingStage — composes with UtilityStage (β·V + u)" begin
    layout = GriddedLayout(:z => Discrete([1.0, 2.0, 3.0]))
    discount = TimeDiscountingStage(layout; β = 0.5)
    util     = UtilityStage(layout; utility = (; z) -> z)

    # `∘` is left-to-right time-ordered, so the backward sweep runs the
    # rightmost stage (util) first then discount: V_start = 0.5·(z + 0) = 0.5·z.
    chain = discount ∘ util
    V_start = backward!(chain, zeros(3), nothing)
    @test V_start ≈ 0.5 .* [1.0, 2.0, 3.0]
end

@testset "TimeDiscountingStage — static_env_deps empty by default" begin
    layout = GriddedLayout(:z => Discrete([1, 2]))
    plain = TimeDiscountingStage(layout; β = 0.9)
    @test effective_env_slice(plain) == ()
end
