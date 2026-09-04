using Test
using HouseholdStages
using ForwardDiff

# PointwiseScaleStage: the two-sided diagonal scale `V_start = a·V_end`, `Λ_end = f·Λ_start`. An
# adjoint pair iff a == f; the discount (f=1), reproduction (a=1), and renorm specials ride it.

@testset "PointwiseScale — two-sided semantics (a backward, f forward)" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :z => Discrete([0.5, 1.5]),
    )
    stage = PointwiseScaleStage(layout; backward = 0.9, forward = 2.0)

    V_end = reshape(Float64.(1:6), (3, 2))
    @test backward!(stage, V_end, nothing) ≈ 0.9 .* V_end

    Λ_start = rand(3, 2)
    Λ_end = forward!(stage, Λ_start)
    @test Λ_end ≈ 2.0 .* Λ_start
    @test Λ_end !== Λ_start                       # written into the buffer
end

@testset "PointwiseScale — forward defaults to backward (the self-adjoint diagonal)" begin
    layout = GriddedLayout(:z => Discrete([1, 2, 3]))
    stage = PointwiseScaleStage(layout; backward = 0.5)
    V = [10.0, 20.0, 30.0]
    @test backward!(stage, V, nothing) ≈ 0.5 .* V
    Λ = [0.2, 0.5, 0.3]
    @test forward!(stage, Λ) ≈ 0.5 .* Λ           # f = backward = 0.5
end

@testset "PointwiseScale — FromEnv scales (both directions)" begin
    layout = GriddedLayout(:z => Discrete([1, 2, 3]))
    stage = PointwiseScaleStage(layout; backward = FromEnv(:a), forward = FromEnv(:f))
    @test Set(effective_env_slice(stage)) == Set((:a, :f))

    V = [1.0, 2.0, 4.0]
    @test backward!(stage, V, (a = 0.5, f = 3.0)) ≈ 0.5 .* V
    Λ = [0.1, 0.2, 0.7]
    @test forward!(stage, Λ) ≈ 3.0 .* Λ           # f seated at the same backward
end

@testset "PointwiseScale — adjoint dot-product identity (a ≠ f)" begin
    # ⟨backward!(V_end), dV_start⟩ = ⟨V_end, backward_adjoint!(dV_start)⟩ (backward K = a·Id),
    # ⟨forward!(Λ_start), dΛ_end⟩  = ⟨Λ_start, forward_adjoint!(dΛ_end)⟩  (forward  K = f·Id).
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :z => Discrete([0.5, 1.5]),
    )
    a, f = 0.97, 1.8
    stage = PointwiseScaleStage(layout; backward = a, forward = f)

    V_end    = randn(3, 2)
    dV_start = randn(3, 2)
    V_start  = backward!(stage, V_end, nothing)
    dV_end   = backward_adjoint!(stage, dV_start)
    @test dV_end ≈ a .* dV_start
    @test isapprox(sum(V_start .* dV_start), sum(V_end .* dV_end); atol = 1e-12)

    dΛ_end  = randn(3, 2)
    Λ_start = randn(3, 2)
    Λ_end   = forward!(stage, Λ_start)
    dΛ_start = forward_adjoint!(stage, dΛ_end)
    @test dΛ_start ≈ f .* dΛ_end
    @test isapprox(sum(Λ_end .* dΛ_end), sum(Λ_start .* dΛ_start); atol = 1e-12)
end

@testset "PointwiseScale — a == f is a genuine transpose pair (duality)" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :income => Discrete([0.5, 1.0, 1.5]),
    )
    stage = PointwiseScaleStage(layout; backward = 0.8)     # f = backward ⇒ self-adjoint
    V_out = randn(4, 3)
    Λ_in  = rand(4, 3)
    V_in  = copy(backward!(stage, V_out, nothing))
    Λ_out = copy(forward!(stage, Λ_in))
    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out); atol = 1e-12)
end

@testset "PointwiseScale — AD through a FromEnv scale (Dual β)" begin
    layout = GriddedLayout(:z => Discrete([1, 2, 3]))
    stage = lift_jacobian(PointwiseScaleStage(layout; backward = FromEnv(:a), forward = 1); n_dual = 1)
    D  = eltype(V_start_buffer(stage))
    ad = D(0.95, ForwardDiff.Partials((1.0,)))
    V  = [1.0, 2.0, 3.0]
    out = backward!(stage, V, (a = ad,))
    @test ForwardDiff.value.(out) ≈ 0.95 .* V
    @test [ForwardDiff.partials(o)[1] for o in out] ≈ V   # d/da (a·V) = V
end

@testset "PointwiseScale ≡ TimeDiscountingStage (backward = β, forward = 1)" begin
    layout = GriddedLayout(:z => Discrete([1, 2, 3]))
    β = 0.93
    td = TimeDiscountingStage(layout; β = β)
    ps = PointwiseScaleStage(layout; backward = β, forward = 1)
    @test td isa PointwiseScaleStage                       # TimeDiscounting is a thin constructor
    V = [1.0, 2.0, 3.0]; Λ = [0.2, 0.5, 0.3]
    @test backward!(td, V, nothing) ≈ backward!(ps, copy(V), nothing)
    @test forward!(td, Λ) ≈ forward!(ps, copy(Λ)) ≈ Λ      # forward = 1 ⇒ Λ copied
end
