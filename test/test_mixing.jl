using Test
using HouseholdStages
using LinearAlgebra

# MixingStage (closed-form rung a): V = b + c*(a−b), a = K_A·W, b = K_B·W, two Markov applies + a
# pointwise conjugate (no θ grid). RetentionStage = MixingStage with K_A = I.

nx = 3
xs    = [1.0, 2.0, 3.0]
block = GriddedLayout(:x => Discrete(xs))
K_A   = [0.9 0.1 0.0; 0.1 0.8 0.1; 0.0 0.1 0.9]
K_B   = [0.4 0.4 0.2; 0.3 0.4 0.3; 0.2 0.4 0.4]
κ     = 2.0

# Closed-form reference for the quadratic cost c(θ)=θ²/(2κ), θ∈[0,1].
ref_value(W) = begin
    a = K_A * W; b = K_B * W; y = a .- b
    θ = clamp.(κ .* y, 0.0, 1.0)
    (b .+ (θ .* y .- θ.^2 ./ (2κ)), θ)
end

@testset "mixing — value = b + c*(a−b) and θ* = clamp(κ(a−b),0,1)" begin
    stage   = MixingStage(block; axis = :x, K_A = K_A, K_B = K_B, cost_curvature = κ)
    W       = [4.0, 1.0, 2.0]
    V_start = backward!(stage, W, NamedTuple())
    Vref, θref = ref_value(W)
    @test V_start ≈ Vref
    @test policy(stage) ≈ θref

    # Brute max over a fine θ grid agrees with the closed form (the conjugate IS the max).
    θgrid = range(0, 1; length = 2001)
    brute = [maximum(θ * (K_A * W)[x] + (1 - θ) * (K_B * W)[x] - θ^2 / (2κ) for θ in θgrid) for x in 1:nx]
    @test V_start ≈ brute atol = 1e-3
end

@testset "mixing — frozen-policy duality ⟨V,Λ⟩ = ⟨−c(θ*),Λ⟩ + ⟨W,Λ_end⟩" begin
    stage   = MixingStage(block; axis = :x, K_A = K_A, K_B = K_B, cost_curvature = κ)
    W       = [4.0, 1.0, 2.0]
    V_start = backward!(stage, W, NamedTuple())
    Λ_start = [0.2, 0.5, 0.3]
    Λ_end   = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ sum(Λ_start)                              # both corners conserve mass ⇒ mixture does
    rew = HouseholdStages.reward(stage)                          # −c(θ*)
    @test sum(V_start .* Λ_start) ≈ sum(rew .* Λ_start) + sum(W .* Λ_end)
end

@testset "retention — K_A = I, pay to stay; V = exit + c*(W − exit)" begin
    K_exit = K_B
    stage  = RetentionStage(block; axis = :x, exit_kernel = K_exit, cost_curvature = κ)
    W      = [4.0, 1.0, 2.0]
    V      = backward!(stage, W, NamedTuple())
    exitV  = K_exit * W
    y      = W .- exitV
    θ      = clamp.(κ .* y, 0.0, 1.0)
    @test V ≈ exitV .+ (θ .* y .- θ.^2 ./ (2κ))
    @test policy(stage) ≈ θ
    # Where staying dominates (W > exit value), θ* > 0; where it doesn't, θ* = 0.
    @test all((policy(stage) .> 0) .== (y .> 0))
end

@testset "mixing — forward-mode AD through the conjugate" begin
    # The pointwise conjugate is Dual-compatible; lift_jacobian(:forward) propagates a tangent.
    stage = MixingStage(block; axis = :x, K_A = K_A, K_B = K_B, cost_curvature = κ)
    dstage = lift_jacobian(stage; mode = :forward)
    W = [4.0, 1.0, 2.0]
    V = backward!(dstage, W, NamedTuple())
    @test eltype(V) <: HouseholdStages.ForwardDiff.Dual
end
