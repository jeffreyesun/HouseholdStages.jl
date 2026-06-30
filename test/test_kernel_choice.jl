using Test
using HouseholdStages
using LinearAlgebra

# KernelChoiceStage (gridded rung d): the household chooses θ in a family {K_θ} of x-transitions
# at cost c(θ). Derived as Collapse[θ] ∘ Markov[x|θ-dep] ∘ ForgetfulSum[θ] over a θ-FULL internal
# layout; the choice axis is a singleton at the block boundary. (The hand-checked prototype this
# pins lived in proto_kernelchoice.jl.)

nx, nθ = 3, 2
xs = [1.0, 2.0, 3.0]
K  = [[0.8 0.2 0.0; 0.1 0.8 0.1; 0.0 0.2 0.8],          # K_θ[from, to], row-stochastic
      [0.5 0.5 0.0; 0.3 0.4 0.3; 0.0 0.5 0.5]]
c  = [0.0, 0.3]

# The block carries the choice axis (:θ) as a size-1 singleton.
block = GriddedLayout(:x => Discrete(xs), :θ => Discrete([1]))

@testset "kernel-choice — hard (argmax) value = max_θ[-c(θ) + K_θ W]" begin
    stage   = KernelChoiceStage(block; choice_axis = :θ, x_axis = :x, kernels = K, cost = c)
    W       = reshape([4.0, 1.0, 2.0], nx, 1)
    V_start = backward!(stage, W, NamedTuple())
    expected = [maximum(-c[θ] + dot(K[θ][x, :], vec(W)) for θ in 1:nθ) for x in 1:nx]
    @test size(V_start) == (nx, 1)                       # θ collapsed back to a singleton
    @test vec(V_start) ≈ expected

    # Forward + frozen-policy duality (reward-carrying: ⟨V_start,Λ⟩ = ⟨reward(θ*),Λ⟩ + ⟨W,Λ_end⟩).
    Λ_start  = reshape([0.2, 0.5, 0.3], nx, 1)
    Λ_end    = forward!(stage, Λ_start)
    θstar    = [argmax([-c[θ] + dot(K[θ][x, :], vec(W)) for θ in 1:nθ]) for x in 1:nx]
    reward_θ = [-c[θstar[x]] for x in 1:nx]
    @test size(Λ_end) == (nx, 1)
    @test sum(V_start .* Λ_start) ≈ sum(reward_θ .* vec(Λ_start)) + sum(W .* Λ_end)
end

@testset "kernel-choice — soft (logit) → hard as ε → 0" begin
    W        = reshape([4.0, 1.0, 2.0], nx, 1)
    hard     = vec(backward!(KernelChoiceStage(block; choice_axis = :θ, x_axis = :x, kernels = K, cost = c), W, NamedTuple()))
    softV    = vec(backward!(KernelChoiceStage(block; choice_axis = :θ, x_axis = :x, kernels = K, cost = c, soft = true, ε = 0.01), W, NamedTuple()))
    @test softV ≈ hard atol = 0.05                       # low temperature ⇒ logsumexp → max

    # Soft value is the explicit logsumexp over θ at a moderate temperature.
    ε        = 0.5
    softstg  = KernelChoiceStage(block; choice_axis = :θ, x_axis = :x, kernels = K, cost = c, soft = true, ε = ε)
    Vsoft    = vec(backward!(softstg, W, NamedTuple()))
    expected = [ε * log(sum(exp((-c[θ] + dot(K[θ][x, :], vec(W))) / ε) for θ in 1:nθ)) for x in 1:nx]
    @test Vsoft ≈ expected
end

@testset "kernel-choice — singleton-axis invariant enforced" begin
    bad = GriddedLayout(:x => Discrete(xs), :θ => Discrete([1, 2]))
    @test_throws AssertionError KernelChoiceStage(bad; choice_axis = :θ, x_axis = :x, kernels = K, cost = c)
end

@testset "PortfolioStage — frontier instantiation" begin
    pblock = GriddedLayout(:wealth => Discrete(xs), :share => Discrete([1]))
    ps     = PortfolioStage(pblock; kernels = K, cost = c)
    W      = reshape([4.0, 1.0, 2.0], nx, 1)
    V      = vec(backward!(ps, W, NamedTuple()))
    @test V ≈ [maximum(-c[θ] + dot(K[θ][x, :], vec(W)) for θ in 1:nθ) for x in 1:nx]
end
