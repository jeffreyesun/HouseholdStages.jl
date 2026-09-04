using Test
using HouseholdStages

@testset "ArgmaxStage — 2-state re-choose, action 2 always preferred (+1)" begin
    layout = GriddedLayout(:s => Discrete([:A, :B]))
    # Reward matrix on :s: choosing :B (after index 2) scores +1, :A scores 0, for any origin.
    stage = ArgmaxStage(layout; axis = :s, reward = [0.0 0.0; 1.0 1.0])

    V_end = Float64[0.0, 0.0]
    V_start = backward!(stage, V_end, nothing)
    @test V_start == [1.0, 1.0]
    @test policy(stage) == [2, 2]

    Λ_start = Float64[1.0, 0.0]
    Λ_end = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ 1.0
    @test Λ_end == [0.0, 1.0]
end

@testset "ArgmaxStage — -Inf reward is skipped (unavailable action)" begin
    layout = GriddedLayout(:s => Discrete([1, 2]))
    # Action 2 (after = 2) is unavailable from origin 1 (before = 1): M[2, 1] = -Inf.
    stage = ArgmaxStage(layout; axis = :s, reward = [0.0 0.0; -Inf 0.0])
    V_end = zeros(2)
    V_start = backward!(stage, V_end, nothing)
    @test V_start == [0.0, 0.0]
    @test policy(stage)[1] == 1
end

# Backward computes `V_start = max_a Q` straight off the reward field, so the flow-at-argmax
# `r[s] = R[s, σ(s)]` is recomputed inline here to assert (a) the affine split
# `V_start = r + Kᵀ·V_end`, where `Kᵀ·V_end` is the kernel's `backward!` (gather through the
# policy), and (b) that the forward push equals the kernel's `forward!` scatter.
@testset "ArgmaxStage — flow-at-argmax decomposition + transition duality" begin
    # Multi-dim layout so the choice axis is not first and the off-choice state
    # varies — exercises the per-cell scatter/gather, not a trivial 1-D case.
    layout = GriddedLayout(
        :z => Discrete([10.0, 20.0]),
        :s => Discrete([:A, :B, :C]),
    )
    # A genuinely state-dependent flow (varies along `z`) with a couple of unavailable actions,
    # so the policy is non-degenerate across the off-choice (`z`) axis. The reward is the
    # `(after, before)` matrix on `:s`, varying along the declared `:z` dep (and here independent
    # of the origin `:s`, so it broadcasts over `before`).
    sval = [:A, :B, :C]
    flow(a, z) = (a == :C && z == 10.0) ? -Inf : (a == :A ? 0.0 : a == :B ? 0.3 : 0.6)
    reward = (; z, env) -> [flow(sval[after], z) for after in 1:3, before in 1:3]
    stage = ArgmaxStage(layout; axis = :s, reward = reward)

    V_out = [0.1 * z + (s == 1 ? 0.0 : s == 2 ? 0.5 : -0.2) for z in 1:2, s in 1:3]
    V_in  = copy(backward!(stage, V_out, nothing))

    pol = policy(stage)

    # The argmax computes `V_start = max_a Q` directly (no stored reward), but the decomposition
    # identity `V_start = flow_at_argmax + Kᵀ·V_out` still holds. Recompute the flow at the chosen
    # action inline (independent of the kernel) and use it as the affine intercept.
    r_check = [flow(sval[pol[z_i, s_i]], (z_i == 1 ? 10.0 : 20.0))
               for z_i in 1:2, s_i in 1:3]

    # (a) Affine split: V_in == flow_at_argmax + Kᵀ·V_out (the policy gather).
    KtV = similar(V_in)
    HouseholdStages.backward!(KtV, stage.kernel, V_out; scratch = stage.scratch.kernel_scratch)
    @test V_in ≈ r_check .+ KtV atol = 1e-12

    # (b) Forward push equals the kernel's scatter.
    Λ_start = [0.4 0.1 0.05; 0.2 0.15 0.1]
    Λ_end   = copy(forward!(stage, Λ_start))
    Λ_op    = similar(Λ_end)
    HouseholdStages.forward!(Λ_op, stage.kernel, Λ_start; scratch = stage.scratch.kernel_scratch)
    @test Λ_end ≈ Λ_op atol = 1e-12
    @test sum(Λ_end) ≈ sum(Λ_start) atol = 1e-12

    # Scatter/gather are duals: ⟨Λ_end, V_out⟩ = ⟨Λ_start, Kᵀ·V_out⟩.
    @test sum(Λ_end .* V_out) ≈ sum(Λ_start .* KtV) atol = 1e-12
end

@testset "ArgmaxStage — rectangular n_start = 1 (max-marginalize / choice-collapse)" begin
    # Collapse :θ (size 3) by max with a passive :s axis; a `before`-singleton reward column.
    layout = GriddedLayout(:θ => Discrete([10.0, 20.0, 30.0]),
                           :s => Discrete([1.0, 2.0]))
    stage = ArgmaxStage(resize_axis(layout, :θ, 1), layout;
                        axis = :θ, reward = zeros(3, 1))   # (after = 3, before = 1)
    # Input collapses θ to size 1; output keeps it full.
    @test size(stage.scratch.V_start, 1) == 1
    @test size(stage.scratch.Λ_end, 1) == 3

    V_end = Float64[11 12; 23 21; 32 39]                                  # (θ, s)
    V_start = copy(backward!(stage, V_end, nothing))
    @test vec(V_start) == [32.0, 39.0]                                    # max over θ per s
    @test vec(policy(stage)) == [3, 3]                                    # argmax θ

    Λ_start = reshape(Float64[0.4, 0.6], 1, 2)                            # mass on (θ = 1, s)
    Λ_end = copy(forward!(stage, Λ_start))
    @test size(Λ_end) == (3, 2)
    @test Λ_end[3, 1] ≈ 0.4 && Λ_end[3, 2] ≈ 0.6                          # scattered to θ* = 3
    @test sum(Λ_end) ≈ 1.0
    # Scatter↔gather duality: ⟨Λ_end, V_end⟩ = ⟨Λ_start, V_start⟩.
    @test sum(Λ_end .* V_end) ≈ sum(Λ_start .* V_start) atol = 1e-12

    # A non-trivial reward (a −c(θ) column) shifts the argmax: penalise θ = 3.
    stage2 = ArgmaxStage(resize_axis(layout, :θ, 1), layout;
                         axis = :θ, reward = reshape([0.0, 0.0, -100.0], 3, 1))
    V2 = copy(backward!(stage2, V_end, nothing))
    @test vec(policy(stage2)) == [2, 2]
    @test vec(V2) == [23.0, 21.0]
end

# These layouts are the choice axis only, so `choice_probabilities` is the `(origin, dest)` matrix
# `P[i, j] = P(j | origin i)`.

@testset "LogitChoiceStage — transition-cost logit, V_end tilts choice" begin
    # Zero cost matrix + V_end[2] = 1 gives destination 2 a +1 advantage
    # (payoff −C[i,j] + V_end[j] = V_end[j]). The closed form is then
    # V_pre = ε log(1 + exp(1/ε)) for every origin.
    layout = GriddedLayout(:a => Discrete([1, 2]))
    ε = 0.5
    stage = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.0; 0.0 0.0],
        ε           = ε,
    )
    V_end = Float64[0.0, 1.0]
    V_start = backward!(stage, V_end, nothing)
    expected = 1.0 + ε * log(1 + exp(-1/ε))
    @test V_start[1] ≈ expected
    @test V_start[2] ≈ expected

    P = choice_probabilities(stage)
    @test sum(P[1, :]) ≈ 1.0
    @test P[1, 2] > P[1, 1]      # the higher V_end favours destination 2

    # Forward from a unit mass at origin 1 lands per the origin-1 probs.
    Λ_start = Float64[1.0, 0.0]
    Λ_end = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ 1.0
    @test Λ_end[1] ≈ P[1, 1]
    @test Λ_end[2] ≈ P[1, 2]
end

@testset "LogitChoiceStage — cost matrix penalises switching" begin
    # Symmetric off-diagonal cost with V_end favouring destination 2:
    # the cost of moving 1→2 partly offsets the value gain, so origin 1
    # is less likely to switch than origin 2 is to stay.
    layout = GriddedLayout(:a => Discrete([1, 2]))
    C = [0.0 0.5; 0.5 0.0]
    stage = LogitChoiceStage(layout; axis = :a, cost_matrix = C, ε = 1.0)

    V_end = Float64[0.0, 0.3]
    V_pre = copy(backward!(stage, V_end, nothing))
    for i in 1:2
        @test V_pre[i] ≈ log(sum(exp(-C[i, j] + V_end[j]) for j in 1:2)) atol = 1e-12
    end

    P = choice_probabilities(stage)
    @test all(sum(P; dims = 2) .≈ 1.0)
    # Origin 1 pays a cost to reach destination 2; origin 2 does not.
    @test P[2, 2] > P[1, 2]
end

@testset "LogitChoiceStage — ε → 0 concentrates on the best destination" begin
    layout = GriddedLayout(:a => Discrete([1, 2]))
    # Destination 2 is the strict argmax of −C[i,j] + V_end[j].
    stage = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.0; 0.0 0.0],
        ε           = 1e-4,
    )
    V_start_sharp = copy(backward!(stage, Float64[0.0, 1.0], nothing))
    @test V_start_sharp[1] ≈ 1.0 atol = 0.05    # → max payoff

    P = choice_probabilities(stage)
    for i in 1:2
        @test P[i, 2] > 0.999
        @test P[i, 1] < 1e-3
    end
end

@testset "LogitChoiceStage — ε swept via env" begin
    layout = GriddedLayout(:a => Discrete([1, 2]))
    stage = LogitChoiceStage(layout;
        axis = :a,
        cost_matrix = [0.0 0.0; 0.0 0.0],
        ε           = FromEnv(:ξ),
    )
    V_end = Float64[0.0, 1.0]
    V_start_sharp  = copy(backward!(stage, V_end, (ξ = 0.01,)))
    V_start_smooth = copy(backward!(stage, V_end, (ξ = 1.0,)))
    @test V_start_sharp[1] < V_start_smooth[1]
    @test V_start_sharp[1] ≈ 1.0 atol = 0.05
end

@testset "ArgmaxStage / LogitChoiceStage — static_env_deps" begin
    @test static_env_deps(HouseholdStages.ArgmaxStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.LogitChoiceStageSpec) === NamedTuple()
end

# Gibbs option-value separation + LSE identity.
# The logit computes V_in = ε·log Σ_j exp((−C[i,j] + V_out[j,s])/ε) (the
# LSE) directly — no stored Gibbs reward. But the option-value decomposition
# V_in = r + Kᵀ·V_out still holds, with the Gibbs reward r = E_π[−C] + ε·H(π)
# RECOVERED as r = V_in − Kᵀ·V_out (the factored pull through the seated kernel).
# We assert (a) V_in equals the analytic LSE, and (b) the recovered r matches the
# independently-computed Gibbs value E_π[−C] + ε·H(π).
@testset "LogitChoiceStage — Gibbs reward separation + LSE identity" begin
    # A genuinely multi-dimensional layout so the choice axis is not first and
    # the off-choice state varies — exercises the permute + per-state matmul.
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 1.0, 2.0]),
        :a => Discrete([1, 2, 3]),
    )
    n  = 3
    C  = [0.0 0.4 0.9; 0.4 0.0 0.5; 0.9 0.5 0.0]
    ε  = 0.7
    stage = LogitChoiceStage(layout; axis = :a, cost_matrix = C, ε = ε)

    V_out = randn(3, 3)                              # (w, a)

    V_in = copy(backward!(stage, V_out, nothing))

    # (a) V_in == analytic LSE = ε·log Σ_j exp((−C[i,j] + V_out[j,s])/ε).
    LSE = similar(V_in)
    for s in 1:3, i in 1:n
        LSE[s, i] = ε * log(sum(exp((-C[i, j] + V_out[s, j]) / ε) for j in 1:n))
    end
    @test V_in ≈ LSE atol = 1e-12

    # Recover the option-value reward via the kernel pull: r = V_in − Kᵀ·V_out.
    # `backward!(stage, …)` above seated the kernel; `forward_adjoint!` is its frozen-kernel
    # Kᵀ pull, Kᵀ·V_out = Σ_j π(j|i,s)·V_out[j,s], factored over the dep batch.
    KtV = forward_adjoint!(stage, V_out)
    r = V_in .- KtV                              # (w, a) over the start layout

    # The policy π for the independent Gibbs value, over the (w, a) start layout with the
    # destination appended: π[s, i, j].
    π = choice_probabilities(stage)

    # (b) r = V_in − Kᵀ·V_out matches the independent Gibbs value E_π[−C] + εH(π).
    H      = [-sum(π[s, i, j] * log(π[s, i, j]) for j in 1:n) for s in 1:3, i in 1:n]
    EπC    = [sum(π[s, i, j] * (-C[i, j]) for j in 1:n) for s in 1:3, i in 1:n]
    r_gibbs = EπC .+ ε .* H
    @test r ≈ r_gibbs atol = 1e-12

    # The factored pull agrees with the explicit πᵀV_out contraction.
    KtV_check = [sum(π[s, i, j] * V_out[s, j] for j in 1:n) for s in 1:3, i in 1:n]
    @test KtV ≈ KtV_check atol = 1e-12
end

# Decomposition identity — the soundness proof for removing `amenity`. A
# destination payoff shifter u[j] is exactly composition with a UtilityStage:
# `LogitChoiceStage ∘ UtilityStage(u)` reproduces what an old `amenity = u` did.
# We check the composed stage's V_in, choice π, and Gibbs reward all match the
# closed form `ε·log Σ_j exp((−C[i,j] + u[j] + V_out[j,s])/ε)`.
@testset "LogitChoiceStage ∘ UtilityStage(u) reproduces an amenity u (decomposition)" begin
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 1.0, 2.0]),
        :a => Discrete([1, 2, 3]),
    )
    n = 3
    C = [0.0 0.4 0.9; 0.4 0.0 0.5; 0.9 0.5 0.0]
    u = [0.0, 0.2, -0.1]                              # the destination amenity
    ε = 0.7

    # u depends only on the destination index (the choice axis), so the
    # UtilityStage closure reads `cell.a` and indexes u.
    stage = LogitChoiceStage(layout; axis = :a, cost_matrix = C, ε = ε) ∘
            UtilityStage(layout; utility = (; a) -> u[a])

    V_out = randn(3, 3)                               # (w, a)
    V_in  = copy(backward!(stage, V_out, nothing))

    # V_in matches the closed form an `amenity = u` logit would have produced.
    LSE = similar(V_in)
    for s in 1:3, i in 1:n
        LSE[s, i] = ε * log(sum(exp((-C[i, j] + u[j] + V_out[s, j]) / ε) for j in 1:n))
    end
    @test V_in ≈ LSE atol = 1e-12

    # The logit's stored policy is the amenity-shifted softmax over (u + V_out).
    # ChainStage is `logit ∘ utility`, so the logit sub-stage is stages[1].
    π = choice_probabilities(stage.buffer.stages[1])
    for s in 1:3, i in 1:n
        denom = sum(exp((-C[i, jj] + u[jj] + V_out[s, jj]) / ε) for jj in 1:n)
        for j in 1:n
            @test π[s, i, j] ≈ exp((-C[i, j] + u[j] + V_out[s, j]) / ε) / denom atol = 1e-12
        end
    end
end

@testset "LogitChoiceStage — cost_matrix as a matrix-returning closure" begin
    layout = GriddedLayout(:a => Discrete([1, 2, 3]))
    C = [0.0 0.4 0.8; 0.4 0.0 0.4; 0.8 0.4 0.0]
    V = randn(3)
    base = backward!(LogitChoiceStage(layout; axis = :a, cost_matrix = C, ε = 0.6), V, nothing)
    # A closure returning the same matrix reproduces the constant-cost result.
    s_cl = LogitChoiceStage(layout; axis = :a, cost_matrix = (; env) -> C, ε = 0.6)
    @test backward!(s_cl, V, nothing) ≈ base
    @test_throws UndefKeywordError LogitChoiceStage(layout; axis = :a, ε = 0.6)  # cost required
end

@testset "LogitChoiceStage — negative ε is the robust soft-MIN (entropic risk)" begin
    layout = GriddedLayout(:a => Discrete([1, 2]))
    K = [0.7 0.3; 0.4 0.6]                      # a transition prior
    W = Float64[1.0, 5.0]
    for θ in (1.0, 0.1)
        ε = -θ
        # Encode the prior K as the Gibbs neg-log-prior cost C = −ε·log K, so the choice weights
        # exp(−C[i,j]/ε) are K[i,j].
        stage = LogitChoiceStage(layout; axis = :a, cost_matrix = -ε .* log.(K), ε = ε)
        V = copy(backward!(stage, W, nothing))
        analytic = [-θ * log(sum(K[i, j] * exp(-W[j] / θ) for j in 1:2)) for i in 1:2]
        @test V ≈ analytic atol = 1e-10
        # Pessimistic: the entropic-risk CE is below the linear expectation E[W|i].
        @test all(V .< [sum(K[i, j] * W[j] for j in 1:2) for i in 1:2])
    end
    # ε → 0⁻ concentrates on the worst (min) reachable continuation.
    sharp = LogitChoiceStage(layout; axis = :a, cost_matrix = 1e-3 .* log.(K), ε = -1e-3)
    @test all(isapprox.(copy(backward!(sharp, W, nothing)), minimum(W); atol = 1e-2))
    # The forward still pushes mass through a valid (worst-case-tilted) kernel.
    Λ = forward!(sharp, Float64[0.6, 0.4])
    @test sum(Λ) ≈ 1.0 atol = 1e-12
end

@testset "LogitChoiceStage — rectangular origin=1 (logsumexp-collapse: primal + adjoint)" begin
    layout = GriddedLayout(:θ => Discrete([1, 2, 3]), :s => Discrete([1.0, 2.0]))
    ε = 0.7
    stage = LogitChoiceStage(resize_axis(layout, :θ, 1), layout;
                             axis = :θ, cost_matrix = zeros(1, 3), ε = ε)
    @test size(stage.scratch.V_start, 1) == 1 && size(stage.scratch.Λ_end, 1) == 3

    V_end = Float64[1 4; 3 2; 2 5]
    V_start = copy(backward!(stage, V_end, nothing))
    @test V_start ≈ reshape([ε * log(sum(exp(V_end[θ, s] / ε) for θ in 1:3)) for s in 1:2], 1, 2) atol = 1e-12

    Λ_end = copy(forward!(stage, reshape(Float64[0.4, 0.6], 1, 2)))
    @test size(Λ_end) == (3, 2)
    @test sum(Λ_end) ≈ 1.0 atol = 1e-12

    # Both adjoints work (the part missed before) and satisfy ⟨dΛ_end, K·dV⟩ = ⟨Kᵀ·dΛ_end, dV⟩.
    dV_start = reshape(Float64[0.1, 0.2], 1, 2); dΛ_end = Float64[0.3 0.1; 0.2 0.4; 0.5 0.15]
    fa = forward_adjoint!(stage, dΛ_end); ba = backward_adjoint!(stage, dV_start)
    @test size(fa) == (1, 2) && size(ba) == (3, 2)
    @test sum(dΛ_end .* ba) ≈ sum(fa .* dV_start) atol = 1e-12
end

# The stored fiber's ORIENTATION, stated once. The fill's size assert validates its SHAPE, but at
# `n_start == n_end` a transposed cost passes every shape check — and passes every test whose cost
# is symmetric or zero. So the proposition needs an ASYMMETRIC literal cost to have any content,
# and it is the one place the tree says which way round `eψC` sits.
@testset "LogitChoiceStage — eψC is stored as exp(−Cᵀ/ε) (asymmetric cost)" begin
    layout = GriddedLayout(:a => Discrete([1, 2, 3]))
    C = [0.0 0.7 1.9;                          # C[i, j] ≠ C[j, i] at every off-diagonal entry
         0.2 0.0 0.4;
         1.1 0.3 0.0]
    ε = 0.6
    stage = LogitChoiceStage(layout; axis = :a, cost_matrix = C, ε = ε)
    V_end = Float64[0.3, -0.2, 0.8]
    backward!(stage, V_end, nothing)

    @test parent(stage.kernel.eψC) ≈ permutedims(exp.(.- C ./ ε))
    @test !(parent(stage.kernel.eψC) ≈ exp.(.- C ./ ε))     # the transpose is not a no-op here

    # `choice_probabilities` reads that orientation: π(j|i) ∝ exp((−C[i,j] + V_end[j])/ε).
    P = choice_probabilities(stage)
    @test size(P) == (3, 3)
    for i in 1:3
        denom = sum(exp((-C[i, jj] + V_end[jj]) / ε) for jj in 1:3)
        for j in 1:3
            @test P[i, j] ≈ exp((-C[i, j] + V_end[j]) / ε) / denom atol = 1e-12
        end
    end
end

@testset "choice_probabilities — dep-varying asymmetric cost, leading operative axis" begin
    # The operative axis leads and the cost varies along the trailing dep, so the join over the
    # compact `(dest, origin, dep)` fiber and the full layout-shaped buffers is non-trivial in
    # both index positions.
    layout = GriddedLayout(:a => Discrete([1, 2, 3]), :z => Discrete([:lo, :hi]))
    Clo = [0.0 0.7 1.9; 0.2 0.0 0.4; 1.1 0.3 0.0]
    Chi = [0.0 0.1 2.5; 1.3 0.0 0.6; 0.4 1.7 0.0]
    cost(; z) = z == :lo ? Clo : Chi
    ε = 0.8
    stage = LogitChoiceStage(layout; axis = :a, cost_matrix = cost, ε = ε)
    V_end = [0.1 * i + (z == 1 ? 0.0 : 0.5) for i in 1:3, z in 1:2]
    backward!(stage, V_end, NamedTuple())

    P = choice_probabilities(stage)
    @test size(P) == (3, 2, 3)                              # (origin a, dep z, dest a)
    for (zi, Cz) in ((1, Clo), (2, Chi)), i in 1:3
        denom = sum(exp((-Cz[i, jj] + V_end[jj, zi]) / ε) for jj in 1:3)
        for j in 1:3
            @test P[i, zi, j] ≈ exp((-Cz[i, j] + V_end[j, zi]) / ε) / denom atol = 1e-12
        end
        @test sum(P[i, zi, :]) ≈ 1.0 atol = 1e-12
    end
end

@testset "choice_probabilities — rectangular 1 → n origin collapse" begin
    layout = GriddedLayout(:θ => Discrete([1, 2, 3]), :s => Discrete([1.0, 2.0]))
    ε = 0.7
    stage = LogitChoiceStage(resize_axis(layout, :θ, 1), layout;
                             axis = :θ, cost_matrix = [0.0 0.5 1.25], ε = ε)
    V_end = Float64[1 4; 3 2; 2 5]
    backward!(stage, V_end, nothing)

    C = [0.0, 0.5, 1.25]
    P = choice_probabilities(stage)
    @test size(P) == (1, 2, 3)                              # start layout (1, 2), dest appended
    for s in 1:2
        denom = sum(exp((-C[jj] + V_end[jj, s]) / ε) for jj in 1:3)
        for j in 1:3
            @test P[1, s, j] ≈ exp((-C[j] + V_end[j, s]) / ε) / denom atol = 1e-12
        end
        @test sum(P[1, s, :]) ≈ 1.0 atol = 1e-12
    end
end
