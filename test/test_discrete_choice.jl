using Test
using HouseholdStages

@testset "ArgmaxStage — 2-state re-choose, action 2 always preferred (+1)" begin
    layout = GriddedLayout(StateAxis(:s, categorical([:A, :B])))
    stage = ArgmaxStage(layout;
        choice_axis = :s,
        flow_payoff = (cell, a; env) -> (a == :B ? 1.0 : 0.0),
        next_state_idx = (cell, a) -> a == :A ? 1 : 2,
    )

    V_end = Float64[0.0, 0.0]
    V_start = backward!(stage, V_end, nothing)
    @test V_start == [1.0, 1.0]
    @test policy(stage) == [2, 2]

    Λ_start = Float64[1.0, 0.0]
    Λ_end = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ 1.0
    @test Λ_end == [0.0, 1.0]
end

@testset "ArgmaxStage — -Inf payoff is skipped (unavailable action)" begin
    layout = GriddedLayout(StateAxis(:s, discrete_finite([1, 2])))
    stage = ArgmaxStage(layout;
        choice_axis = :s,
        flow_payoff = (cell, a; env) -> (a == 2 && cell.s == 1) ? -Inf : 0.0,
        next_state_idx = (cell, a) -> a,
    )
    V_end = zeros(2)
    V_start = backward!(stage, V_end, nothing)
    @test V_start == [0.0, 0.0]
    @test policy(stage)[1] == 1
end

# Modern argmax: backward computes `V_start = max_a Q` directly (no stored reward).
# The decomposition identity still holds — we recompute the flow-at-argmax
# `r[s] = R[s, σ(s)]` inline and assert (a) the affine split V_start = r + Kᵀ·V_out,
# where Kᵀ·V_out is the kernel's `backward!` (gather through the policy), and (b) the
# forward push equals the kernel's `forward!` scatter.
@testset "ArgmaxStage — flow-at-argmax decomposition + transition duality" begin
    # Multi-dim layout so the choice axis is not first and the off-choice state
    # varies — exercises the per-cell scatter/gather, not a trivial 1-D case.
    layout = GriddedLayout(
        StateAxis(:z,  discrete_finite([10.0, 20.0])),
        StateAxis(:s,  categorical([:A, :B, :C])),
    )
    # A genuinely state-dependent flow with a couple of unavailable actions, so
    # the policy is non-degenerate across the off-choice (`z`) axis.
    fp = (cell, a; env) ->
        (a == :C && cell.z == 10.0) ? -Inf :
        (a == :A ? 0.0 : a == :B ? 0.3 : 0.6)
    nidx = Dict(:A => 1, :B => 2, :C => 3)
    stage = ArgmaxStage(layout;
        choice_axis = :s,
        flow_payoff = fp,
        next_state_idx = (cell, a) -> nidx[a],
    )

    V_out = [0.1 * z + (s == 1 ? 0.0 : s == 2 ? 0.5 : -0.2) for z in 1:2, s in 1:3]
    V_in  = copy(backward!(stage, V_out, nothing))

    pol = policy(stage)

    # The modern argmax computes `V_start = max_a Q` directly (no stored reward),
    # but the decomposition identity `V_start = flow_at_argmax + Kᵀ·V_out` still
    # holds. Recompute the flow at the chosen action inline (independent of the
    # kernel) and use it as the affine intercept.
    actions = [:A, :B, :C]
    r_check = [fp((z = (z_i == 1 ? 10.0 : 20.0), s = actions[pol[z_i, s_i]]),
                  actions[pol[z_i, s_i]]; env = nothing)
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

# Recompute the per-origin choice probabilities from a solved kernel.
# These layouts are the choice axis only, so the non-choice column is s = 1:
#     P(j | origin i) = eψC[i,j] · W[j,1] / res[i,1].
function _logit_prob(stage, n)
    k = stage.kernel
    return [k.eψC[i, j] * k.value_weight[j, 1] / k.normalizer[i, 1] for i in 1:n, j in 1:n]
end

@testset "LogitChoiceStage — transition-cost logit, V_end tilts choice" begin
    # Zero cost matrix + V_end[2] = 1 gives destination 2 a +1 advantage
    # (payoff −C[i,j] + V_end[j] = V_end[j]). The closed form is then
    # V_pre = ε log(1 + exp(1/ε)) for every origin.
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    ε = 0.5
    stage = LogitChoiceStage(layout;
        choice_axis = :a,
        cost_matrix = [0.0 0.0; 0.0 0.0],
        ε           = ε,
    )
    V_end = Float64[0.0, 1.0]
    V_start = backward!(stage, V_end, nothing)
    expected = 1.0 + ε * log(1 + exp(-1/ε))
    @test V_start[1] ≈ expected
    @test V_start[2] ≈ expected

    P = _logit_prob(stage, 2)
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
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    C = [0.0 0.5; 0.5 0.0]
    stage = LogitChoiceStage(layout; choice_axis = :a, cost_matrix = C, ε = 1.0)

    V_end = Float64[0.0, 0.3]
    V_pre = copy(backward!(stage, V_end, nothing))
    for i in 1:2
        @test V_pre[i] ≈ log(sum(exp(-C[i, j] + V_end[j]) for j in 1:2)) atol = 1e-12
    end

    P = _logit_prob(stage, 2)
    @test all(sum(P; dims = 2) .≈ 1.0)
    # Origin 1 pays a cost to reach destination 2; origin 2 does not.
    @test P[2, 2] > P[1, 2]
end

@testset "LogitChoiceStage — ε → 0 concentrates on the best destination" begin
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    # Destination 2 is the strict argmax of −C[i,j] + V_end[j].
    stage = LogitChoiceStage(layout;
        choice_axis = :a,
        cost_matrix = [0.0 0.0; 0.0 0.0],
        ε           = 1e-4,
    )
    V_start_sharp = copy(backward!(stage, Float64[0.0, 1.0], nothing))
    @test V_start_sharp[1] ≈ 1.0 atol = 0.05    # → max payoff

    P = _logit_prob(stage, 2)
    for i in 1:2
        @test P[i, 2] > 0.999
        @test P[i, 1] < 1e-3
    end
end

@testset "LogitChoiceStage — ε swept via env" begin
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoiceStage(layout;
        choice_axis = :a,
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
# The modern logit computes V_in = ε·log Σ_j exp((−C[i,j] + V_out[j,s])/ε) (the
# LSE) directly — no stored Gibbs reward. But the option-value decomposition
# V_in = r + Kᵀ·V_out still holds, with the Gibbs reward r = E_π[−C] + ε·H(π)
# RECOVERED as r = V_in − Kᵀ·V_out (the factored pull through the seated kernel).
# We assert (a) V_in equals the analytic LSE, and (b) the recovered r matches the
# independently-computed Gibbs value E_π[−C] + ε·H(π).
@testset "LogitChoiceStage — Gibbs reward separation + LSE identity" begin
    # A genuinely multi-dimensional layout so the choice axis is not first and
    # the off-choice state varies — exercises the permute + per-state matmul.
    layout = GriddedLayout(
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:a, discrete_finite([1, 2, 3])),
    )
    n  = 3
    C  = [0.0 0.4 0.9; 0.4 0.0 0.5; 0.9 0.5 0.0]
    ε  = 0.7
    stage = LogitChoiceStage(layout; choice_axis = :a, cost_matrix = C, ε = ε)

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
    r = V_in .- KtV                              # (w, a) over the input layout

    # The policy π (choice axis = dim 2) for the independent Gibbs value. eψC's compact
    # parent is (origin, dest); value_weight/normalizer are layout-shaped (w = s, a = choice).
    k = stage.kernel
    eC = reshape(parent(k.eψC), n, n)
    π = [eC[i, j] * k.value_weight[s, j] / k.normalizer[s, i] for s in 1:3, i in 1:n, j in 1:n]  # π[s,i,j]

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
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:a, discrete_finite([1, 2, 3])),
    )
    n = 3
    C = [0.0 0.4 0.9; 0.4 0.0 0.5; 0.9 0.5 0.0]
    u = [0.0, 0.2, -0.1]                              # the old amenity vector
    ε = 0.7

    # u depends only on the destination index (the choice axis), so the
    # UtilityStage closure reads `cell.a` and indexes u.
    stage = LogitChoiceStage(layout; choice_axis = :a, cost_matrix = C, ε = ε) ∘
            UtilityStage(layout; utility = (cell; env) -> u[cell.a])

    V_out = randn(3, 3)                               # (w, a)
    V_in  = copy(backward!(stage, V_out, nothing))

    # V_in matches the closed form an `amenity = u` logit would have produced.
    LSE = similar(V_in)
    for s in 1:3, i in 1:n
        LSE[s, i] = ε * log(sum(exp((-C[i, j] + u[j] + V_out[s, j]) / ε) for j in 1:n))
    end
    @test V_in ≈ LSE atol = 1e-12

    # The logit's stored policy is the amenity-shifted softmax over (u + V_out).
    # ChainStage is `logit ∘ utility`, so the logit sub-stage is stages[1] (modern).
    k = stage.buffer.stages[1].kernel
    eC = reshape(parent(k.eψC), n, n)
    π = [eC[i, j] * k.value_weight[s, j] / k.normalizer[s, i] for s in 1:3, i in 1:n, j in 1:n]
    for s in 1:3, i in 1:n
        denom = sum(exp((-C[i, jj] + u[jj] + V_out[s, jj]) / ε) for jj in 1:n)
        for j in 1:n
            @test π[s, i, j] ≈ exp((-C[i, j] + u[j] + V_out[s, j]) / ε) / denom atol = 1e-12
        end
    end
end

@testset "LogitChoiceStage — cost_matrix as a matrix-returning closure" begin
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2, 3])))
    C = [0.0 0.4 0.8; 0.4 0.0 0.4; 0.8 0.4 0.0]
    V = randn(3)
    base = backward!(LogitChoiceStage(layout; choice_axis = :a, cost_matrix = C, ε = 0.6), V, nothing)
    # A closure returning the same matrix reproduces the constant-cost result.
    s_cl = LogitChoiceStage(layout; choice_axis = :a, cost_matrix = (; env) -> C, ε = 0.6)
    @test backward!(s_cl, V, nothing) ≈ base
    @test_throws UndefKeywordError LogitChoiceStage(layout; choice_axis = :a, ε = 0.6)  # cost required
end
