using Test
using HouseholdStages

# Rational inattention as a COMPOSITION (not a stage). The canonical Matějka–McKay
# (2015) static Shannon-cost discrete choice IS a generalized logit:
#
#   P(a | s) ∝ q(a) · exp(u(s,a)/λ),   value(s) = λ·log Σ_a q(a)·exp(u(s,a)/λ).
#
# This is reproduced EXACTLY by `LogitChoiceStage(ε=λ, cost=0) ∘ UtilityStage(u)`
# where the UtilityStage's per-action payoff carries `λ·log q(a)` (the log
# attention-prior) — the old `RationalInattentionStage`'s amenity. The destination
# payoff being V-additive is precisely why RI needs no dedicated stage: it is
# composition. The endogenous prior `q(a) = ∫ P(a|s) dΛ(s)` is an outer-loop
# aggregate carried in env and held fixed within a solve, so the UtilityStage's
# `λ·log q` reads `q` from env. These tests prove the composition IS RI:
#   (a) it reproduces the Matějka–McKay value λ·log Σ_a q(a)·exp(V/λ);
#   (b) the choice probabilities are the M–M posterior P(a|s) ∝ q(a)·exp(V/λ);
#   (c) the Gibbs / LSE identity holds with the env-supplied prior;
#   (d) duality and mass conservation hold.

"""
Build the rational-inattention composition `LogitChoice(ε=λ) ∘ UtilityStage(λ·log q)`,
with the log attention-prior read from `env.q` so the outer loop can converge it.

The per-action amenity `λ·log q(a)` is a dep closure on the (runtime-named) choice axis reading the
env-supplied prior `env.q` at the axis's grid value; it rides a `DepClosure` carrying `(choice_axis, :env)`
for the Sources path.
"""
ri_composition(layout; choice_axis, λ) =
    LogitChoiceStage(layout; axis = choice_axis, cost_matrix = zeros(_ri_n(layout, choice_axis), _ri_n(layout, choice_axis)), ε = λ) ∘
    UtilityStage(layout; utility = HouseholdStages.DepClosure(nt -> λ * log(nt.env.q[nt[choice_axis]]), (choice_axis,), true))

_ri_n(layout, choice_axis) = axissize(layout.axes[axis_position(layout, choice_axis)])

# Per-origin choice probabilities from the logit sub-buffer (choice axis only ⇒
# s = 1): P(j | origin i) = eψC[i,j] · W[j,1] / res[i,1]. For a zero-cost logit
# eψC ≡ 1, so this is just the softmax over the destination weights.
function _ri_prob(chain, n)
    k = chain.buffer.stages[1].kernel    # stages[1] is the logit (modern)
    return [parent(k.eψC)[i, j] * k.value_weight[j, 1] / k.normalizer[i, 1] for i in 1:n, j in 1:n]
end

@testset "RI composition — reproduces the Matějka–McKay option value (env prior)" begin
    # Three actions; the RI value with prior q and payoff (carried in V_end) is
    #   value(i) = λ·log Σ_a q(a)·exp(V_end[a]/λ)
    #            = λ·log Σ_a exp((λ·log q(a) + V_end[a])/λ),
    # i.e. exactly the zero-cost logit LSE with the UtilityStage adding λ·log q.
    layout = GriddedLayout(:a => Discrete([1, 2, 3]))
    λ = 0.7
    q = [0.2, 0.5, 0.3]                                # an attention prior
    chain = ri_composition(layout; choice_axis = :a, λ = λ)
    @test chain isa ChainStage
    # The prior flows through env at solve time via the UtilityStage closure; the
    # dep closure declares only `:env` (not the specific key `:q`, read dynamically),
    # so the env slice stays empty and the numerics below are the proof that the
    # composition IS RI.
    @test isempty(effective_env_slice(chain))

    V_end = Float64[0.0, 0.4, -0.2]
    V_in  = copy(backward!(chain, V_end, (q = q,)))

    expected = [λ * log(sum(q[a] * exp(V_end[a] / λ) for a in 1:3)) for _ in 1:3]
    @test V_in ≈ expected atol = 1e-12

    # Choice probabilities are the Matějka–McKay posterior P(a|s) ∝ q(a)·exp(V/λ).
    P = _ri_prob(chain, 3)
    for i in 1:3
        denom = sum(q[a] * exp(V_end[a] / λ) for a in 1:3)
        for j in 1:3
            @test P[i, j] ≈ q[j] * exp(V_end[j] / λ) / denom atol = 1e-12
        end
    end
end

@testset "RI composition — Gibbs / LSE identity with the env prior" begin
    # Multi-dim layout so the off-choice state varies (exercises the permute +
    # per-state matmul) and the logit cost is identically zero (RI has no friction).
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 1.0, 2.0]),
        :a => Discrete([1, 2, 3]),
    )
    n = 3
    λ = 0.5
    q = [0.5, 0.3, 0.2]
    u = λ .* log.(q)                                   # what the UtilityStage adds

    chain = ri_composition(layout; choice_axis = :a, λ = λ)
    V_out = randn(3, 3)                                # (w, a)
    V_in  = copy(backward!(chain, V_out, (q = q,)))

    # (a) V_in == the LSE = λ·log Σ_a exp((λ·log q(a) + V_out[a])/λ) (cost = 0).
    LSE = similar(V_in)
    for s in 1:3, i in 1:n
        LSE[s, i] = λ * log(sum(exp((u[j] + V_out[s, j]) / λ) for j in 1:n))
    end
    @test V_in ≈ LSE atol = 1e-12

    # (b) The logit Gibbs reward r = V_in_logit − Kᵀ(V_out + u) = λ·H(π) (cost = 0).
    # No stored reward on the modern logit: recover it via the kernel pull. The logit
    # sub-stage ran on the UtilityStage's output V_out + u (u on the choice axis), and
    # the chain V_in IS the logit's V_start, so r_logit = V_in − Kᵀ(V_out + u).
    logit = chain.buffer.stages[1]
    k = logit.kernel
    eC = reshape(parent(k.eψC), n, n)
    π = [eC[i, j] * k.value_weight[s, j] / k.normalizer[s, i] for s in 1:3, i in 1:n, j in 1:n]
    H = [-sum(π[s, i, j] * log(π[s, i, j]) for j in 1:n) for s in 1:3, i in 1:n]

    V_logit_out = V_out .+ reshape(u, 1, n)              # utility adds u on the choice axis
    KtVu = forward_adjoint!(logit, V_logit_out)          # Kᵀ(V_out + u) = Σ_j π(j|i,s)·(V_out + u)[j,s]
    r_logit = V_in .- KtVu                               # the pure logit Gibbs reward
    @test r_logit ≈ λ .* H atol = 1e-12                # cost = 0 ⇒ reward is pure entropy

    # The full RI reward (option value over the policy) is E_π[u] + λ·H(π).
    Eπu = [sum(π[s, i, j] * u[j] for j in 1:n) for s in 1:3, i in 1:n]
    KtV = [sum(π[s, i, j] * V_out[s, j] for j in 1:n) for s in 1:3, i in 1:n]
    @test V_in ≈ KtV .+ Eπu .+ λ .* H atol = 1e-12
end

@testset "RI composition — duality + mass conservation" begin
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 0.5, 1.0]),
        :a => Discrete([1, 2]),
    )
    λ = 1.0
    q = [0.6, 0.4]
    chain = ri_composition(layout; choice_axis = :a, λ = λ)

    env    = (q = q,)
    V_out  = randn(3, 2)
    Λ_in   = rand(3, 2); Λ_in ./= sum(Λ_in)

    V_in  = copy(backward!(chain, V_out, env))
    Λ_out = copy(forward!(chain, Λ_in))

    # Mass conserved (the action→outcome is a row-stochastic logit scatter).
    @test isapprox(sum(Λ_out), sum(Λ_in); atol = 1e-12)

    # Duality with the chain's per-stage rewards: ⟨V_in,Λ_in⟩ = ⟨V_out,Λ_out⟩ + reward work.
    # The utility (identity on Λ) and the logit (the only scatter) compose, so the
    # standard chain duality ⟨V_in − r_eff, Λ_in⟩ = ⟨V_out, Λ_out⟩ holds, where the
    # effective reward absorbs both u and the entropy. We verify it via the policy.
    logit = chain.buffer.stages[1]
    k = logit.kernel
    n = 2
    eC = reshape(parent(k.eψC), n, n)
    π = [eC[i, j] * k.value_weight[s, j] / k.normalizer[s, i] for s in 1:3, i in 1:n, j in 1:n]
    u = λ .* log.(q)
    H = [-sum(π[s, i, j] * log(π[s, i, j]) for j in 1:n) for s in 1:3, i in 1:n]
    Eπu = [sum(π[s, i, j] * u[j] for j in 1:n) for s in 1:3, i in 1:n]
    r_eff = Eπu .+ λ .* H
    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out) + sum(r_eff .* Λ_in); atol = 1e-12)
end

@testset "RI composition — uniform prior is a plain temperature-λ logit" begin
    # With q ∝ 1 the log-prior is constant, so RI reduces to the temperature-λ
    # logit with no shifter: value(i) = λ·log Σ_a exp(V_end[a]/λ).
    layout = GriddedLayout(:a => Discrete([1, 2, 3]))
    λ = 0.8
    q = fill(1 / 3, 3)
    chain = ri_composition(layout; choice_axis = :a, λ = λ)

    V_end = Float64[0.1, 0.5, -0.3]
    V_in  = copy(backward!(chain, V_end, (q = q,)))
    # A uniform prior shifts the value by a constant λ·log(1/3); the *choice*
    # probabilities are the plain logit, and the value differs from the
    # no-prior logit only by that constant.
    plain = [λ * log(sum(exp(V_end[a] / λ) for a in 1:3)) for _ in 1:3]
    @test V_in ≈ plain .+ λ * log(1 / 3) atol = 1e-12
end
