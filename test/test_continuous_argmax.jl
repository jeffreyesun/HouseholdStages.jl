using Test
using HouseholdStages

# Direct tests of `ArgmaxStage` in its monotone-search mode (`search = :divide_conquer`,
# the generalisation behind ConsumptionSavingsStage). The CS-specific numeric tests live in
# test_monotone_argmax.jl; here we exercise the primitive's reward `M`-source API
# and the CS ⇔ primitive equivalence that the wrapper relies on.
#
# The reward is an ordinary matrix source `M[after, before]` on the choice axis — a
# `Matrix`, a `FromEnv`, or a closure `(; dep…, env) -> Matrix` — exactly like Markov's
# transition and Logit's cost. The declared deps size the U table.

@testset "ArgmaxStage(:divide_conquer) — equals CS with the budget reward matrix" begin
    n_w = 64
    g = [exp(t) - 1.0 for t in range(0.0, log(101.0); length = n_w)]
    layout = GriddedLayout(
        :wealth => GriddedContinuous(g),
        :income => Discrete([0.6, 1.0, 1.4]),
    )
    u = (cell, c; env) -> log(c)

    cs = ConsumptionSavingsStage(layout; β = 0.96, utility = u, axis = :wealth)

    # The reward as the ordinary (after, before) matrix on wealth: consumption
    # c = wealth_before − wealth_after, log utility, non-positive masked to -Inf —
    # exactly the matrix the CS wrapper builds, so the two agree bit-for-bit.
    reward = [ g[before] - g[after] > 0 ? log(g[before] - g[after]) : -Inf
               for after in 1:n_w, before in 1:n_w ]
    ca = ArgmaxStage(layout; reward = reward, axis = :wealth, search = :divide_conquer) ∘
         TimeDiscountingStage(layout; β = 0.96)

    V_end = [0.1 * w_i + 0.05 * y_j for w_i in 1:n_w, y_j in 1:3]
    env   = NamedTuple()

    V_cs = copy(backward!(cs, V_end, env))
    V_ca = copy(backward!(ca, V_end, env))

    @test policy(cs) == policy(ca)
    @test V_cs == V_ca                      # bit-identical (same arithmetic)

    Λ_start = rand(n_w, 3); Λ_start ./= sum(Λ_start)
    Λ_cs = copy(forward!(cs, Λ_start))
    Λ_ca = copy(forward!(ca, Λ_start))
    @test Λ_cs == Λ_ca
end

@testset "ArgmaxStage(:divide_conquer) — composed discount scales V_end (diagonal-forced policy)" begin
    layout = GriddedLayout(:k => GriddedContinuous([0.0, 1.0, 2.0, 3.0]))
    # reward 0 only on the diagonal (after == before), -Inf elsewhere ⇒ the policy is
    # forced to a = s, so V_start[s] = 0 + β·V_end[s] isolates the discount. Discount is now its
    # own composed `TimeDiscountingStage` (end-goal §1), run before the argmax in the backward sweep.
    reward = [ a == b ? 0.0 : -Inf for a in 1:4, b in 1:4 ]
    V_end  = [10.0, 20.0, 30.0, 40.0]

    ca1 = ArgmaxStage(layout; reward = reward, axis = :k,
                      search = :divide_conquer, assume_monotone = true)
    @test copy(backward!(ca1, V_end, NamedTuple())) == V_end          # no discount

    ca2 = ArgmaxStage(layout; reward = reward, axis = :k,
                      search = :divide_conquer, assume_monotone = true) ∘
          TimeDiscountingStage(layout; β = 0.5)
    @test copy(backward!(ca2, V_end, NamedTuple())) == 0.5 .* V_end
    @test policy(ca1) == [1, 2, 3, 4]                     # diagonal
end

@testset "ArgmaxStage(:divide_conquer) — reward deps size the U table" begin
    g = [0.0, 1.0, 2.0, 3.0]
    layout = GriddedLayout(
        :wealth => GriddedContinuous(g),
        :income => Discrete([1.0, 2.0]),
    )
    # A closure reward varying along income declares the income dep (a `+ 0.0 * income`
    # no-op): U must be full (size 2) along the income axis, dim 3 of the compact table.
    reward = (; income, env) -> [ g[b] - g[a] > 0 ? log(g[b] - g[a]) + 0.0 * income : -Inf
                                  for a in 1:4, b in 1:4 ]
    ca = ArgmaxStage(layout; reward = reward, axis = :wealth, search = :divide_conquer)
    @test size(parent(ca.scratch.U), 3) == 2

    # A constant reward matrix has no deps ⇒ U is singleton along income.
    reward_c = [ g[b] - g[a] > 0 ? log(g[b] - g[a]) : -Inf for a in 1:4, b in 1:4 ]
    ca_c = ArgmaxStage(layout; reward = reward_c, axis = :wealth, search = :divide_conquer)
    @test size(parent(ca_c.scratch.U), 3) == 1

    # An undeclared kwarg that is not a layout axis / env is rejected.
    reward_bad = (; income, typo, env) -> zeros(4, 4)
    @test_throws ErrorException ArgmaxStage(layout; reward = reward_bad,
                                            axis = :wealth, search = :divide_conquer)
end
