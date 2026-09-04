using Test
using Random
using HouseholdStages

# Direct tests of the brute `ArgmaxStage`: the reward `M`-source API and the closure-lowering ⇔
# matrix equivalence (the construction the grid-snapped CS recipe in examples/additional_stages/
# relies on).
#
# The reward is an ordinary matrix source `M[after, before]` on the choice axis — a
# `Matrix`, a `FromEnv`, or a closure `(; dep…, env) -> Matrix` — exactly like Markov's
# transition and Logit's cost. The declared deps size the U table.

@testset "ArgmaxStage — closure lowering equals the hand-built budget matrix" begin
    n_w = 64
    g = [exp(t) - 1.0 for t in range(0.0, log(101.0); length = n_w)]
    layout = GriddedLayout(
        :wealth => GriddedContinuous(g),
        :income => Discrete([0.6, 1.0, 1.4]),
    )

    # A start-and-end payoff lowered by `to_matrix_source` sweeps the SAME grid values the
    # hand-built matrix below indexes, so the two stages agree bit-for-bit — the
    # closure-lowering ⇔ matrix equivalence this testset exists for. (This lowering into a
    # discrete `ArgmaxStage` is what the removed CS `discrete = true` mode did; the recipe
    # survives in examples/additional_stages/continuous_argmax_gridsnapped.jl.)
    pay = (b, a) -> b - a > 0 ? log(b - a) : -Inf
    cs  = ArgmaxStage(layout; reward = HouseholdStages.to_matrix_source(pay, layout, layout, :wealth),
                      axis = :wealth) ∘
          TimeDiscountingStage(layout; β = 0.96)

    # The reward as the ordinary (after, before) matrix on wealth: consumption
    # c = wealth_before − wealth_after, log utility, non-positive masked to -Inf —
    # exactly the matrix the lowering builds, so the two agree bit-for-bit.
    reward = [ g[before] - g[after] > 0 ? log(g[before] - g[after]) : -Inf
               for after in 1:n_w, before in 1:n_w ]
    ca = ArgmaxStage(layout; reward = reward, axis = :wealth) ∘
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

@testset "ArgmaxStage — composed discount scales V_end (diagonal-forced policy)" begin
    layout = GriddedLayout(:k => GriddedContinuous([0.0, 1.0, 2.0, 3.0]))
    # reward 0 only on the diagonal (after == before), -Inf elsewhere ⇒ the policy is
    # forced to a = s, so V_start[s] = 0 + β·V_end[s] isolates the discount. Discount is now its
    # own composed `TimeDiscountingStage` (end-goal §1), run before the argmax in the backward sweep.
    reward = [ a == b ? 0.0 : -Inf for a in 1:4, b in 1:4 ]
    V_end  = [10.0, 20.0, 30.0, 40.0]

    ca1 = ArgmaxStage(layout; reward = reward, axis = :k)
    @test copy(backward!(ca1, V_end, NamedTuple())) == V_end          # no discount

    ca2 = ArgmaxStage(layout; reward = reward, axis = :k) ∘
          TimeDiscountingStage(layout; β = 0.5)
    @test copy(backward!(ca2, V_end, NamedTuple())) == 0.5 .* V_end
    @test policy(ca1) == [1, 2, 3, 4]                     # diagonal
end

@testset "ArgmaxStage — reward deps size the U table" begin
    g = [0.0, 1.0, 2.0, 3.0]
    layout = GriddedLayout(
        :wealth => GriddedContinuous(g),
        :income => Discrete([1.0, 2.0]),
    )
    # A closure reward varying along income declares the income dep (a `+ 0.0 * income`
    # no-op): U must be full (size 2) along the income axis, dim 3 of the compact table.
    reward = (; income, env) -> [ g[b] - g[a] > 0 ? log(g[b] - g[a]) + 0.0 * income : -Inf
                                  for a in 1:4, b in 1:4 ]
    ca = ArgmaxStage(layout; reward = reward, axis = :wealth)
    @test size(parent(ca.scratch.U), 3) == 2

    # A constant reward matrix has no deps ⇒ U is singleton along income.
    reward_c = [ g[b] - g[a] > 0 ? log(g[b] - g[a]) : -Inf for a in 1:4, b in 1:4 ]
    ca_c = ArgmaxStage(layout; reward = reward_c, axis = :wealth)
    @test size(parent(ca_c.scratch.U), 3) == 1

    # An undeclared kwarg that is not a layout axis / env is rejected.
    reward_bad = (; income, typo, env) -> zeros(4, 4)
    @test_throws ErrorException ArgmaxStage(layout; reward = reward_bad, axis = :wealth)
end

@testset "ArgmaxStage — the fused small-axis sweep stays reachable" begin
    # The generic column path and the fused unit-stride scan must agree bit-for-bit, and the
    # dispatch condition (dep-free reward, non-leading operative axis, host arrays) must still
    # select the fused one — folding the column loop into the driver must not strand it.
    n_h = 5
    layout = GriddedLayout(
        :wealth => GriddedContinuous(collect(range(0.0, 4.0; length = 7))),
        :h      => Discrete(collect(1.0:n_h)),
        :z      => Discrete([0.5, 1.5]),
    )
    reward = [b - a >= 0 ? 0.5 * log1p(b - a) : -Inf for a in 1:n_h, b in 1:n_h]
    stage  = ArgmaxStage(layout; reward = reward, axis = :h)
    V_end  = float.(rand(MersenneTwister(20260730), 0:3, 7, n_h, 2))   # tie-heavy

    fused = copy(backward!(stage, V_end, nothing))
    pol_fused = copy(policy(stage))

    # Same problem forced onto the generic path by handing it a non-`Array` value buffer.
    Vs    = zeros(7, n_h, 2)
    pol   = zeros(Int, 7, n_h, 2)
    U     = HouseholdStages.matrix_field(Float64, layout, layout, :h, reward)
    HouseholdStages.fill_field!(U, reward, layout, :h, nothing)
    HouseholdStages.stratified!(HouseholdStages.BruteSolveOp(), Vs, pol, V_end, U; dims = Val(2))

    @test fused == Vs                       # bit-identical, not approximate
    @test pol_fused == pol
end
