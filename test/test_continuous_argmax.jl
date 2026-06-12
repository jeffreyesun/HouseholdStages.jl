using Test
using HouseholdStages

# Direct tests of the ContinuousArgmaxStage primitive (the generalisation behind
# ConsumptionSavingsStage). The CS-specific numeric tests live in
# test_monotone_argmax.jl; here we exercise the primitive's general payoff API
# and the CS ⇔ primitive equivalence that the wrapper relies on.
#
# The payoff is a *kwarg* closure: it declares the reserved `choice` kwarg (the
# choice-axis value), the input layout axes it reads, and optionally `env`. The
# declared input axes size the U table (read via `Base.kwarg_decl` — no IR walk).

@testset "ContinuousArgmaxStage — equals CS with the budget payoff" begin
    n_w = 64
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid(
            [exp(t) - 1.0 for t in range(0.0, log(101.0); length = n_w)])),
        StateAxis(:income, discrete_finite([0.6, 1.0, 1.4])),
    )
    u = (cell, c; env) -> log(c)

    cs = ConsumptionSavingsStage(layout; β = 0.96, utility = u, wealth_axis = :wealth)

    # Hand-built primitive with the budget payoff in kwarg form: declares the
    # `wealth` input axis (b_in) and the `choice` value (b_end); consumption
    # `c = wealth - choice`, non-positive masked to -Inf exactly as the wrapper.
    pay = (; wealth, choice, env) -> begin
        c = wealth - choice
        c > 0 ? log(c) : -Inf
    end
    ca = ContinuousArgmaxStage(layout; β = 0.96, payoff = pay, choice_axis = :wealth)

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

@testset "ContinuousArgmaxStage — β discounts V_end (diagonal-forced policy)" begin
    layout = GriddedLayout(StateAxis(:k, continuous_grid([0.0, 1.0, 2.0, 3.0])))
    # payoff 0 only on the diagonal (choice == k), -Inf elsewhere ⇒ the policy is
    # forced to a = s, so V_start[s] = 0 + β·V_end[s] isolates the discount.
    pay = (; k, choice, env) -> k == choice ? 0.0 : -Inf
    V_end = [10.0, 20.0, 30.0, 40.0]

    ca1 = ContinuousArgmaxStage(layout; payoff = pay, choice_axis = :k,
                                assume_monotone = true)
    @test copy(backward!(ca1, V_end, NamedTuple())) == V_end          # β = 1

    ca2 = ContinuousArgmaxStage(layout; payoff = pay, choice_axis = :k, β = 0.5,
                                assume_monotone = true)
    @test copy(backward!(ca2, V_end, NamedTuple())) == 0.5 .* V_end
    @test policy(ca1) == [1, 2, 3, 4]                     # diagonal
end

@testset "ContinuousArgmaxStage — payoff deps from the kwarg signature size U" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([1.0, 2.0])),
    )
    # The payoff declares it reads the income axis (a `+ 0.0 * income` no-op):
    # the kwarg signature is the dependency record. U must be full along the
    # income axis (dim 3 of the table), not singleton.
    pay = (; wealth, income, choice, env) ->
        (wealth - choice) > 0 ? log(wealth - choice) + 0.0 * income : -Inf
    ca = ContinuousArgmaxStage(layout; payoff = pay, choice_axis = :wealth)
    # U's compact parent is (dest, origin, dep…); the declared income dep is dim 3, size 2.
    @test size(parent(ca.scratch.U), 3) == 2

    # A payoff that does *not* declare the income kwarg leaves U singleton there
    # (the dep mechanism reads the declared kwargs, not the body).
    pay_w = (; wealth, choice, env) ->
        (wealth - choice) > 0 ? log(wealth - choice) : -Inf
    ca_w = ContinuousArgmaxStage(layout; payoff = pay_w, choice_axis = :wealth)
    @test size(parent(ca_w.scratch.U), 3) == 1

    # An undeclared kwarg that is not a layout axis / choice / env is rejected.
    pay_bad = (; wealth, choice, typo, env) -> log(wealth - choice)
    @test_throws ErrorException ContinuousArgmaxStage(layout; payoff = pay_bad,
                                                      choice_axis = :wealth)

    # A payoff missing the reserved `choice` kwarg is rejected.
    pay_nochoice = (; wealth, env) -> log(wealth)
    @test_throws ErrorException ContinuousArgmaxStage(layout; payoff = pay_nochoice,
                                                      choice_axis = :wealth)
end
