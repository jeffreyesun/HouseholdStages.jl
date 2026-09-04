using Test
using HouseholdStages
using Statistics

# End-to-end validation: the continuous MeanPreservingSpreadStage composed in a real household chain
# and solved to a stationary steady state through the library outer loop (VFI fixed point + forward Λ
# stationarity + moments). A stylized mean-reverting income-fluctuation block with a risk/attention
# choice — the economics are deliberately simple; the point is that the primitive composes with `∘`
# and runs through `solve_steady_state_given_env!` and `define_moments!`/`compute_moments` as is.

@testset "streaming-choice — steady state in a ∘-composed chain" begin
    ws     = collect(range(0.0, 10.0; length = 60))
    ys     = [0.8, 1.2]
    P      = [0.8 0.2; 0.2 0.8]
    layout = GriddedLayout(:wealth => GriddedContinuous(ws), :income => Discrete(ys))

    shock   = MarkovStage(layout; axis = :income, transition_matrix = P)
    receipt = WealthChangeStage(layout; axis = :wealth,                          # ρ via env ⇒ an SSJ input
                                wealth_post = (; wealth, income, env) -> env.ρ * wealth + income)
    util    = UtilityStage(layout; utility = (; wealth) -> sqrt(wealth))   # concave ⇒ risk-averse
    risk    = MeanPreservingSpreadStage(layout; axis = :wealth, θ_max = 1.5,
                                        cost = (θ; env) -> 0.05 * θ^2)
    disc    = TimeDiscountingStage(layout; β = 0.95)

    hh  = shock ∘ receipt ∘ util ∘ risk ∘ disc
    hh  = define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
    env = (; ρ = 0.85)                                               # mean-reversion ⇒ bounded stationary wealth

    res = solve_steady_state_given_env!(hh, env)
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-6)                     # stationary distribution, mass conserved
    @test all(isfinite, res.V)                                       # VFI fixed point is finite

    m = compute_moments(hh, res.Λ, env)
    @test m.mean_wealth > 0                                          # moment path runs end-to-end

    # Risk-averse (concave) continuation ⇒ the attention/risk choice mostly insures (low dispersion).
    @test mean(policy(risk)) < 1.5 / 2

    # The sequence-space Jacobian runs THROUGH the chain containing the streaming stage and returns
    # a finite, non-trivial d(mean_wealth)/dρ.
    J = compute_fake_news_ssj(hh, env, 3; inputs = (:ρ,), outputs = (:mean_wealth,))
    @test size(J) == (3, 3)
    @test all(isfinite, J)
    @test abs(J[1, 1]) > 0                                           # ρ moves the date-0 distribution
end
