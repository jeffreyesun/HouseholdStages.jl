using Test, HouseholdStages

# The model file defines top-level names (`params`, `u_*`, helpers); wrap the
# include in a module so those globals don't clash with other examples'
# identically-named definitions when the whole suite runs.
module ChooSiowExampleTest
  using Test, HouseholdStages
  include(joinpath(@__DIR__, "..", "examples", "choo_siow", "model.jl"))

  @testset "example: choo_siow — existing stages only" begin
      p = params

      # The household block is assembled entirely from library stages —
      # Flow (UtilityStage) ∘ Discount (TimeDiscountingStage) ∘
      # PartnerChoice (LogitChoiceStage) ∘ Dissolution (MarkovStage).
      hh = choo_siow_household(p)

      n   = ones(length(p.partner_types))          # symmetric partner availability
      env = choo_siow_env(n, p)
      res = solve_steady_state_given_env!(hh, env)

      # Stationary distribution: a proper probability mass, finite values.
      @test sum(res.Λ) ≈ 1.0 atol = 1e-6
      @test all(res.Λ .>= -1e-12)
      @test all(isfinite, res.V)

      # The marital chain is ergodic, so the steady state is non-degenerate:
      # some singles, some matched (neither corner is absorbing).
      @test 0 < res.moments.single_share < 1
      @test res.moments.match_rate ≈ 1 - res.moments.single_share atol = 1e-8

      # Steady-state accounting: at the stationary distribution the inflow to
      # singles (dissolutions) balances the outflow (new matches). The match
      # hazard out of single is `1 − π(single | single)`; dissolutions in are
      # `δ · matched_mass`. Both equal the cross-period single flow, so
      #   δ · (1 − single_share) = (single inflow) implies single_share is
      # pinned by δ and the match propensity — here we just check the share is
      # decreasing in the dissolution hazard δ (more dissolution ⇒ more singles).
      p_hi = ChooSiowParams(; δ = 2 * p.δ)
      hh_hi = choo_siow_household(p_hi)
      res_hi = solve_steady_state_given_env!(hh_hi, choo_siow_env(n, p_hi))
      @test res_hi.moments.single_share > res.moments.single_share

      # Choo-Siow comparative static: raising the availability of one partner
      # type pulls stationary mass toward matches with that type. Position k+1
      # on the marital axis is `matched-k`; bump type 2's availability.
      states = marital_states(p)
      k2_idx = findfirst(==(Symbol("matched_", p.partner_types[2])), states)
      n_more2 = copy(n); n_more2[2] = 3.0
      res2 = solve_steady_state_given_env!(hh, choo_siow_env(n_more2, p))
      @test res2.Λ[k2_idx] > res.Λ[k2_idx]

      # Policy sanity: with ε = 1 the partner choice is a genuine logit, so the
      # single state assigns strictly positive probability to every match
      # destination AND to staying single (no degenerate corner). Read it off
      # the seated logit kernel (stage 3 in the chain), single = origin 1.
      choice = hh.buffer.stages[3]
      kern = choice.kernel
      K1 = length(states)
      single_probs = [parent(kern.eψC)[1, j] * kern.value_weight[j, 1] / kern.normalizer[1, 1]
                      for j in 1:K1]
      @test sum(single_probs) ≈ 1.0 atol = 1e-8
      @test all(single_probs .> 0)
  end
end
