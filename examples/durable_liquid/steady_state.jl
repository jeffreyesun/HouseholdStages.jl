####################################################
# Durable + liquid (S,s) steady state — partial eq. #
####################################################

# Returns and the wage are exogenous, so there is no market to clear: the "outer
# loop" is a single inner V/Λ fixed-point solve. The household block — one durable
# choice that sets the durable axis AND debits liquid by the down-payment — is built
# from existing stages via the auxiliary-choice-axis pattern (see model.jl).

include("model.jl")

using Printf

"""
The adjustment rate `∫ 1{d' ≠ d} dΛ_pre` — the mass that re-optimises its durable this period.
A *transition* statistic (both adjusters and keepers land on `cell.durable = d'`, so it is invisible
to an `at_end` integrand): recovered driver-side from the choice policy and the distribution entering
the choice. `stages = (depreciate, shock, receipt, CHOOSE, …)`, so the pre-choice Λ is `Λ` pushed
through `depreciate`, `shock`, `receipt`, and `policy(choose)` gives the chosen durable index per cell
(durable measured AFTER depreciation, so the count includes both replacements and voluntary upgrades).
"""
function durable_adjustment_rate(hh, Λ_end)
    stages = hh.buffer.stages
    Λ_pre  = forward!(stages[3], forward!(stages[2], forward!(stages[1], copy(Λ_end))))  # entering CHOOSE
    pol    = HouseholdStages.policy(stages[4])                        # chosen :durable_choice index per cell
    # block axes order: liquid(1), durable(2), income(3), durable_choice(4 — singleton here).
    adj = 0.0
    for ci in CartesianIndices(Λ_pre)
        pol[ci] != ci[2] && (adj += Λ_pre[ci])                       # chosen durable index ≠ current durable index
    end
    return adj / sum(Λ_pre)
end

"""
Solve the durable+liquid (S,s) steady state at the exogenous env and report mean liquid, mean durable,
and the adjustment rate. Returns the stationary `(V, Λ)` and the moments.
"""
function durable_liquid_steady_state(p = durable_liquid_params; verbosity = 1)
    hh  = durable_liquid_household(p)
    env = durable_liquid_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    adj = durable_adjustment_rate(hh, res.Λ)

    if verbosity > 0
        @printf "Durable+liquid (S,s) steady state (β=%.2f, σ=%.1f, F=%.2f, p=%.2f, θ=%.2f)\n" p.β p.σ p.F p.p p.θ
        @printf "  mass(Λ)         = %.6f\n" sum(res.Λ)
        @printf "  mean liquid     = %.4f\n" m.mean_liquid
        @printf "  mean durable    = %.4f\n" m.mean_durable
        @printf "  adjustment rate = %.4f\n" adj
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, m.mean_liquid, m.mean_durable,
              adjustment_rate = adj, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving durable+liquid (S,s) steady state…")
    @time durable_liquid_steady_state()
end
