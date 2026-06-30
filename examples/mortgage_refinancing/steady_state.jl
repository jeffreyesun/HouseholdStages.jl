######################################################
# Mortgage-refinancing steady state — partial eq.    #
######################################################

# Prices (the liquid return, the mortgage rate, the wage) are exogenous, so there
# is no market to clear: the outer loop is a single inner V/Λ fixed-point solve.
# The household block — one LTV-gated refi choice that sets the mortgage balance
# AND moves liquid by the principal change + fixed cost — is built from existing
# stages via the auxiliary-choice-axis pattern (see model.jl).

include("model.jl")

using Printf

"""
The refinancing rate `∫ 1{m' ≠ m} dΛ_pre` — the mass that adjusts its mortgage balance this period
(cash-out or prepay). A *transition* statistic (both refinancers and keepers land on
`cell.mortgage = m'`, invisible to an `at_end` integrand): recovered driver-side from the choice policy
and the distribution entering the choice. `stages = (shock_y, shock_h, receipt, CHOOSE, …)`, so the
pre-choice Λ is `Λ` pushed through the three shocks/receipt, and `policy(choose)` gives the chosen
balance index per origin cell.
"""
function refinancing_rate(hh, Λ_end)
    stages = hh.buffer.stages
    Λ_pre  = forward!(stages[3], forward!(stages[2], forward!(stages[1], copy(Λ_end))))  # entering CHOOSE
    pol    = HouseholdStages.policy(stages[4])                       # chosen :refi_choice index per cell
    # block axes order: wealth(1), mortgage(2), income(3), hp(4), refi_choice(5 — singleton here).
    adj = 0.0
    for ci in CartesianIndices(Λ_pre)
        pol[ci] != ci[2] && (adj += Λ_pre[ci])                      # chosen balance index ≠ current balance index
    end
    return adj / sum(Λ_pre)
end

"""
Solve the mortgage-refinancing steady state at the exogenous env and report mean wealth, mean mortgage
balance, and the refinancing rate. Returns the stationary `(V, Λ)` and the moments.
"""
function mortgage_steady_state(p = mortgage_params; verbosity = 1)
    hh  = mortgage_household(p)
    env = mortgage_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    refi = refinancing_rate(hh, res.Λ)

    if verbosity > 0
        @printf "Mortgage-refinancing steady state (β=%.2f, r_m=%.3f, κ=%.3f, θ_ltv=%.2f)\n" p.β p.r_m p.κ p.θ_ltv
        @printf "  mass(Λ)          = %.6f\n" sum(res.Λ)
        @printf "  mean wealth      = %.4f\n" m.mean_wealth
        @printf "  mean mortgage    = %.4f\n" m.mean_mortgage
        @printf "  refinancing rate = %.4f\n" refi
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, m.mean_wealth, m.mean_mortgage,
              refinancing_rate = refi, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving mortgage-refinancing steady state…")
    @time mortgage_steady_state()
end
