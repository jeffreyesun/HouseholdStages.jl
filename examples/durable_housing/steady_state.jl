###################################################
# Durable-housing steady state — partial equilibrium #
###################################################

# Returns, wage, and the house price are exogenous, so there is no market
# to clear: the "outer loop" is a single inner V/Λ fixed-point solve at the
# given env. (Contrast the Aiyagari / Krusell–Smith examples, which roll a
# tatonnement on K̄.) The whole point is that the household block is library
# stages only — see `model.jl`.

include("model.jl")

using Printf

"""
Solve the durable-housing household steady state at the exogenous env and report
wealth, the housing stock, and the homeownership rate. Returns the stationary
`(V, Λ)`, the attached moments, and the seated housing-size policy.
"""
function durable_housing_steady_state(p = durable_housing_params; verbosity = 1)
    hh  = durable_housing_household(p)
    env = durable_housing_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    # The buy choice is the 3rd leaf (move, shock, BUY, receipt, usercost, savings).
    h_pol = HouseholdStages.policy(hh.buffer.stages[3])

    if verbosity > 0
        @printf "Durable-housing steady state (r = %.3f, w = %.2f, q = %.2f, σ = %.1f)\n" p.r p.w p.q p.σ
        @printf "  mass(Λ)        = %.6f\n" sum(res.Λ)
        @printf "  mean wealth    = %.4f\n" m.mean_wealth
        @printf "  mean house     = %.4f\n" m.mean_house
        @printf "  ownership rate = %.4f\n" m.own_rate
        @printf "  size policy    = [%d, %d] (housing index)\n" minimum(h_pol) maximum(h_pol)
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ,
              mean_wealth = m.mean_wealth, mean_house = m.mean_house,
              own_rate = m.own_rate, h_pol, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving durable-housing (S,s) steady state…")
    @time durable_housing_steady_state()
end
