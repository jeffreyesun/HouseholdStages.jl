###################################################
# Default steady state — partial equilibrium       #
###################################################

# Prices are exogenous (a risk-free unit bond), so there is no market to
# clear: the "outer loop" is a single inner V/Λ fixed-point solve at the
# given env. (Contrast Aiyagari/Krusell–Smith, which roll a tatonnement on
# K̄.) The whole point is that the household block — including the
# repay/default choice — is library stages only; see `model.jl`.

include("model.jl")

using Printf

"""
Solve the default household steady state at the exogenous env and report mean assets, the
stationary default rate, and the borrowing-vs-saving split. Returns the stationary `(V, Λ)` and
the two moments.
"""
function default_steady_state(p = default_params; verbosity = 1)
    hh  = default_household(p)
    env = default_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "Default steady state (β = %.2f, σ = %.1f, haircut λ = %.2f, readmit ψ = %.2f)\n" p.β p.σ p.λ p.ψ
        @printf "  mass(Λ)        = %.6f\n"  sum(res.Λ)
        @printf "  mean assets    = %.4f\n"  m.mean_assets
        @printf "  excluded rate  = %.4f\n"  m.excluded_rate
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_assets = m.mean_assets,
              excluded_rate = m.excluded_rate, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving sovereign/consumer-default steady state…")
    @time default_steady_state()
end
