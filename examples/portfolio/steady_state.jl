###################################################
# Portfolio steady state — partial equilibrium     #
###################################################

# Returns are exogenous, so there is no market to clear: the "outer loop"
# is a single inner V/Λ fixed-point solve at the given env. (Contrast the
# Aiyagari/Krusell–Smith examples, which roll a tatonnement on K̄.) The
# whole point is that the household block is library stages only —
# see `model.jl`.

include("model.jl")

using Printf

"""
Solve the portfolio household steady state at the given wage `w` and report wealth and the risky
share. Returns the stationary `(V, Λ)`, the `mean_wealth` moment, and the population summary of the
seated risky-share policy `θ*(x)`.
"""
function portfolio_steady_state(p = portfolio_params; w = p.w, verbosity = 1)
    hh  = portfolio_household(p)
    env = (; w)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    θ = HouseholdStages.policy(hh.buffer.stages[end])      # the Portfolio (MeanVarianceStage) leaf
    share_lo, share_hi, share_mean = minimum(θ), maximum(θ), sum(θ) / length(θ)

    if verbosity > 0
        @printf "Portfolio steady state (w = %.2f, σ = %.1f, premium = %.3f)\n" w p.σ (sum(p.p_risky .* p.R_risky) - p.R_f)
        @printf "  mass(Λ)          = %.6f\n"     sum(res.Λ)
        @printf "  mean wealth      = %.4f\n"     m.mean_wealth
        @printf "  risky share θ*   = [%.2f, %.2f], grid-mean %.3f\n" share_lo share_hi share_mean
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              risky_share = θ, share_mean, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving portfolio-choice steady state…")
    @time portfolio_steady_state()
end
