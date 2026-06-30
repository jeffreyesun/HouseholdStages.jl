###################################################
# Insurance steady state — partial equilibrium      #
###################################################

# Returns/wage are exogenous, so there is no market to clear: the "outer loop"
# is a single inner V/Λ fixed-point solve at the given env. (Contrast the
# Aiyagari/Krusell–Smith examples, which roll a tatonnement on K̄.) The whole
# point is that the household block is library stages only — the insurance
# choice is `MixingStage` blending a no-loss and a loss wealth kernel at convex
# cost. See `model.jl`.

include("model.jl")

using Printf

"""
Solve the insurance household steady state at the exogenous env and report wealth and the seated
insurance-coverage policy `θ*(x)` (the share routed through the no-loss kernel `K_A = I`). Returns
the stationary `(V, Λ)`, the `mean_wealth` moment, and the coverage policy summary.
"""
function insurance_steady_state(p = insurance_params; env = insurance_env(p), verbosity = 1)
    hh  = insurance_household(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    θ = HouseholdStages.policy(hh.buffer.stages[2])        # the Insurance (MixingStage) leaf
    cov_lo, cov_hi, cov_mean = minimum(θ), maximum(θ), sum(θ) / length(θ)

    if verbosity > 0
        @printf "Insurance steady state (r = %.3f, w = %.2f, σ = %.1f, loss = %.2f, κ = %.1f)\n" env.r env.w p.σ p.loss_factor p.cost_curvature
        @printf "  mass(Λ)          = %.6f\n"     sum(res.Λ)
        @printf "  mean wealth      = %.4f\n"     m.mean_wealth
        @printf "  coverage θ*      = [%.2f, %.2f], grid-mean %.3f\n" cov_lo cov_hi cov_mean
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              coverage = θ, cov_mean, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving insurance-demand steady state…")
    @time insurance_steady_state()
end
