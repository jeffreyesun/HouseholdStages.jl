###################################################################
# Regime-switching steady state — single solve at fixed r          #
###################################################################
#
# Fixed-r partial-equilibrium solve. The regime is an ENDOGENOUS axis with
# its own Markov law, so the single stationary Λ carries both aggregate
# states jointly — the recession share below is the regime chain's own
# ergodic mass, recovered as a moment of the joint distribution.

include("model.jl")

using Printf

"""
Solve the regime-switching self-insurance steady state at the fixed
exogenous return `r` (one inner V/Λ fixed point over the joint
`(regime, income, wealth)` state) and report the aggregate buffer stock and
the recession population share.
"""
function regime_switching_steady_state(p = regime_switching_params; verbosity = 1)
    hh  = regime_switching_household(p)
    env = regime_switching_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    # Ergodic recession share of the regime chain alone, for cross-check.
    π_regime = let T = p.P_regime, π = [0.5, 0.5]
        for _ in 1:10_000; π = vec(π' * T); end
        π ./ sum(π)
    end

    if verbosity > 0
        @printf "Regime-switching income steady state (Hamilton-style)\n"
        @printf "  regimes : %s   income : %s\n" string(p.regime_grid) string(p.y_grid)
        @printf "  r = %.4f   (1/β − 1 = %.4f, impatience gap = %.4f)\n" p.r (1/p.β - 1) (1/p.β - 1 - p.r)
        @printf "  ΣΛ              = %.6f\n" sum(res.Λ)
        @printf "  A_mean          = %.4f\n" m.A_mean
        @printf "  recession_share = %.4f   (regime-chain ergodic = %.4f)\n" m.recession_share π_regime[2]
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r = p.r, V = res.V, Λ = res.Λ,
              A_mean = m.A_mean, recession_share = m.recession_share, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving regime-switching self-insurance steady state…")
    @time res = regime_switching_steady_state()
    @printf "  ΣΛ = %.6f, A_mean = %.4f, recession_share = %.4f\n" sum(res.Λ) res.A_mean res.recession_share
end
