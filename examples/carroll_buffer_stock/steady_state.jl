###############################################
# Carroll (1997) buffer-stock — steady state   #
###############################################
#
# Partial equilibrium in NORMALIZED (ratio-to-permanent-income) units: a single
# inner V/Λ solve at fixed prices delivers the stationary distribution of
# normalized cash-on-hand m — the buffer-stock target. There is no market to
# clear here (this is the partial-equilibrium buffer-stock experiment); the
# permanent-income normalization is what renders the unit-root problem
# stationary on a fixed grid. See model.jl for the chain and the normalization.

include("model.jl")

using Printf

function carroll_steady_state(p = carroll_params; verbosity = 1)
    hh  = carroll_household(p)
    env = carroll_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    if verbosity > 0
        @printf "Carroll buffer-stock steady state (β=%.3f, σ=%.1f, R=%.3f, G=%.3f)\n" p.β p.σ p.R p.G
        @printf "  mass(Λ)            = %.6f\n" sum(res.Λ)
        @printf "  mean norm. m       = %.4f\n" m.mean_m
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_m = m.mean_m, history = res.history)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Carroll buffer-stock steady state…")
    @time carroll_steady_state()
end
