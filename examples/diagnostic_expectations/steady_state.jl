###################################################################
# Diagnostic expectations — distorted vs rational, fixed-r solves   #
###################################################################
#
# Fixed-r partial-equilibrium solve. We solve the SAME block twice — once
# with the diagnostic tilt (θ > 0), once rational (θ = 0) — and compare the
# stationary buffer stock. The only difference between the two runs is the
# offline-distorted transition matrix; the household block and the solver
# are identical.

include("model.jl")

using Printf

"""
Solve the diagnostic-expectations self-insurance steady state at the fixed
exogenous return `r` for a given diagnosticity `θ` (one inner V/Λ fixed
point) and report the aggregate buffer stock.
"""
function diagnostic_steady_state(p = diagnostic_params; θ = p.θ, verbosity = 1)
    hh  = diagnostic_household(p; θ)
    env = diagnostic_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "  θ = %.2f : ΣΛ = %.6f, A_mean = %.4f  (VFI %d, Λ %d)\n" θ sum(res.Λ) m.A_mean res.history.vfi_iters res.history.lambda_iters
    end
    return (; θ, A_mean = m.A_mean, V = res.V, Λ = res.Λ, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = diagnostic_params
    println("Diagnostic expectations (Bordalo-Gennaioli-Shleifer 2018) — fixed-r solves")
    @printf "  r = %.4f (1/β − 1 = %.4f), diagnosticity θ = %.2f\n" p.r (1/p.β - 1) p.θ
    @time begin
        rational   = diagnostic_steady_state(p; θ = 0.0)
        diagnostic = diagnostic_steady_state(p; θ = p.θ)
    end
    @printf "  ΔA (diagnostic − rational) = %+.4f  (%.1f%% vs rational)\n" (diagnostic.A_mean - rational.A_mean) (100 * (diagnostic.A_mean / rational.A_mean - 1))
end
