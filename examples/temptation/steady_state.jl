###############################################################
# Temptation steady state — single fixed-r inner solve         #
###############################################################
#
# Gul–Pesendorfer temptation as a partial-equilibrium experiment at fixed
# real return `r` and wage `w`: no market clears, so the stationary
# distribution follows from ONE `solve_steady_state_given_env!`. The household
# block (`model.jl`) is the Aiyagari spine; the temptation content is the
# closed-form corner self-control cost folded into the savings utility closure.

include("model.jl")

using Printf

"""
Solve the temptation household at the fixed env `(r, w)`: one inner V/Λ solve.
Returns the converged `V`, `Λ`, aggregate wealth `K`, and a top-of-grid mass
diagnostic. Temptation tilts toward present consumption, so `K` here is lower
than the no-temptation (λ = 0) Aiyagari counterpart at the same env.
"""
function temptation_steady_state(p = temptation_params; verbosity = 1)
    hh  = temptation_household(p)
    env = temptation_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    K = compute_moments(hh, Λ, env).K_supplied
    top_mass = sum(Λ[end, :]) / sum(Λ)

    if verbosity > 0
        @printf "Temptation steady state (β = %.2f, σ = %.1f, σ_t = %.1f, λ = %.2f, r = %.3f)\n" p.β p.σ p.σ_t p.λ p.r
        @printf "  ΣΛ (total mass)        : %.6f\n" sum(Λ)
        @printf "  V finite               : %s\n" all(isfinite, V)
        @printf "  aggregate wealth K     : %.4f\n" K
        @printf "  top-cell mass fraction : %.2e\n" top_mass
        @printf "  VFI iters / Λ iters    : %d / %d\n" history.vfi_iters history.lambda_iters
    end

    return (; V, Λ, K, top_mass, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving temptation steady state…")
    @time temptation_steady_state()
end
