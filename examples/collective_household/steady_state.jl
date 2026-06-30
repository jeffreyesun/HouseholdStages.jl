###############################################################
# Collective-household steady state — single fixed-r solve     #
###############################################################
#
# Fixed-Pareto-weight collective household as a partial-equilibrium experiment
# at fixed real return `r` and wage `w`: no market clears, so the stationary
# distribution follows from ONE `solve_steady_state_given_env!`. The household
# block (`model.jl`) is the Aiyagari spine; the collective content is the
# Pareto-weighted two-member objective inside the savings utility closure.

include("model.jl")

using Printf

"""
Solve the collective household at the fixed env `(r, w)`: one inner V/Λ solve.
Returns the converged `V`, `Λ`, aggregate wealth `K`, and a top-of-grid mass
diagnostic (fraction of mass in the topmost wealth cell).
"""
function collective_steady_state(p = collective_params; verbosity = 1)
    hh  = collective_household(p)
    env = collective_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    K = compute_moments(hh, Λ, env).K_supplied
    top_mass = sum(Λ[end, :]) / sum(Λ)

    if verbosity > 0
        @printf "Collective-household steady state (β = %.2f, σ_A = %.1f, σ_B = %.1f, μ = %.2f, s = %.2f, r = %.3f)\n" p.β p.σ_A p.σ_B p.μ p.s p.r
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
    println("Solving collective-household steady state…")
    @time collective_steady_state()
end
