###############################################################
# Wealth-in-utility steady state — single fixed-r inner solve  #
###############################################################
#
# Wealth-in-utility is run here as a partial-equilibrium experiment at fixed
# real return `r` and wage `w`: there is no market to clear, so the stationary
# distribution follows from ONE `solve_steady_state_given_env!`. The household
# block (`model.jl`) is the Aiyagari spine; the capitalist-spirit content is
# the direct taste-for-wealth term inside the savings utility closure.

include("model.jl")

using Printf

"""
Solve the wealth-in-utility household at the fixed env `(r, w)`: one inner
V/Λ solve. Returns the converged `V`, `Λ`, aggregate wealth `K`, plus a
top-of-grid mass diagnostic (the fraction of mass in the topmost wealth cell;
small ⇒ the grid contains the stationary distribution).
"""
function wiu_steady_state(p = wiu_params; verbosity = 1)
    hh  = wiu_household(p)
    env = wiu_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    K = compute_moments(hh, Λ, env).K_supplied
    top_mass = sum(Λ[end, :]) / sum(Λ)

    if verbosity > 0
        @printf "Wealth-in-utility steady state (β = %.2f, σ = %.1f, σ_w = %.1f, χ = %.2f, r = %.3f)\n" p.β p.σ p.σ_w p.χ p.r
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
    println("Solving wealth-in-utility steady state…")
    @time wiu_steady_state()
end
