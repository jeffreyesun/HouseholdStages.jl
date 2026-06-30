##################################################################
# Perpetual Youth — Partial-Equilibrium Stationary Distribution  #
##################################################################

# Partial equilibrium: factor prices `(r, w)` are fixed (no market to
# clear), so the steady state is a single V/Λ fixed point delivered by
# `solve_steady_state_given_env!`. The interesting object is the
# stationary cross-section Λ — in particular its total mass, which the
# mass-preserving rebirth (`Σg = δ`, survival `1−δ`) pins at ≈ 1.

include("model.jl")

using Printf

"""
Solve the perpetual-youth stationary distribution at fixed prices
`(r, w)`. Returns the household block, the converged `(V, Λ)`, and the
mean-wealth / population-mass moments.
"""
function perpetual_youth_steady_state(p = perpetual_youth_params;
                                      r = 0.03, w = 1.0)
    hh  = perpetual_youth_household(p)
    env = (; r, w)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    mean_wealth = m.A_total / m.pop
    return (; hh, env, V = res.V, Λ = res.Λ,
              pop = m.pop, A_total = m.A_total, mean_wealth,
              history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving perpetual-youth (Blanchard–Yaari) stationary distribution…")
    p = perpetual_youth_params
    @time res = perpetual_youth_steady_state(p; r = 0.03, w = 1.0)

    (; vfi_iters, lambda_iters) = res.history
    @printf "  δ (death hazard)      = %.3f\n"  p.δ
    @printf "  annuity gross return  = %.4f   (= (1+r)/(1−δ))\n" (1 + res.env.r) / (1 - p.δ)
    @printf "  VFI iters / Λ iters   = %d / %d\n" vfi_iters lambda_iters
    @printf "  population mass  ΣΛ   = %.6f   (target ≈ 1)\n" res.pop
    @printf "  mean wealth E[w]      = %.4f\n" res.mean_wealth
    @printf "  ∫ wealth dΛ           = %.4f\n" res.A_total
    @printf "  V finite everywhere   = %s\n"  all(isfinite, res.V)
end
