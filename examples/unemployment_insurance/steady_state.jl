############################################################
# Hansen–İmrohoroğlu UI Steady State — single solve at fixed r #
############################################################

# Partial-equilibrium self-insurance against employment risk, with an
# explicit UI scheme (replacement rate `ρ`, payroll tax `τ`) in the budget.
# The return `r` is FIXED and exogenous (strictly below `1/β − 1`), so the
# whole solve is a single inner V/Λ fixed point at the given env.

include("model.jl")

using Printf

"""
Solve the Hansen–İmrohoroğlu UI steady state at the fixed exogenous return
`r` and report the precautionary summary: the aggregate buffer stock
`A_mean`, the share `frac_constrained` at the liquidity constraint, and the
UI policy `(ρ, τ)`.
"""
function ui_steady_state(p = ui_params; verbosity = 1)
    hh  = ui_household(p)
    env = ui_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    gap = 1 / p.β - 1 - p.r

    if verbosity > 0
        @printf "Hansen–İmrohoroğlu UI steady state (β = %.3f, σ = %.1f)\n" p.β p.σ
        @printf "  r                  = %.4f   (1/β − 1 = %.4f, impatience gap = %.4f)\n" p.r (1 / p.β - 1) gap
        @printf "  UI policy          : replacement ρ = %.2f, payroll tax τ = %.2f\n" p.ρ p.τ
        @printf "  ΣΛ                 = %.6f\n" sum(res.Λ)
        @printf "  A_mean (buffer)    = %.4f\n" m.A_mean
        @printf "  frac at constraint = %.4f\n" m.frac_constrained
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r = p.r, V = res.V, Λ = res.Λ,
              A_mean = m.A_mean, frac_constrained = m.frac_constrained,
              gap, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Hansen–İmrohoroğlu UI steady state…")
    @time res = ui_steady_state()
    @printf "  impatience gap 1/β − 1 − r = %.4f  (> 0 ⇒ stationary)\n" res.gap
end
