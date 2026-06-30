############################################################
# Bewley Steady State — single solve at fixed exogenous r   #
############################################################

# Pure partial-equilibrium self-insurance experiment. The interest rate `r`
# is FIXED and exogenous (strictly below `1/β − 1`), so there is no market
# to clear: the whole "outer loop" is a single inner V/Λ fixed-point solve
# at the given env. (Contrast Aiyagari/Huggett, which roll a tatonnement /
# bisection to clear a market; the Bewley point is the self-insurance
# mechanism itself, not general equilibrium.) The per-env inner work — V
# backward to a fixed point, then Λ forward to the stationary distribution —
# is delegated to `HouseholdStages.solve_steady_state_given_env!`.

include("model.jl")

using Printf

"""
Solve the Bewley self-insurance steady state at the fixed exogenous return
`r` (a single inner V/Λ fixed-point solve — no market clearing) and report
the precautionary-savings summary: the aggregate buffer stock `A_mean`, the
hand-to-mouth share `frac_constrained` pinned at the borrowing constraint,
and the impatience gap `1/β − 1 − r`.

Returns the stationary `(V, Λ)`, the attached moments, and the inner-solve
history.
"""
function bewley_steady_state(p = bewley_params; r = p.r, verbosity = 1)
    hh  = bewley_household(p)
    env = bewley_env(r)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    gap = 1 / p.β - 1 - r   # impatience gap; > 0 ⇒ stationary distribution exists

    if verbosity > 0
        @printf "Bewley self-insurance steady state (β = %.3f, σ = %.1f)\n" p.β p.σ
        @printf "  r                  = %.4f   (1/β − 1 = %.4f, impatience gap = %.4f)\n" r (1 / p.β - 1) gap
        @printf "  ΣΛ                 = %.6f\n" sum(res.Λ)
        @printf "  A_mean (buffer)    = %.4f\n" m.A_mean
        @printf "  frac at constraint = %.4f\n" m.frac_constrained
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r, V = res.V, Λ = res.Λ,
              A_mean = m.A_mean, frac_constrained = m.frac_constrained,
              gap, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Bewley self-insurance steady state…")
    @time res = bewley_steady_state()
    @printf "  impatience gap 1/β − 1 − r = %.4f  (> 0 ⇒ stationary)\n" res.gap
end
