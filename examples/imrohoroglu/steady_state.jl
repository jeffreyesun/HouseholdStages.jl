############################################################
# İmrohoroğlu Steady State — single solve at fixed exogenous r #
############################################################

# Partial-equilibrium self-insurance against employment risk. The return
# `r` is FIXED and exogenous (strictly below `1/β − 1`), so there is no
# market to clear — the "outer loop" is a single inner V/Λ fixed-point
# solve at the given env. The per-env inner work is delegated to
# `HouseholdStages.solve_steady_state_given_env!`.

include("model.jl")

using Printf

"""
Solve the İmrohoroğlu self-insurance steady state at the fixed exogenous
return `r` and report the precautionary summary: the aggregate buffer stock
`A_mean`, the share `frac_constrained` pinned at the liquidity constraint,
and the impatience gap `1/β − 1 − r`.
"""
function imrohoroglu_steady_state(p = imrohoroglu_params; verbosity = 1)
    hh  = imrohoroglu_household(p)
    env = imrohoroglu_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    gap = 1 / p.β - 1 - p.r

    if verbosity > 0
        @printf "İmrohoroğlu self-insurance steady state (β = %.3f, σ = %.1f)\n" p.β p.σ
        @printf "  r                  = %.4f   (1/β − 1 = %.4f, impatience gap = %.4f)\n" p.r (1 / p.β - 1) gap
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
    println("Solving İmrohoroğlu self-insurance steady state…")
    @time res = imrohoroglu_steady_state()
    @printf "  impatience gap 1/β − 1 − r = %.4f  (> 0 ⇒ stationary)\n" res.gap
end
