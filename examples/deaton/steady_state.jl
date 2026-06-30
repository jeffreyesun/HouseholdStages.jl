############################################################
# Deaton Steady State — single solve at fixed exogenous r   #
############################################################

# Buffer-stock saving in partial equilibrium. The return `r` is FIXED and
# exogenous with `β(1+r) < 1` (impatience), so a single inner V/Λ fixed-
# point solve at the given env delivers the stationary buffer-stock wealth
# distribution. No market is cleared.

include("model.jl")

using Printf

"""
Solve the Deaton buffer-stock steady state at the fixed exogenous return
`r` and report the buffer-stock summary: the impatience factor `β(1+r)`,
the aggregate buffer stock `A_mean`, and the share `frac_constrained` at the
binding `a = 0` constraint (large under buffer-stock behaviour).
"""
function deaton_steady_state(p = deaton_params; verbosity = 1)
    hh  = deaton_household(p)
    env = deaton_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    impatience = p.β * (1 + p.r)

    if verbosity > 0
        @printf "Deaton buffer-stock steady state (β = %.3f, σ = %.1f)\n" p.β p.σ
        @printf "  r                  = %.4f   β(1+r) = %.4f  (< 1 ⇒ impatient)\n" p.r impatience
        @printf "  ΣΛ                 = %.6f\n" sum(res.Λ)
        @printf "  A_mean (buffer)    = %.4f\n" m.A_mean
        @printf "  frac at constraint = %.4f\n" m.frac_constrained
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r = p.r, V = res.V, Λ = res.Λ,
              A_mean = m.A_mean, frac_constrained = m.frac_constrained,
              impatience, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Deaton buffer-stock steady state…")
    @time res = deaton_steady_state()
    @printf "  β(1+r) = %.4f  (< 1 ⇒ buffer-stock saving)\n" res.impatience
end
