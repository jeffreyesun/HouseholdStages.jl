####################################################################
# Countercyclical risk — solve at z = boom AND z = recession        #
####################################################################
#
# In a steady state the aggregate state `z` is fixed, so we run two
# independent fixed-r solves — one per `z` — on the SAME household object.
# Both calls hit `solve_steady_state_given_env!` with a different `env.z`;
# the env-closure transition re-seats accordingly. Comparing the two
# stationary wealth distributions demonstrates the re-seating: higher
# idiosyncratic risk in recession ⇒ a stronger precautionary motive ⇒ more
# self-insurance wealth.

include("model.jl")

using Printf

"""
Solve the countercyclical-risk steady state at a fixed aggregate state `z`
(one inner V/Λ fixed point) and return the buffer-stock moment. The same
`hh` object is reused across states; only `env.z` differs, exercising the
env-closure re-seat.
"""
function countercyclical_steady_state(z::Symbol, p = countercyclical_params;
                                      hh = countercyclical_household(p), verbosity = 1)
    env = countercyclical_env(z, p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "  z = %-10s : ΣΛ = %.6f, A_mean = %.4f  (VFI %d, Λ %d)\n" string(z) sum(res.Λ) m.A_mean res.history.vfi_iters res.history.lambda_iters
    end
    return (; z, A_mean = m.A_mean, V = res.V, Λ = res.Λ, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p  = countercyclical_params
    hh = countercyclical_household(p)
    println("Countercyclical idiosyncratic risk — fixed-r solves at z = boom and z = recession")
    @printf "  r = %.4f (1/β − 1 = %.4f), σ_recession = %.2f > σ_boom = %.2f (ρ = %.2f fixed)\n" p.r (1/p.β - 1) p.σ_recession p.σ_boom p.ρ
    @time begin
        boom = countercyclical_steady_state(:boom,      p; hh)
        rec  = countercyclical_steady_state(:recession, p; hh)
    end
    @printf "  ΔA (recession − boom) = %+.4f  (%.1f%% more self-insurance wealth in recession)\n" (rec.A_mean - boom.A_mean) (100 * (rec.A_mean / boom.A_mean - 1))
end
