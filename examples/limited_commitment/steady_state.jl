###############################################################
# Limited-Commitment Steady State — single solve at fixed r    #
###############################################################

# Fixed-`r` partial-equilibrium solve of the DECENTRALIZED household block under
# given per-state solvency bounds (the planner's fixed point that derives those
# bounds is out of scope). A single inner V/Λ fixed point at the given env. The
# per-env inner work is delegated to `HouseholdStages.solve_steady_state_given_env!`.

include("model.jl")

using Printf

"""
Solve the decentralized limited-commitment steady state at the fixed return `r`
(single inner V/Λ solve) and report the aggregate position `A_mean` together with
mean assets in the low- vs high-income states — the per-state solvency bounds let
high-income households carry more debt, which shows up here.
"""
function limited_commitment_steady_state(p = limited_commitment_params; verbosity = 1)
    hh  = limited_commitment_household(p)
    env = limited_commitment_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "Limited-commitment steady state (β = %.2f, σ = %.1f, r = %.4f)\n" p.β p.σ p.r
        @printf "  solvency bounds B(y) = %s  (y = %s)\n" string(p.solvency_bounds) string(p.y_grid)
        @printf "  ΣΛ                   = %.6f\n" sum(res.Λ)
        @printf "  A_mean               = %+.4f\n" m.A_mean
        @printf "  mean a | low income  = %+.4f   (bound %.2f)\n" m.A_lowy p.solvency_bounds[1]
        @printf "  mean a | high income = %+.4f   (bound %.2f)\n" m.A_highy p.solvency_bounds[end]
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r = p.r, V = res.V, Λ = res.Λ,
              A_mean = m.A_mean, A_lowy = m.A_lowy, A_highy = m.A_highy,
              history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving limited-commitment (decentralized) steady state…")
    @time res = limited_commitment_steady_state()
    @printf "  ΣΛ = %.6f\n" sum(res.Λ)
end
