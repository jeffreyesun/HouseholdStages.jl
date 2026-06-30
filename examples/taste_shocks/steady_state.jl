############################################################
# Taste-Shock Steady State — single solve at fixed r        #
############################################################

# Fixed-`r` partial-equilibrium solve (the Bewley framing). A single inner V/Λ
# fixed point at the exogenous return delivers the stationary joint distribution
# over (wealth, income, taste). The per-env inner work is delegated to
# `HouseholdStages.solve_steady_state_given_env!`.

include("model.jl")

using Printf

"""
Solve the taste-shock self-insurance steady state at the fixed exogenous return
`r` (single inner V/Λ solve, no market clearing) and report the aggregate buffer
stock `A_mean` and how mean assets split across taste states (the precautionary
buffer is drawn down in high-taste / high-marginal-value periods).
"""
function taste_shock_steady_state(p = taste_shock_params; verbosity = 1)
    hh  = taste_shock_household(p)
    env = taste_shock_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "Taste-shock self-insurance steady state (β = %.3f, σ = %.1f, r = %.4f)\n" p.β p.σ p.r
        @printf "  taste grid (flow shifts) = %s\n" string(p.taste_grid)
        @printf "  ΣΛ                       = %.6f\n" sum(res.Λ)
        @printf "  A_mean (buffer stock)    = %.4f\n" m.A_mean
        @printf "  mean assets|high taste   = %.4f\n" m.A_hightaste
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r = p.r, V = res.V, Λ = res.Λ,
              A_mean = m.A_mean, A_hightaste = m.A_hightaste, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving taste-shock self-insurance steady state…")
    @time res = taste_shock_steady_state()
    @printf "  ΣΛ = %.6f\n" sum(res.Λ)
end
