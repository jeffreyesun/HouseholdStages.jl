##############################################################
# Collateral-constrained investment — Partial-Equilibrium SS #
##############################################################

# Partial equilibrium: the rental/saving rate `r` is exogenous, so there is no market to
# clear and the "outer loop" is a single inner solve — `solve_steady_state_given_env!`
# runs V backward to its fixed point and Λ forward to stationarity over `(net worth, z)`.
# (A general-equilibrium close would wrap this in a tatonnement on `r`, with the wealth
# distribution as the aggregate state; see aiyagari's driver for that pattern.)
#
# The reported moments demonstrate the financial-friction mechanism: a non-trivial fraction
# of firms hit the collateral constraint (`λa < k*`), and the resulting dispersion of the
# marginal product of capital — zero in the frictionless allocation, where every firm sets
# MPK = r+δ — measures the capital misallocation the friction creates.

include("model.jl")

using Printf
using Statistics: std

"""
Solve the partial-equilibrium collateral-investment steady state at exogenous `r`: build the
firm block, run the single inner V/Λ solve, and return the value, stationary distribution,
and the production/misallocation moments.
"""
function collateral_investment_steady_state(p = collateral_investment_params)
    firm = collateral_investment_household(p)
    env  = collateral_investment_env(p)
    res  = solve_steady_state_given_env!(firm, env)
    m    = compute_moments(firm, res.Λ, env)
    return (; firm, env, res.V, res.Λ, res.history, moments = m)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = collateral_investment_params
    println("Solving collateral-constrained investment steady state (partial equilibrium)…")
    @printf "  λ = %.2f, r = %.3f, δ = %.3f, α = %.3f, β = %.3f\n" p.λ p.r p.δ p.α p.β
    @time out = collateral_investment_steady_state(p)
    (; V, Λ, history, moments) = out

    mass     = sum(Λ)
    std_logmpk = sqrt(max(0.0, moments.mean_log_mpk_sq - moments.mean_log_mpk^2))

    println("\n--- Solve diagnostics ---")
    @printf "  V finite           : %s (range [%.3f, %.3f])\n" all(isfinite, V) minimum(V) maximum(V)
    @printf "  mass ΣΛ            : %.8f\n" mass
    @printf "  VFI iters / Λ iters: %d / %d\n" history.vfi_iters history.lambda_iters

    println("\n--- Aggregates (per-firm means, ΣΛ = 1) ---")
    @printf "  mean net worth  E[a]      : %.4f\n" moments.mean_a
    @printf "  mean productivity E[z]    : %.4f\n" moments.mean_z
    @printf "  mean profit  E[π]         : %.4f\n" moments.mean_profit
    @printf "  mean capital operated E[k]: %.4f\n" moments.mean_k

    println("\n--- Financial friction / misallocation ---")
    @printf "  fraction constrained (λa<k*): %.4f\n" moments.frac_constrained
    @printf "  mean log MPK                : %.4f   (frictionless: log(r+δ) = %.4f)\n" moments.mean_log_mpk log(p.r + p.δ)
    @printf "  std  log MPK  (misalloc.)   : %.4f   (frictionless: 0)\n" std_logmpk

    # Capital operated rises in BOTH net worth a and productivity z — print a small (a × z) table.
    println("\n--- k(a, z): capital operated rises in a and z (✓ checks below) ---")
    env       = out.env
    layout, _ = collateral_investment_layout(p)
    z_grid    = axis_grid(layout, :z)
    a_grid    = axis_grid(layout, :wealth)
    a_show = a_grid[round.(Int, range(1, length(a_grid); length = 5))]
    z_show = z_grid[[1, (length(z_grid) + 1) ÷ 2, length(z_grid)]]
    @printf "  %10s" "a \\ z"
    for z in z_show; @printf " %10.3f" z; end
    println()
    for a in a_show
        @printf "  %10.3f" a
        for z in z_show; @printf " %10.3f" k_operated(a, z, env); end
        println()
    end
    k_up_in_a = all(diff([k_operated(a, z_show[end], env) for a in a_grid]) .>= -1e-9)
    k_up_in_z = all(diff([k_operated(a_grid[end], z, env) for z in z_grid]) .>= -1e-9)
    @printf "  k nondecreasing in a (top z): %s\n" k_up_in_a
    @printf "  k nondecreasing in z (top a): %s\n" k_up_in_z
end
