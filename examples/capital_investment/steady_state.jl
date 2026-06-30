###############################################################
# Capital investment with convex adjustment cost — steady state #
###############################################################

# Partial equilibrium (the discount rate `r` is exogenous), so there is no
# market to clear. The "outer loop" is a single stationary V/Λ solve over the
# joint `(k, z)` state via `solve_steady_state_given_env!`: the value function is
# iterated to its fixed point and the distribution forward to stationarity. The
# persistent profitability chain `:z` gives a genuine non-degenerate stationary
# distribution over `(k, z)` — capital tracks profitability with convex-cost
# smoothing, so the k-marginal is spread (not a point mass). See `model.jl`.

include("model.jl")

using Printf

"""
Solve the capital-investment stationary equilibrium at the exogenous discount
rate `r`. Runs one `solve_steady_state_given_env!` over the `(k, z)` block, then
reports the joint stationary distribution's key features: V finiteness, Λ
convergence + mass, mean capital / profit / profitability, the spread of the
k-marginal (its support and Gini-free dispersion), and the monotonicity of the
mean-capital-by-z policy profile (capital should rise in profitability z).
"""
function capital_investment_steady_state(p = capital_investment_params; verbosity = 1)
    hh  = capital_investment_household(p)
    env = capital_investment_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res

    cells  = cell_array(output_layout(hh))            # (N_k, N_z) cells with fields (:k, :z)
    k_vals = getfield.(cells, :k)
    z_vals = getfield.(cells, :z)

    m         = compute_moments(hh, Λ, env)
    k_marg    = vec(sum(Λ; dims = 2))                 # marginal distribution over k
    z_marg    = vec(sum(Λ; dims = 1))                 # marginal distribution over z
    # Mean capital conditional on each profitability state (the k–z policy in the cross-section).
    mean_k_by_z = vec(sum(k_vals .* Λ; dims = 1)) ./ max.(z_marg, eps())
    z_axis      = z_vals[1, :]

    # Spread of the k-marginal: number of grid points carrying mass, and the
    # interquantile capital range — a point mass would put all weight on one point.
    occupied = count(>(1e-10), k_marg)
    k_sorted_idx = sortperm(vec(k_vals[:, 1]))
    k_grid = vec(k_vals[:, 1])
    cdf    = cumsum(k_marg)
    q10    = k_grid[findfirst(>=(0.10), cdf)]
    q50    = k_grid[findfirst(>=(0.50), cdf)]
    q90    = k_grid[findfirst(>=(0.90), cdf)]

    k_rising_in_z = issorted(mean_k_by_z)

    if verbosity > 0
        @printf "Capital investment, convex adjustment (α = %.2f, φ = %.1f, δ = %.2f, r = %.3f)\n" p.α p.φ p.δ p.r
        @printf "  grid: N_k = %d ∈ [%.2f, %.2f], N_z = %d\n" p.N_k p.k_min p.k_max p.N_z
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere      = %s\n" all(isfinite, V)
        @printf "  ΣΛ (mass conserved)      = %.8f\n" sum(Λ)
        @printf "  mean k                   = %.4f\n" m.mean_k
        @printf "  mean profit (z·k^α)      = %.4f\n" m.mean_profit
        @printf "  mean z                   = %.4f\n" m.mean_z
        @printf "  k-marginal: %d / %d grid points carry mass\n" occupied p.N_k
        @printf "  k quantiles q10/q50/q90  = %.3f / %.3f / %.3f\n" q10 q50 q90
        @printf "  k NON-degenerate         = %s\n" (occupied > 1 && q90 > q10)
        @printf "  mean k by z              = [%s]\n" join((@sprintf("%.3f", x) for x in mean_k_by_z), ", ")
        @printf "  k rising in z            = %s\n" k_rising_in_z
    end

    return (; V, Λ, m, k_marg, z_marg, mean_k_by_z, z_axis,
              q10, q50, q90, occupied, k_rising_in_z, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving capital-investment (convex adjustment) stationary equilibrium…")
    @time capital_investment_steady_state()
end
