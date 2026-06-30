###############################################################
# R&D / knowledge investment — stationary equilibrium           #
###############################################################

# Partial equilibrium (no prices to clear), so the "outer loop" is a single
# stationary V/Λ solve over the joint `(knowledge, shock)` state via
# `solve_steady_state_given_env!`: the value function is iterated to its fixed
# point and the distribution forward to stationarity. The persistent demand chain
# `:shock` gives a genuine non-degenerate stationary distribution over
# `(knowledge, shock)` — knowledge tracks demand with convex-cost smoothing, so the
# knowledge-marginal is spread (not a point mass). See `model.jl`.

include("model.jl")

using Printf

"""
Solve the R&D / knowledge-investment stationary equilibrium. Runs one
`solve_steady_state_given_env!` over the `(knowledge, shock)` block, then reports
the joint stationary distribution's key features: V finiteness, Λ convergence +
mass, mean knowledge / revenue / demand, the spread of the knowledge-marginal (its
occupied support and interquantile range), and the monotonicity of mean knowledge
by demand state (knowledge should rise in the demand shock).
"""
function rnd_investment_steady_state(p = rnd_investment_params; verbosity = 1)
    hh  = rnd_investment_household(p)
    env = rnd_investment_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res

    cells  = cell_array(output_layout(hh))            # (N_k, N_s) cells with fields (:knowledge, :shock)
    k_vals = getfield.(cells, :knowledge)
    s_vals = getfield.(cells, :shock)

    m      = compute_moments(hh, Λ, env)
    k_marg = vec(sum(Λ; dims = 2))                    # marginal over knowledge
    s_marg = vec(sum(Λ; dims = 1))                    # marginal over the demand shock
    mean_k_by_s = vec(sum(k_vals .* Λ; dims = 1)) ./ max.(s_marg, eps())
    s_axis      = s_vals[1, :]

    occupied = count(>(1e-10), k_marg)
    k_grid   = vec(k_vals[:, 1])
    cdf      = cumsum(k_marg)
    q10      = k_grid[findfirst(>=(0.10), cdf)]
    q50      = k_grid[findfirst(>=(0.50), cdf)]
    q90      = k_grid[findfirst(>=(0.90), cdf)]

    k_rising_in_s = issorted(mean_k_by_s)

    if verbosity > 0
        @printf "R&D / knowledge investment (β = %.2f, γ = %.2f, η = %.2f, δ_z = %.2f, c_rnd = %.1f)\n" p.β p.γ p.η p.δ_z p.c_rnd
        @printf "  grid: N_k = %d ∈ [%.2f, %.2f], N_shock = %d\n" p.N_k p.k_min p.k_max p.N_s
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere      = %s\n" all(isfinite, V)
        @printf "  ΣΛ (mass conserved)      = %.8f\n" sum(Λ)
        @printf "  mean knowledge           = %.4f\n" m.mean_knowledge
        @printf "  mean revenue (s·k^η)     = %.4f\n" m.mean_revenue
        @printf "  mean shock               = %.4f\n" m.mean_shock
        @printf "  knowledge-marginal: %d / %d grid points carry mass\n" occupied p.N_k
        @printf "  knowledge q10/q50/q90    = %.3f / %.3f / %.3f\n" q10 q50 q90
        @printf "  knowledge NON-degenerate = %s\n" (occupied > 1 && q90 > q10)
        @printf "  mean knowledge by shock  = [%s]\n" join((@sprintf("%.3f", x) for x in mean_k_by_s), ", ")
        @printf "  knowledge rising in shock= %s\n" k_rising_in_s
    end

    return (; V, Λ, m, k_marg, s_marg, mean_k_by_s, s_axis,
              q10, q50, q90, occupied, k_rising_in_s, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving R&D / knowledge-investment stationary equilibrium…")
    @time rnd_investment_steady_state()
end
