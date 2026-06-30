#####################################################################
# Uninsured investment risk steady state — the precautionary wedge    #
#####################################################################

# The capital-return distribution, the wage, and the safe return are
# exogenous, so there is no market to clear: the "outer loop" is a single
# inner V/Λ fixed-point solve. The whole point is that the household block is
# library stages only — see `model.jl`.
#
# The report SWEEPS the idiosyncratic capital-return variance (the half-spread
# Δ) from a near-zero benchmark upward, holding the MEAN capital return μ_k
# fixed, and shows the precautionary investment wedge: the risky-capital share
# θ* and aggregate risky capital both FALL as undiversifiable variance rises.

include("model.jl")

using Printf

"""
Solve the uninsured-investment-risk household at one capital-return half-spread
`Δ` (mean μ_k held fixed) and return the stationary `(V, Λ)`, mean wealth, and
risky-share summary. The wealth-weighted risky share is the aggregate fraction
of wealth in risky capital; mean risky capital is `∫ θ*(x)·wealth dΛ`.
"""
function investment_risk_solve(p = investment_risk_params; Δ = p.Δ)
    hh  = investment_risk_household(p; Δ)
    env = investment_risk_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    θ        = HouseholdStages.policy(investment_risk_stage(hh))   # θ*(x), shape (N_w, N_income)
    w_vals   = getfield.(cell_array(output_layout(hh)), :wealth)
    mass     = sum(res.Λ)
    θ_grid   = sum(θ) / length(θ)                                  # grid-mean share (portfolio convention)
    θ_wtd    = sum(θ .* res.Λ) / mass                              # wealth-weighted aggregate share
    risky_K  = sum(θ .* w_vals .* res.Λ) / mass                   # mean risky capital ∫ θ·w dΛ

    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth, θ,
              θ_grid, θ_wtd, risky_K, mean_w = m.mean_wealth, history = res.history)
end

"""
Sweep the idiosyncratic capital-return half-spread `Δ` and print the
precautionary investment wedge: risky share and mean risky capital falling as
variance rises (mean return held fixed).
"""
function investment_risk_steady_state(p = investment_risk_params;
                                      Δ_grid = [0.005, 0.10, 0.20, 0.30, 0.40], verbosity = 1)
    rows = [(; Δ, r = investment_risk_solve(p; Δ)) for Δ in Δ_grid]

    if verbosity > 0
        prem = p.μ_k - p.R_f
        @printf "Uninsured investment risk (β = %.2f, σ = %.1f, R_f = %.2f, μ_k = %.2f, premium = %.3f)\n" p.β p.σ p.R_f p.μ_k prem
        @printf "  the risky asset is the agent's OWN capital; θ = share of wealth in risky capital\n\n"
        @printf "  %8s %10s %14s %16s %12s\n" "Δ (var↑)" "θ* (grid)" "θ* (wtd)" "mean risky K" "mean wealth"
        for row in rows
            @printf "  %8.3f %10.3f %14.3f %16.4f %12.4f\n" row.Δ row.r.θ_grid row.r.θ_wtd row.r.risky_K row.r.mean_w
        end
        lo, hi = rows[1].r, rows[end].r
        wedge = hi.θ_wtd < lo.θ_wtd && hi.risky_K < lo.risky_K
        @printf "\n  ⇒ precautionary wedge: as idiosyncratic variance rises, risky share & capital FALL: %s\n" wedge
        @printf "    risky share θ* (wtd): %.3f (near-zero var) → %.3f (Δ=%.2f), a %.0f%% drop\n" lo.θ_wtd hi.θ_wtd rows[end].Δ 100 * (1 - hi.θ_wtd / lo.θ_wtd)
        @printf "    mean risky capital:   %.3f → %.3f, a %.0f%% drop\n" lo.risky_K hi.risky_K 100 * (1 - hi.risky_K / lo.risky_K)
    end
    return rows
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving uninsured-investment-risk steady state (variance sweep)…")
    @time investment_risk_steady_state()
end
