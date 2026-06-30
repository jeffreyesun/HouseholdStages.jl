##############################################################
# Risk-shifting steady state — partial equilibrium            #
##############################################################

# Returns, productivity, and the wage are exogenous, so there is no market to
# clear: the "outer loop" is a single inner V/Λ fixed-point solve at the given
# env. (Contrast the Aiyagari/Krusell–Smith examples, which roll a tatonnement
# on K̄.) The whole point is that the household block is library stages only —
# see `model.jl`. The report shows the Vereshchagina–Hopenhayn comparative
# static: the seated project-risk policy θ*(x) is HIGH near the limited-
# liability floor (gambling for resurrection) and LOW for well-capitalized
# entrepreneurs.

include("model.jl")

using Printf

"""
Solve the risk-shifting household steady state at the given env and report
wealth and the risk-shifting policy. Returns the stationary `(V, Λ)`, the
`mean_wealth` moment, and a poor-vs-rich summary of the seated project-risk
policy `θ*(x)`.
"""
function risk_shifting_steady_state(p = risk_shifting_params; w = p.w, a_floor = p.a_floor,
                                    verbosity = 1)
    hh  = risk_shifting_household(p)
    env = risk_shifting_env(p; w, a_floor)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    θ    = HouseholdStages.policy(risk_shifting_gamble_stage(hh))   # project-risk θ*(x)
    grid = GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log).grid

    # Poor (near the floor) vs rich (top decile) average project risk.
    fl       = findfirst(>=(a_floor), grid)
    poor     = fl:min(fl + 4, p.N_a)
    rich     = round(Int, 0.9 * p.N_a):p.N_a
    θ_poor   = sum(@view θ[poor, :]) / (length(poor) * length(p.z_grid))
    θ_rich   = sum(@view θ[rich, :]) / (length(rich) * length(p.z_grid))

    if verbosity > 0
        meanrisky = sum([p.p_up, 1 - p.p_up] .* [p.R_up, p.R_dn])
        @printf "Risk-shifting steady state (σ = %.1f, a_floor = %.2f, E[R_k] = %.3f vs R_f = %.2f)\n" p.σ a_floor meanrisky p.R_f
        @printf "  mass(Λ)               = %.6f\n"  sum(res.Λ)
        @printf "  mean wealth           = %.4f\n"  m.mean_wealth
        @printf "  project risk θ*: poor = %.3f  (near floor a≈%.2f)\n" θ_poor a_floor
        @printf "  project risk θ*: rich = %.3f  (top decile a≈%.0f)\n" θ_rich grid[end]
        @printf "  ⇒ risk-shifting: poor gamble %.1f× more than rich\n" (θ_poor / max(θ_rich, 1e-3))
        @printf "  VFI iters = %d, Λ iters = %d\n"  res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              risk_policy = θ, θ_poor, θ_rich, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving risk-shifting (gambling-for-resurrection) steady state…")
    @time risk_shifting_steady_state()
end
