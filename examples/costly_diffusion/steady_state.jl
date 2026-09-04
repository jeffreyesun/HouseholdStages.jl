####################################################################
# Costly-diffusion steady state — partial equilibrium              #
####################################################################

# Returns and income are exogenous, so there is no market to clear: the "outer
# loop" is a single inner V/Λ fixed-point solve at the given env. (Contrast the
# Aiyagari/Krusell–Smith examples, which roll a tatonnement on K̄.) The whole
# point is that the household block is library stages only — the diffusion leaf
# is a `MeanPreservingSpreadStage` in the θ↑ ("diffuse") reading, no bespoke household
# stage. See `model.jl`.
#
# The report shows the diffusion comparative static: the seated dispersion
# policy θ*(x) is HIGH near the limited-liability floor (deliberate diffusion —
# the convex region) and → 0 for well-capitalized households (concave V). It is
# the mirror image of `examples/risk_shifting` (θ* DECREASING in net worth), but
# via an ADDITIVE mean-preserving spread (`MeanPreservingSpreadStage`) rather than a
# multiplicative risky share (`GaussianLoadingStage`).

include("model.jl")

using Printf

"""
Solve the costly-diffusion household steady state at the given env and report
wealth and the seated dispersion policy θ*(x). Returns the stationary `(V, Λ)`,
the `mean_wealth` moment, and a near-floor-vs-rich summary of θ*.
"""
function costly_diffusion_steady_state(p = costly_diffusion_params;
                                       r = 0.0, w = p.w, λ = p.λ, a_floor = p.a_floor,
                                       verbosity = 1)
    hh  = costly_diffusion_household(p)
    env = costly_diffusion_env(p; r, w, λ, a_floor)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    θ    = HouseholdStages.policy(costly_diffusion_diffuse_stage(hh))   # dispersion θ*(x)
    grid = GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log).grid

    # Near the floor (convex V) vs rich (top decile, concave V) average diffusion.
    fl     = findfirst(>=(a_floor), grid)
    poor   = fl:min(fl + 9, p.N_w)
    rich   = round(Int, 0.9 * p.N_w):p.N_w
    θ_poor = sum(@view θ[poor, :]) / (length(poor) * length(p.y_grid))
    θ_rich = sum(@view θ[rich, :]) / (length(rich) * length(p.y_grid))
    θ_mean = sum(θ) / length(θ)
    diffuse_frac = sum(θ .> 0) / length(θ)

    if verbosity > 0
        @printf "Costly-diffusion steady state (σ = %.1f, a_floor = %.2f, λ = %.4f)\n" p.σ a_floor λ
        @printf "  mass(Λ)              = %.6f\n"  sum(res.Λ)
        @printf "  mean wealth          = %.4f\n"  m.mean_wealth
        @printf "  dispersion θ*: poor  = %.3f  (near floor a≈%.2f, convex V)\n" θ_poor a_floor
        @printf "  dispersion θ*: rich  = %.3f  (top decile a≈%.1f, concave V)\n" θ_rich grid[end]
        @printf "  mean θ*              = %.3f,  frac(θ*>0) = %.3f\n" θ_mean diffuse_frac
        @printf "  ⇒ deliberate diffusion: poor spread %.1f× more than rich\n" (θ_poor / max(θ_rich, 1e-3))
        @printf "  VFI iters = %d, Λ iters = %d\n"  res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              dispersion = θ, θ_poor, θ_rich, θ_mean, diffuse_frac, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving costly-diffusion (deliberate variance-increase) steady state…")
    @time costly_diffusion_steady_state()

    # The diffusion comparative static: mean chosen dispersion falls as the
    # dispersion cost λ rises (the cost disciplines the deliberate spread).
    println("\nComparative static — mean chosen dispersion vs. dispersion cost λ:")
    for λ in (0.0, 0.005, 0.02, 0.08)
        r = costly_diffusion_steady_state(; λ, verbosity = 0)
        @printf "  λ = %.4f  →  mean θ* = %.4f,  θ*(poor) = %.3f,  frac(θ*>0) = %.3f\n" λ r.θ_mean r.θ_poor r.diffuse_frac
    end
end
