###################################################################
# Variance-RI steady state — partial equilibrium                   #
###################################################################

# Returns are exogenous, so there is no market to clear: the "outer loop"
# is a single inner V/Λ fixed-point solve at the given env. (Contrast the
# Aiyagari/Krusell–Smith examples, which roll a tatonnement on K̄.) The
# whole point is that the household block is library stages only — the
# attention leaf is a `MeanPreservingSpreadStage`, no bespoke household stage.
# See `model.jl`.

include("model.jl")

using Printf

"""
Solve the variance-RI household steady state at the given returns/wage and information cost, and
report wealth and the seated attention (dispersion) policy θ*(x). Returns the stationary `(V, Λ)`,
the `mean_wealth` moment, and the population summary of θ*.
"""
function rational_inattention_steady_state(p = rational_inattention_params;
                                           r = p.r, w = p.w, λ = p.λ, verbosity = 1)
    hh  = rational_inattention_household(p)
    env = (; r, w, λ)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    θ = HouseholdStages.policy(hh.buffer.stages[end])      # the Attention (MeanPreservingSpreadStage) leaf
    θ_lo, θ_hi, θ_mean = minimum(θ), maximum(θ), sum(θ) / length(θ)
    attentive_frac = sum(θ .> 0) / length(θ)               # share of cells choosing positive dispersion

    if verbosity > 0
        @printf "Variance-RI steady state (r = %.3f, w = %.2f, σ = %.1f, λ = %.4f)\n" r w p.σ λ
        @printf "  mass(Λ)          = %.6f\n"     sum(res.Λ)
        @printf "  mean wealth      = %.4f\n"     m.mean_wealth
        @printf "  dispersion θ*    = [%.2f, %.2f], cell-mean %.3f\n" θ_lo θ_hi θ_mean
        @printf "  frac(θ* > 0)     = %.3f  (rest choose perfect attention)\n" attentive_frac
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              dispersion = θ, θ_mean, attentive_frac, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving variance-RI steady state…")
    @time res = rational_inattention_steady_state()

    # The RI comparative static: θ* falls as the information cost λ rises.
    println("\nComparative static — mean chosen dispersion vs. information cost λ:")
    for λ in (0.0, 0.001, 0.01, 0.05)
        r = rational_inattention_steady_state(; λ, verbosity = 0)
        @printf "  λ = %.4f  →  mean θ* = %.4f,  frac(θ*>0) = %.3f\n" λ r.θ_mean r.attentive_frac
    end
end
