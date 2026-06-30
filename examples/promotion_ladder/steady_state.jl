#########################################################################
# Promotion ladder — stationary steady state (partial equilibrium)      #
#########################################################################

# The return r is exogenous, so there is no market to clear: the "outer loop" is a
# single inner V/Λ fixed-point solve at the given env (cf. the insurance example). The
# point of interest is the seated promotion policy θ*(rung) — read from the MixingStage
# leaf after the solve — which should be INTERIOR in (0,1) and HIGHER for low rungs
# (they have the most to gain from climbing).

include("model.jl")

using Printf, Statistics

"""
Solve the promotion-ladder household at the exogenous env and report the rung
distribution, mean wealth, and the seated promotion policy `θ*(rung)`. The latter is read
from the `MixingStage` leaf (`hh.buffer.stages[1]`) and averaged over wealth per rung —
it should be interior in (0,1) and decreasing in rung.
"""
function promotion_steady_state(p = promotion_params; env = promotion_env(p), verbosity = 1)
    hh  = promotion_household(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = res.moments

    n_rung   = length(p.wage_grid)
    rung_pmf = [getfield(m, Symbol("rung_$k")) for k in 1:n_rung]   # mass on each rung

    # Seated promotion intensity θ*(wealth, rung) from the MixingStage leaf; average over
    # the wealth dimension (dim 1) to get θ*(rung).
    θ        = HouseholdStages.policy(hh.buffer.stages[1])          # shape (N_w, n_rung)
    θ_byrung = vec(mean(θ; dims = 1))

    if verbosity > 0
        @printf "Promotion-ladder steady state (r = %.3f, σ = %.1f, κ = %.2f)\n" env.r p.σ p.cost_curvature
        @printf "  mass(Λ)        = %.6f\n" sum(res.Λ)
        @printf "  mean wealth    = %.4f\n" m.mean_wealth
        @printf "  mean rung wage = %.4f\n" m.mean_rung
        println("  rung distribution (wage ⇒ mass):")
        for k in 1:n_rung
            @printf "    rung %d (w = %.2f): mass = %.4f,  θ*(promotion) = %.4f\n" k p.wage_grid[k] rung_pmf[k] θ_byrung[k]
        end
        @printf "  θ* range = [%.4f, %.4f]  (interior ⇒ in (0,1))\n" minimum(θ) maximum(θ)
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth, mean_rung = m.mean_rung,
              rung_pmf, θ_byrung, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving promotion-ladder steady state…")
    @time promotion_steady_state()
end
