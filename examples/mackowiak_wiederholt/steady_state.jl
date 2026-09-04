####################################################################
# Single-margin MW steady state — partial equilibrium               #
####################################################################

# Returns are exogenous, so there is no market to clear: the outer loop is a
# single inner V/Λ fixed-point solve at the given env. The whole point is that
# the household block is library stages only — the attention leaf is a
# `MeanPreservingSpreadStage`, no bespoke household stage. See `model.jl`. The report
# shows the MW single-margin attention gradient: θ*(x) is HIGH for the
# constrained poor (option value near the borrowing constraint) and → 0 for the
# wealthy, and falls everywhere as the information cost λ rises.

include("model.jl")

using Printf

"""
Solve the single-margin MW household steady state at the given returns/wage and
information cost, and report wealth and the seated attention policy θ*(x).
Returns the stationary `(V, Λ)`, the `mean_wealth` moment, and poor-vs-rich and
population summaries of the attention dispersion θ*.
"""
function mw_steady_state(p = mw_params; r = p.r, w = p.w, λ = p.λ, verbosity = 1)
    hh  = mw_household(p)
    env = (; r, w, λ)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    θ = HouseholdStages.policy(mw_attention_stage(hh))     # θ*(x) over (wealth, income, z)
    θ_lo, θ_hi, θ_mean = minimum(θ), maximum(θ), sum(θ) / length(θ)
    attentive_frac = sum(θ .> 0) / length(θ)

    # Attention gradient in wealth: poorest decile vs richest decile.
    poor = 1:max(1, p.N_w ÷ 10)
    rich = round(Int, 0.9 * p.N_w):p.N_w
    θ_poor = sum(@view θ[poor, :, :]) / length(@view θ[poor, :, :])
    θ_rich = sum(@view θ[rich, :, :]) / length(@view θ[rich, :, :])

    if verbosity > 0
        @printf "Single-margin MW steady state (r = %.3f, w = %.2f, σ = %.1f, λ = %.4f)\n" r w p.σ λ
        @printf "  mass(Λ)             = %.6f\n"     sum(res.Λ)
        @printf "  mean wealth         = %.4f\n"     m.mean_wealth
        @printf "  dispersion θ*       = [%.2f, %.2f], cell-mean %.3f\n" θ_lo θ_hi θ_mean
        @printf "  frac(θ* > 0)        = %.3f  (rest choose perfect attention)\n" attentive_frac
        @printf "  attention gradient  : poor θ* = %.3f, rich θ* = %.3f  (poor attend less precisely)\n" θ_poor θ_rich
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              dispersion = θ, θ_mean, θ_poor, θ_rich, attentive_frac, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving single-margin Maćkowiak–Wiederholt steady state…")
    @time res = mw_steady_state()

    # The RI comparative static: the chosen attention dispersion falls as the
    # information cost λ rises (precision is cheaper ⇒ more dispersion taken).
    println("\nComparative static — mean chosen dispersion vs. information cost λ:")
    for λ in (0.0, 0.001, 0.01, 0.05)
        r = mw_steady_state(; λ, verbosity = 0)
        @printf "  λ = %.4f  →  mean θ* = %.4f,  frac(θ*>0) = %.3f\n" λ r.θ_mean r.attentive_frac
    end
end
