###################################################################
# Hansen–Sargent robust savings — precautionary comparative static #
###################################################################
#
# Partial equilibrium: returns are exogenous, so the "outer loop" is a single
# inner V/Λ solve per robustness multiplier θ = |ε|. We sweep θ from very robust
# (θ = 0.5) to near-risk-neutral (θ = 1e8) and report TWO objects:
#
#   (1) The PRECAUTIONARY policy statistic — the average chosen next-period
#       wealth b'(state), evaluated under a COMMON fixed reference distribution
#       (the non-robust stationary Λ). This is a pure function of the savings
#       policy / robust value function, so it isolates "how cautious is the
#       household" from "what measure generates the data." It is cleanly
#       MONOTONE: more robust (smaller θ) ⇒ chooses to carry more wealth — the
#       precautionary effect.
#
#   (2) The own-measure stationary mean wealth — the mean of the model's own
#       Λ. This is NON-monotone in θ, and deliberately reported as such: the
#       `LogitChoiceStage.forward!` pushes Λ through the seated WORST-CASE
#       kernel π(j|i) ∝ P[i,j]·exp(−V_end[j]/θ), so the ergodic income measure
#       gets more pessimistic exactly as the policy gets more cautious. The two
#       forces (cautious policy ↑ wealth; pessimistic simulated income ↓
#       resources) move oppositely, so the own-measure mean is not a clean
#       precaution read. (1) is the right object; (2) is shown for honesty and
#       to flag a genuine framework characteristic — see README.
#
# The θ = 1e8 solve is the risk-neutral limit and is cross-checked against the
# ordinary-expectation `MarkovStage` reference chain (matches to ~1e-7).

include("model.jl")

using Printf

"""
Solve the robust-savings steady state at robustness multiplier `θ` (logit scale
`ε = −θ`) and return the savings policy, the own-measure mean wealth, mean V,
and iteration counts.
"""
function hansen_sargent_solve(p = hansen_sargent_params; θ::Float64 = 1.0)
    hh  = hansen_sargent_household(p; θ)
    env = hansen_sargent_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    bprime = hansen_sargent_savings_policy(hh, p)
    return (; θ, bprime, mean_wealth = m.mean_wealth, mean_V = sum(res.V) / length(res.V),
              Λ = res.Λ, history = res.history)
end

"""
Solve the ordinary-expectation reference (non-robust) `MarkovStage(:income)`
steady state — supplies both the θ→∞ mean-wealth cross-check and the common
reference distribution `Λref` under which the precautionary policy statistic is
evaluated.
"""
function reference_solve(p = hansen_sargent_params)
    hh  = reference_household(p)
    env = hansen_sargent_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    return (; mean_wealth = m.mean_wealth, Λ = res.Λ, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = hansen_sargent_params
    println("Hansen–Sargent robust savings — precautionary comparative static in θ = |ε|")
    @printf "  β = %.3f, r = %.3f (1/β−1 = %.4f), σ = %.1f, income = %s\n" p.β p.r (1/p.β-1) p.σ string(p.y_grid)

    ref  = reference_solve(p)                       # non-robust: cross-check + reference distribution
    Λref = ref.Λ
    mean_bprime_ref(b) = sum(b .* Λref) / sum(Λref)  # average chosen next-wealth under the common measure

    θ_grid  = [0.5, 1.0, 2.0, 10.0, 1e8]
    results = [hansen_sargent_solve(p; θ) for θ in θ_grid]

    println("\n  θ = |ε|     E_ref[b'(state)]   mean V      own-measure ΣΛ-mean wealth   (VFI, Λ iters)")
    println("  " * "-"^88)
    for r in results
        @printf "  %-10.4g  %12.4f      %9.4f     %12.4f                 (%d, %d)\n" r.θ mean_bprime_ref(r.bprime) r.mean_V r.mean_wealth r.history.vfi_iters r.history.lambda_iters
    end

    println()
    bp = [mean_bprime_ref(r.bprime) for r in results]
    @printf "Precautionary policy effect : E_ref[b'] = %.4f at θ=0.5 vs %.4f at θ=1e8  →  +%.2f%% more wealth carried\n" bp[1] bp[end] 100*(bp[1]/bp[end]-1)
    @printf "Monotone DECREASING in θ (more robust ⇒ saves more) : %s\n" all(diff(bp) .< 0)
    @printf "Sanity (θ→∞ limit) : robust mean wealth at θ=1e8 = %.6f  vs MarkovStage reference = %.6f  (|Δ| = %.2e)\n" results[end].mean_wealth ref.mean_wealth abs(results[end].mean_wealth - ref.mean_wealth)
end
