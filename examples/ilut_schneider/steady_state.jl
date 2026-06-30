###################################################################
# Ilut–Schneider ambiguous business cycles — comparative static    #
###################################################################
#
# Partial equilibrium: a single inner V/Λ solve per ambiguity multiplier
# θ = |ε|. We sweep θ from highly ambiguous (θ = 0.5) to near-expected-utility
# (θ = 1e8) and report three objects:
#
#   (1) The PRECAUTIONARY policy statistic E_ref[b'(state)] — average chosen
#       next-period wealth under a COMMON fixed reference distribution (the
#       non-ambiguous stationary Λ). A pure policy read, cleanly MONOTONE: more
#       ambiguity (smaller θ) ⇒ carries more wealth.
#
#   (2) The ergodic RECESSION SHARE under the household's own (worst-case)
#       cycle measure. Here the worst-case forward push is the economically
#       interesting object: ambiguity tilts the ergodic cycle distribution
#       toward recessions, so the recession share RISES as θ falls — the
#       Ilut–Schneider "ambiguous business cycles" amplification.
#
#   (3) The θ = 1e8 limit, cross-checked against the ordinary-expectation
#       `MarkovStage(:z)` reference chain (mean wealth and recession share).
#
# The smoothed entropic worst-case operator is what we build; the literal
# set-based max-min is its θ → 0⁺ (ε → 0⁻) limit (catalog status ◐: entropic
# form ✅).

include("model.jl")

using Printf

"""
Solve the ambiguous-cycle steady state at ambiguity multiplier `θ` (logit scale
`ε = −θ`); return the savings policy, own-measure mean wealth and recession
share, and iteration counts.
"""
function ilut_schneider_solve(p = ilut_schneider_params; θ::Float64 = 1.0)
    hh  = ilut_schneider_household(p; θ)
    env = ilut_schneider_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    bprime = ilut_schneider_savings_policy(hh, p)
    return (; θ, bprime, mean_wealth = m.mean_wealth, recession_share = m.recession_share,
              Λ = res.Λ, history = res.history)
end

"""
Solve the non-ambiguous `MarkovStage(:z)` reference steady state — supplies the
θ→∞ cross-check and the reference distribution for the precautionary statistic.
"""
function ilut_schneider_reference_solve(p = ilut_schneider_params)
    hh  = ilut_schneider_reference(p)
    env = ilut_schneider_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    return (; mean_wealth = m.mean_wealth, recession_share = m.recession_share, Λ = res.Λ)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = ilut_schneider_params
    println("Ilut–Schneider ambiguous business cycles — comparative static in θ = |ε|")
    @printf "  β = %.3f, r = %.3f, σ = %.1f;  cycle P_z = %s\n" p.β p.r p.σ string(p.P_z[1,:]) * "/" * string(p.P_z[2,:])

    # Reference (no ambiguity): cross-check + common reference distribution.
    ref  = ilut_schneider_reference_solve(p)
    Λref = ref.Λ
    mean_bprime_ref(b) = sum(b .* Λref) / sum(Λref)

    θ_grid  = [0.5, 1.0, 2.0, 10.0, 1e8]
    results = [ilut_schneider_solve(p; θ) for θ in θ_grid]

    println("\n  θ = |ε|     E_ref[b'(state)]   worst-case recession share   own-measure mean wealth   (VFI, Λ)")
    println("  " * "-"^92)
    for r in results
        @printf "  %-10.4g  %12.4f      %16.4f          %14.4f             (%d, %d)\n" r.θ mean_bprime_ref(r.bprime) r.recession_share r.mean_wealth r.history.vfi_iters r.history.lambda_iters
    end
    @printf "\n  reference (MarkovStage cycle, ordinary E): recession share = %.4f, mean wealth = %.4f\n" ref.recession_share ref.mean_wealth

    bp = [mean_bprime_ref(r.bprime) for r in results]
    rs = [r.recession_share for r in results]
    println()
    @printf "Precautionary policy effect : E_ref[b'] = %.4f at θ=0.5 vs %.4f at θ=1e8  →  +%.2f%% more wealth carried\n" bp[1] bp[end] 100*(bp[1]/bp[end]-1)
    @printf "  monotone DECREASING in θ (more ambiguity ⇒ saves more)        : %s\n" all(diff(bp) .< 0)
    @printf "Amplification : worst-case recession share %.4f (θ=0.5) vs %.4f (θ=1e8) vs %.4f (reference)\n" rs[1] rs[end] ref.recession_share
    @printf "  monotone DECREASING in θ (more ambiguity ⇒ more recession mass): %s\n" all(diff(rs) .< 0)
    @printf "Sanity (θ→∞ limit) : θ=1e8 mean wealth = %.6f vs reference %.6f (|Δ|=%.2e); recession share %.6f vs %.6f (|Δ|=%.2e)\n" results[end].mean_wealth ref.mean_wealth abs(results[end].mean_wealth-ref.mean_wealth) results[end].recession_share ref.recession_share abs(results[end].recession_share-ref.recession_share)
end
