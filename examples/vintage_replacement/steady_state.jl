###################################################
# Vintage replacement — partial-equilibrium steady state #
###################################################

# Returns, wage, and the technology parameters are exogenous, so there is no
# market to clear: the "outer loop" is a single inner V/Λ fixed-point solve at
# the given env. The whole point is that the household block is library stages
# only (regenerative stopping via the auxiliary-choice-axis pattern) — see
# `model.jl`.
#
# The per-period ADOPTION RATE is not a moment of the end-of-period distribution
# (the `:adopt_choice` axis is collapsed by `Forget`), so the driver reads it off
# the `Choose` policy weighted by the distribution ENTERING the choice — exactly
# the kind of intermediate-distribution bookkeeping the driver is meant to own.

include("model.jl")

using Printf

"""
Per-period adoption (replacement) rate: the mass that chooses ADOPT on entering the period.
The stationary distribution `Λ_ss` is the chain's start-of-period law (input = output layout at the
fixed point); push it through the leading `IncomeShock` to get the distribution entering `Choose`,
then weight the `Choose` policy (`2` = adopt) by that distribution.
"""
function adoption_rate(hh, Λ_ss)
    income_stage = hh.buffer.stages[1]                       # IncomeShock
    choose_stage = hh.buffer.stages[2]                       # Choose (ArgmaxStage on :adopt_choice)
    Λ_pre = copy(forward!(income_stage, copy(Λ_ss)))         # distribution entering Choose
    pol   = HouseholdStages.policy(choose_stage)             # chosen index ∈ {1 = keep, 2 = adopt}
    return sum(Λ_pre[pol .== 2]) / sum(Λ_pre)
end

"""
Solve the vintage-replacement household steady state at the exogenous env and report the
mean vintage, the share at the top (newest) vintage, and the per-period adoption rate.
Returns the stationary `(V, Λ)`, the attached moments, and the adoption rate.
"""
function vintage_replacement_steady_state(p = vintage_replacement_params; verbosity = 1)
    hh  = vintage_replacement_household(p)
    env = vintage_replacement_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    adopt = adoption_rate(hh, res.Λ)

    if verbosity > 0
        @printf "Vintage-replacement steady state (β=%.2f, σ=%.1f, F=%.2f, π_dep=%.2f, θ=%.2f)\n" p.β p.σ p.F p.π_dep p.θ
        @printf "  mass(Λ)        = %.6f\n" sum(res.Λ)
        @printf "  mean vintage   = %.4f  (grid %s)\n" m.mean_vintage p.v_grid
        @printf "  top-vintage sh = %.4f\n" m.top_share
        @printf "  adoption rate  = %.4f  (per period)\n" adopt
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ,
              mean_vintage = m.mean_vintage, top_share = m.top_share,
              adoption_rate = adopt, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving vintage-replacement (regenerative stopping) steady state…")
    @time vintage_replacement_steady_state()
end
