#########################################################################
# Skill depreciation — stationary steady state (partial equilibrium)    #
#########################################################################

# Tightness θ, return r, wage scale w and benefit b_u are exogenous, so there is no
# market to clear: the "outer loop" is a single inner V/Λ fixed-point solve at the
# given env (cf. the insurance example). The decay mechanism is read off the stationary
# distribution: mean skill among the unemployed sits BELOW that among the employed.

include("model.jl")

using Printf

"""
Solve the skill-depreciation household at the exogenous env and report the employment
rate, mean wealth, and employment-conditional mean skill (employed vs unemployed — the
latter should be LOWER, the decay mechanism). The seated job-finding policy `p*` is
read from the `Matching` `MixingStage` leaf (search is a choice only for the
unemployed; the employed cells are degenerate at `p* = 0`).
"""
function skill_dep_steady_state(p = skill_dep_params; env = skill_dep_env(p), verbosity = 1)
    hh  = skill_dep_household(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = res.moments

    mean_skill_emp   = m.employment   > 0 ? m.skill_emp_sum   / m.employment   : NaN
    mean_skill_unemp = m.unemployment > 0 ? m.skill_unemp_sum / m.unemployment : NaN

    # Seated job-finding policy p*(wealth, skill) on the UNEMPLOYED slice, from the
    # Matching MixingStage leaf (stage 2; the employed cells are degenerate p* = 0).
    # The implied search effort inverts the matching technology p = 1 − exp(−A·e·θ):
    # e(p) = −log(1−p)/(A·θ) — the effort a worker would have to expend to buy p*.
    pstar         = HouseholdStages.policy(hh.buffer.stages[2])[:, :, 1]
    mean_pfind_u  = sum(pstar) / length(pstar)
    mean_effort_u = sum(-log1p.(-pstar) ./ (p.A_match * env.θ)) / length(pstar)

    if verbosity > 0
        @printf "Skill-depreciation steady state (θ = %.2f, r = %.3f, w = %.2f, b_u = %.2f)\n" env.θ env.r env.w env.b_u
        @printf "  mass(Λ)                 = %.6f\n"  sum(res.Λ)
        @printf "  employment rate         = %.4f\n"  m.employment
        @printf "  mean wealth             = %.4f\n"  m.mean_wealth
        @printf "  mean skill | employed   = %.4f\n"  mean_skill_emp
        @printf "  mean skill | unemployed = %.4f   (LOWER ⇒ decay mechanism)\n" mean_skill_unemp
        @printf "  mean p* | unemployed    = %.4f\n"  mean_pfind_u
        @printf "  implied effort | unemp  = %.4f\n"  mean_effort_u
        @printf "  VFI iters = %d, Λ iters = %d\n"    res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, employment = m.employment, mean_wealth = m.mean_wealth,
              mean_skill_emp, mean_skill_unemp, mean_pfind_u, mean_effort_u, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving skill-depreciation steady state…")
    @time skill_dep_steady_state()
end
