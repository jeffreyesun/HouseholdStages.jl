##################################################################
# HHJK occupational choice — stationary steady state              #
##################################################################

# Partial equilibrium: occupation wages clear in the talent-allocation GE
# (the caller's outer loop). Here we solve the household block at a fixed
# `env` and report the wedge-driven occupational segregation between groups.

include("model.jl")

using Printf

"Solve the HHJK occupation-choice household block at fixed wages."
function hhjk_steady_state(p = params)
    hh  = hhjk_household(p)
    env = hhjk_env(p)
    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    moments = compute_moments(hh, Λ, env)
    return (; Λ, V, moments, env, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving HHJK occupational-choice steady state…")
    @time res = hhjk_steady_state()
    m = res.moments
    @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    @printf "  ΣΛ          = %.6f\n" sum(res.Λ)
    println("  MISALLOCATION — occupation shares within each group (each group has mass 1/2):")
    @printf "    skilled : advantaged = %.4f,  disadvantaged = %.4f\n" m.skilled_adv m.skilled_dis
    @printf "    home    : advantaged = %.4f,  disadvantaged = %.4f\n" m.home_adv    m.home_dis
    println("  (the wedge pushes the disadvantaged group OUT of the skilled occupation)")
end
