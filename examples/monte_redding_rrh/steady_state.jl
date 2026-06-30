##################################################################
# MRRH commuting — partial-equilibrium stationary steady state    #
##################################################################

# Wages, rents, and the residential measure clear in the spatial GE
# (the caller's outer loop); here we solve the household block at a fixed
# `env`, exactly as `../sectoral/steady_state.jl` does. The within-period
# work (V backward + Λ forward to stationarity) is delegated to
# `HouseholdStages.solve_steady_state_given_env!`.

include("model.jl")

using Printf

"Solve the MRRH commuting household block at fixed prices."
function mrrh_steady_state(p = params)
    hh  = mrrh_household(p)
    env = mrrh_env(p)
    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    moments = compute_moments(hh, Λ, env)
    return (; Λ, V, moments, env, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving MRRH commuting stationary steady state…")
    @time res = mrrh_steady_state()
    m = res.moments
    @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    @printf "  ΣΛ          = %.6f\n" sum(res.Λ)
    @printf "  K_supplied  = %.4f\n" m.K_supplied
    println("  Employment by WORKPLACE (commute-destination share):")
    @printf "    west   = %.4f  (wage %.2f)\n" m.emp_west   params.wage[1]
    @printf "    center = %.4f  (wage %.2f)\n" m.emp_center params.wage[2]
    @printf "    east   = %.4f  (wage %.2f)\n" m.emp_east   params.wage[3]
end
