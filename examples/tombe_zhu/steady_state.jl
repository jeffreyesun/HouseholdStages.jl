##################################################################
# Tombe–Zhu region×sector migration — stationary steady state     #
##################################################################

# Partial equilibrium: the region×sector real wages (which embed trade) and
# the spatial measure clear in the trade-and-migration GE (the caller's
# outer loop). Here we solve the household block at a fixed `env`.

include("model.jl")

using Printf

"Solve the Tombe–Zhu composite-migration household block at fixed prices."
function tombe_zhu_steady_state(p = params)
    hh  = tombe_zhu_household(p)
    env = tombe_zhu_env(p)
    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    moments = compute_moments(hh, Λ, env)
    return (; Λ, V, moments, env, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Tombe–Zhu region×sector migration steady state…")
    @time res = tombe_zhu_steady_state()
    m = res.moments
    @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    @printf "  ΣΛ          = %.6f\n" sum(res.Λ)
    @printf "  K_supplied  = %.4f\n" m.K_supplied
    println("  Population by (region, sector):")
    @printf "    (coast, ag)     = %.4f  (w %.2f)\n" m.pop_coast_ag  params.real_wage[1]
    @printf "    (coast, mfg)    = %.4f  (w %.2f)\n" m.pop_coast_mfg params.real_wage[2]
    @printf "    (interior, ag)  = %.4f  (w %.2f)\n" m.pop_int_ag    params.real_wage[3]
    @printf "    (interior, mfg) = %.4f  (w %.2f)\n" m.pop_int_mfg   params.real_wage[4]
end
