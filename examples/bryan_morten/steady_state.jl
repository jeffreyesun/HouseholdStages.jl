##################################################################
# Bryan–Morten internal migration — stationary steady state       #
##################################################################

# Partial equilibrium: location wages and the spatial measure clear in the
# regional GE (the caller's outer loop). Here we solve the discrete-choice
# migration household block at a fixed `env`, delegating the inner V/Λ solve
# to `HouseholdStages.solve_steady_state_given_env!`.

include("model.jl")

using Printf

"Solve the Bryan–Morten migration household block at fixed prices."
function bryan_morten_steady_state(p = params)
    hh  = bryan_morten_household(p)
    env = bryan_morten_env(p)
    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    moments = compute_moments(hh, Λ, env)
    return (; Λ, V, moments, env, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Bryan–Morten internal-migration steady state…")
    @time res = bryan_morten_steady_state()
    m = res.moments
    @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    @printf "  ΣΛ          = %.6f\n" sum(res.Λ)
    println("  Population by location:")
    @printf "    rural = %.4f,  town = %.4f,  city = %.4f\n" m.pop_rural m.pop_town m.pop_city
    println("  SELECTION — city population by ability type (each type has mass 1/3):")
    @printf "    low ability  in city = %.4f\n" m.city_lo
    @printf "    mid ability  in city = %.4f\n" m.city_mid
    @printf "    high ability in city = %.4f\n" m.city_hi
    println("  (positive selection ⇔ city_hi > city_mid > city_lo)")
end
