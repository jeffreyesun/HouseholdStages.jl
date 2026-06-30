#######################################################
# Roy Occupational Choice — Stationary Steady State    #
#######################################################

# Partial-equilibrium stationary solve at fixed sector wages and return.
# The inner V/Λ fixed-point work is delegated to
# `HouseholdStages.solve_steady_state_given_env!`; there is no outer
# tatonnement here (prices are exogenous). A market-clearing closure on
# the wages would be a plain outer loop in the style of
# `../spatial/steady_state.jl`, but is not needed to exercise the
# household block.

include("model.jl")

using Printf

"""
Solve the Roy occupational-choice stationary distribution at fixed prices:
run the inner VFI-to-fixed-point + Λ-to-stationarity solve once at the
exogenous env, then read out the per-sector employment shares, the wealth
held in each sector, and aggregate wealth.
"""
function sectoral_steady_state(p = params; verbosity::Int = 1)
    hh  = sectoral_household(p)
    env = sectoral_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    moments = compute_moments(hh, Λ, env)

    verbosity > 0 && @printf(
        "  VFI iters = %d, Λ iters = %d\n", history.vfi_iters, history.lambda_iters)

    return (; V, Λ, env, moments,
              vfi_iters = history.vfi_iters,
              lambda_iters = history.lambda_iters)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Roy occupational-choice stationary steady state…")
    @time res = sectoral_steady_state()
    m = res.moments
    pop_tot = m.pop_ag + m.pop_mfg + m.pop_svc
    @printf "  ΣΛ            = %.6f\n" sum(res.Λ)
    @printf "  K_supplied    = %.4f\n" m.K_supplied
    println("  Sector employment shares:")
    @printf "    :ag  = %.4f  (K = %.4f)\n" m.pop_ag / pop_tot  m.K_ag
    @printf "    :mfg = %.4f  (K = %.4f)\n" m.pop_mfg / pop_tot m.K_mfg
    @printf "    :svc = %.4f  (K = %.4f)\n" m.pop_svc / pop_tot m.K_svc
end
