######################################################################
# JMP household block — steady state (partial equilibrium)            #
######################################################################

# Returns, rents, and the house price are exogenous, so there is no
# market to clear: the "outer loop" is a single inner V/Λ fixed-point
# solve at the given env. The GE tatonnement on aggregate capital /
# housing and the aggregate-shock loop that close the full JMP model
# live OUTSIDE the household block and are not built here — the point
# of this example is that the household block itself is library stages
# only (see `model.jl`).

include("model.jl")

using Printf

"""
Solve the JMP household block to its stationary `(V, Λ)` at the exogenous env and
report mass, wealth, the housing stock, the homeownership rate, and the home-
location population share.
"""
function jmp_steady_state(p = jmp_params; verbosity = 1)
    hh  = jmp_household(p)
    env = jmp_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "JMP household steady state (r = %.3f, r_m = %.3f, q = %.2f)\n" p.r p.r_m p.q
        @printf "  mass(Λ)          = %.8f\n" sum(res.Λ)
        @printf "  all V finite     = %s\n" all(isfinite, res.V)
        @printf "  mean wealth      = %.4f\n" m.mean_wealth
        @printf "  mean house size  = %.4f\n" m.mean_house
        @printf "  ownership rate   = %.4f\n" m.own_rate
        @printf "  pop share :home  = %.4f\n" m.pop_home
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ,
              mean_wealth = m.mean_wealth, mean_house = m.mean_house,
              own_rate = m.own_rate, pop_home = m.pop_home,
              history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving JMP household-block steady state…")
    @time jmp_steady_state()
end
