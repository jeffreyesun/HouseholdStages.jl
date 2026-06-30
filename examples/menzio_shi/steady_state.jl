#######################################################################
# Menzio–Shi on-the-job directed search — partial-equilibrium solve    #
#######################################################################

# Prices `(r, b)` and the fill schedule `f(·)` (the free-entry tightness
# summary) are exogenous, so there is no market to clear: the "outer loop" is
# a single inner V/Λ fixed-point solve at the given env. The point is that the
# household block — savings AND on-the-job directed search with a fall-back to
# the current wage — is library stages only; see `model.jl`.

include("model.jl")

using Printf

"""
Solve the Menzio–Shi on-the-job directed-search steady state at exogenous
`(r, benefit)` and report wealth, the unemployment rate, the mean employed
wage, and the wage-tier distribution of the employed (the OTJ ladder).
"""
function menzio_shi_steady_state(p = params; verbosity = 1)
    hh  = menzio_shi_household(p)
    env = menzio_shi_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    emp_rate  = 1 - m.unemp_rate
    mean_wage = m.wage_bill / emp_rate

    # Λ is (wealth, submarket, target). Marginal over :submarket = the wage ladder.
    sub = submarket_levels(p)
    by_sub = vec(sum(res.Λ; dims = (1, 3)))   # mass at each submarket level
    by_sub ./= sum(by_sub)

    if verbosity > 0
        @printf "Menzio–Shi OTJ directed search (r = %.3f, b = %.2f, ε = %.3f, sep = %.2f)\n" p.r p.benefit p.ε p.sep
        @printf "  mass(Λ)        = %.6f\n"  sum(res.Λ)
        @printf "  mean wealth    = %.4f\n"  m.mean_wealth
        @printf "  unemp rate     = %.4f\n"  m.unemp_rate
        @printf "  mean wage(emp) = %.4f  (posted-wage range [%.2f, %.2f])\n" mean_wage first(p.wages) last(p.wages)
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
        println("  submarket       : ", "unemp  ", join((@sprintf("%.2f", w) for w in p.wages), "  "))
        println("  fill prob f(w)  : ", "       ", join((@sprintf("%.2f", fill_prob(w, p)) for w in p.wages), "  "))
        println("  mass at tier    : ", join((@sprintf("%.3f", x) for x in by_sub), "  "))
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              unemp_rate = m.unemp_rate, mean_wage, by_sub, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Menzio–Shi on-the-job directed-search steady state…")
    @time menzio_shi_steady_state()
end
