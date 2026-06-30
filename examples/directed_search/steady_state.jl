###################################################
# Directed-search steady state — partial equilibrium #
###################################################

# Prices `(r, b)` and the fill schedule `f(·)` are exogenous (the Moen/Menzio–Shi
# free-entry tightness schedule is summarised by `f(w)`), so there is no market to
# clear: the "outer loop" is a single inner V/Λ fixed-point solve at the given env.
# (Contrast the Aiyagari/Krusell–Smith examples, which roll a tatonnement on K̄.)
# The whole point is that the household block — savings AND the directed-search
# submarket choice — is library stages only; see `model.jl`.

include("model.jl")

using Printf

"""
Solve the directed-search household steady state at the exogenous `(r, benefit)`
and report wealth, the unemployment rate, and the submarket-choice distribution.
Returns the stationary `(V, Λ)`, the attached moments, and the per-submarket mass
of the unemployed (the directed-search choice) and the employed.
"""
function directed_search_steady_state(p = directed_search_params;
                                      r = p.r, benefit = p.benefit, verbosity = 1)
    hh  = directed_search_household(p)
    env = (; r, benefit)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    emp_rate  = 1 - m.unemp_rate
    mean_wage = m.wage_bill / emp_rate

    # Submarket distribution of the unemployed (their directed-search choice) and
    # the employed (the wage they hold). Λ is (wealth, employment, submarket).
    unemp_by_sub = vec(sum(res.Λ[:, 1, :], dims = 1)); unemp_by_sub ./= sum(unemp_by_sub)
    emp_by_sub   = vec(sum(res.Λ[:, 2, :], dims = 1)); emp_by_sub   ./= sum(emp_by_sub)

    if verbosity > 0
        @printf "Directed-search steady state (r = %.3f, b = %.2f, ε = %.3f, sep = %.2f)\n" r benefit p.ε p.sep
        @printf "  mass(Λ)        = %.6f\n"  sum(res.Λ)
        @printf "  mean wealth    = %.4f\n"  m.mean_wealth
        @printf "  unemp rate     = %.4f\n"  m.unemp_rate
        @printf "  mean wage(emp) = %.4f  (posted-wage range [%.2f, %.2f])\n" mean_wage first(p.wages) last(p.wages)
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
        println("  posted wage     : ", join((@sprintf("%.2f", w) for w in p.wages), "  "))
        println("  fill prob f(w)  : ", join((@sprintf("%.2f", fill_prob(w, p)) for w in p.wages), "  "))
        println("  unemp search →  : ", join((@sprintf("%.2f", x) for x in unemp_by_sub), "  "))
        println("  employed at     : ", join((@sprintf("%.2f", x) for x in emp_by_sub), "  "))
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth,
              unemp_rate = m.unemp_rate, mean_wage,
              unemp_by_sub, emp_by_sub, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving directed-search steady state…")
    @time directed_search_steady_state()
end
