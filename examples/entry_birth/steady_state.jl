###################################################################
# Entry / Birth Demonstrator — Population Mass under Two Regimes   #
###################################################################

# Partial equilibrium (prices `(r, w)` fixed). The forward mass map is
# the affine recursion `M_{t+1} = s·M_t + Σg`, fixed point `Σg/(1−s)`.
# We exhibit two regimes:
#
#   (i)  replacement  `Σg = 1−s`     ⇒ stationary mass ≈ 1
#   (ii) birth-heavy  `Σg = 2(1−s)`  ⇒ stationary mass ≈ 2 (a LEVEL shift)
#
# For (ii) we also print the transient mass trajectory: starting from the
# replacement steady state (mass 1) and iterating the block's own
# `forward!`, the population climbs toward `Σg/(1−s) = 2`. Because the
# birth source is a fixed additive inflow (not proportional to the
# current mass), the per-pass mass ratio relaxes to 1 as the level
# approaches its fixed point — affine convergence, not geometric growth.

include("model.jl")

using Printf

"""
Solve the stationary distribution for a given newborn mass `birth_mass`
at fixed prices `(r, w)`. Returns the seated household block plus the
converged `(V, Λ)` and the population / mean-wealth moments.
"""
function entry_birth_steady_state(p = entry_birth_params;
                                  birth_mass = 1 - p.s, r = 0.03, w = 1.0)
    hh  = entry_birth_household(p; birth_mass)
    env = (; r, w)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    return (; hh, env, V = res.V, Λ = res.Λ,
              pop = m.pop, mean_wealth = m.A_total / m.pop)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = entry_birth_params
    println("Entry/birth demonstrator — population mass under two regimes")
    @printf "  survival s = %.3f  (death hazard 1−s = %.3f)\n\n" p.s (1 - p.s)

    # Regime (i): replacement, Σg = 1−s ⇒ stationary mass ≈ 1.
    @time res_i = entry_birth_steady_state(p; birth_mass = 1 - p.s)
    println("Regime (i) — replacement  Σg = 1−s")
    @printf "    stationary mass ΣΛ = %.6f   (target 1)\n"   res_i.pop
    @printf "    mean wealth E[w]   = %.4f\n\n"              res_i.mean_wealth

    # Regime (ii): birth-heavy, Σg = 2(1−s) ⇒ stationary mass ≈ 2.
    birth2 = 2 * (1 - p.s)
    res_ii = entry_birth_steady_state(p; birth_mass = birth2)
    M_star = birth2 / (1 - p.s)
    println("Regime (ii) — birth-heavy  Σg = 2(1−s)")
    @printf "    stationary mass ΣΛ = %.6f   (analytic Σg/(1−s) = %.4f)\n" res_ii.pop M_star
    @printf "    mean wealth E[w]   = %.4f\n\n"              res_ii.mean_wealth

    # Transient: start from the replacement steady state (mass 1) and
    # iterate the birth-heavy block's own forward map. The population
    # climbs from 1 toward Σg/(1−s) = 2; the per-pass ratio relaxes to 1.
    println("Regime (ii) transient — mass trajectory from M₀ = 1 (block's own forward!):")
    let Λ = copy(res_i.Λ), prev = sum(copy(res_i.Λ))
        @printf "    pass %2d:  M = %.6f\n" 0 prev
        for t in 1:12
            Λ = forward!(res_ii.hh, Λ)
            M = sum(Λ)
            @printf "    pass %2d:  M = %.6f   (M_t/M_{t-1} = %.5f)\n" t M (M / prev)
            prev = M
        end
    end
    @printf "\n  ⇒ birth ≠ death moves the stationary population: %.4f vs %.4f.\n" res_i.pop res_ii.pop
end
