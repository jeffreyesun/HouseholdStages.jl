###################################################################
# One-sided Choo-Siow Partner Choice — stationary marriage market #
###################################################################

# Solve the stationary marriage-state distribution of the one-sided
# Choo-Siow searcher at a given exogenous partner-availability vector `n`.
# The inner V/Λ fixed-point solve is delegated to
# `HouseholdStages.solve_steady_state_given_env!`; there is no outer loop to
# roll here in the one-sided model — the partner side is exogenous, so a
# single inner solve at the env IS the steady state.
#
# (The TWO-sided Choo-Siow equilibrium WOULD add an outer loop: a fixed
# point on the availabilities `n[k]` that clears each partner type's
# matching market. That loop would live here, wrapping the inner solve
# exactly as Aiyagari's tatonnement on K wraps its inner solve. It is a
# known gap, not implemented — see model.jl's scope note.)

include("model.jl")

using Printf

"""
Solve the one-sided Choo-Siow stationary marriage market at the exogenous
partner-availability vector `n` (length = number of partner types). Returns
the stationary value `V` and distribution `Λ` over marital states, plus the
single share and overall match rate.
"""
function choo_siow_steady_state(n::AbstractVector = ones(length(params.partner_types)),
                                p = params)
    hh  = choo_siow_household(p)
    env = choo_siow_env(n, p)
    res = solve_steady_state_given_env!(hh, env)
    return (; V = res.V, Λ = res.Λ,
              moments = res.moments,
              states  = marital_states(p),
              Π       = match_payoff(n, p),
              n)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving one-sided Choo-Siow stationary marriage market…")
    n = ones(length(params.partner_types))          # symmetric partner availability
    @time res = choo_siow_steady_state(n)

    @printf "  ΣΛ            = %.6f\n" sum(res.Λ)
    @printf "  single share  = %.4f\n" res.moments.single_share
    @printf "  match rate    = %.4f\n" res.moments.match_rate
    println("  stationary marital distribution:")
    for (s, λ, π) in zip(res.states, res.Λ, vcat(NaN, res.Π))
        if s == :single
            @printf "    %-12s  Λ = %.4f\n" s λ
        else
            @printf "    %-12s  Λ = %.4f   (Π = %+.3f)\n" s λ π
        end
    end
end
