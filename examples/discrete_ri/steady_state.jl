###################################################################
# Discrete-RI steady state — Shannon-cost occupation choice        #
###################################################################

# Returns are exogenous, so there is no price to clear: at a FIXED attention
# prior `q` the model closes with a single inner V/Λ fixed-point solve
# (`solve_steady_state_given_env!`). The headline solve fixes a uniform prior
# `q ∝ 1` — the canonical Matějka–McKay reference point at which RI is the
# plain temperature-λ logit and the comparative static in λ is cleanest: as
# the Shannon cost λ rises, attention is costlier and the posterior collapses
# toward the (uniform) prior, so the occupation shares compress toward 1/n.
#
# The ENDOGENOUS-prior fixed point (the full Matějka–McKay consistency
# condition `q(a) = ∫ 1{occ = a} dΛ`) is also provided
# (`discrete_ri_endogenous_prior`): the prior the strategy is optimized
# against must equal the realized choice share. It is a damped iteration on q
# around the same inner solve — the discrete-RI analogue of tatonnement,
# rolled with the consumer per the library's "close-the-model loop belongs to
# the consumer" principle. It tends to a corner when one occupation dominates
# (the well-known endogenous-prior degeneracy), so the fixed-prior solve is
# the headline. See `model.jl`.

include("model.jl")

using Printf

"""
Marginal occupation share of `Λ`: `q(a) = ∫ 1{occupation = a} dΛ` (the
occupation axis is the second of the (income, occupation) layout). This is
the realized unconditional choice distribution.
"""
occupation_share(Λ) = vec(sum(Λ; dims = 1))

"""
Solve the discrete-RI occupation-choice steady state at a FIXED attention
prior `q` (default uniform). One inner V/Λ solve at `env = (; λ, q)`; reports
occupation shares, mean income, and the RI choice policy. Returns the
stationary `(V, Λ)`, the moments, and the per-state choice probabilities.
"""
function discrete_ri_steady_state(p = discrete_ri_params;
                                  λ = p.λ, q = fill(1 / length(p.premium), length(p.premium)),
                                  verbosity = 1)
    hh  = discrete_ri_household(p)
    env = (; λ, q)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    P   = ri_choice_probs(hh)                 # (income, origin-occ, dest-occ)

    if verbosity > 0
        @printf "Discrete-RI steady state (λ = %.3f, κ = %.2f, β = %.2f, prior q = [%.3f, %.3f, %.3f])\n" λ p.κ p.β q[1] q[2] q[3]
        @printf "  mass(Λ)            = %.6f\n"  sum(res.Λ)
        @printf "  occupation shares  = safe %.3f | persistent %.3f | career %.3f\n" m.safe_share m.persistent_share m.career_share
        @printf "  mean income        = %.4f\n"  m.mean_income
        @printf "  P(career | low inc,  origin safe) = %.3f\n" P[1, 1, 3]
        @printf "  P(career | high inc, origin safe) = %.3f\n" P[end, 1, 3]
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; V = res.V, Λ = res.Λ, q, mean_income = m.mean_income,
              safe_share = m.safe_share, persistent_share = m.persistent_share,
              career_share = m.career_share, choice_probs = P, history = res.history)
end

"""
Solve the full Matějka–McKay ENDOGENOUS-prior fixed point by damped iteration
on `q`: run the inner V/Λ solve at the current prior, read the realized
occupation share `q(a) = ∫ 1{occ = a} dΛ`, and nudge `q` toward it until the
consistency condition holds. Returns the converged prior and the solution.
"""
function discrete_ri_endogenous_prior(p = discrete_ri_params;
                                      λ = p.λ, update_speed = 0.5,
                                      tol = 1e-8, max_iter = 200, verbosity = 1)
    hh    = discrete_ri_household(p)
    n_occ = length(p.premium)
    q     = fill(1 / n_occ, n_occ)
    q_err = Inf; iters = 0; V, Λ = nothing, nothing

    while iters < max_iter
        env = (; λ, q)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ) = res
        q_new = occupation_share(Λ)
        q_err = maximum(abs, q_new .- q); iters += 1
        verbosity > 1 && @printf "  iter %d: q = [%.4f, %.4f, %.4f], q_err = %.2e\n" iters q[1] q[2] q[3] q_err
        q_err <= tol && (q = q_new; break)
        q = (1 - update_speed) .* q .+ update_speed .* q_new
    end

    converged = q_err <= tol
    converged || @warn "Discrete-RI endogenous-prior iteration stuck at tol $tol after $iters iterations."
    sol = discrete_ri_steady_state(p; λ, q, verbosity = 0)
    verbosity > 0 && @printf "Endogenous prior q* = [%.4f, %.4f, %.4f] (converged = %s, %d iters)\n" q[1] q[2] q[3] converged iters
    return (; sol..., q, iters, converged)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving discrete-RI occupation-choice steady state (fixed uniform prior)…")
    @time res = discrete_ri_steady_state()

    # The Matějka–McKay comparative static: as the Shannon cost λ rises,
    # attention is costlier, so the posterior collapses toward the uniform
    # prior — occupation shares compress toward 1/3 and the high-payoff career
    # share falls.
    println("\nComparative static — occupation shares vs. Shannon cost λ (fixed uniform prior):")
    for λ in (0.10, 0.30, 0.60, 1.20, 3.0, 8.0)
        r = discrete_ri_steady_state(; λ, verbosity = 0)
        @printf "  λ = %.2f  →  shares [safe %.3f, persistent %.3f, career %.3f], mean income %.4f\n" λ r.safe_share r.persistent_share r.career_share r.mean_income
    end

    println("\nEndogenous-prior fixed point (full Matějka–McKay consistency q = realized share):")
    discrete_ri_endogenous_prior(; verbosity = 1)
end
