######################################################################
# Diamond city choice — amenity fixed point (endogenous sorting)      #
######################################################################

# The household block (city choice) is exercised inside an OUTER loop that
# closes the ENDOGENOUS-AMENITY fixed point: a city's amenity rises with its
# high-skill population share,
#
#     A[c] = amenity_base[c] + spillover · (high-skill share in city c),
#
# so high-skill inflows raise amenities, which attract more high skill — the
# Diamond sorting feedback. Wages and rents are held exogenous (a full GE
# would also clear those in this same outer loop). The per-amenity inner work
# (V backward + Λ forward) is delegated to `solve_steady_state_given_env!`;
# the amenity iteration is a damped fixed point rolled here.

include("model.jl")

using Printf

"""
Solve the Diamond city-choice steady state by damped iteration on the city
amenity vector. At each pass: solve the household block at the current
amenities, recompute amenities from the realized high-skill shares, and damp.
"""
function diamond_steady_state(p = params;
                              damping  = 0.5,
                              tol      = 1e-4,
                              maxiter  = 200,
                              verbosity = 1)
    hh = diamond_household(p)
    nC = length(p.cities)

    A = copy(p.amenity_base)         # initial amenity guess
    V, Λ = nothing, nothing
    moments = nothing
    iters = 0
    converged = false

    while iters < maxiter
        env = diamond_env(A, p)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ) = res
        moments = compute_moments(hh, Λ, env)

        pop  = [moments.pop_rustbelt, moments.pop_sunbelt, moments.pop_hub]
        hi   = [moments.hi_rustbelt,  moments.hi_sunbelt,  moments.hi_hub]
        hi_share = hi ./ max.(pop, eps())
        A_new = p.amenity_base .+ p.amenity_spillover .* hi_share

        gap = maximum(abs.(A_new .- A))
        iters += 1
        verbosity > 0 && (iters <= 4 || iters % 10 == 0) &&
            @printf "  iter %d: amenity gap = %.6f, A = [%.3f, %.3f, %.3f]\n" iters gap A...

        if gap < tol
            converged = true
            A = A_new
            break
        end
        A = (1 - damping) .* A .+ damping .* A_new
    end

    converged || @warn "Diamond amenity fixed point not converged in $iters iterations."
    return (; A, V, Λ, moments, iters, converged)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Diamond city-choice steady state (endogenous amenities)…")
    @time res = diamond_steady_state()
    m = res.moments
    println(res.converged ? "Amenity fixed point converged in $(res.iters) iterations." :
                            "DID NOT CONVERGE in $(res.iters) iterations.")
    pop  = [m.pop_rustbelt, m.pop_sunbelt, m.pop_hub]
    hi   = [m.hi_rustbelt,  m.hi_sunbelt,  m.hi_hub]
    println("  Endogenous amenities A = [$(join(round.(res.A; digits=3), ", "))]")
    println("  Population by city:  rustbelt=$(round(pop[1];digits=3))  sunbelt=$(round(pop[2];digits=3))  hub=$(round(pop[3];digits=3))")
    @printf "  High-skill share by city: rustbelt=%.3f  sunbelt=%.3f  hub=%.3f\n" (hi ./ max.(pop, eps()))...
    @printf "  K_supplied = %.4f,  ΣΛ = %.6f\n" m.K_supplied sum(res.Λ)
end
