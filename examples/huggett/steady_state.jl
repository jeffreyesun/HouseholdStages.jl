############################################################
# Huggett Steady State — Bisection on r (bond clears to 0) #
############################################################

# Bisection on the bond interest rate r. The per-r inner work (V backward
# to fixed point + Λ forward to stationarity) is delegated to
# `HouseholdStages.solve_steady_state_given_env!`; the outer
# bond-market-clearing loop is rolled here.
#
# The bond is in ZERO NET SUPPLY, so equilibrium r is where aggregate
# bond demand A_supplied(r) = ∫ a dΛ clears to zero. A_supplied(r) is
# increasing in r (higher returns ⇒ more saving / less borrowing), so a
# bracket [r_lo, r_hi] with A_supplied(r_lo) < 0 < A_supplied(r_hi) is
# refined by bisection. The natural upper bound is r < 1/β − 1 (above it
# everyone wants to save without bound and demand never clears). This
# mirrors the Aiyagari tatonnement but clears assets-to-zero rather than
# capital-to-demand; per the library's principle the close-the-model loop
# lives with the consumer, not in src/.

include("model.jl")

using Printf

"""
Solve the Huggett bond-economy steady state by bisection on the interest
rate r so that aggregate bond demand `A_supplied(r) = ∫ a dΛ` clears to
zero (the bond is in zero net supply).

`A_supplied` is increasing in r; the loop brackets the root in
`[r_lo, r_hi]` (default below `1/β − 1`) and bisects, reusing the per-r
inner V/Λ solve via `solve_steady_state_given_env!`. Warm-starts each
inner solve from the previous (V, Λ).
"""
function huggett_steady_state(p = huggett_params;
                              r_lo      = -0.02,
                              r_hi      = 1 / p.β - 1 - 1e-3,
                              atol      = 1e-3,
                              max_iter  = 60,
                              verbosity = 1)
    hh = huggett_household(p)

    V, Λ = nothing, nothing   # warm-start placeholders across r values

    "Inner solve at rate r; returns (A_supplied, V, Λ)."
    function demand_at(r)
        env = huggett_env(r, p)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ) = res
        A = compute_moments(hh, Λ, env).A_supplied
        return A
    end

    A_lo = demand_at(r_lo)
    A_hi = demand_at(r_hi)
    verbosity > 0 && @printf "  bracket: A(r=%.4f) = %.4f, A(r=%.4f) = %.4f\n" r_lo A_lo r_hi A_hi

    if A_lo > 0 || A_hi < 0
        @warn "Bond demand not bracketed on [$r_lo, $r_hi]: A_lo=$A_lo, A_hi=$A_hi. Returning closer endpoint."
        r = abs(A_lo) < abs(A_hi) ? r_lo : r_hi
        A = abs(A_lo) < abs(A_hi) ? A_lo : A_hi
        return (; r, A_supplied = A, V, Λ, iterations = 0, converged = false)
    end

    r = (r_lo + r_hi) / 2
    A = Inf
    iterations = 0
    residual_history = Float64[]

    while iterations < max_iter
        r = (r_lo + r_hi) / 2
        A = demand_at(r)
        push!(residual_history, abs(A))
        iterations += 1

        verbosity > 0 && @printf "  iter %d: r = %.5f → A_supplied = %.6f  [%.5f, %.5f]\n" iterations r A r_lo r_hi

        abs(A) <= atol && break

        A > 0 ? (r_hi = r) : (r_lo = r)   # demand increasing in r
    end

    converged = abs(A) <= atol
    converged || @warn "Huggett bisection stuck at tolerance $atol after $iterations iterations."

    return (; r, A_supplied = A, V, Λ,
              iterations, converged, residual_history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Huggett bond-economy steady state…")
    @time res = huggett_steady_state()
    println(res.converged ? "Converged in $(res.iterations) bisection steps." :
                            "DID NOT CONVERGE in $(res.iterations) bisection steps.")
    @printf "  r          = %.5f   (1/β − 1 = %.5f)\n" res.r (1 / huggett_params.β - 1)
    @printf "  A_supplied = %.6f   (target ≈ 0)\n" res.A_supplied
    @printf "  ΣΛ         = %.6f\n" sum(res.Λ)
end
