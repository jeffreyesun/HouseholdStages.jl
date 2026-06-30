###################################################################
# Aiyagari with endogenous labor — tatonnement on κ = K/L          #
###################################################################

# With GHH endogenous labor BOTH factor markets must clear: capital `K` and
# effective labor `L = ∫ ε·n*(w,ε) dΛ`. But the firm's two prices `(r, w)` are
# pinned by the SINGLE capital-labor ratio `κ = K/L`, so the equilibrium reduces
# to a 1-D fixed point: guess `κ`, price it, solve the household block, read off
# capital supply `A = ∫ wealth` and labor supply `L_s = ∫ ε·n*`, and update
# `κ → A / L_s`. (`L_s` is endogenous because `n*` rises with the wage that `κ`
# sets.) The per-κ inner V/Λ solve is delegated to
# `solve_steady_state_given_env!`; the outer loop is rolled here, as in
# `../aiyagari/steady_state.jl`.

include("model.jl")

using Printf

"""
Solve the Aiyagari-with-endogenous-labor steady state by damped tatonnement on the
capital-labor ratio `κ = K/L`. Each pass prices `κ`, runs the inner V/Λ solve, reads
capital supply `A = ∫ wealth` and effective-labor supply `L_s = ∫ ε·n*`, and nudges
`κ` toward the implied `A / L_s`. Returns the cleared `(κ, K, L, r, w)`, the value /
distribution, and aggregate / mean hours.
"""
function labor_supply_steady_state(p = labor_supply_params;
                                   κ_init       = 5.5,
                                   update_speed = 0.05,
                                   rtol         = 1e-2,
                                   max_iter     = 200,
                                   verbosity    = 1)
    hh = labor_supply_household(p)

    κ = κ_init
    iterations = 0
    κ_err = Inf
    V, Λ = nothing, nothing

    while iterations < max_iter
        env = (; labor_supply_prices(κ, p)...)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ, history) = res; (; vfi_iters, lambda_iters) = history

        m   = compute_moments(hh, Λ, env)
        A   = m.K_supplied
        L_s = m.L_supplied
        κ_implied = A / L_s
        κ_err = abs(κ_implied - κ) / κ
        iterations += 1

        verbosity > 0 && @printf "  iter %d: κ = %.4f → A = %.4f, L = %.4f, κ_implied = %.4f, κ_err = %.6f (VFI %d, Λ %d)\n" iterations κ A L_s κ_implied κ_err vfi_iters lambda_iters

        κ_err <= rtol && break
        κ += update_speed * (κ_implied - κ)
    end

    converged = κ_err <= rtol
    converged || @warn "labor_supply tatonnement stuck at tolerance $rtol after $iterations iterations."

    env = (; labor_supply_prices(κ, p)...)
    m   = compute_moments(hh, Λ, env)
    mass = sum(Λ)
    (; κ, K = m.K_supplied, L = m.L_supplied,
       env.r, env.w, V, Λ,
       mean_wealth = m.K_supplied / mass,
       mean_hours  = m.hours / mass,
       iterations, converged)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = labor_supply_params
    println("Solving Aiyagari-with-endogenous-labor (GHH) steady state…")
    @time res = labor_supply_steady_state(p)
    println(res.converged ? "Converged in $(res.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(res.iterations) outer iterations.")
    @printf "  κ = K/L        = %.4f\n" res.κ
    @printf "  K (capital)    = %.4f\n" res.K
    @printf "  L (eff. labor) = %.4f\n" res.L
    @printf "  r              = %.4f\n" res.r
    @printf "  w              = %.4f\n" res.w
    @printf "  mean wealth    = %.4f\n" res.mean_wealth
    @printf "  mean hours     = %.4f\n" res.mean_hours
    @printf "  ΣΛ (mass)      = %.6f\n" sum(res.Λ)
    @printf "  V finite       = %s\n" all(isfinite, res.V)

    # Hours respond to the productivity state: n*(w·ε) rises with ε at the cleared wage.
    println("  hours by productivity state ε (at cleared w = $(round(res.w, digits=4))):")
    for ε in p.y_grid
        @printf "    ε = %.2f  →  n* = %.4f\n" ε n_star(res.w * ε, p)
    end
end
