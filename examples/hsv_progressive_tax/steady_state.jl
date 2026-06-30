############################################################
# HSV Steady State — Tatonnement on K (GE Aiyagari + HSV tax) #
############################################################

# Damped tatonnement on aggregate capital K, copied from examples/aiyagari.
# The per-K inner work (V backward to fixed point + Λ forward to
# stationarity) is delegated to `solve_steady_state_given_env!`; the env
# carries the HSV tax parameters `λ, τ` alongside the cleared prices `r, w`.

include("model.jl")

using Printf

"""
Solve the HSV-tax Aiyagari steady state by damped tatonnement on aggregate
K: at each pass run the inner V/Λ solve at the current K (with the HSV tax
parameters `λ, τ` in `env`), and nudge K toward the implied supply by
`update_speed · (K_supplied − K)`. Same structure as `aiyagari_steady_state`.
"""
function hsv_steady_state(p = hsv_params;
                          K_init       = 5.0,
                          update_speed = 0.05,
                          rtol         = 2e-2,
                          max_iter     = 500,
                          verbosity    = 1)
    hh = hsv_household(p)

    K = K_init
    iterations = 0
    K_err = Inf
    residual_history = Float64[]
    V, Λ = nothing, nothing

    while iterations < max_iter
        env = (; K, λ = p.λ, τ = p.τ, hsv_prices(K, p)...)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ, history) = res; (; vfi_iters, lambda_iters) = history

        K_supplied = compute_moments(hh, Λ, env).K_supplied
        K_err = abs(K_supplied - K) / K
        push!(residual_history, K_err)
        iterations += 1

        verbosity > 0 && @printf "  iter %d: K = %.4f → K_supplied = %.4f, K_err = %.6f, VFI iters = %d, Λ iters = %d\n" iterations K K_supplied K_err vfi_iters lambda_iters

        K_err <= rtol && break
        K += update_speed * (K_supplied - K)
    end

    converged = K_err <= rtol
    converged || @warn "HSV tatonnement stuck at tolerance $rtol after $iterations iterations. Returning current K."

    (; r, w) = hsv_prices(K, p)
    return (; K, r, w, V, Λ, iterations, converged, residual_history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving HSV-tax Aiyagari steady state (λ = $(hsv_params.λ), τ = $(hsv_params.τ))…")
    @time res = hsv_steady_state()
    println(res.converged ? "Converged in $(res.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(res.iterations) outer iterations.")
    @printf "  K   = %.4f\n" res.K
    @printf "  r   = %.4f\n" res.r
    @printf "  w   = %.4f\n" res.w
    @printf "  ΣΛ  = %.6f\n" sum(res.Λ)
end
